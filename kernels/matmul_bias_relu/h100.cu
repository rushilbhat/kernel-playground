#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    using  bias_vec       = sv_bf<64*N_BLOCK>;
    using  bias_layout    = gl<bf16, 1, 1, 1, -1, bias_vec>;
    struct globals        { 
        global_layout A, B, C; 
        bias_layout   bias;
    };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct scratch_block  { bias_vec bias; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, 64> accum[N_BLOCK]; };
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
      // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks) // 32*16 = 512
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M }; 
        else if (task_id < Rblocks*Cblocks) { // 512
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;  // 64
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.common.coord.y+i, args.iter}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            for (int n = 0; n < N_BLOCK; n++) 
                zero(args.state.accum[n]);
            // Every consumer warp in the CTA needs the same 256-element bias slice, so we have all eight consumer warps cooperate on that single load. 
            // group<NUM_CONSUMER_WARPS>::load just divides the slice into equal chunks: each thread gets its own subset of elements to copy into registers then shared memory 
            // (done under the hood with ld.global and st.shard).

            // Since args.scratch.bias is type sv_bf<64*N_BLOCK>, the coord passed as the third arg to load (i.e. starting idx of the load) must be coord<sv_bf<64*N_BLOCK>>.
            // Thus, we want the starting index of the slice into the bias to be in terms of tasks (i.e. width 256), which is why we divide by N_BLOCK.
            // [See defition of .unit_coord<-1, 3>() to get more clarity if confused]

            // To clarify why the slice of the bias vector is being loaded collaboratively into shared memory instead of each warp issuing the load for the entire slice in the epilogue:
            // If each warp just does its own global load in the epilogue, all eight warps load the same 256 bias values independently - identical work and unnecessary bandwidth.
            // Even if you tried to split the load warps cooperatively (each warp loads a subset into its registers), you’d still need a way for every warp to access all 256 entries
            // (remember registers are private to a thread).
            // By loading once into shared memory, we're doing a single global transfer, and then each warp grabs what it needs in the epilogue.
            int start_idx = args.common.coord.y / N_BLOCK;
            group<NUM_CONSUMER_WARPS>::load(args.scratch.bias, args.globals.bias, {start_idx});
            group<NUM_CONSUMER_WARPS>::sync(0);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            using wide_rt = rt_fl<16, 64*N_BLOCK>;
            using tall_st = st_bf<64*N_BLOCK, 64>;
            // dispatch the largest possible tensor core instruction to maximize TFLOPS (64x16x256 on M_BLOCK=2, N_BLOCK=4)
            warpgroup::mma_ABt(
                reinterpret_cast<wide_rt&>(args.state.accum),
                args.input.a[warpgroup::groupid()],
                reinterpret_cast<tall_st&>(args.input.b)
            );
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            for(int n = 0; n < N_BLOCK; n++) {
                // Logically carve out 64-scalar chunk, load into register vector, perform broadcasted-sum.
                auto &bias_sv = subvec_inplace<64>(args.scratch.bias, n);
                rt_fl<16, 64>::row_vec bias_rv;
                load(bias_rv, bias_sv);
                col_map<base_ops::sum>(args.state.accum[n], args.state.accum[n], bias_rv);
                // Apply ReLU activation
                relu(args.state.accum[n], args.state.accum[n]);
                warpgroup::store(args.finish.c[warpgroup::groupid()][n], args.state.accum[n]);
            }
            warpgroup::sync(warpgroup::groupid()+4);
            
            if(warpgroup::warpid() == 0) {
                for(int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                   {args.common.coord.x, args.common.coord.y+i});
                    tma::store_async_read_wait();
                }
            }

            // Zero the accumulators
            for(int n = 0; n < N_BLOCK; n++) {
                zero(args.state.accum[n]);
            }
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, bf16 *d_bias, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using bias_layout   = typename mmt::layout::bias_layout;
    using globals  = typename mmt::layout::globals;
    // printf("M: %d, N: %d, K: %d\n", M, N, K);
    global_layout Ag{d_A, nullptr, nullptr, M, K};
    global_layout Bg{d_B, nullptr, nullptr, N, K};
    global_layout Cg{d_C, nullptr, nullptr, M, N};
    bias_layout   biasg{d_bias, nullptr, nullptr, nullptr, N};
    globals G{Ag, Bg, Cg, biasg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

#ifdef TORCH_COMPILE
#include <torch/extension.h>

torch::Tensor matmul_bias_relu(torch::Tensor A, torch::Tensor B, torch::Tensor bias){
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M,N}, A.options());

    c10::BFloat16* A_ptr = A.data_ptr<c10::BFloat16>();
    c10::BFloat16* B_ptr = B.data_ptr<c10::BFloat16>();
    c10::BFloat16* C_ptr = C.data_ptr<c10::BFloat16>();
    c10::BFloat16* bias_ptr = bias.data_ptr<c10::BFloat16>();

    bf16* d_A = reinterpret_cast<bf16*>(A_ptr);
    bf16* d_B = reinterpret_cast<bf16*>(B_ptr);
    bf16* d_C = reinterpret_cast<bf16*>(C_ptr);
    bf16* d_bias = reinterpret_cast<bf16*>(bias_ptr);

    using mmt = matmul_template<2,4,8>;
    dim3 grid(mmt::grid(M,N,K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    inner_run<mmt>(d_A, d_B, d_C, d_bias, M, N, K, grid, block);
    // cudaDeviceSynchronize();
    return C;
}
#endif

#ifdef STANDALONE_COMPILE
constexpr bool NCU = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

void cpu_gemm_with_bias(float* a, float* b, float* bias, float* c, int M, int N, int K) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[j * K + k];
                // sum += a[i * K + k] * b[k * N + j];
            }
            float result = sum + bias[j];
            c[i * N + j] = std::max(result, 0.0f);
        }
    }
}

template<typename mmt>
int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cudaStatus;

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << mmt::M_BLOCK*64 << "x" << mmt::N_BLOCK*64 << "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    float *h_bias = new float[N];

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    for (int i=0; i < N; ++i) h_bias[i] = dis(gen);

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm_with_bias(h_A, h_B, h_bias, h_C_ref,  M, N, K);

    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C, *d_bias;
    cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16));
    cudaMalloc(&d_bias, N*sizeof(__nv_bfloat16));

    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    __nv_bfloat16 *h_bias_bf16 = new __nv_bfloat16[N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    for (int i = 0; i < N; ++i) h_bias_bf16[i] = __float2bfloat16(h_bias[i]);

    cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias_bf16, N, cudaMemcpyHostToDevice);

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch kernel
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    for(int i = 0; i < (NCU ? 0 : 2); i++) { // warmup
        inner_run<mmt>(d_A, d_B, d_C, d_bias, M, N, K, grid, block);
    }

    // Start timing
    cudaDeviceSynchronize();
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int ITERS = (NCU ? 1 : 10);
    for(int i = 0; i < ITERS; i++) {
        inner_run<mmt>(d_A, d_B, d_C, d_bias, M, N, K, grid, block);
    }
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy result back to host
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

    std::cout << "Converted result back to float" << std::endl;

    // Check result
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if(error > 1.0) { // large because of bf16 vs fp32 numerics
            if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Total elements: " << M*N << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    // for (int i=0; i < 100; ++i) {
    //     std::cout << h_C[i] << ", " << h_C_ref[i] << std::endl;
    // }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_bias;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    delete[] h_bias_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_bias);

    return 0;
}

int main() {
    int M = 2048, N = 4096, K = 8192;
    run_benchmark<matmul_template<2,4,8>>(M, N, K);
    return 0;
}
#endif
