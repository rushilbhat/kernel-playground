#include "kittens.cuh"

constexpr int CONSUMER_WARPGROUPS = 2;
constexpr int PRODUCER_WARPGROUPS = 1;
constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS; 

constexpr int QO_CHUNK = 64;
constexpr int KV_CHUNK = 64;  

constexpr int NUM_STAGES = 2;

using namespace kittens;

template<int HEAD_DIM> struct fwd_globals {
    using qo_tile = st_bf<QO_CHUNK, HEAD_DIM>;
    using kv_tile = st_bf<KV_CHUNK, HEAD_DIM>;
    using l_vec  = col_vec<st_fl<QO_CHUNK, HEAD_DIM>>;

    gl<bf16, -1, -1, -1, HEAD_DIM, qo_tile> q;
    gl<bf16, -1, -1, -1, HEAD_DIM, kv_tile> k;
    gl<bf16, -1, -1, -1, HEAD_DIM, kv_tile> v;
    gl<bf16, -1, -1, -1, HEAD_DIM, qo_tile> o;
    // Working backwards from the constraint: vector create_tensor_map forces smem_shape[0] =
    // sv_tma_dim1<sv_fl<QO_CHUNK>> = 64. That 64 is carved out of the sequence axis, so
    // gmem_shape[0] must line up with sequence length. Because gmem_shape[0] is populated
    // from the `cols` argument, we pass seq_len into `cols` and leave the singleton axis
    // in `rows`.
    gl<float, -1, -1, 1, -1, l_vec> l;

};

__device__ static inline int4 get_task_coord(fwd_globals<64> globals, int task_iter, bool is_consumer){
    int task_id = task_iter * gridDim.x + blockIdx.x;
    int chunked_seqlen = globals.q.rows() / (QO_CHUNK * CONSUMER_WARPGROUPS);
    int row = task_id % (chunked_seqlen);
    int depth = (task_id / chunked_seqlen) % globals.q.depth();
    int batch = task_id / (chunked_seqlen * globals.q.depth());

    if (batch < globals.q.batch()){
        return {batch, depth, row * CONSUMER_WARPGROUPS + (is_consumer ? warpgroup::groupid() : 0), 0};
    }
    else {
        return {-1, -1, -1, -1};
    }
}

__device__ static inline void print_coords_debug(int task_iter, int4 coord) {
    for (int i = 0; i < gridDim.x; i++){
        for (int j = 0; j < blockDim.x; j += 128){
    // for (int i = 0; i < 1; i++){
    //     for (int j = 0; j < 1; j += 128){
            if (blockIdx.x == i && threadIdx.x == j){
                printf(
                    "taskIter %d blockId %d threadId %d warpgroupId %d (%d, %d, %d, %d)\n", 
                    task_iter, blockIdx.x, threadIdx.x, warpgroup::groupid(), coord.x, coord.y, coord.z, coord.w
                );
            }
        }   
    }
}

__global__ __launch_bounds__(NUM_WARPGROUPS*WARPGROUP_THREADS, 1)
void fwd_kernel(const __grid_constant__ fwd_globals<64> globals) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al(&__shm[0]);

    using everyone = group<NUM_WARPGROUPS*WARPGROUP_WARPS>;

    using fwd_types = fwd_globals<64>;
    using qo_tile = typename fwd_types::qo_tile;
    using kv_tile = typename fwd_types::kv_tile;
    using l_vec = typename fwd_types::l_vec;
    qo_tile (&q_smem) [CONSUMER_WARPGROUPS] = al.allocate<qo_tile, CONSUMER_WARPGROUPS>();
    kv_tile (&k_smem) [NUM_STAGES]          = al.allocate<kv_tile, NUM_STAGES>();
    kv_tile (&v_smem) [NUM_STAGES]          = al.allocate<kv_tile, NUM_STAGES>();
    l_vec   (&l_smem) [CONSUMER_WARPGROUPS] = al.allocate<l_vec,   CONSUMER_WARPGROUPS>();
    qo_tile (*o_smem) = &q_smem[0];

    __shared__ kittens::semaphore q_tiles_ready, o_tiles_flushed;
    __shared__ kittens::semaphore k_stage_arrived[NUM_STAGES], k_stage_finished[NUM_STAGES];
    __shared__ kittens::semaphore v_stage_arrived[NUM_STAGES], v_stage_finished[NUM_STAGES];
    
    uint32_t semaphore_bitfield = 0xFFFF0000;

    if (warpgroup::groupid() >= CONSUMER_WARPGROUPS){
        if (warpgroup::warpid() == 0){
            init_semaphore(q_tiles_ready, PRODUCER_WARPGROUPS, 0);
            init_semaphore(o_tiles_flushed, CONSUMER_WARPGROUPS, 0);
            for(int i = 0; i < NUM_STAGES; i++){
                init_semaphore(k_stage_arrived[i], PRODUCER_WARPGROUPS, 0);
                init_semaphore(k_stage_finished[i], CONSUMER_WARPGROUPS, 0);
                init_semaphore(v_stage_arrived[i], PRODUCER_WARPGROUPS, 0);
                init_semaphore(v_stage_finished[i], CONSUMER_WARPGROUPS, 0);
            }
        }
        // REVISIT - id, everyone::sync vs __syncthreads 
        everyone::sync(15);

        // INSERT: warpgroup::decrease_registers
        if (warpgroup::warpid() == 0){
            for(int task_iter=0; true; task_iter++){
                int4 task_coord = get_task_coord(globals, task_iter, false);
                // print_coords_debug(task_iter, task_coord);
                if (task_coord.x == -1) break;
                wait(o_tiles_flushed, (task_iter^1)%2);
                // if (blockIdx.x == 0 && warpgroup::laneid() == 0) printf("Cleared wait on o_tiles_flushed\n");
                // return
                tma::expect(q_tiles_ready, q_smem);
                for(int cwg = 0; cwg < CONSUMER_WARPGROUPS; cwg++){
                    tma::load_async(q_smem[cwg], globals.q, 
                        kittens::coord<qo_tile>{task_coord.x, task_coord.y, task_coord.z + cwg, task_coord.w}, q_tiles_ready);
                }
                int iters_per_task = globals.k.rows() / KV_CHUNK;
                for(int it = 0; it < iters_per_task; it++){
                    kittens::coord<kv_tile> kv_idx = {task_coord.x, task_coord.y, it, task_coord.w};
                    // if (blockIdx.x == 0 && warpgroup::laneid() == 0) printf("Coord from KV %d %d %d %d \n", kv_idx.b, kv_idx.d, kv_idx.r, kv_idx.c);
                    
                    wait(k_stage_finished[it % NUM_STAGES], ((it / NUM_STAGES) + 1) % 2);
                    // if (blockIdx.x == 0 && warpgroup::laneid() == 0) printf("Cleared wait on k_stage_finished %d\n", it % NUM_STAGES);
                    tma::expect(k_stage_arrived[it % NUM_STAGES], k_smem[it % NUM_STAGES]);
                    tma::load_async(k_smem[it % NUM_STAGES], globals.k, kv_idx, k_stage_arrived[it % NUM_STAGES]);
                    wait(v_stage_finished[it % NUM_STAGES], ((it / NUM_STAGES) + 1) % 2);
                    // if (blockIdx.x == 0 && warpgroup::laneid() == 0) printf("Cleared wait on v_stage_finished %d\n", it % NUM_STAGES);
                    tma::expect(v_stage_arrived[it % NUM_STAGES], v_smem[it % NUM_STAGES]);
                    tma::load_async(v_smem[it % NUM_STAGES], globals.v, kv_idx, v_stage_arrived[it % NUM_STAGES]);
                    // if (it == iters_per_task-1) return;
                }
            }
            // Not sure about whether we need to sync all producer threads before continuing with the next task like in H100 matmul.
            // Currently following B200's blueprint which doesn't have a sync. If we need a sync, then would need to pull task_iter loop
            // outside of this conditional `if (warpgroup::warpid() == 0)`.
        }

    }
    else {
        everyone::sync(15);
        rt_fl<16, KV_CHUNK> attn;
        rt_bf<16, KV_CHUNK> attn_mma;
        rt_fl<16, 64> o_reg; // <- REVISIT: Should be HEAD_DIM cols

        col_vec<rt_fl<16, 64>> m_vec, m_vec_old, cf_vec, l_reg;
        
        for(int task_iter=0; true; task_iter++){
            int4 task_coord = get_task_coord(globals, task_iter, true);
            // print_coords_debug(task_iter, task_coord);
            if (task_coord.x == -1) break;

            zero(o_reg);
            zero(l_reg);
            neg_infty(m_vec);

            wait(q_tiles_ready, (task_iter^0)%2);
            if (blockIdx.x == 0 && warpgroup::groupid() == 0 && warpgroup::laneid() == 0) printf("Task %d. Cleared wait on q_tiles_ready\n", task_iter);
            int iters_per_task = globals.k.rows() / KV_CHUNK;
            for(int it = 0; it < iters_per_task; it++){
                wait(k_stage_arrived[it % NUM_STAGES], (it / NUM_STAGES) % 2);
                if (blockIdx.x == 0 && warpgroup::groupid() == 0 && warpgroup::laneid() == 0) printf("Task %d. Cleared wait on k_stage_arrived %d\n", task_iter, it);
                warpgroup::mm_ABt(attn, q_smem[warpgroup::groupid()], k_smem[it % NUM_STAGES]);
                warpgroup::mma_async_wait();
                if (warpgroup::laneid() == 0) arrive(k_stage_finished[it % NUM_STAGES]);
                mul(attn, attn, 0.125f); // <- REVISIT: 0.125f = float(1/sqrt(64))
                copy(m_vec_old, m_vec);
                row_max(m_vec, attn, m_vec_old);
                sub_row(attn, attn, m_vec);
                exp(attn, attn);
                sub(cf_vec, m_vec_old, m_vec);
                exp(cf_vec, cf_vec);
                mul(l_reg, l_reg, cf_vec);
                row_sum(l_reg, attn, l_reg);
                mul_row(o_reg, o_reg, cf_vec);
                copy(attn_mma, attn);
                wait(v_stage_arrived[it % NUM_STAGES], (it / NUM_STAGES) % 2);
                // if (blockIdx.x == 0 && warpgroup::groupid() == 0 && warpgroup::laneid() == 0) printf("Task %d. Cleared wait on v_stage_arrived %d\n", task_iter, it);
                // if (it == iters_per_task-1) return;
                warpgroup::mma_AB(o_reg, attn_mma, v_smem[it % NUM_STAGES]);
                warpgroup::mma_async_wait();
                if (warpgroup::laneid() == 0) arrive(v_stage_finished[it % NUM_STAGES]);
            }
            // Not sure about whether we need to sync all consumer threads before doing stores like in H100 matmul.
            div_row(o_reg, o_reg, l_reg);
            warpgroup::store(o_smem[warpgroup::groupid()], o_reg);
            warpgroup::sync(warpgroup::groupid()+4); // REVISIT: Why do we need this?
            if (warpgroup::warpid() == 0) tma::store_async(globals.o, o_smem[warpgroup::groupid()], kittens::coord<qo_tile>(task_coord));
            log(l_reg, l_reg);
            add(l_reg, l_reg, m_vec);
            warpgroup::store(l_smem[warpgroup::groupid()], l_reg);
            warpgroup::sync(warpgroup::groupid()+4); // REVISIT: Why do we need this?
            // l has different coords system has o (remember batch, head, 1, seq vs batch, head, seq, head_dim)
            if (warpgroup::warpid() == 0) tma::store_async(globals.l, l_smem[warpgroup::groupid()], kittens::coord<l_vec>({task_coord.x, task_coord.y, task_coord.w, task_coord.z}));
             tma::store_async_read_wait();
            if (warpgroup::laneid() == 0) arrive(o_tiles_flushed);
            // Not sure about whether we need to sync all consumer threads before continuing with the next task like in H100 matmul.
            // I think we might just for semantics but not for correctness. The concern is that without the sync, the threads which didn't
            // do tma::store_async will race ahead but thread 0 is still blocked by tma::store_async_read_wait(); and thus won't arrive(o_tiles_flushed);
            // which keeps everything still correct.
        }
    }
}

#ifdef TORCH_COMPILE
#include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>

torch::Tensor flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v){
    const int batch = q.size(0);
    const int heads = q.size(1);
    const int seq   = q.size(2);
    const int dim   = q.size(3);

    auto o = torch::empty({batch, heads, seq, dim}, q.options());
    auto l = torch::empty({batch, heads, seq, 1}, q.options().dtype(torch::kFloat32));

    bf16* d_q = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16* d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    float* d_l = l.data_ptr<float>();

    // REVISIT
    // cudaDeviceSynchronize();
    // auto stream = at::cuda::getCurrentCUDAStream().stream(); 

    TORCH_CHECK(dim == 64, "temporary limitation: only head_dim == 64 implemented");
    // Explicit cast keeps nvcc from warning about narrowing int -> unsigned long
    fwd_globals<64> globals{
        {d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(heads), static_cast<unsigned int>(seq), nullptr},
        {d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(heads), static_cast<unsigned int>(seq), nullptr},
        {d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(heads), static_cast<unsigned int>(seq), nullptr},
        {d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(heads), static_cast<unsigned int>(seq), nullptr},
        {d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(heads), nullptr, static_cast<unsigned int>(seq)}
    };

    dim3 grid(114);
    dim3 block(NUM_WARPGROUPS*WARPGROUP_THREADS);

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // fwd_kernel<<<grid, block, mem_size, stream>>>();
    fwd_kernel<<<grid, block, mem_size>>>(globals);

    // REVISIT
    // CHECK_CUDA_ERROR(cudaGetLastError());
    // cudaStreamSynchronize(stream);

    return o;
}

#else

#include "harness.impl"

#endif
