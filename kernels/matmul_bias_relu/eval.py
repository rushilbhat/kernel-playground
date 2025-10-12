import torch
import kp
import time

M, N, K = 2048, 4096, 8192
device = 'cuda'

print(f"M={M}, N={N}, K={K}")

torch.cuda.manual_seed(42)
A = (torch.rand(M, K, device=device) - 0.5)
B = (torch.rand(N, K, device=device) - 0.5) 
bias = (torch.rand(N, device=device) - 0.5)

C_ref = torch.relu((A @ B.T) + bias.unsqueeze(0))

A_bf16 = A.to(torch.bfloat16)
B_bf16 = B.to(torch.bfloat16)
bias_bf16 = bias.to(torch.bfloat16)

# Warmup
for _ in range(2):
    C_bf16 = kp.matmul_bias_relu(A_bf16, B_bf16, bias_bf16)

# Benchmark
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    C_bf16 = kp.matmul_bias_relu(A_bf16, B_bf16, bias_bf16)
torch.cuda.synchronize()
end = time.perf_counter()

diff = (end - start) * 1e6 / 10
flops = 2.0 * M * N * K
tflops = (flops / diff) / 1e6
print(f"Avg Kernel execution time: {diff:.2f} us")
print(f"Achieved performance: {tflops:.2f} TFLOPs")

# Correctness check
C = C_bf16.to(torch.float32)
max_error = torch.max(torch.abs(C - C_ref)).item()
error_count = torch.sum(torch.abs(C- C_ref) > 1.0).item()

print(f"Total elements: {M*N}")
print(f"Max error: {max_error:.6f}")
print(f"Error count: {error_count}")