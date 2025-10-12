import torch
import kp
import time

M, N, K = 2048, 4096, 8192
device = 'cuda'

print(f"M={M}, N={N}, K={K}")

torch.cuda.manual_seed(42)
a = (torch.rand(M, K, device=device) - 0.5)
b = (torch.rand(N, K, device=device) - 0.5) 
bias = (torch.rand(N, device=device) - 0.5)

c_ref = torch.relu((a @ b.T) + bias.unsqueeze(0))

a_bf16 = a.to(torch.bfloat16)
b_bf16 = b.to(torch.bfloat16)
bias_bf16 = bias.to(torch.bfloat16)

# Warmup
for _ in range(2):
    c_bf16 = kp.matmul_bias_relu(a_bf16, b_bf16, bias_bf16)

# Benchmark
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    c_bf16 = kp.matmul_bias_relu(a_bf16, b_bf16, bias_bf16)
torch.cuda.synchronize()
end = time.perf_counter()

diff = (end - start) * 1e6 / 10
flops = 2.0 * M * N * K
tflops = (flops / diff) / 1e6
print(f"Avg Kernel execution time: {diff:.2f} us")
print(f"Achieved performance: {tflops:.2f} TFLOPs")

# Correctness check
c = c_bf16.to(torch.float32)
max_error = torch.max(torch.abs(c - c_ref)).item()
error_count = torch.sum(torch.abs(c - c_ref) > 1.0).item()

print(f"Total elements: {M*N}")
print(f"Max error: {max_error:.6f}")
print(f"Error count: {error_count}")