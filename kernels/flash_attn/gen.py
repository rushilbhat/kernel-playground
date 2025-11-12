import torch
import torch.nn.functional as F
from tqdm import trange

batch_size = 1
n_heads = 1
seq_len = 256
head_dim = 64
device = 'cuda'
dtype = torch.bfloat16
causal = True

print(f"batch={batch_size}, nheads={n_heads}, seqlen={seq_len}, headdim={head_dim}")

torch.cuda.manual_seed(42)
q = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
k = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
v = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)


scores = (q.float() @ k.float().transpose(-2, -1)) * (head_dim ** -0.5)
if causal:
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    scores = scores.masked_fill(causal_mask, float('-inf'))
l = torch.logsumexp(scores, dim=-1, keepdim=True)

o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

fn = f'{batch_size}_{n_heads}_{seq_len}_{head_dim}.txt'
with open(fn, 'w') as f:
    q_out = q.float().flatten().detach().cpu().numpy().tolist()
    k_out = k.float().flatten().detach().cpu().numpy().tolist()
    v_out = v.float().flatten().detach().cpu().numpy().tolist()
    o_out = o.float().flatten().detach().cpu().numpy().tolist()
    # import pandas as pd
    # for k in range(batch_size):
    #     for j in range(n_heads):
    #         for i in range(seq_len//head_dim):
                # print(f"CTA {k*n_heads*seq_len/head_dim + j*seq_len//head_dim + i//2}, CWG {i%2}")
                # print(f"batch {k}, head {j}, seq {i}")
                # print(pd.DataFrame(o[k, j, (64*i):(64*(i+1)), :].float().cpu().numpy()))

    l_out = l.flatten().detach().cpu().numpy().tolist()

    total_elements = batch_size * n_heads * seq_len * head_dim
    for tensor in (q_out, k_out, v_out, o_out):
        for i in trange(total_elements):
            f.write(repr(tensor[i]))
            f.write(' ')
    
    l_elements = batch_size * n_heads * seq_len
    for i in trange(l_elements):
        f.write(repr(l_out[i]))
        f.write(' ')
