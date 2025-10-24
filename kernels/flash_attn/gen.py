import torch
import torch.nn.functional as F
from tqdm import trange

batch_size = 12
n_heads = 12
seq_len = 1024
head_dim = 64
device = 'cuda'
dtype = torch.bfloat16

print(f"batch={batch_size}, nheads={n_heads}, seqlen={seq_len}, headdim={head_dim}")

torch.cuda.manual_seed(42)
q = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
k = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)
v = torch.randn((batch_size, n_heads, seq_len, head_dim), dtype=dtype, device=device)

# print(q)

scores = (q.float()@k.float().transpose(-2,-1)) * (head_dim**-0.5)
l = torch.logsumexp(scores, dim=-1, keepdim=True)

o = F.scaled_dot_product_attention(q, k, v, is_causal=False)

fn = f'{batch_size}_{n_heads}_{seq_len}_{head_dim}.txt'
with open(fn, 'w') as f:
    q_out = q.float().flatten().detach().cpu().numpy().tolist()
    k_out = k.float().flatten().detach().cpu().numpy().tolist()
    v_out = v.float().flatten().detach().cpu().numpy().tolist()
    o_out = o.float().flatten().detach().cpu().numpy().tolist()
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
