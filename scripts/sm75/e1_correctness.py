"""Proper correctness: E1 fp16 dequant+cuBLAS vs fp32 reference, via relative-L2 (robust to
near-zero GEMM outputs) and cosine. Also compare against the CURRENT Triton W8A16 path so we
know E1 stays in the same numeric ballpark as what's already deployed."""
import torch
from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

dev = "cuda"
torch.manual_seed(0)
M, K, N = 2048, 2048, 12288
w_fp8 = (torch.randn(N, K, device=dev) * 0.1).clamp(-448, 448).to(torch.float8_e4m3fn)
scale = torch.rand(N, device=dev) * 0.5 + 0.2
x = torch.randn(M, K, device=dev, dtype=torch.float16)

# fp32 reference
w_ref = w_fp8.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)
ref = x.to(torch.float32) @ w_ref.t()

# E1: dequant fp16 + cuBLAS fp16
w16 = (w_fp8.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)).to(torch.float16)
e1 = x @ w16.t()

# current Triton W8A16 (the deployed path)
cur = fp8_pertensor_linear(x, w_fp8, scale, None, None, False)

def rel_l2(a, b):
    a = a.float(); b = b.float()
    return (a - b).norm() / (b.norm() + 1e-9)
def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

print(f"rel-L2  E1  vs ref : {rel_l2(e1, ref):.5f}   cos: {cos(e1, ref):.6f}")
print(f"rel-L2  cur vs ref : {rel_l2(cur, ref):.5f}   cos: {cos(cur, ref):.6f}")
print(f"rel-L2  E1  vs cur : {rel_l2(e1, cur):.5f}   cos: {cos(e1, cur):.6f}")
print(f"max abs diff E1 vs ref: {(e1.float()-ref).abs().max().item():.5f}  (|ref| max {ref.abs().max().item():.3f})")
