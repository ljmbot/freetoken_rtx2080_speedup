"""Verify E1 primitive on SM75: fp8->fp16 dequant + cuBLAS fp16 GEMM.
Confirms (a) the cast works on Turing, (b) numerics match the reference W8A16,
(c) the per-forward dequant cost is negligible, (d) memory transient is small.
"""
import time, statistics
import torch

dev = "cuda"
print("cap:", torch.cuda.get_device_capability(dev))

# Use the real dominant GDN in_proj shape [N=12288, K=2048], M=2048.
M, K, N = 2048, 2048, 12288
torch.manual_seed(0)

# fp8 weight + per-row scale (mimic the checkpoint layout)
w_fp8 = (torch.randn(N, K, device=dev) * 0.1).clamp(-448, 448).to(torch.float8_e4m3fn)
scale = (torch.rand(N, device=dev) * 0.5 + 0.2)  # per-row fp32
x = torch.randn(M, K, device=dev, dtype=torch.float16)

# (a) cast works?
w32 = w_fp8.to(torch.float32)
w16 = w_fp8.to(torch.float16)
print("fp8->fp32 OK, fp8->fp16 OK")

# Reference: dequant in fp32, matmul in fp32 (the ground truth)
w_ref = w32 * scale.to(torch.float32).unsqueeze(-1)          # [N,K] fp32
ref = (x.to(torch.float32) @ w_ref.t())                       # [M,N] fp32

# E1 path: dequant to fp16 (fp32 intermediate for the scale multiply), cuBLAS fp16
def dequant_fp16(w, s):
    return (w.to(torch.float32) * s.to(torch.float32).unsqueeze(-1)).to(torch.float16)
w16d = dequant_fp16(w_fp8, scale)
out_e1 = x @ w16d.t()                                         # cuBLAS fp16

# also the "cast-first" variant (cheaper, no fp32 temp)
def dequant_fp16_fast(w, s):
    return w.to(torch.float16) * s.to(torch.float16).unsqueeze(-1)
w16f = dequant_fp16_fast(w_fp8, scale)
out_fast = x @ w16f.t()

def relerr(a, b):
    return ((a - b).abs() / (b.abs() + 1e-3)).max().item()

print(f"relerr (fp32-intermediate dequant) vs ref: {relerr(out_e1.float(), ref):.5f}")
print(f"relerr (fast cast-first dequant)   vs ref: {relerr(out_fast.float(), ref):.5f}")
print(f"relerr (fast vs fp32-int):         {relerr(out_fast.float(), out_e1.float()):.5f}")

# (c) per-forward dequant cost (all 3 shape classes, one forward's worth)
def bench(fn, iters=10, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

# one forward's dense projections: 40x[2048,2048] + 30x[12288,2048] + 10x[9216,2048]
shapes = [(2048, 2048)] * 40 + [(12288, 2048)] * 30 + [(9216, 2048)] * 10
ws = [((torch.randn(n, k, device=dev) * 0.1).clamp(-448, 448).to(torch.float8_e4m3fn),
       torch.rand(n, device=dev) * 0.5 + 0.2) for (n, k) in shapes]
xs = [torch.randn(M, k, device=dev, dtype=torch.float16) for (_, k) in shapes]

def full_forward_dequant_gemm():
    for (w, s), xx in zip(ws, xs):
        wd = dequant_fp16(w, s)
        _ = xx @ wd.t()
        del wd
t_dq_gemm = bench(full_forward_dequant_gemm)
print(f"\nfull prefill dense (dequant+cuBLAS fp16, 80 GEMMs, M=2048): {t_dq_gemm:8.1f} ms")

# compare: pure cuBLAS fp16 without dequant (lower bound)
def full_forward_gemm_only():
    for (w, s), xx in zip(ws, xs):
        wd = w.to(torch.float16)
        _ = xx @ wd.t()
        del wd
t_gemm = bench(full_forward_gemm_only)
print(f"full prefill dense (cuBLAS fp16, no dequant)          : {t_gemm:8.1f} ms")

# (d) transient memory for one dequant (largest weight)
big = ws[40][0]  # [12288,2048] fp8
before = torch.cuda.memory_allocated()
wd = dequant_fp16(big, ws[40][1])
peak = torch.cuda.memory_allocated() - before
del wd
print(f"\ntransient for one in_proj_qkvz dequant [12288x2048]: {peak/1e6:.1f} MB")
total_fp16 = sum(n * k * 2 for (n, k) in shapes) / 1e6
total_fp8 = sum(n * k for (n, k) in shapes) / 1e6
print(f"all dense projections: fp8={total_fp8:.0f} MB  fp16={total_fp16:.0f} MB  (+{total_fp16-total_fp8:.0f} MB if cached)")
