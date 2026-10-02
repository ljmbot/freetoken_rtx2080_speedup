"""E2: does ANY Triton config emit tensor-core mma on SM75?

For each (BM,BN,BK,warps,stages) x {transpose, no-transpose}, compile once, dump
PTX, count mma/hmma vs ffma/fma. Time any config that emits mma. This separates the
slow compile step from timing so the sweep stays bounded.
"""
import re, time, statistics
import torch, triton, triton.language as tl

dev = "cuda"
print("cap:", torch.cuda.get_device_capability(dev))

M, K, N = 2048, 2048, 12288   # dominant GDN in_proj shape
a = torch.randn(M, K, device=dev, dtype=torch.float16)
w = torch.randn(N, K, device=dev, dtype=torch.float16)   # [N, K], like the weight
out = torch.empty(M, N, device=dev, dtype=torch.float16)

# ---- Variant A: current style, weight loaded [BN,BK] then tl.trans -> [BK,BN] ----
@triton.jit
def gA(a_ptr, w_ptr, c_ptr, M, N, K, sam, sak, swn, swk, scm, scn,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    om = pm*BM + tl.arange(0,BM); on = pn*BN + tl.arange(0,BN); ok = tl.arange(0,BK)
    mm = om<M; nm = on<N
    ap = a_ptr + om[:,None]*sam + ok[None,:]*sak
    wp = w_ptr + on[:,None]*swn + ok[None,:]*swk
    acc = tl.zeros((BM,BN), dtype=tl.float32)
    for k in range(tl.cdiv(K,BK)):
        kr = K - k*BK
        aa = tl.load(ap, mask=mm[:,None] & (ok[None,:]<kr), other=0.0)
        ww = tl.load(wp, mask=nm[:,None] & (ok[None,:]<kr), other=0.0)
        acc += tl.dot(aa, tl.trans(ww), out_dtype=tl.float32)
        ap += BK*sak; wp += BK*swk
    cp = c_ptr + om[:,None]*scm + on[None,:]*scn
    tl.store(cp, acc.to(tl.float16), mask=mm[:,None] & nm[None,:])

# ---- Variant B: no in-kernel transpose, weight loaded directly [BK,BN] ----
@triton.jit
def gB(a_ptr, w_ptr, c_ptr, M, N, K, sam, sak, swn, swk, scm, scn,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    om = pm*BM + tl.arange(0,BM); on = pn*BN + tl.arange(0,BN); ok = tl.arange(0,BK)
    mm = om<M; nm = on<N
    ap = a_ptr + om[:,None]*sam + ok[None,:]*sak
    # weight is [N,K]; we want [BK,BN] tile -> index [ok, on]
    wp = w_ptr + on[None,:]*swn + ok[:,None]*swk
    acc = tl.zeros((BM,BN), dtype=tl.float32)
    for k in range(tl.cdiv(K,BK)):
        kr = K - k*BK
        aa = tl.load(ap, mask=mm[:,None] & (ok[None,:]<kr), other=0.0)
        ww = tl.load(wp, mask=(ok[:,None]<kr) & nm[None,:], other=0.0)
        acc += tl.dot(aa, ww, out_dtype=tl.float32)
        ap += BK*sak; wp += BK*swk
    cp = c_ptr + om[:,None]*scm + on[None,:]*scn
    tl.store(cp, acc.to(tl.float16), mask=mm[:,None] & nm[None,:])

def count_ptx(h):
    ptx = h.asm.get("ptx", "")
    hmma = len(re.findall(r"\bmma\.[\w.]+|hmma\.[\w.]+", ptx))
    ffma = len(re.findall(r"\bffma\.[\w.]+", ptx))
    fma  = len(re.findall(r"(?<!f)fma\.[\w.]+", ptx))
    target = re.search(r"\.target\s+(\S+)", ptx)
    return hmma, ffma, fma, (target.group(1) if target else "?"), ptx

def bench(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)
def tflops(ms): return 2*M*N*K/1e9/(ms/1000)/1e12

configs = []
for BM in (64, 128):
    for BN in (128, 256):
        for BK in (64, 128):
            for warps in (4, 8):
                for stages in (2, 3):
                    configs.append((BM, BN, BK, warps, stages))
# add the exact current config + a couple small ones
configs += [(64,128,64,8,3),(32,128,64,4,3),(64,64,64,4,3),(128,64,64,4,3)]

winners = []
print(f"\n{'variant':8s} {'BM':>4s} {'BN':>4s} {'BK':>4s} {'w':>2s} {'st':>2s} | {'mma':>5s} {'ffma':>5s} {'fma':>5s} | target")
print("-"*78)
seen = set()
for name, kern in (("A(trans)", gA), ("B(no-tr)", gB)):
    for (BM,BN,BK,warps,stages) in configs:
        if (name,BM,BN,BK,warps,stages) in seen: continue
        seen.add((name,BM,BN,BK,warps,stages))
        try:
            h = kern[(triton.cdiv(M,BM), triton.cdiv(N,BN))](
                a, w, out, M,N,K, a.stride(0),a.stride(1), w.stride(0),w.stride(1),
                out.stride(0),out.stride(1), BM=BM,BN=BN,BK=BK,
                num_warps=warps, num_stages=stages)
            hmma, ffma, fma, target, _ = count_ptx(h)
            flag = "  <-- MMA" if hmma>0 else ""
            print(f"{name:8s} {BM:>4d} {BN:>4d} {BK:>4d} {warps:>2d} {stages:>2d} | {hmma:>5d} {ffma:>5d} {fma:>5d} | {target}{flag}")
            if hmma>0:
                winners.append((name,BM,BN,BK,warps,stages,kern))
        except Exception as e:
            print(f"{name:8s} {BM:>4d} {BN:>4d} {BK:>4d} {warps:>2d} {stages:>2d} | COMPILE FAIL: {str(e)[:40]}")

print(f"\n=== configs emitting tensor-core mma: {len(winners)} ===")
# time the winners + cuBLAS reference
ref = bench(lambda: torch.matmul(a, w.t(), out=out))
print(f"cuBLAS fp16 [2048x2048]x[12288x2048]: {ref:8.2f} ms  {tflops(ref):6.2f} TFLOPS")
for (name,BM,BN,BK,warps,stages,kern) in winners[:12]:
    def f():
        kern[(triton.cdiv(M,BM), triton.cdiv(N,BN))](
            a, w, out, M,N,K, a.stride(0),a.stride(1), w.stride(0),w.stride(1),
            out.stride(0),out.stride(1), BM=BM,BN=BN,BK=BK,
            num_warps=warps, num_stages=stages)
    t = bench(f)
    print(f"  {name} BM={BM} BN={BN} BK={BK} w={warps} st={stages}: {t:8.2f} ms  {tflops(t):6.2f} TFLOPS  ({ref/t:.2f}x vs cuBLAS)")

if not winners:
    print("\nNO config emits tensor-core mma on SM75 -> E2 (fix Triton codegen) is a dead end.")
    print("The tl.dot -> CUDA-core FMA fallback is a hard codegen limitation of this build.")
