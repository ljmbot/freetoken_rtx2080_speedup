#!/usr/bin/env python
"""GDN recurrence microbenchmark (idea-4 gate 4.2) — faithful to production.

Sweeps tile/warp/stage configs for the two dominant GDN kernels on sm75 (RTX 2080)
and reports best vs the current default. No engine, no model — synthetic tensors
at the real model geometry.

Kernels:
  chunk_fwd_kernel_o   (22.3% of device at n8192) — currently hardcoded
                       BK=128, BV=64, num_warps=4, num_stages=2 (autotune off).
                       Tested in the VARLEN path (IS_VARLEN=True, real cu_seqlens)
                       to match how the engine calls it.
  chunk_gated_delta_rule_fwd_kernel_h_blockdim64   (9.0%) — wrapped in
                       @triton.autotune whose single config is read from env
                       (SGLANG_GDN_CHUNK_H_BV/NUM_WARPS/NUM_STAGES) at import.
                       Swept by spawning a subprocess per env combo.

Gate: >=1.5x on the recurrence before touching the engine.
"""
import os, sys, json, subprocess, time
import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

from freetoken.kernel.fla.chunk_o import chunk_fwd_kernel_o
from freetoken.kernel.fla.index import prepare_chunk_indices

torch.manual_seed(0)
dev = "cuda"
torch.cuda.init()
print("device:", torch.cuda.get_device_name(0), "sm", torch.cuda.get_device_capability(0))

T, B, H, Hg, K, V, BT = 8192, 1, 32, 16, 128, 128, 64
dtype = torch.float16
scale = K ** -0.5

def timeit(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def bench_o_varlen():
    q = torch.randn(B, T, Hg, K, device=dev, dtype=dtype)
    k = torch.randn(B, T, Hg, K, device=dev, dtype=dtype)
    v = torch.randn(B, T, H, V, device=dev, dtype=dtype)
    g = torch.randn(B, T, H, device=dev, dtype=torch.float32) * 0.1  # fp32 log-decay cumsum
    h = torch.empty(B, triton.cdiv(T, BT), H, V, K, device=dev, dtype=dtype)
    o = torch.empty_like(v)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)   # single seq of length T
    ci = prepare_chunk_indices(cu, BT)
    NT = len(ci)
    res = []
    for BV in (32, 64):
        for BK in (64, 128):
            for nw in (4, 8):
                for ns in (2, 3):
                    def f():
                        grid = (triton.cdiv(V, BV), NT, B * H)
                        chunk_fwd_kernel_o[grid](q, k, v, h, g, o, cu, ci, scale,
                            T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=BK, BV=BV,
                            USE_G=True, IS_VARLEN=True, num_warps=nw, num_stages=ns)
                    try:
                        res.append(dict(BV=BV, BK=BK, num_warps=nw, num_stages=ns,
                                        ms=round(timeit(f), 3)))
                    except Exception as ex:
                        res.append(dict(BV=BV, BK=BK, num_warps=nw, num_stages=ns,
                                        ms=None, err=type(ex).__name__))
    return res, NT

def report_o(rows):
    ok = [r for r in rows if r.get("ms")]
    ok.sort(key=lambda r: r["ms"])
    base = next((r for r in ok if r["BV"]==64 and r["BK"]==128
                 and r["num_warps"]==4 and r["num_stages"]==2), None)
    print(f"\n=== chunk_fwd_kernel_o (VARLEN, 22.3% of device) — {len(ok)} valid ===")
    print("top 6:")
    for r in ok[:6]:
        tag = "  <== current" if (base and r is base) else ""
        print(f"  {r['ms']:9.3f} ms  { {k:v for k,v in r.items() if k!='ms'} }{tag}")
    if ok and base:
        best = ok[0]
        print(f"  current {base['ms']:.3f} ms -> best {best['ms']:.3f} ms  "
              f"= {base['ms']/best['ms']:.2f}x   best { {k:v for k,v in best.items() if k!='ms'} }")
        print(f"  projected save on measured 4431.5 ms (n8192): "
              f"~{4431.5*(1-best['ms']/base['ms']):.0f} ms")
    return ok

# ---- chunk_h: subprocess per env combo (autotune reads env at import) ----
H_CHILD = r'''
import sys, os, torch, triton, json
sys.path.insert(0, sys.argv[1])
from freetoken.kernel.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
T,B,H,Hg,K,V,BT=8192,1,32,16,128,128,64; dt=torch.float16; dev="cuda"
k=torch.randn(B,T,Hg,K,device=dev,dtype=dt); w=torch.randn(B,T,H,K,device=dev,dtype=dt)
u=torch.randn(B,T,H,V,device=dev,dtype=dt); g=torch.randn(B,T,H,device=dev,dtype=torch.float32)*0.1
h=torch.empty(B,triton.cdiv(T,BT),H,V,K,device=dev,dtype=dt); v_new=torch.empty_like(u)
st=torch.zeros(H,V,K,device=dev,dtype=dt); idx=torch.zeros(B,device=dev,dtype=torch.int32)
BV=int(os.environ["SGLANG_GDN_CHUNK_H_BV"]); NW=int(os.environ["SGLANG_GDN_CHUNK_H_NUM_WARPS"]); NS=int(os.environ["SGLANG_GDN_CHUNK_H_NUM_STAGES"])
def f():
    stc=st.clone()
    grid=(triton.cdiv(V,BV),B*H)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](k=k,v=u,w=w,v_new=v_new,g=g,gk=None,
        h=h,initial_state=stc,initial_state_indices=idx,cu_seqlens=None,chunk_offsets=None,
        T=T,H=H,Hg=Hg,K=K,V=V,BT=BT,USE_G=True,USE_GK=False,USE_INITIAL_STATE=True,
        INPLACE_UPDATE=True,SAVE_NEW_VALUE=True,IS_VARLEN=False,NT_BUCKET=2)
for _ in range(2): f()
torch.cuda.synchronize()
s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
for _ in range(5): f()
e.record();torch.cuda.synchronize()
print(json.dumps(dict(ms=round(s.elapsed_time(e)/5,3),BV=BV,num_warps=NW,num_stages=NS)))
'''

def bench_h():
    res = []
    base_env = dict(os.environ)
    for BV in (32, 64):
        for NW in (4, 8):
            for NS in (2, 3):
                env = dict(base_env)
                env.update({"SGLANG_GDN_CHUNK_H_BV": str(BV),
                            "SGLANG_GDN_CHUNK_H_NUM_WARPS": str(NW),
                            "SGLANG_GDN_CHUNK_H_NUM_STAGES": str(NS)})
                py = os.path.join(HERE, "..", "..", ".venv", "bin", "python")
                try:
                    out = subprocess.run([py, "-c", H_CHILD, os.path.join(HERE, "..", "..", "python")],
                                         env=env, capture_output=True, text=True, timeout=300)
                    line = [l for l in out.stdout.splitlines() if l.strip().startswith("{")]
                    if line:
                        res.append(json.loads(line[-1]))
                    else:
                        res.append(dict(BV=BV, num_warps=NW, num_stages=NS, ms=None,
                                        err=out.stderr.strip().splitlines()[-1][:80] if out.stderr else "?"))
                except Exception as ex:
                    res.append(dict(BV=BV, num_warps=NW, num_stages=NS, ms=None, err=type(ex).__name__))
    return res

def report_h(rows):
    ok = [r for r in rows if r.get("ms")]
    ok.sort(key=lambda r: r["ms"])
    base = next((r for r in ok if r["BV"]==32 and r["num_warps"]==4 and r["num_stages"]==2), None)
    print(f"\n=== chunk_gated_delta_rule_fwd_kernel_h (9.0% of device) — {len(ok)} valid ===")
    print("top 6:")
    for r in ok[:6]:
        tag = "  <== current" if (base and r is base) else ""
        print(f"  {r['ms']:9.3f} ms  { {k:v for k,v in r.items() if k!='ms'} }{tag}")
    if ok and base:
        best = ok[0]
        print(f"  current {base['ms']:.3f} ms -> best {best['ms']:.3f} ms  "
              f"= {base['ms']/best['ms']:.2f}x   best { {k:v for k,v in best.items() if k!='ms'} }")
        print(f"  projected save on measured 1790.2 ms (n8192): "
              f"~{1790.2*(1-best['ms']/base['ms']):.0f} ms")
    return ok

def main():
    print(f"inputs: T={T} B={B} H={H} Hg={Hg} K={K} V={V} BT={BT} dtype={dtype}")
    o_rows, NT = bench_o_varlen()
    print(f"  varlen NT (chunks) = {NT}")
    o_ok = report_o(o_rows)
    h_rows = bench_h()
    h_ok = report_h(h_rows)
    out = dict(o_varlen=o_rows, h=h_rows)
    if o_ok and h_ok:
        o_cur = next(r for r in o_ok if r["BV"]==64 and r["BK"]==128 and r["num_warps"]==4 and r["num_stages"]==2)
        h_cur = next(r for r in h_ok if r["BV"]==32 and r["num_warps"]==4 and r["num_stages"]==2)
        o_best, h_best = o_ok[0], h_ok[0]
        cur = o_cur["ms"] + h_cur["ms"]; best = o_best["ms"] + h_best["ms"]
        out["combined"] = dict(current_ms=round(cur,3), best_ms=round(best,3),
                               speedup=round(cur/best,3),
                               proj_save_n8192_ms=round(6221.7*(1-best/cur),1))
        print(f"\n=== combined (o+h) at n8192 ===")
        print(f"  current {cur:.3f} ms -> best {best:.3f} ms  = {cur/best:.2f}x")
        print(f"  projected save on measured 6221.7 ms (o+h, n8192): "
              f"~{6221.7*(1-best/cur):.0f} ms per n8192 prefill")
    json.dump(out, open(os.path.join(HERE, "..", "..", "artifacts", "sm75-prefill", "gdn-microbench.json"), "w"), indent=2)
    print("\nwrote artifacts/sm75-prefill/gdn-microbench.json")

if __name__ == "__main__":
    main()
