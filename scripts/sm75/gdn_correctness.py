#!/usr/bin/env python
"""Correctness gate for the sm75 GDN tile changes (idea-4).

Compares the FULL gdn_prefill_chunk_fla output (the model's GDN output `o`)
between the original and new tile/warp configs on identical, bounded inputs.
rel-L2 < 1e-3 per the plan gate.

chunk_fwd_kernel_o: BK/num_warps change (direct kernel call).
chunk_h: num_warps change (env-driven; compare via the full op in subprocesses).
"""
import os, sys, json, subprocess, tempfile
import torch, triton

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
PY = os.path.join(ROOT, ".venv", "bin", "python")
sys.path.insert(0, os.path.join(ROOT, "python"))

torch.manual_seed(1)
dev = "cuda"
T, B, H, Hg, K, V, BT = 8192, 1, 32, 16, 128, 128, 64
dt = torch.float16
scale = K ** -0.5

def rel_l2(a, b):
    a = a.float(); b = b.float()
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    return (a - b).norm() / (b.norm() + 1e-9)

# Bounded, well-conditioned inputs (decay g negative => stable recurrence).
def make_inputs(seed=1):
    torch.manual_seed(seed)
    q = torch.randn(B, T, Hg, K, device=dev, dtype=dt) * 0.1
    k = torch.randn(B, T, Hg, K, device=dev, dtype=dt) * 0.1
    v = torch.randn(B, T, H, V, device=dev, dtype=dt) * 0.1
    g = -torch.rand(B, T, H, device=dev, dtype=torch.float32) * 0.5  # negative decay
    beta = torch.rand(B, T, H, device=dev, dtype=torch.float32) * 0.5
    return q, k, v, g, beta

# ---- chunk_fwd_kernel_o (direct) ----
from freetoken.kernel.fla.chunk_o import chunk_fwd_kernel_o
from freetoken.kernel.fla.index import prepare_chunk_indices
q, k, v, g, beta = make_inputs(1)
h = torch.randn(B, triton.cdiv(T, BT), H, V, K, device=dev, dtype=dt) * 0.05
cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
ci = prepare_chunk_indices(cu, BT)
NT = len(ci)

def run_o(BK, BV, nw, ns):
    o = torch.empty_like(v)
    grid = (triton.cdiv(V, BV), NT, B * H)
    chunk_fwd_kernel_o[grid](q, k, v, h, g, o, cu, ci, scale,
        T=T, H=H, Hg=Hg, K=K, V=V, BT=BT, BK=BK, BV=BV,
        USE_G=True, IS_VARLEN=True, num_warps=nw, num_stages=ns)
    torch.cuda.synchronize()
    return o

o_old = run_o(128, 64, 4, 2)
o_new = run_o(64, 64, 8, 2)
ro = rel_l2(o_new, o_old)
print(f"chunk_fwd_kernel_o  old(128,64,w4,s2) vs new(64,64,w8,s2):  rel-L2 = {ro:.3e}  "
      f"{'PASS' if ro < 1e-3 else 'FAIL'}  (max abs {(o_new.float()-o_old.float()).abs().max().item():.2e})")

# ---- full op (covers chunk_h num_warps change) ----
CHILD = r'''
import sys, os, torch, triton
sys.path.insert(0, sys.argv[1])
from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla
torch.manual_seed(1); dev="cuda"
T,B,H,Hg,K,V,BT=8192,1,32,16,128,128,64; dt=torch.float16
q=torch.randn(B,T,Hg,K,device=dev,dtype=dt)*0.1
k=torch.randn(B,T,Hg,K,device=dev,dtype=dt)*0.1
v=torch.randn(B,T,H,V,device=dev,dtype=dt)*0.1
g=-(torch.rand(B,T,H,device=dev,dtype=torch.float32)*0.5)
beta=torch.rand(B,T,H,device=dev,dtype=torch.float32)*0.5
cu=torch.tensor([0,T],device=dev,dtype=torch.int64)
st=torch.zeros(1,H,V,K,device=dev,dtype=torch.float32); idx=torch.zeros(B,device=dev,dtype=torch.int32)
o=gdn_prefill_chunk_fla(q,k,v,g,beta,state_source=st,indices=idx,cu_seqlens=cu,scale=K**-0.5)
torch.cuda.synchronize()
torch.save(o.detach().cpu(), sys.argv[2])
'''
def run_full(nw, out):
    env = dict(os.environ, SGLANG_GDN_CHUNK_H_BV="32", SGLANG_GDN_CHUNK_H_NUM_WARPS=str(nw),
               SGLANG_GDN_CHUNK_H_NUM_STAGES="2", CUDA_VISIBLE_DEVICES="0")
    subprocess.run([PY, "-c", CHILD, os.path.join(ROOT, "python"), out], env=env,
                   check=True, capture_output=True, text=True)
    return torch.load(out)

d = tempfile.mkdtemp()
f_old = run_full(4, os.path.join(d, "old.pt"))
f_new = run_full(8, os.path.join(d, "new.pt"))
rf = rel_l2(f_new, f_old)
print(f"full gdn_prefill    old(w4) vs new(w8):  o rel-L2 = {rf:.3e}  "
      f"{'PASS' if rf < 1e-3 else 'FAIL'}  (max abs {(f_new.float()-f_old.float()).abs().max().item():.2e})")
print("\nALL PASS" if (ro < 1e-3 and rf < 1e-3) else "\nSOME FAILED")
