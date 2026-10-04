#!/usr/bin/env python
"""Parse raw chrome trace for the n8192 device breakdown (kernel + gpu_memcpy + gpu_memset).

Compares the GDN-tuned capture against the pre-tune baseline.
"""
import json, sys
from collections import defaultdict

def ms(v):  # chrome trace ts/dur are in us
    return v / 1000.0

def breakdown(path):
    with open(path) as f:
        data = json.load(f)
    ev = data["traceEvents"] if isinstance(data, dict) else data
    byname = defaultdict(float)
    total = 0.0
    for e in ev:
        if not isinstance(e, dict):
            continue
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            d = ms(e.get("dur", 0))
            nm = e.get("name", "?")
            byname[nm] += d
            total += d
    return byname, total

def group(name):
    n = name.lower()
    if "chunk_fwd_kernel_o" in n: return "GDN chunk_fwd_kernel_o"
    if "chunk_gated_delta_rule_fwd_kernel_h" in n or "chunk_h" in n: return "GDN chunk_h"
    if "kkt" in n or "recompute" in n or "solve_tril" in n or "l2norm" in n or "intra" in n: return "GDN kkt/w_u/l2norm"
    if "extend_attention" in n or "attention" in n or "fmha" in n or "flash" in n: return "full attention"
    if "memcpy" in n or "h2d" in n or "d2h" in n: return "memcpy (HtoD/D2D)"
    if "memset" in n: return "memset"
    if "bmm" in n or "gemm" in n or "matmul" in n or "mm_" in n or "sgemm" in n or "hgemm" in n: return "GEMM (bmm/mm)"
    return "other"

def report(path, label):
    byname, total = breakdown(path)
    g = defaultdict(float)
    for nm, d in byname.items():
        g[group(nm)] += d
    print(f"\n=== {label}  (total device {total:.0f} ms) ===")
    for k, d in sorted(g.items(), key=lambda x: -x[1]):
        print(f"  {d:9.0f} ms  {100*d/total:5.1f}%  {k}")
    return g, total

base = "/home/lmeek/src/freetoken-sm75-native-prefill/artifacts/sm75-prefill/profile-8192/prefill-06-n8192.json"
tuned = "/home/lmeek/src/freetoken-sm75-native-prefill/artifacts/sm75-prefill/profile-8192-gdn/prefill-06-n8192.json"
gb, tb = report(base, "PRE-TUNE (Hopper/Ampere GDN config)")
gt, tt = report(tuned, "GDN-TUNED (FREETOKEN_GDN_SM75_TUNE=1)")

print("\n=== delta (tuned vs pre) ===")
for k in sorted(set(gb) | set(gt), key=lambda k: -(gt.get(k,0)-gb.get(k,0))):
    d = gt.get(k,0) - gb.get(k,0)
    print(f"  {d:+9.0f} ms  {100*(gt.get(k,0)/tt - gb.get(k,0)/tb):+5.1f} pts  {k}")
print(f"  total device: {tb:.0f} ms -> {tt:.0f} ms  ({tt/tb:.3f}x, {100*(1-tt/tb):+.1f}%)")
