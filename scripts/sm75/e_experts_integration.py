"""Integration: run the REAL production entry point fused_experts_nvfp4 with the
E3 gate on vs off (separate processes; the gate is read at import). Confirms the
gated tensor-core path is a faithful drop-in AND faster, through the actual
dispatch in layers/moe.py -> fused_nvfp4.py."""
import os, sys, time
import torch

torch.manual_seed(0)
dev = torch.device("cuda")
H, I, E, TOP_K, M = 2048, 512, 256, 8, 2048
two_i = 2 * I

mode = sys.argv[1]  # "off" | "on" | "compare"
# set the gate BEFORE importing the module (the flag is read at import time)
if mode == "on":
    os.environ["FREETOKEN_E3_CUBLAS"] = "1"
else:
    os.environ.pop("FREETOKEN_E3_CUBLAS", None)

valid = torch.tensor([0,1,2,3,4,5,6,8,9,10,11,12,13,14], dtype=torch.uint8, device=dev)
gu_packed = valid[torch.randint(0, 14, (E, two_i, H // 2), device=dev)]
gu_scale = torch.randint(0x38, 0x3C, (E, two_i, H // 16), dtype=torch.uint8, device=dev)
gu_global = (torch.rand(E, two_i, device=dev) * 0.5 + 0.5).half()
dn_packed = valid[torch.randint(0, 14, (E, H, I // 2), device=dev)]
dn_scale = torch.randint(0x38, 0x3C, (E, H, I // 16), dtype=torch.uint8, device=dev)
dn_global = (torch.rand(E, H, device=dev) * 0.5 + 0.5).half()
hidden = (torch.randn(M, H, device=dev) * 0.01).half()
topk_ids = torch.randint(0, E, (M, TOP_K), dtype=torch.int32, device=dev)
topk_w = torch.softmax(torch.randn(M, TOP_K, device=dev), dim=-1)

from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

def run():
    for _ in range(2):
        out = fused_experts_nvfp4(hidden, gu_packed, gu_scale, gu_global,
                                  dn_packed, dn_scale, dn_global, topk_w, topk_ids,
                                  E, "silu", False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 5
    for _ in range(n):
        out = fused_experts_nvfp4(hidden, gu_packed, gu_scale, gu_global,
                                  dn_packed, dn_scale, dn_global, topk_w, topk_ids,
                                  E, "silu", False)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / n * 1000

if mode in ("off", "on"):
    out, ms = run()
    torch.save(out, f"/tmp/e3_{mode}.pt")
    print(f"{mode}: {ms:.1f} ms/layer")
else:  # compare
    a = torch.load("/tmp/e3_off.pt").float()
    b = torch.load("/tmp/e3_on.pt").float()
    rel = (a - b).norm() / (b.norm() + 1e-12)
    cs = (a.flatten() @ b.flatten()) / (a.norm() * b.norm())
    print(f"rel-L2 (off vs on): {rel:.6f}")
    print(f"cosine: {cs:.6f}")
    print(f"max abs diff: {(a - b).abs().max():.6f}")
