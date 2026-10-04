"""Does PyTorch mem-efficient (CUTLASS) SDPA support GQA natively (q=16, kv=2)
without the repeat_interleave expansion? If yes, the 8x KV-memory blowup
disappears and CUTLASS becomes viable at 130k context.

Also measures the GQA-native path vs the expanded path at a large prefix to find
the memory ceiling of the expansion approach.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DEV = "cuda"
DT = torch.float16
QH, KVH, HD = 16, 2, 256
SCALE = 1.0 / (HD ** 0.5)


def try_gqa_native(t, kv_len, prefix):
    """q=[1,QH,t,HD], k/v=[1,KVH,kv_len,HD] -- GQA natively (no expansion)."""
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn(1, QH, t, HD, generator=g, device=DEV, dtype=DT) * 0.3
    k = torch.randn(1, KVH, kv_len, HD, generator=g, device=DEV, dtype=DT) * 0.3
    v = torch.randn(1, KVH, kv_len, HD, generator=g, device=DEV, dtype=DT) * 0.3
    try:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=(prefix == 0), scale=SCALE)
        return o.shape, "OK"
    except Exception as e:
        return None, f"ERROR: {str(e)[:120]}"


def gqa_expanded_mem(t, kv_len):
    """Peak extra memory of the repeat_interleave expansion (k+v), in MB."""
    return 2 * (kv_len * QH * HD * 2) / 1e6  # k and v, fp16


def main():
    print(f"geometry QH={QH} KVH={KVH} HD={HD}")
    # 1. Does GQA-native work at all? (small)
    for (t, kv, pre) in [(512, 512, 0), (512, 1024, 512)]:
        shape, status = try_gqa_native(t, kv, pre)
        print(f"GQA-native q={t} kv={kv} prefix={pre}: {status}  {shape}")

    # 2. Correctness of GQA-native vs a reference (head-by-head, small)
    t, kv, pre = 256, 512, 256
    g = torch.Generator(device=DEV).manual_seed(1)
    q = torch.randn(1, QH, t, HD, generator=g, device=DEV, dtype=DT) * 0.3
    k = torch.randn(1, KVH, kv, HD, generator=g, device=DEV, dtype=DT) * 0.3
    v = torch.randn(1, KVH, kv, HD, generator=g, device=DEV, dtype=DT) * 0.3
    # reference: expand and use math
    grp = QH // KVH
    kx = k.repeat_interleave(grp, dim=1).float()
    vx = v.repeat_interleave(grp, dim=1).float()
    qq = q.float()
    # offset-causal mask: row i attends to cols <= pre+i
    rows = torch.arange(t, device=DEV)
    cols = torch.arange(kv, device=DEV)
    mask = cols[None, :] <= (pre + rows)[:, None]
    scores = torch.einsum('bhid,bhjd->bhij', qq, kx) * SCALE
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = torch.where(torch.isnan(p), torch.zeros_like(p), p)
    ref = torch.einsum('bhij,bhjd->bhid', p, vx)
    try:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], scale=SCALE).float()
        rel = (o - ref).norm() / (ref.norm() + 1e-9)
        print(f"GQA-native correctness (offset-causal mask): relL2={rel:.2e}  {'OK' if rel<1e-2 else 'MISMATCH'}")
    except Exception as e:
        print(f"GQA-native correctness: ERROR {str(e)[:100]}")

    # 3. Memory ceiling of the EXPANSION approach at various prefix sizes
    print("\nExpansion-approach peak KV memory (would need this free):")
    for prefix in [4096, 8192, 16384, 32768, 65536, 131072]:
        t = 4096
        kv = prefix + t
        mb = gqa_expanded_mem(t, kv)
        print(f"  prefix={prefix:6d}  kv_len={kv:6d}  -> {mb:8.0f} MB  "
              f"{'OK(<850MB)' if mb < 850 else 'WILL OOM'}")


if __name__ == "__main__":
    main()
