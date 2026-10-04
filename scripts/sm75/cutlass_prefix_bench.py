"""Decisive test: can CUTLASS (PyTorch mem-efficient SDPA) cover the PREFIX/paged
extend case (chunk 2+ of a chunked prefill), where most long-context attention work
lives? Compare against the production Triton extend kernel at the true geometry
(16 qh / 2 kvh / hd=256), T=4096 new tokens, prefix=4096 cached.

If CUTLASS-with-mask is fast here, it covers the whole 8192/130k prefill. If it
falls back to a slow math path, CUTLASS is a dead end for long context.
"""
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from freetoken.kernel.triton.attention import _extend_attention_split_kernel
import triton

DEV = "cuda"
DT = torch.float16
QH, KVH, HD = 16, 2, 256
GROUP = QH // KVH
T = 4096
PREFIX = 4096
SCALE = 1.0 / (HD ** 0.5)


def build(t, prefix, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(t, QH, HD, generator=g, device=DEV, dtype=DT) * 0.3
    k_ext = torch.randn(t, KVH, HD, generator=g, device=DEV, dtype=DT) * 0.3
    v_ext = torch.randn(t, KVH, HD, generator=g, device=DEV, dtype=DT) * 0.3
    kc = torch.randn(prefix, KVH, HD, generator=g, device=DEV, dtype=DT) * 0.3
    vc = torch.randn(prefix, KVH, HD, generator=g, device=DEV, dtype=DT) * 0.3
    return q, k_ext, v_ext, kc, vc


def triton_extend(q, k_ext, v_ext, kc, vc, prefix):
    t = q.shape[0]
    o = torch.empty_like(q)
    qo = torch.tensor([0, t], device=DEV, dtype=torch.int32)
    kv = torch.tensor([0, prefix], device=DEV, dtype=torch.int32)
    kvi = torch.arange(prefix, device=DEV, dtype=torch.int32)
    pl = torch.tensor([prefix], device=DEV, dtype=torch.int32)
    block_m, block_n, w, s = 32, 16, 8, 1
    grid = (1, QH, triton.cdiv(t, block_m))
    _extend_attention_split_kernel[grid](
        q, k_ext, v_ext, kc, vc, o, qo, kv, kvi, pl, SCALE,
        q, q.stride(0), q.stride(1), k_ext.stride(0), k_ext.stride(1),
        v_ext.stride(0), v_ext.stride(1), kc.stride(0), kc.stride(1),
        vc.stride(0), vc.stride(1), o.stride(0), o.stride(1),
        GROUP=GROUP, D=HD, BLOCK_D=triton.next_power_of_2(HD),
        BLOCK_DV=triton.next_power_of_2(HD), BLOCK_M=block_m, BLOCK_N=block_n,
        SLIDING_WINDOW=0, HAS_SINKS=False, num_warps=w, num_stages=s,
    )
    return o


def cutlass_offset_causal(q, k_ext, v_ext, kc, vc, prefix):
    """Offset-causal: row i of q (new tokens) attends to kv cols 0..prefix+i.
    Trick: prepend `prefix` dummy q rows -> total q_len = prefix+T, standard
    is_causal then gives row (prefix+i) attends to cols 0..prefix+i. Discard the
    first `prefix` output rows.
    """
    t = q.shape[0]
    dummy = torch.zeros(prefix, QH, HD, device=DEV, dtype=DT)
    q_padded = torch.cat([dummy, q], dim=0)           # [prefix+T, QH, HD]
    k = torch.cat([kc, k_ext], dim=0)                 # [prefix+T, KVH, HD]
    v = torch.cat([vc, v_ext], dim=0)                 # [prefix+T, KVH, HD]
    q4 = q_padded.permute(1, 0, 2).unsqueeze(0)       # [1, QH, prefix+T, HD]
    k4 = k.permute(1, 0, 2).unsqueeze(0).repeat_interleave(GROUP, dim=1)
    v4 = v.permute(1, 0, 2).unsqueeze(0).repeat_interleave(GROUP, dim=1)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        o4 = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True, scale=SCALE)
    o = o4.squeeze(0).permute(1, 0, 2)               # [prefix+T, QH, HD]
    return o[prefix:]                                 # [T, QH, HD]


def cutlass_custom_mask(q, k_ext, v_ext, kc, vc, prefix):
    """Explicit bool mask (row i attends to cols <= prefix+i)."""
    t = q.shape[0]
    k = torch.cat([kc, k_ext], dim=0)
    v = torch.cat([vc, v_ext], dim=0)
    q4 = q.permute(1, 0, 2).unsqueeze(0)
    k4 = k.permute(1, 0, 2).unsqueeze(0).repeat_interleave(GROUP, dim=1)
    v4 = v.permute(1, 0, 2).unsqueeze(0).repeat_interleave(GROUP, dim=1)
    kv_len = prefix + t
    rows = torch.arange(t, device=DEV)
    cols = torch.arange(kv_len, device=DEV)
    mask = cols[None, :] <= (prefix + rows)[:, None]  # [T, kv_len] True
    mask4 = mask.unsqueeze(0).unsqueeze(0)            # [1,1,T,kv_len]
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        o4 = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask4, scale=SCALE)
    return o4.squeeze(0).permute(1, 0, 2)


def timeit(fn, *a, reps=3):
    fn(*a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    torch.cuda.empty_cache()
    print(f"geometry QH={QH} KVH={KVH} HD={HD} T={T} prefix={PREFIX} (chunk 2 of 8192)")
    q, k_ext, v_ext, kc, vc = build(T, PREFIX)

    # reference (small T to bound [T,kv] memory)
    tr, ker, ver, kcr, vcr = build(512, 512, seed=1)
    ref = triton_extend(tr, ker, ver, kcr, vcr, 512).float()

    trit_ms = timeit(triton_extend, q, k_ext, v_ext, kc, vc, PREFIX)
    print(f"Triton extend (prefix):   {trit_ms:8.1f} ms   (production chunk-2 path)")

    for name, fn in [("CUTLASS offset-causal", cutlass_offset_causal),
                     ("CUTLASS custom-mask", cutlass_custom_mask)]:
        try:
            o = fn(tr, ker, ver, kcr, vcr, 512).float()
            rel = (o - ref).norm() / (ref.norm() + 1e-9)
            ms = timeit(fn, q, k_ext, v_ext, kc, vc, PREFIX)
            print(f"{name:24s}: {ms:8.1f} ms   relL2={rel:.2e}  "
                  f"{'OK' if rel < 1e-2 else 'MISMATCH'}   "
                  f"({trit_ms/ms:.2f}x vs triton)")
        except Exception as e:
            print(f"{name:24s}: ERROR {str(e)[:90]}")


if __name__ == "__main__":
    main()
