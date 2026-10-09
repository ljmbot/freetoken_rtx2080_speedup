"""Idea-5 CUTLASS attention backend (sm75).

PyTorch's built-in **memory-efficient (CUTLASS) SDPA backend** runs on sm75 and,
at the true production geometry (16 qh / 2 kvh / hd=256, T=4096, fresh prefill),
is ~13x faster than the Triton extend kernel (12.1 ms vs 162.0 ms; measured in
scripts/sm75/attn_microbench.py + a same-process A/B, relL2 1.73e-04 vs Triton).

The only wrinkle: the CUTLASS backend requires equal q/kv head counts, so GQA
(16 qh / 2 kvh) must be expanded to MHA via ``repeat_interleave`` (0.4 ms + 34 MB
at T=4096 — negligible). Decode (M==1) is untouched: this backend routes decode
to the Triton paged kernel, so it is safe to use as the single (P+D) backend.

Design: env-gated and default-off. It is registered under the name ``cutlass`` and
is only selected when the user explicitly passes ``--attention-backend cutlass``
(or ``cutlass,triton``). ``qwen.sh`` keeps ``--attention-backend triton`` by
default, so this is a no-op until opted in.

Two prefill paths use CUTLASS mem-efficient SDPA; decode is always Triton:
  * fresh (prefix==0): dense, is_causal=True.
  * prefix (chunk 2+ of a chunked prefill): gather the request's full KV from the
    paged cache, GQA->MHA expand, and run with an offset-causal mask. A memory guard
    bails to the Triton extend kernel if the expansion would exceed the free VRAM
    (it scales with context length).

Measured (sm75, 16 qh / 2 kvh / hd=256, 10 full-attn layers of 40):
  * kernel: fresh 4096 = 13x, prefix 4096 = 10x vs the Triton extend kernel.
  * end-to-end: flat at <=4096 (attention is ~2% of prefill), ~2.9x at 61K tokens
    (attention dominates; Triton 941 s vs CUTLASS 325 s cold). Output is
    byte-for-byte identical to Triton (greedy, seed=1234).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import os
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .triton import TritonAttentionBackend, TritonMetadata

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


class CutlassAttentionBackend(TritonAttentionBackend):
    """Triton backend whose fresh-prefill (dense, no-prefix) attention uses PyTorch's
    CUTLASS memory-efficient SDPA instead of the Triton extend kernel. Decode and the
    paged/prefix path fall back to the Triton kernels (inherited).

    Inherits all of TritonAttentionBackend's metadata / CUDA-graph machinery verbatim;
    only ``forward`` is overridden to route the dense fresh-prefill case to CUTLASS.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._cutlass_enabled = True  # gated by explicit --attention-backend cutlass
        # KV segment size for the segmented GQA extend path (working set scales
        # with this, not with context length). 8192 -> ~134 MB expansion at
        # hd=256; measured 1.02x the full-context path at t=4096/kv=65536.
        self.EXTEND_SEG = int(os.environ.get("FREETOKEN_CUTLASS_EXTEND_SEG", "8192"))
        # The segmented GQA extend path needs the aten efficient-attention op with
        # LSE output (per-segment online-softmax combine). If a future/older torch
        # lacks it, _cutlass_extend falls back to the full-context expansion.
        self._has_aten_lse = hasattr(
            torch.ops.aten, "_scaled_dot_product_efficient_attention"
        )

    @staticmethod
    def _unpack_v(vb: torch.Tensor, head_dim: int) -> torch.Tensor:
        """Unpack a packed sub-byte V tile to signed int values (int32).

        Mirrors the Triton ``_deq_v4``/``_deq_v5`` helpers (and the inverse of
        ``mha_pool._pack_v``). ``vb`` is ``[slots, kv_heads, v_dim]`` uint8 with
        ``v_dim = head_dim * bits // 8``; returns ``[slots, kv_heads, head_dim]``
        int32 in ``[-2^(bits-1), 2^(bits-1)-1]``. int4: two's-complement nibbles,
        low nibble = field 2i (sign-extend ``(nib ^ 8) - 8``). int5: 8 fields per
        5-byte group, field j at bit ``5j % 8`` of byte ``5j // 8``; a field
        straddles the boundary when ``r > 3`` so stitch ``((b0 >> r) | (b1 <<
        (8 - r))) & 0x1F`` and sign-extend ``(f ^ 16) - 16``.
        """
        v = vb.to(torch.int32)
        w = v.shape[-1]
        if w * 2 == head_dim:  # int4
            lo = (v & 0xF) ^ 8
            hi = (v >> 4) ^ 8
            out = torch.stack([lo, hi], dim=-1).reshape(*v.shape[:-1], -1) - 8
            return out
        assert w * 8 == 5 * head_dim, f"unexpected V slab width {w} for head_dim {head_dim}"
        # int5
        j = torch.arange(head_dim, device=v.device, dtype=torch.int32)
        b0i = (5 * j) >> 3
        r = (5 * j) & 7
        b0 = v.index_select(-1, b0i)
        b1 = v.index_select(-1, (b0i + 1).clamp(max=w - 1))
        lo = b0 >> r
        hi = torch.where(r > 3, b1 << (8 - r), torch.zeros_like(b1))
        f = (lo | hi) & 0x1F
        return (f ^ 16) - 16

    def _dequant_kv(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale_flat: torch.Tensor,
        v_scale_flat: torch.Tensor,
        slots: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequant the K/V at paged-cache slots ``[num_slots]`` to ``dtype`` (fp16).

        The Q8_0 cache is int8 with a per-(token, head) fp32 scale; the CUTLASS SDPA
        paths need fp16, so dequant the selected subset here (the Triton kernels
        dequantize in-kernel instead). With sub-byte V (``--kv-v-bits 4/5``) the V
        slab is packed uint8 (width ``head_dim*bits//8``) and is unpacked here
        first — reading it as int8 values would silently mix two fields per byte.
        Returns ``(k, v)`` each ``[num_slots, kv_heads, head_dim]`` in ``dtype``.
        """
        k = k_cache[slots].to(torch.float32)
        v_raw = v_cache[slots]
        if v_raw.shape[-1] == k.shape[-1]:
            v = v_raw.to(torch.float32)  # int8 V (v_bits=8)
        else:
            v = self._unpack_v(v_raw, k.shape[-1])  # packed int4/int5
        ks = k_scale_flat[slots].unsqueeze(-1)  # [num_slots, kv_heads, 1]
        vs = v_scale_flat[slots].unsqueeze(-1)
        return (k * ks).to(dtype), (v * vs).to(dtype)

    def _cutlass_prefill(self, q, k_cache, v_cache, metadata, scale, kv_heads, head_dim,
                         k_scale_flat=None, v_scale_flat=None):
        """Dense fresh-prefill (prefix==0 for every request) via CUTLASS mem-efficient.

        Returns the output in the same layout as the Triton extend kernel
        (``[num_q_tokens, num_q_heads, head_dim]``), or ``None`` if the batch is not
        eligible (any request has a cached prefix) so the caller falls back to Triton.
        """
        # Eligibility: single-request, fresh (no cached prefix), fp16/bf16.
        # The production config is --max-running-requests 1, so bs==1 here.
        if metadata.indptr.numel() - 1 != 1:
            return None
        if bool((metadata.prefix_lens != 0).any().item()):
            return None
        if q.dtype not in (torch.float16, torch.bfloat16):
            return None

        num_q_tokens, num_q_heads, hd = q.shape
        if num_q_heads % kv_heads != 0:
            return None
        g = num_q_heads // kv_heads
        if num_q_tokens == 0:
            return q

        # q: [T, Hq, D] -> [1, Hq, T, D]
        q4 = q.permute(1, 0, 2).unsqueeze(0)
        # Fresh prefill: the KV for this request is exactly the extend k/v (dense),
        # so read them directly rather than gathering from the paged cache.
        k4 = k_cache[metadata.indptr[0] : metadata.indptr[1]].permute(1, 0, 2).unsqueeze(0)
        v4 = v_cache[metadata.indptr[0] : metadata.indptr[1]].permute(1, 0, 2).unsqueeze(0)
        if k_scale_flat is not None:
            # Q8_0: dequant the dense slice to fp16 before SDPA.
            k4, v4 = self._dequant_kv(
                k_cache, v_cache, k_scale_flat, v_scale_flat,
                torch.arange(metadata.indptr[0], metadata.indptr[1], device=q.device),
                q.dtype,
            )
            k4 = k4.permute(1, 0, 2).unsqueeze(0)
            v4 = v4.permute(1, 0, 2).unsqueeze(0)
        # GQA -> MHA (CUTLASS mem-efficient requires equal q/kv head counts).
        if g != 1:
            k4 = k4.repeat_interleave(g, dim=1)
            v4 = v4.repeat_interleave(g, dim=1)

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            o4 = F.scaled_dot_product_attention(
                q4, k4, v4, is_causal=True, scale=scale
            )
        # [1, Hq, T, D] -> [T, Hq, D]
        return o4.permute(0, 2, 1, 3).squeeze(0)

    def _cutlass_extend_seg(
        self, q, k_cache, v_cache, indices, metadata, scale, kv_heads, head_dim,
        k_scale_flat=None, v_scale_flat=None,
    ):
        """Segmented GQA extend: KV segments expanded 8x, per-segment LSE, online
        softmax combine. Bounded working set (~0.5-0.6 GB at t=4096, SEG=8192)
        regardless of context length, vs ~1.7 GB for the full-context expansion.
        Same speed (measured 1.02x at t=4096/kv=65536), same fp16 relL2 (~4e-4).
        """
        t, num_q_heads, hd = q.shape
        g = num_q_heads // kv_heads
        device_len = int(metadata.indptr[1].item())  # prefix + t
        prefix = int(metadata.prefix_lens[0].item())

        # Static guard: the per-segment working set is context-independent
        # (expansion + bias + kernel workspace), so check free VRAM once.
        try:
            free_bytes = torch.cuda.mem_get_info()[0]
        except Exception:
            free_bytes = None
        seg_peak = (
            2 * self.EXTEND_SEG * num_q_heads * hd * q.element_size()  # k+v expanded
            + t * self.EXTEND_SEG * q.element_size()  # bias base (stride-0 expanded)
            + 32 * 1024 * 1024  # out + running M/S/O buffers
            + 256 * 1024 * 1024  # kernel workspace estimate
        )
        if free_bytes is not None and free_bytes < seg_peak + 150 * 1024 * 1024:
            return None

        q4 = q.permute(1, 0, 2).unsqueeze(0)  # [1, Hq, t, hd]
        rows = torch.arange(t, device=q.device)
        bound = prefix + rows  # row i attends to kv cols 0..bound[i]
        neg = float("-inf")
        # Accumulate the online-softmax combine in fp32 (the aten op returns LSE in
        # fp32; fp32 running buffers are also more stable). The final output is cast
        # back to q.dtype so downstream fp16 layers (MoE gate) see the right dtype.
        M = torch.full((1, num_q_heads, t), neg, device=q.device, dtype=torch.float32)
        S = torch.zeros(1, num_q_heads, t, device=q.device, dtype=torch.float32)
        O = torch.zeros(1, num_q_heads, t, hd, device=q.device, dtype=torch.float32)

        for s in range(0, device_len, self.EXTEND_SEG):
            e_real = min(s + self.EXTEND_SEG, device_len)
            if s > bound[-1]:
                break  # no row sees any col in this segment
            seg_real = e_real - s
            # The mem-efficient kernel requires the KV segment length to be a
            # multiple of 8 (attn_bias strideM alignment). Pad the last partial
            # segment up to a multiple of 8; the extra positions are >= device_len
            # so they're masked out by the causal bound for every row.
            seg_pad = (seg_real + 7) // 8 * 8
            sl = slice(s, e_real)
            if k_scale_flat is not None:
                k_seg, v_seg = self._dequant_kv(
                    k_cache, v_cache, k_scale_flat, v_scale_flat, indices[sl], q.dtype
                )
            else:
                k_seg = k_cache[indices[sl]]
                v_seg = v_cache[indices[sl]]
            if seg_pad > seg_real:
                pad = seg_pad - seg_real
                k_seg = torch.cat(
                    [k_seg, torch.zeros(pad, *k_seg.shape[1:], device=k_seg.device, dtype=k_seg.dtype)], dim=0)
                v_seg = torch.cat(
                    [v_seg, torch.zeros(pad, *v_seg.shape[1:], device=v_seg.device, dtype=v_seg.dtype)], dim=0)
            if g != 1:
                ke = k_seg.permute(1, 0, 2).unsqueeze(0).repeat_interleave(g, dim=1)
                ve = v_seg.permute(1, 0, 2).unsqueeze(0).repeat_interleave(g, dim=1)
            else:
                ke = k_seg.permute(1, 0, 2).unsqueeze(0)
                ve = v_seg.permute(1, 0, 2).unsqueeze(0)
            if seg_pad == seg_real and e_real <= bound[0]:
                # every row sees the whole (unpadded) segment -> dense
                out, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
                    q4, ke, ve, None, True, 0.0, False, scale=scale
                )
            else:
                # rows with bound[i] > s see cols s..min(e_real, bound[i]); padded
                # cols (>= seg_real) are masked. The bias must be full BMHK
                # ([1, Hq, t, seg_pad]) for the aten op.
                cols = torch.arange(seg_pad, device=q.device)
                m = (cols[None, :] < seg_real) & (cols[None, :] <= (bound - s)[:, None])
                bias = torch.where(m, 0.0, neg).to(q.dtype).unsqueeze(0).unsqueeze(0) \
                    .expand(1, num_q_heads, t, seg_pad)
                out, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
                    q4, ke, ve, bias, True, 0.0, False, scale=scale
                )
            del ke, ve
            # The mem-efficient kernel pads LSE to a multiple of 64 (e.g. t=2668 ->
            # 2688) while out keeps the true length; slice LSE back to t so the
            # combine (buffers sized to t) lines up.
            if lse.shape[2] != t:
                lse = lse[:, :, :t]
            # Fully-masked rows give lse=-inf/NaN and out=NaN; zero their weight.
            lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, neg))
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            # Online softmax merge (split-KV combine): M' = max(M, lse);
            # O' = O*exp(M-M') + out*exp(lse-M'); S' = S*exp(M-M') + exp(lse-M').
            M_new = torch.maximum(M, lse)
            a = torch.exp(M - M_new)
            b = torch.exp(lse - M_new)
            O = O * a.unsqueeze(-1) + out * b.unsqueeze(-1)
            S = S * a + b
            M = M_new
        return (O / S.unsqueeze(-1)).to(q.dtype).permute(0, 2, 1, 3).squeeze(0)

    def _cutlass_extend(
        self, q, k_cache, v_cache, indices, metadata, scale, kv_heads, head_dim,
        k_scale_flat=None, v_scale_flat=None,
    ):
        """Paged prefix-extend (chunk 2+ of a chunked prefill) via CUTLASS mem-efficient.

        This is where most long-context attention work lives: each chunk's new tokens
        attend to the full cached prefix + the new tokens. The Triton extend kernel is
        the slow path here (~482 ms vs ~48 ms for CUTLASS at T=4096, prefix=4096,
        measured in scripts/sm75/cutlass_prefix_bench.py, relL2 2.2e-04).

        Default path (torch with the aten efficient-attention LSE op): segmented GQA
        (``_cutlass_extend_seg``) — bounded ~0.5-0.6 GB working set at any context
        length. Fallback (older torch): full-context GQA->MHA expansion with a
        free-VRAM guard that bails to Triton when the expansion would exceed 60% of
        the free VRAM (it scales with context length, ~1.7 GB at 64K).

        Returns the output in the same layout as the Triton extend kernel
        (``[num_q_tokens, num_q_heads, head_dim]``), or ``None`` to fall back to
        Triton (not eligible, or the memory guard bailed).
        """
        # Eligibility: single-request, has a cached prefix, fp16/bf16.
        if metadata.indptr.numel() - 1 != 1:
            return None
        prefix = int(metadata.prefix_lens[0].item())
        if prefix <= 0:
            return None  # fresh -> handled by _cutlass_prefill
        if q.dtype not in (torch.float16, torch.bfloat16):
            return None

        t, num_q_heads, hd = q.shape
        if num_q_heads % kv_heads != 0 or t == 0:
            return None
        g = num_q_heads // kv_heads
        device_len = int(metadata.indptr[1].item())  # prefix + t

        if self._has_aten_lse:
            return self._cutlass_extend_seg(
                q, k_cache, v_cache, indices, metadata, scale, kv_heads, head_dim,
                k_scale_flat, v_scale_flat,
            )

        # Legacy full-context expansion (torch without the aten LSE op).
        # Memory guard: the GQA->MHA expansion materializes k and v at full q-head
        # count for the whole device_len, plus the [t, device_len] bool mask. This
        # scales with context length (e.g. ~1.4 GB at 64K), so bail to Triton if it
        # would consume more than a fraction of the free VRAM (avoids an OOM mid-run).
        try:
            free_bytes = torch.cuda.mem_get_info()[0]
        except Exception:
            free_bytes = None
        peak_bytes = (
            2 * device_len * num_q_heads * hd * q.element_size()  # k + v expanded
            + t * device_len  # bool mask (1 byte)
            + t * num_q_heads * hd * q.element_size()  # q4
        )
        if free_bytes is not None and peak_bytes > 0.6 * free_bytes:
            return None

        # Gather the request's full KV (prefix + extend) in token order.
        if k_scale_flat is not None:
            # Q8_0: dequant the gathered int8 KV to fp16 before SDPA.
            k_full, v_full = self._dequant_kv(
                k_cache, v_cache, k_scale_flat, v_scale_flat, indices, q.dtype
            )
        else:
            k_full = k_cache[indices]  # [device_len, kv_heads, hd]
            v_full = v_cache[indices]  # [device_len, kv_heads, hd]
        q4 = q.permute(1, 0, 2).unsqueeze(0)  # [1, Hq, t, hd]
        if g != 1:
            k4 = k_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(g, dim=1)
            v4 = v_full.permute(1, 0, 2).unsqueeze(0).repeat_interleave(g, dim=1)
        else:
            k4 = k_full.permute(1, 0, 2).unsqueeze(0)
            v4 = v_full.permute(1, 0, 2).unsqueeze(0)
        # Offset-causal mask: new-token row i attends to kv cols 0..prefix+i.
        rows = torch.arange(t, device=q.device)
        cols = torch.arange(device_len, device=q.device)
        mask = (cols[None, :] <= (prefix + rows)[:, None]).unsqueeze(0).unsqueeze(0)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            o4 = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask, scale=scale)
        return o4.permute(0, 2, 1, 3).squeeze(0)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.attention import (
            decode_paged_attention,
            extend_paged_attention,
            paged_attention,
        )

        metadata = batch.attn_metadata
        assert isinstance(metadata, TritonMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads, head_dim = k_raw.shape[-2], k_raw.shape[-1]
        assert head_dim == q.shape[-1]
        k_cache = k_raw.view(-1, kv_heads, head_dim)
        # Packed sub-byte V (v_bits<8) rows are narrower than head_dim — view at
        # the raw slab width (the kernels/_dequant_kv infer the bit width from it).
        v_cache = v_raw.view(-1, kv_heads, v_raw.shape[-1])

        # Q8_0 KV cache: the cache is int8 + per-(token, head) fp32 scales. The Triton
        # kernels dequantize in-kernel (they take k_scale/v_scale); the CUTLASS SDPA
        # paths dequantize the gathered subset to fp16 first (they cannot read int8).
        q8 = getattr(self.kvcache, "q8", False)
        if q8:
            k_scale_flat = self.kvcache.k_scale(layer_id).reshape(-1, kv_heads)
            v_scale_flat = self.kvcache.v_scale(layer_id).reshape(-1, kv_heads)
        else:
            k_scale_flat, v_scale_flat = None, None

        spec = attn_spec or AttentionSpec()
        indices = metadata.indices
        if spec.sliding_window is not None and metadata.swa_indices is not None:
            indices = metadata.swa_indices
        scale = spec.sm_scale if spec.sm_scale is not None else q.shape[-1] ** -0.5

        # Decode: identical to Triton (untouched).
        if metadata.is_decode and q.dtype in (torch.float16, torch.bfloat16):
            bs = metadata.indptr.numel() - 1
            self._ensure_decode_scratch(metadata, bs, q.shape[1], q.shape[-1])
            assert metadata.attn_logits is not None
            assert metadata.attn_lse is not None
            assert metadata.num_kv_splits is not None
            return decode_paged_attention(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                indptr=metadata.indptr,
                indices=indices,
                q_positions=metadata.q_positions,
                attn_logits=metadata.attn_logits[:bs],
                attn_lse=metadata.attn_lse[:bs],
                num_kv_splits=metadata.num_kv_splits[:bs],
                max_kv_splits=self.max_kv_splits,
                sm_scale=scale,
                sliding_window=spec.sliding_window,
                sinks=spec.sinks,
                k_scale=k_scale_flat,
                v_scale=v_scale_flat,
            )

        # Prefill (dense, no sliding window, no sinks): CUTLASS mem-efficient.
        # Fresh (prefix==0) -> _cutlass_prefill; chunk 2+ (prefix>0) -> _cutlass_extend.
        # Both return None if ineligible (e.g. memory guard), falling back to Triton.
        if (
            self._cutlass_enabled
            and (not metadata.is_decode)
            and q.dtype in (torch.float16, torch.bfloat16)
            and spec.sliding_window is None
            and spec.sinks is None
        ):
            o = self._cutlass_prefill(
                q, k_cache, v_cache, metadata, scale, kv_heads, head_dim,
                k_scale_flat, v_scale_flat,
            )
            if o is not None:
                return o
            o = self._cutlass_extend(
                q, k_cache, v_cache, indices, metadata, scale, kv_heads, head_dim,
                k_scale_flat, v_scale_flat,
            )
            if o is not None:
                return o

        # Fallback: Triton extend/paged (paged prefix path, SWA, or ineligible).
        if (
            (not metadata.is_decode)
            and q.dtype in (torch.float16, torch.bfloat16)
            and (q.shape[-1] <= 256 or metadata.max_q_len >= self.prefill_tile_min_q)
        ):
            return extend_paged_attention(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                qo_indptr=metadata.cu_seqlens_q_gpu,
                kv_indptr=metadata.indptr,
                kv_indices=indices,
                prefix_lens=metadata.prefix_lens,
                max_q_len=metadata.max_q_len,
                sm_scale=scale,
                sliding_window=spec.sliding_window,
                sinks=spec.sinks,
                k_extend=k.view(q.shape[0], kv_heads, head_dim),
                v_extend=v.view(q.shape[0], kv_heads, head_dim),
                k_scale=k_scale_flat,
                v_scale=v_scale_flat,
            )
        return paged_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            indptr=metadata.indptr,
            indices=indices,
            q_to_req=metadata.q_to_req,
            q_positions=metadata.q_positions,
            sm_scale=scale,
            sliding_window=spec.sliding_window,
            sinks=spec.sinks,
            k_scale=k_scale_flat,
            v_scale=v_scale_flat,
        )
