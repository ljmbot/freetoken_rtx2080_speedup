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

    def _cutlass_prefill(self, q, k_cache, v_cache, metadata, scale, kv_heads, head_dim):
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

    def _cutlass_extend(
        self, q, k_cache, v_cache, indices, metadata, scale, kv_heads, head_dim
    ):
        """Paged prefix-extend (chunk 2+ of a chunked prefill) via CUTLASS mem-efficient.

        This is where most long-context attention work lives: each chunk's new tokens
        attend to the full cached prefix + the new tokens. The Triton extend kernel is
        the slow path here (~482 ms vs ~48 ms for CUTLASS at T=4096, prefix=4096,
        measured in scripts/sm75/cutlass_prefix_bench.py, relL2 2.2e-04).

        Approach: gather the request's full KV (prefix + extend) from the paged cache
        into a dense tensor (``indices`` is the in-order slot list for all
        ``device_len`` tokens), expand GQA->MHA, and run SDPA with an explicit
        offset-causal mask (row i of the T new tokens attends to kv cols 0..prefix+i).

        Returns the output in the same layout as the Triton extend kernel
        (``[num_q_tokens, num_q_heads, head_dim]``), or ``None`` to fall back to
        Triton (not eligible, or the GQA expansion would exceed the free-memory guard).
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
        v_cache = v_raw.view(-1, kv_heads, head_dim)

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
                q, k_cache, v_cache, metadata, scale, kv_heads, head_dim
            )
            if o is not None:
                return o
            o = self._cutlass_extend(
                q, k_cache, v_cache, indices, metadata, scale, kv_heads, head_dim
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
        )
