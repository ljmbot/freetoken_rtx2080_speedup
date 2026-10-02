"""Phase B instrumentation: observable backend selection + prefill profiling.

Gated entirely on ``FREETOKEN_PROFILE=1`` (read once at import). When unset, every
helper here is a no-op with negligible cost, so normal serving is untouched.

Env knobs (all optional):
  FREETOKEN_PROFILE=1            enable
  FREETOKEN_PROFILE_AFTER=<n>    skip the first n prefill forwards (warmup/compile)
  FREETOKEN_PROFILE_COUNT=<c>    capture the next c prefill forwards
  FREETOKEN_PROFILE_OUT=<dir>    where to write traces + tables (default under artifacts)

The engine runs in a spawned subprocess, so this is driven by env at server launch,
not by an HTTP endpoint.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Callable

PROFILE = os.environ.get("FREETOKEN_PROFILE", "0") == "1"
ARM_AFTER = int(os.environ.get("FREETOKEN_PROFILE_AFTER", "0"))
ARM_COUNT = int(os.environ.get("FREETOKEN_PROFILE_COUNT", "1"))
OUT_DIR = os.environ.get(
    "FREETOKEN_PROFILE_OUT", "artifacts/sm75-prefill/profile"
)


@contextmanager
def nvtx_range(name: str):
    """Push an NVTX range, always on (matches the codebase's unconditional
    ``nvtx_annotate`` convention). Cheap CPU push/pop; no GPU work. Consumed by
    both torch.profiler and nsys."""
    import torch.cuda.nvtx as nvtx

    with nvtx.range(name):
        yield


# ---------------------------------------------------------------------------
# Prefill capture
# ---------------------------------------------------------------------------
_prefill_count = 0
_captured = 0


def maybe_profile_prefill(batch: Any, forward_fn: Callable[[], Any]) -> Any:
    """Run one forward under torch.profiler (CPU+CUDA) when this prefill is armed.

    ``forward_fn`` is the zero-arg ``self.model.forward``. Returns its output. Skips
    decode (only prefill is profiled) and the first ``ARM_AFTER`` prefills (warmup /
    Triton autotune / cuBLAS heuristics).
    """
    global _prefill_count, _captured
    if not PROFILE:
        return forward_fn()
    if not getattr(batch, "is_prefill", False):
        return forward_fn()

    _prefill_count += 1
    armed = ARM_AFTER < _prefill_count <= ARM_AFTER + ARM_COUNT
    if not armed:
        return forward_fn()

    import torch
    from torch.profiler import ProfilerActivity, profile

    os.makedirs(OUT_DIR, exist_ok=True)
    label = f"n{int(batch.input_ids.shape[0])}" if hasattr(batch, "input_ids") else "?"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        out = forward_fn()
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    trace_path = f"{OUT_DIR}/prefill-{_prefill_count:02d}-{label}.json"
    prof.export_chrome_trace(trace_path)
    table = build_kernel_table(prof, trace_path)
    table.update(
        label=label,
        prefill_idx=_prefill_count,
        elapsed_ms=round(elapsed_ms, 2),
        total_input_tokens=int(batch.input_ids.shape[0]) if hasattr(batch, "input_ids") else None,
    )
    table_path = trace_path + ".table.json"
    with open(table_path, "w") as fh:
        json.dump(table, fh, indent=2)
    _captured += 1
    print(
        f"[profile] captured prefill #{_prefill_count} ({label}, "
        f"{table['total_input_tokens']} tok) {elapsed_ms:.1f} ms -> {trace_path}",
        flush=True,
    )
    return out


def build_kernel_table(prof: Any, trace_path: str) -> dict:
    """Aggregate a torch profiler into (a) top CUDA ops by device time and
    (b) GPU time attributed to each NVTX sub-stage (via the chrome trace)."""
    table: dict = {}

    # (a) per-op table from key_averages (device time = self_device_time_total).
    rows = []
    for ev in prof.key_averages():
        dev_us = getattr(ev, "self_device_time_total", 0) or 0
        if dev_us <= 0:
            continue
        rows.append(
            {
                "op": ev.key,
                "count": ev.count,
                "dev_us": round(dev_us, 1),
                "dev_ms": round(dev_us / 1000.0, 3),
            }
        )
    rows.sort(key=lambda r: r["dev_us"], reverse=True)
    table["top_ops_by_device_time"] = rows[:40]
    total_dev_us = sum(r["dev_us"] for r in rows)
    table["total_device_us"] = round(total_dev_us, 1)

    # (b) NVTX-stage attribution: parse the chrome trace, match each CUDA kernel to
    #     the innermost enclosing NVTX range.
    try:
        table["by_nvtx_stage"] = _nvtx_stage_attribution(trace_path)
    except Exception as exc:  # noqa: BLE001
        table["by_nvtx_stage"] = {"error": f"{type(exc).__name__}: {exc}"}
    return table


def _nvtx_stage_attribution(trace_path: str) -> dict:
    with open(trace_path) as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", [])

    # NVTX ranges: torch exports them as X events with cat in these buckets.
    nvtx_cats = {"nvtx_push_pop", "nvtx", "cuda_nvtx", "user_annotation"}
    ranges = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in nvtx_cats:
            continue
        dur = e.get("dur")
        if not dur:
            continue
        name = e.get("name", "")
        # Strip layer suffixes like "Layer_12" -> keep, but normalise sub-stage prefixes.
        ranges.append((e["ts"], e["ts"] + dur, name))

    kernels = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "cuda_memcpy", "cuda_memset"):
            continue
        dur = e.get("dur")
        if not dur:
            continue
        kernels.append((e["ts"], dur, e.get("name", ""), e.get("cat")))

    # For each kernel, find the innermost enclosing NVTX range (smallest dur containing ts).
    stage_us: dict[str, float] = {}
    stage_count: dict[str, int] = {}
    unattributed_us = 0.0
    unattributed_count = 0
    for ts, dur, name, cat in kernels:
        best = None
        for s, e, nm in ranges:
            if s <= ts and ts + dur <= e:
                if best is None or (e - s) < (best[1] - best[0]):
                    best = (s, e, nm)
        if best is None:
            unattributed_us += dur
            unattributed_count += 1
        else:
            stage_us[best[2]] = stage_us.get(best[2], 0.0) + dur
            stage_count[best[2]] = stage_count.get(best[2], 0) + 1

    # Group by the leading token of the range name (e.g. "Layer_12" -> "Layer",
    # "GDN_recurrence" stays "GDN_recurrence" since it is already a sub-stage label).
    grouped: dict[str, float] = {}
    for nm, us in stage_us.items():
        key = "Layer" if nm.startswith("Layer_") else nm
        grouped[key] = grouped.get(key, 0.0) + us

    total = sum(stage_us.values()) or 1.0
    out = {
        "total_attributed_us": round(sum(stage_us.values()), 1),
        "unattributed_us": round(unattributed_us, 1),
        "unattributed_count": unattributed_count,
        "num_nvtx_ranges": len(ranges),
        "num_kernels": len(kernels),
        "stages": [
            {"stage": k, "us": round(v, 1), "ms": round(v / 1000.0, 3), "pct": round(100.0 * v / total, 2)}
            for k, v in sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)
        ],
    }
    return out


__all__ = [
    "PROFILE",
    "nvtx_range",
    "maybe_profile_prefill",
    "build_kernel_table",
]
