"""Phase 3 verification: does the production prefill path warm the decode LRU cache?

Production (qwen_q8.sh) runs with --disable-moe-prefill-overlap, so prefill uses
materialize_layer (synchronous full-layer H2D into the first num_experts slots), NOT
the double-buffer overlap path. Decode uses ensure_experts (lru_ensure) which reads
slot_for_id to decide hit vs miss.

Question: after a full prefill (materialize_layer for every layer in order), how many
(layer, expert) entries in slot_for_id are mapped (>=0, a decode hit) vs cold (-1)?
If prefill usefully warmed decode, we'd expect a large fraction mapped. If it doesn't,
the decode cache is effectively cold after prefill.

We exercise the REAL Triton kernels (materialize_layer) on a small real cache.
"""
import torch
from freetoken.moe.offload_cache import OffloadMoeCache

L = 8        # num_layers (small for the test)
E = 16       # num_experts
S = 64       # cache_size (unified pool, shared across all layers)
dev = torch.device("cuda")

cache = OffloadMoeCache(
    num_layers=L, num_experts=E, cache_size=S, device=dev,
    cache_policy="lru", prefill_overlap=False, quant_format="bf16",
)
# Tiny dummy host banks: [num_experts, 4, 4] bf16 per bank per layer.
src = {}
for bank in ("gate_up", "down"):
    src[bank] = [torch.randn(E, 4, 4, dtype=torch.bfloat16, pin_memory=True) for _ in range(L)]
cache.set_bank_sources(src)

def mapped_count():
    """How many (layer, expert) entries decode would treat as a hit (slot_for_id >= 0)."""
    return int((cache.slot_for_id >= 0).sum().item()), L * E

print(f"cache: L={L} E={E} S={S}  (decode uses lru_ensure over slot_for_id)")
print(f"before prefill: mapped={mapped_count()[0]}/{mapped_count()[1]} (expect 0/0, cold)")

# --- Simulate a production prefill: materialize_layer for every layer in order ---
# (materialize_layer is pure LRU-map bookkeeping; copy_missing is just the data copy
#  and does not touch slot_for_id, so it is omitted here.)
for layer_id in range(L):
    cache.materialize_layer(layer_id)

tot_mapped, total = mapped_count()
print(f"after prefill (materialize all {L} layers): mapped={tot_mapped}/{total}")
per_layer = []
for lid in range(L):
    m = int((cache.slot_for_id[lid] >= 0).sum().item())
    per_layer.append(m)
print(f"  per-layer mapped (decode-hit potential): {per_layer}")
print(f"  -> fraction of (layer,expert) pairs a decode token could hit: {tot_mapped/total*100:.1f}%")

# --- Now run a few decode steps (ensure_experts) and measure realized hit rate ---
# Simulate decode routing: each layer routes top_k experts. Use a fixed (deterministic)
# routing so the "warm" benefit is maximized if it exists. We reuse the SAME experts the
# prefill touched (best case for a warm cache).
cache.reset_stats()
cache.collect_stats = True
topk = 8
# Deterministic routing: each layer routes experts [0..topk-1] every token (max locality).
tokens = 64
for t in range(tokens):
    for lid in range(L):
        ids = torch.zeros(topk, dtype=torch.int32, device=dev)
        for k in range(topk):
            ids[k] = k  # route expert k
        cache.ensure_experts(lid, ids)

st = cache.decode_miss_stats()
print(f"\nafter {tokens} decode tokens (topk={topk}, best-case routing):")
print(f"  miss_rate = {st['miss_rate']*100:.1f}%   (active_per_layer={st['active_per_layer']:.2f})")
print(f"  -> realized decode hit rate = {(1-st['miss_rate'])*100:.1f}%")
