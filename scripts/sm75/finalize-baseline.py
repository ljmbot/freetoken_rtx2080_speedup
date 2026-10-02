#!/usr/bin/env python3
"""Finalize baseline results:
  1. Preserve the original (buggy 2058-full) prefix-cached rows as provenance.
  2. Swap the 6 prefix-cached rows in results.jsonl for the fixed 2010-full rows.
  3. Rebuild the corpus's prefix-cached entries (deterministic 2000-content builder).
  4. Print the median table."""
import json, os, shutil, hashlib, collections
from transformers import AutoTokenizer

BASE = "artifacts/sm75-prefill"
MODEL = os.path.expanduser("~/Models/Qwen3.6-35B-A3B-NVFP4")
FIXED = "/tmp/prefix-fixed.jsonl"

def sha_tokens(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]

def _to_ids(x):
    if isinstance(x, (list, tuple)): return [int(t) for t in x]
    if hasattr(x, "input_ids"):
        ids = x.input_ids
        ids = ids[0] if isinstance(ids, (list, tuple)) and len(ids) and isinstance(ids[0], (list, tuple)) else ids
        return [int(t) for t in ids]
    return [int(t) for t in x]

def build_content_token_ids(tok, n_tokens, salt_token=None):
    base_words = []; i = 0
    while len(base_words) < n_tokens + 64:
        base_words.append(f"tok{i:06d}"); i += 1
    text = " ".join(base_words)
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    if len(ids) < n_tokens:
        pad = tok.encode(" pad", add_special_tokens=False)
        while len(ids) < n_tokens:
            ids.extend(pad)
        ids = ids[:n_tokens]
    if salt_token is not None:
        ids = [salt_token] + ids[1:]
    return ids

def apply_chat(tok, content_ids):
    text = tok.decode(content_ids, skip_special_tokens=False)
    full = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                   add_generation_prompt=True,
                                   chat_template_kwargs={"enable_thinking": False})
    return _to_ids(full)

# 1. preserve buggy rows
orig = f"{BASE}/results.jsonl"
prov = f"{BASE}/results-prefixcached-2058-bug.jsonl"
rows = [json.loads(l) for l in open(orig)]
buggy = [r for r in rows if r.get("workload") == "prefix-cached"]
with open(prov, "w") as f:
    for r in buggy:
        f.write(json.dumps(r) + "\n")
print(f"preserved {len(buggy)} buggy prefix-cached rows -> {prov}")

# 2. swap in fixed rows
fixed = [json.loads(l) for l in open(FIXED)]
assert len(fixed) == 6, f"expected 6 fixed rows, got {len(fixed)}"
kept = [r for r in rows if r.get("workload") != "prefix-cached"]
# rebuild in original workload order: prefill-128,512,2048, mixed-512,2048, prefix-cached, agent
order = ["prefill-128", "prefill-512", "prefill-2048", "mixed-512", "mixed-2048", "prefix-cached", "agent-envelope"]
by = collections.defaultdict(list)
for r in kept:
    by[r["workload"]].append(r)
for r in fixed:
    by["prefix-cached"].append(r)
final = []
for w in order:
    final.extend(by.get(w, []))
with open(orig, "w") as f:
    for r in final:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(final)} rows -> {orig}")

# 3. rebuild corpus prefix-cached entries
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
corpus = json.load(open(f"{BASE}/benchmark-corpus.json"))
ids = build_content_token_ids(tok, 2000, salt_token=None)
full_ids = apply_chat(tok, ids)
ph = sha_tokens(full_ids)
new_pc = []
for run in (1, 2, 3, 4, 5):
    new_pc.append({"workload": "prefix-cached", "run": run, "n_content_tokens": 2000,
                   "full_token_ids": full_ids, "prompt_hash": ph,
                   "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16}, "seed": 1234})
corpus["prompts"] = [p for p in corpus["prompts"] if p["workload"] != "prefix-cached"] + new_pc
with open(f"{BASE}/benchmark-corpus.json", "w") as f:
    json.dump(corpus, f, indent=2)
print(f"rebuilt corpus: {len(corpus['prompts'])} prompts; prefix-cached full_ids_len={len(full_ids)} hash={ph}")

# 4. median table
rows = [json.loads(l) for l in open(orig)]
print("\n=== BASELINE MEDIAN TABLE (7 workloads) ===")
print(f"{'workload':16s} {'runs':>4s} {'median_ttft_ms':>15s} {'median_dec_tps':>16s} {'median_in':>10s} {'cached':>7s}")
for w in order:
    rs = [r for r in rows if r["workload"] == w and not r.get("warmup") and r.get("client_ttft_ms") is not None]
    if not rs:
        print(f"{w:16s} (no timed rows)"); continue
    tt = sorted(r["client_ttft_ms"] for r in rs)
    med_tt = tt[len(tt)//2]
    dec = sorted(r["decode_tokens_per_second"] for r in rs if r.get("decode_tokens_per_second"))
    med_dec = f"{dec[len(dec)//2]:.1f}" if dec else "-"
    ins = [r["input_tokens"] for r in rs if r.get("input_tokens")]
    med_in = ins[len(ins)//2] if ins else "-"
    cached = rs[0].get("prefix_cached_tokens")
    print(f"{w:16s} {len(rs):>4d} {med_tt:>15.0f} {med_dec:>16s} {str(med_in):>10s} {str(cached):>7s}")
