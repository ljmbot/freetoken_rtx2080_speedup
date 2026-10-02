#!/usr/bin/env python3
"""Phase A/B benchmark harness for the FreeToken SM75 prefill project.

Design follows freetoken_project.md:
  - fixed tokenised prompts, same checkpoint for every A/B comparison
  - records token IDs/hash, prompt text, chat-template settings, seed, sampling params,
    and ACTUAL input/output token counts
  - 1 warm-up + N timed repetitions; reports medians and every raw sample
  - separates: process-cold / kernel-warm-uncached / prefix-cached
  - does NOT count an empty SSE role/header event as a generated token
  - client_TTFT = time to first token of ANY kind (content or reasoning); the model is run
    with enable_thinking=False so the first token is a real content token (prefill proxy)

Workloads (per the plan):
  prefill-128 / prefill-512 / prefill-2048 / prefill-4096   (max_tokens=1, uncached)
  mixed-512 / mixed-2048                                     (max_tokens=128)
  prefix-cached                                              (fixed prompt, repeated -> cache hit)
  agent-envelope                                             (representative Hermes system/tool prompt, 64-128 out)

Usage:
  .venv/bin/python scripts/sm75/bench.py \
      --host 127.0.0.1 --port 1920 \
      --model /home/lmeek/Models/Qwen3.6-35B-A3B-NVFP4 \
      --variant baseline --reps 5 --warmup 1 \
      --out artifacts/sm75-prefill/results.jsonl
"""
from __future__ import annotations
import argparse, json, hashlib, time, sys, os

import httpx
from transformers import AutoTokenizer

# sampling params (fixed for all A/B comparisons)
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": None}  # max_tokens set per workload
SEED = 1234


def sha_tokens(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def build_content_token_ids(tok, n_tokens, salt_token=None):
    """Produce exactly n_tokens content tokens (deterministic base, optional per-run salt at pos 0)."""
    # deterministic base text long enough to tokenize to >= n_tokens
    base_words = []
    i = 0
    while len(base_words) < n_tokens + 64:
        base_words.append(f"tok{i:06d}")
        i += 1
    text = " ".join(base_words)
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    if len(ids) < n_tokens:
        # pad deterministically
        pad = tok.encode(" pad", add_special_tokens=False)
        while len(ids) < n_tokens:
            ids.extend(pad)
        ids = ids[:n_tokens]
    if salt_token is not None:
        ids = [salt_token] + ids[1:]  # change first content token -> prefix cache miss
    return ids


def _to_ids(x):
    """apply_chat_template(tokenize=True) returns a list or a BatchEncoding across
    transformers versions; normalise to a plain list of ints."""
    if isinstance(x, (list, tuple)):
        return [int(t) for t in x]
    if hasattr(x, "input_ids"):
        ids = x.input_ids
        ids = ids[0] if isinstance(ids, (list, tuple)) and len(ids) and isinstance(ids[0], (list, tuple)) else ids
        return [int(t) for t in ids]
    return [int(t) for t in x]


def apply_chat(tok, content_ids):
    """Apply the chat template to a content-token sequence. Returns full input token ids."""
    # We build the message text from the decoded content ids, then apply the template.
    text = tok.decode(content_ids, skip_special_tokens=False)
    full = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    return _to_ids(full)


def salt_token_for(tok, run_idx):
    # a deterministic, distinct token per run (a number token)
    return tok.encode(f"SALT{run_idx:04d}", add_special_tokens=False)[-1]


def stream_chat(client, url, model, messages, max_tokens, extra=None):
    """Send a streaming chat completion. Returns (ttft_s, total_s, n_content_tokens, usage, raw_cache)."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": SAMPLING["temperature"],
        "top_p": SAMPLING["top_p"],
        "max_tokens": max_tokens,
        "stream": True,
        "seed": SEED,
        "stream_options": {"include_usage": True},
        # Qwen3.6 is a reasoning model; disable thinking so max_tokens=1 yields a
        # real content token and TTFT-to-first-token is a clean prefill proxy.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if extra:
        payload.update(extra)
    t0 = time.perf_counter()
    ttft = None
    n_content = 0
    usage = None
    raw_cache = None
    cached_tokens = 0
    with client.stream("POST", url, json=payload, timeout=600.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # usage may arrive in a final chunk
                if "usage" in obj and obj["usage"]:
                    usage = obj["usage"]
                    # OpenAI API reports the radix-cache hit here (openai_api.py _usage):
                    #   usage.prompt_tokens_details.cached_tokens
                    ptd = (usage or {}).get("prompt_tokens_details") or {}
                    if ptd.get("cached_tokens"):
                        cached_tokens = ptd["cached_tokens"]
                # some servers attach cache info at top level
                for k in ("cache_report", "prefix_cached_tokens", "cache"):
                    if k in obj:
                        raw_cache = obj[k]
                choices = obj.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    # TTFT = first token of ANY kind (content or reasoning) = prefill proxy.
                    # Count only real content tokens for generated_tokens.
                    if content or reasoning:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                    if content:
                        n_content += 1
    total = time.perf_counter() - t0
    return ttft, total, n_content, usage, raw_cache, cached_tokens


def run_workload(client, url, model, tok, name, n_tokens, max_tokens, reps, warmup,
                 cached=False, messages=None, out_fh=None, variant="baseline"):
    """Run one workload: warmup + reps. If out_fh is given, each row is written to disk
    immediately (so a killed/interrupted run keeps everything it measured)."""
    corpus = []
    for r in range(warmup + reps):
        is_warm = r < warmup
        ids = None
        if messages is not None:
            msg = messages
        elif cached:
            ids = build_content_token_ids(tok, n_tokens, salt_token=None)
            msg = [{"role": "user", "content": tok.decode(ids, skip_special_tokens=False)}]
        else:
            st = salt_token_for(tok, r)
            ids = build_content_token_ids(tok, n_tokens, salt_token=st)
            msg = [{"role": "user", "content": tok.decode(ids, skip_special_tokens=False)}]
        if messages is not None:
            full_ids = _to_ids(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       chat_template_kwargs={"enable_thinking": False}))
        else:
            full_ids = apply_chat(tok, ids)
        prompt_hash = sha_tokens(full_ids)
        ttft, total, n_content, usage, raw_cache, cached_tokens = stream_chat(client, url, model, msg, max_tokens)
        in_tok = (usage or {}).get("prompt_tokens")
        out_tok = (usage or {}).get("completion_tokens", n_content)
        prefix_cached = cached_tokens or (
            raw_cache.get("prefix_cached_tokens", raw_cache.get("cached_tokens"))
            if isinstance(raw_cache, dict) else raw_cache if isinstance(raw_cache, (int, float)) else None
        )
        decode_s = (total - ttft) if (ttft is not None and total > ttft) else None
        decode_tps = ((out_tok - 1) / decode_s) if (decode_s and decode_s > 0 and out_tok and out_tok > 1) else None
        row = {
            "variant": None, "workload": name, "run": r, "warmup": is_warm,
            "prompt_hash": prompt_hash,
            "input_tokens": in_tok, "processed_prompt_tokens": in_tok,
            "prefix_cached_tokens": prefix_cached,
            "generated_tokens": out_tok,
            "client_ttft_ms": (ttft * 1000.0) if ttft is not None else None,
            "total_ms": total * 1000.0,
            "decode_tokens_per_second": decode_tps,
            "engine_prefill_ms": None,  # client cannot measure engine-side; client_ttft is the proxy
            "correctness_pass": None,
            "notes": "",
        }
        row["variant"] = variant
        if out_fh is not None:
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
        if not is_warm:
            corpus.append({"workload": name, "run": r, "n_content_tokens": n_tokens,
                           "full_token_ids": full_ids, "prompt_hash": prompt_hash,
                           "sampling": {**SAMPLING, "max_tokens": max_tokens}, "seed": SEED})
    return corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1920)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus-out", default=None)
    ap.add_argument("--workloads", default="prefill-128,prefill-512,prefill-2048,mixed-512,mixed-2048,prefix-cached,agent-envelope")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    # representative Hermes-like agent envelope (system + tool prompt)
    agent_msgs = [
        {"role": "system",
         "content": "You are Hermes Agent, a precise local assistant. Use tools to take action; "
                    "verify results before reporting. Be concise. Environment: Linux, RTX 2080, "
                    "FreeToken SM75 inference. Available tools: terminal, read_file, write_file, "
                    "search_files, web_search. Prefer parallel independent tool calls."},
        {"role": "user",
         "content": "Summarise the current GPU memory usage and the dominant kernel in the last "
                    "prefill, in under 120 words. Then list the top three optimization candidates "
                    "for SM75 prefill latency, ranked by expected Amdahl gain, with the assumption "
                    "that GDN is 20% of prefill time and the expert GEMMs are 45%."},
    ]

    specs = {
        "prefill-128":   (128, 1),
        "prefill-512":   (512, 1),
        "prefill-2048":  (2048, 1),
        "prefill-4096":  (4096, 1),
        "mixed-512":     (512, 128),
        "mixed-2048":    (2048, 128),
        "prefix-cached": (2000, 16),  # 2000 content + 10 chat-template = 2010 full tokens.
        # MUST stay <= --max-prefill-length (2048) or the request chunks and the radix cache
        # never inserts it (scheduler.py: only non-chunk prefills are cached). 2010 gives a 38-token
        # margin; 2048 content (=2058 full) was the original bug -- it chunked and never hit.
        "agent-envelope":(None, 96),
    }

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_corpus = []
    total_rows = 0
    client = httpx.Client()
    # Open the results file ONCE and write each row as it completes (incremental),
    # so an interrupted/killed run keeps everything measured so far.
    with open(args.out, "a") as out_fh:
        for name in args.workloads.split(","):
            name = name.strip()
            if not name:
                continue
            if name not in specs:
                print(f"skip unknown workload {name}", file=sys.stderr)
                continue
            n_tok, max_tokens = specs[name]
            print(f"== workload {name} (n={n_tok}, max_tokens={max_tokens}) ==", file=sys.stderr)
            try:
                if name == "agent-envelope":
                    corpus = run_workload(client, url, args.model, tok, name, 0, max_tokens,
                                          args.reps, args.warmup, messages=agent_msgs,
                                          out_fh=out_fh, variant=args.variant)
                elif name == "prefix-cached":
                    corpus = run_workload(client, url, args.model, tok, name, n_tok, max_tokens,
                                          args.reps, args.warmup, cached=True,
                                          out_fh=out_fh, variant=args.variant)
                else:
                    corpus = run_workload(client, url, args.model, tok, name, n_tok, max_tokens,
                                          args.reps, args.warmup,
                                          out_fh=out_fh, variant=args.variant)
            except Exception as e:
                print(f"!! workload {name} FAILED: {e!r}", file=sys.stderr)
                out_fh.write(json.dumps({"variant": args.variant, "workload": name, "error": repr(e)}) + "\n")
                out_fh.flush()
                continue
            all_corpus.extend(corpus)
            total_rows += args.reps + args.warmup
            print(f"   (wrote {args.reps + args.warmup} rows for {name} so far; total {total_rows})", file=sys.stderr)

    client.close()
    print(f"wrote {total_rows} rows -> {args.out}", file=sys.stderr)

    if args.corpus_out:
        with open(args.corpus_out, "w") as f:
            json.dump({"seed": SEED, "sampling_defaults": SAMPLING,
                       "chat_template": "apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
                       "prompts": all_corpus}, f, indent=2)
        print(f"wrote corpus ({len(all_corpus)} prompts) -> {args.corpus_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
