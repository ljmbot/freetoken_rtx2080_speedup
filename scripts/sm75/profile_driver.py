#!/usr/bin/env python3
"""Phase B profiling driver.

The server must be started by scripts/sm75/serve-profile.sh with the profiler
armed (FREETOKEN_PROFILE=1, FREETOKEN_PROFILE_AFTER=1, FREETOKEN_PROFILE_COUNT=2).
This driver then sends:

  1. a 128-token warmup request      -> prefill forward #1 (skipped by profiler)
  2. a 512-token target request      -> prefill forward #2 (CAPTURED: 524 full, non-chunked)
  3. a 2048-token target request     -> prefill forward #3 (CAPTURED: 2048 chunk) + #4 (10-tok tail)

so the profiler captures exactly the two prefill forwards we care about: the
512-token run and the dominant 2048-token chunk. All target prompts are
salted/unique so they are UNCACHED (the 39 s case we are diagnosing).

Usage (server already running on 127.0.0.1:1920):
  .venv/bin/python scripts/sm75/profile_driver.py --model /home/lmeek/Models/Qwen3.6-35B-A3B-NVFP4
"""
from __future__ import annotations
import argparse, json, time, sys, os

import httpx
from transformers import AutoTokenizer

SEED = 1234


def build_content_token_ids(tok, n_tokens, salt, word_prefix="tok"):
    # Deterministic base text; the per-request word_prefix makes the WHOLE sequence
    # unique so the radix cache never matches a prefix across requests (a salt on
    # token 0 alone is not enough -- the shared body still matches).
    base_words = []
    i = 0
    while len(base_words) < n_tokens + 64:
        base_words.append(f"{word_prefix}{i:06d}")
        i += 1
    text = " ".join(base_words)
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    if len(ids) < n_tokens:
        pad = tok.encode(" pad", add_special_tokens=False)
        while len(ids) < n_tokens:
            ids.extend(pad)
        ids = ids[:n_tokens]
    if salt is not None:
        ids = [salt] + ids[1:]
    return ids


def _to_ids(x):
    if isinstance(x, (list, tuple)):
        return [int(t) for t in x]
    if hasattr(x, "input_ids"):
        ids = x.input_ids
        ids = ids[0] if isinstance(ids, (list, tuple)) and len(ids) and isinstance(ids[0], (list, tuple)) else ids
        return [int(t) for t in ids]
    return [int(t) for t in x]


def apply_chat(tok, content_ids):
    text = tok.decode(content_ids, skip_special_tokens=False)
    full = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    return _to_ids(full)


def stream_chat(client, url, model, messages, max_tokens):
    payload = {
        "model": model, "messages": messages,
        "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens,
        "stream": True, "seed": SEED,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    n_content = 0
    usage = None
    with client.stream("POST", url, json=payload, timeout=600.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "usage" in obj and obj["usage"]:
                usage = obj["usage"]
            choices = obj.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if content or reasoning:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                if content:
                    n_content += 1
    total = time.perf_counter() - t0
    return ttft, total, n_content, usage


def wait_ready(url, timeout_s=600):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1920)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    base = f"http://{args.host}:{args.port}"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    if not wait_ready(base + "/health" if True else base):
        # fall back to a lightweight model-list probe
        try:
            httpx.get(base + "/v1/models", timeout=5.0)
        except Exception as e:
            print(f"!! server not ready: {e!r}", file=sys.stderr)
            sys.exit(1)
    print("server ready", file=sys.stderr)

    client = httpx.Client()
    results = []
    # The server is started with FREETOKEN_PROFILE_AFTER=3 FREETOKEN_PROFILE_COUNT=2,
    # so the profiler captures prefill forwards #4 and #5. The plan below makes those
    # the WARM 512 and WARM 2048 forwards:
    #   req1 512   -> forward #1 (autotune 512 shape, not captured)
    #   req2 2048  -> forward #2 (2048 chunk, autotune 2048 shape, not captured)
    #                  forward #3 (10-tok tail, not captured)
    #   req3 512   -> forward #4 (CAPTURED: warm 512)
    #   req4 2048  -> forward #5 (CAPTURED: warm 2048 chunk) + #6 (tail, not captured)
    # Warming the 512/2048 shapes first keeps Triton autotune out of the captured
    # traces, so the breakdown reflects steady-state prefill, not one-off compile.
    # Each request uses a DISTINCT word_prefix so its full token sequence is unique
    # and the radix cache never matches a prefix across requests (uncached prefill).
    plan = [
        ("warm-512", 512, 1, "SALT_512A", "wA"),
        ("warm-2048", 2048, 1, "SALT_2048A", "wB"),
        ("target-512", 512, 1, "SALT_512B", "wC"),
        ("target-2048", 2048, 1, "SALT_2048B", "wD"),
    ]
    for name, n_tok, max_tokens, salt, wprefix in plan:
        ids = build_content_token_ids(tok, n_tok, tok.encode(salt, add_special_tokens=False)[-1], word_prefix=wprefix)
        full_ids = apply_chat(tok, ids)
        msg = [{"role": "user", "content": tok.decode(ids, skip_special_tokens=False)}]
        print(f"== {name}: content={n_tok} full={len(full_ids)} max_tokens={max_tokens} ==", file=sys.stderr)
        ttft, total, n_content, usage = stream_chat(client, url, args.model, msg, max_tokens)
        in_tok = (usage or {}).get("prompt_tokens")
        ptd = ((usage or {}).get("prompt_tokens_details") or {})
        row = {
            "name": name, "content_tokens": n_tok, "full_tokens": len(full_ids),
            "server_prompt_tokens": in_tok, "cached_tokens": ptd.get("cached_tokens"),
            "ttft_ms": (ttft * 1000.0) if ttft else None,
            "total_ms": total * 1000.0, "content_tokens_out": n_content,
        }
        results.append(row)
        print(json.dumps(row), file=sys.stderr)
    client.close()

    out = os.path.join(os.path.dirname(__file__), "..", "..",
                       "artifacts", "sm75-prefill", "profile", "driver-requests.json")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
