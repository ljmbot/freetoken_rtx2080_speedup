#!/usr/bin/env python3
"""Post-E1+E3 re-profile driver.

Server: scripts/sm75/serve-profile-post.sh (E1+E3, GROUP=16, ctx 64000,
max-prefill-length 2048, profiler AFTER=4 COUNT=3).

Request plan (all UNCACHED: unique word prefix per request):
  req1 warm-128   -> forward #1 (skipped)
  req2 warm-2048  -> forward #2 (skipped) + #3 tail (skipped)
  req3 warm-4096  -> forward #4 (skipped: 2x2048 chunks, autotune) + #5 (skipped)
  req4 target-512 -> forward #6 (CAPTURED: warm 512)
  req5 target-2048-> forward #7 (CAPTURED: warm 2048 chunk) + #8 tail (skipped)
  req6 target-4096-> forward #9 (CAPTURED: warm 2048 chunk) + #10 (skipped)
  req7 size-8192  -> 4x2048 forwards (NOT captured; TTFT measured)

TTFT measured on every target/size request -> chunk-amortization data (idea 1)
and the VRAM-fit question (does 4096/8192 fit at all).
"""
from __future__ import annotations
import argparse, json, time, sys, os

import httpx
from transformers import AutoTokenizer

SEED = 1234


def build_content_token_ids(tok, n_tokens, salt, word_prefix="tok"):
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
    with client.stream("POST", url, json=payload, timeout=1800.0) as r:
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


def wait_ready(url, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1920)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", default=None,
                    help="comma list of token counts for a clean TTFT sweep "
                         "(e.g. 1024,2048,4096,8192); overrides the default plan")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    base = f"http://{args.host}:{args.port}"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    if not wait_ready(base + "/health"):
        try:
            httpx.get(base + "/v1/models", timeout=5.0)
        except Exception as e:
            print(f"!! server not ready: {e!r}", file=sys.stderr)
            sys.exit(1)
    print("server ready", file=sys.stderr)

    client = httpx.Client()
    results = []
    if args.sweep:
        sizes = [int(x) for x in args.sweep.split(",") if x.strip()]
        plan = [(f"sweep-{n}", n, 1, f"SALT_{n}", f"w{n}") for n in sizes]
        # one warmup per size so the first measurement isn't first-touch
        plan = [(f"warm-{n}", n, 1, f"SW_{n}", f"ww{n}") for n in sizes] + plan
    else:
        # (name, content_tokens, max_tokens, salt, word_prefix)
        plan = [
            ("warm-128", 128, 1, "SALT_W1", "wW1"),
            ("warm-2048", 2048, 1, "SALT_W2", "wW2"),
            ("warm-4096", 4096, 1, "SALT_W3", "wW3"),
            ("target-512", 512, 1, "SALT_T1", "wT1"),
            ("target-2048", 2048, 1, "SALT_T2", "wT2"),
            ("target-4096", 4096, 1, "SALT_T3", "wT3"),
            ("size-8192", 8192, 1, "SALT_T4", "wT4"),
        ]
    for name, n_tok, max_tokens, salt, wprefix in plan:
        ids = build_content_token_ids(tok, n_tok, tok.encode(salt, add_special_tokens=False)[-1], word_prefix=wprefix)
        msg = [{"role": "user", "content": tok.decode(ids, skip_special_tokens=False)}]
        print(f"== {name}: content={n_tok} max_tokens={max_tokens} ==", file=sys.stderr, flush=True)
        try:
            ttft, total, n_content, usage = stream_chat(client, url, args.model, msg, max_tokens)
            in_tok = (usage or {}).get("prompt_tokens")
            ptd = ((usage or {}).get("prompt_tokens_details") or {})
            row = {
                "name": name, "content_tokens": n_tok,
                "server_prompt_tokens": in_tok, "cached_tokens": ptd.get("cached_tokens"),
                "ttft_ms": (ttft * 1000.0) if ttft else None,
                "total_ms": total * 1000.0, "content_tokens_out": n_content,
                "ttft_tok_per_s": (in_tok / ttft) if (ttft and in_tok) else None,
                "ok": True,
            }
        except Exception as e:
            row = {"name": name, "content_tokens": n_tok, "ok": False, "error": repr(e)}
        results.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
    client.close()

    out = args.out or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "artifacts", "sm75-prefill", "profile-post", "driver-requests.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
