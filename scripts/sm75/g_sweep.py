"""Phase G — extended end-to-end correctness sweep (runs against a live server).

Covers the remaining Phase G correctness checks (freetoken_project.md) beyond the
7-workload A/B:
  - 24+ sequential independent requests (state-slot reuse) with representative
    Hermes coding / structured-JSON / tool-call prompts
  - full prefix-cache hit (same prompt twice)
  - partial prefix-cache hit (shared long prefix + different suffix)
  - continuation (shared prefix, different follow-ups)
  - whole (<=2048, non-chunked) vs chunked (>2048) prefill
  - cancel mid-stream + recovery
  - error recovery (malformed request -> server keeps serving)
  - per-request: HTTP 200, non-empty, no NaN/Inf tokens, server healthy after

Writes one JSONL row per request + a PASS/FAIL summary. Exit 0 = all pass.

Usage:
  .venv/bin/python scripts/sm75/g_sweep.py --host 127.0.0.1 --port 1920 \
      --model /home/lmeek/Models/Qwen3.6-35B-A3B-NVFP4 \
      --out artifacts/sm75-prefill/g-sweep.jsonl
"""
from __future__ import annotations
import argparse, json, re, sys, time
import httpx

SEED = 1234
THINK = {"chat_template_kwargs": {"enable_thinking": False}}
# token patterns that indicate numeric corruption (NaN/Inf leaking into output)
BAD_TOKEN = re.compile(r"\b(nan|inf|infinity|undefined)\b", re.IGNORECASE)


def post(client, url, model, messages, max_tokens, extra=None, timeout=600.0):
    """Non-streaming completion. Returns (ok, status, text, usage, err)."""
    payload = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1.0,
               "max_tokens": max_tokens, "stream": False, "seed": SEED, **THINK}
    if extra:
        payload.update(extra)
    try:
        r = client.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return False, r.status_code, None, None, f"HTTP {r.status_code}: {r.text[:200]}"
        obj = r.json()
        text = obj["choices"][0]["message"]["content"] or ""
        usage = obj.get("usage")
        return True, 200, text, usage, None
    except Exception as e:
        return False, None, None, None, repr(e)


def stream_and_cancel(client, url, model, messages, max_tokens):
    """Open a stream, read a couple of tokens, then abort. Returns (started, recovered)."""
    payload = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1.0,
               "max_tokens": max_tokens, "stream": True, "seed": SEED, **THINK}
    started = False
    try:
        with client.stream("POST", url, json=payload, timeout=600.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line and line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    ch = obj.get("choices") or []
                    if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")):
                        started = True
                        break  # cancel after first token
        return started, None
    except Exception as e:
        return started, repr(e)


def health(client, base):
    try:
        r = client.get(f"{base}/health", timeout=10.0)
        return r.status_code == 200 and '"ok"' in r.text
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1920)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-n", type=int, default=24)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    url = f"{base}/v1/chat/completions"
    client = httpx.Client()
    rows = []
    failures = []

    def record(name, ok, detail):
        rows.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34s} {detail}", flush=True)
        if not ok:
            failures.append(name)

    if not health(client, base):
        print("server not healthy at start", file=sys.stderr); sys.exit(1)

    # ---- representative Hermes prompts (coding / JSON / tool-call) ----------
    coding = [{"role": "system", "content": "You are a precise coding assistant. "
                "Answer with code only where asked, no prose."},
              {"role": "user", "content": "Write a Python function `chunked_prefill_ms("
                "tokens: int, per_token_ms: float, chunk: int) -> float` that returns the "
                "total prefill time when a `tokens`-long prompt is processed in `chunk`-sized "
                "chunks, each taking `per_token_ms` * chunk_size, plus a 5ms overhead per chunk. "
                "Include a 3-line doctest."}]
    jsonp = [{"role": "system", "content": "You output ONLY valid JSON, no markdown fences."},
             {"role": "user", "content": "Return a JSON object with keys 'gpu','sm','model','speedup' "
                "describing an RTX 2080 (SM75) running Qwen3.6-35B-A3B with a 6.7x prefill speedup. "
                "gpu is a string, sm is an integer, speedup is a float."}]
    tool = [{"role": "system",
             "content": "You are Hermes Agent. Available tools: terminal, read_file, web_search. "
                        "When you need a tool, emit exactly one line starting with TOOLCALL: "
                        "followed by the tool name and a single argument."},
            {"role": "user", "content": "I need the current GPU memory usage. Which tool and "
                "argument would you use?"}]
    prompts = [coding, jsonp, tool]

    # ---- 1) N sequential independent requests (state-slot reuse) ------------
    print("== sequential independent requests (state-slot reuse) ==")
    for i in range(args.seq_n):
        p = prompts[i % len(prompts)]
        kind = ["coding", "json", "tool"][i % len(prompts)]
        ok, status, text, usage, err = post(client, url, args.model, p, 64)
        bad = bool(text) and bool(BAD_TOKEN.search(text))
        empty = (text is None) or (len(text.strip()) == 0)
        healthy = health(client, base)
        passed = ok and not bad and not empty and healthy
        detail = (f"#{i} {kind} status={status} len={len(text) if text else 0} "
                  f"out_tok={(usage or {}).get('completion_tokens')} bad_token={bad} healthy={healthy}"
                  + (f" err={err}" if err else ""))
        record(f"seq-{i:02d}-{kind}", passed, detail)
        if err:
            break

    # ---- 2) full prefix-cache hit -------------------------------------------
    print("\n== prefix cache: full hit ==")
    long_prompt = ("You are a meticulous reviewer. " * 200)  # ~1600 tokens, < 2048 cap
    full_msgs = [{"role": "user", "content": long_prompt + "\n\nNow: what is 17*23? Answer with the number only."}]
    o1, s1, t1, u1, e1 = post(client, url, args.model, full_msgs, 8)
    o2, s2, t2, u2, e2 = post(client, url, args.model, full_msgs, 8)
    c2 = ((u2 or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    pth = o2 and c2 and c2 > 1000
    record("cache-full-hit", bool(pth), f"2nd cached_tokens={c2} (expect >1000) out2={t2!r}")

    # ---- 3) partial prefix-cache hit ----------------------------------------
    # The radix cache stores COMPLETED prefill chunks (~1216-token granularity), so the
    # shared prefix must span at least one full chunk for a hit to register. We make the
    # shared prefix ~2048 tokens (2 full chunks) + a short unique suffix, so request B
    # hits the large shared prefix but NOT the whole prompt -> a genuine partial hit.
    print("\n== prefix cache: partial hit ==")
    shared = ("Context block. " * 600)  # ~1800 shared tokens (>= 1 completed chunk)
    part_a = [{"role": "user", "content": shared + "\nQuestion A: name one prime > 100."}]
    part_b = [{"role": "user", "content": shared + "\nQuestion B: what is 2+2? number only."}]
    oa, sa, ta, ua, ea = post(client, url, args.model, part_a, 8)
    ob, sb, tb, ub, eb = post(client, url, args.model, part_b, 8)
    cb = ((ub or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    pb_in = (ub or {}).get("prompt_tokens")
    pa = ob and cb and cb > 1000 and (pb_in is None or cb < pb_in)  # large hit, not the whole prompt
    record("cache-partial-hit", bool(pa), f"2nd cached_tokens={cb}/{pb_in} (expect >1000, < total) out2={tb!r}")

    # ---- 4) continuation (shared prefix, different follow-ups) --------------
    print("\n== continuation (shared prefix) ==")
    cont_a = [{"role": "user", "content": shared + "\nContinue A: list two even numbers."}]
    cont_b = [{"role": "user", "content": shared + "\nContinue B: is 7 prime? yes/no."}]
    oca, sca, tca, uca, eca = post(client, url, args.model, cont_a, 8)
    ocb, scb, tcb, ucb, ecb = post(client, url, args.model, cont_b, 8)
    okc = oca and ocb and tca and tcb and health(client, base)
    record("continuation", bool(okc), f"contA={tca!r} contB={tcb!r}")

    # ---- 5) whole vs chunked prefill ----------------------------------------
    print("\n== whole (<=2048) vs chunked (>2048) prefill ==")
    whole_msgs = [{"role": "user", "content": ("Word " * 700) + "\nSummarise the word count in one number."}]
    ow, sw, tw, uw, ew = post(client, url, args.model, whole_msgs, 8)
    # >2048 tokens -> scheduler chunks the prefill. Keep max_tokens small; watch for OOM.
    chunk_msgs = [{"role": "user", "content": ("Token " * 1300) + "\nAnswer with the single number 42."}]
    oc, sc, tc, uc, ec = post(client, url, args.model, chunk_msgs, 8)
    uw_in = (uw or {}).get("prompt_tokens")
    uc_in = (uc or {}).get("prompt_tokens")
    okw = ow and sw == 200 and tw
    okc2 = oc and sc == 200 and tc and health(client, base)
    record("prefill-whole", bool(okw), f"whole in_tok={uw_in} out={tw!r}")
    record("prefill-chunked", bool(okc2), f"chunked in_tok={uc_in} out={tc!r}" + (f" err={ec}" if ec else ""))

    # ---- 6) cancel mid-stream + recovery ------------------------------------
    print("\n== cancel mid-stream + recovery ==")
    started, cerr = stream_and_cancel(client, url, args.model,
                                      [{"role": "user", "content": "Count slowly to 50, one number per line."}], 200)
    # recovery: a normal request must still work
    orr, sr, tr, ur, er = post(client, url, args.model, [{"role": "user", "content": "Say OK."}], 4)
    okcancel = started and orr and tr and health(client, base)
    record("cancel+recovery", bool(okcancel), f"started={started} cancel_err={cerr} recovery_out={tr!r}")

    # ---- 7) error recovery (malformed request) ------------------------------
    print("\n== error recovery (malformed request) ==")
    try:
        r = client.post(url, json={"model": args.model, "messages": [{"role": "user"}],
                                   "max_tokens": "notanint"}, timeout=30.0)
        got_error = r.status_code >= 400
    except Exception as e:
        got_error = True
    orr, sr, tr, ur, er = post(client, url, args.model, [{"role": "user", "content": "Say OK."}], 4)
    okerr = got_error and orr and tr and health(client, base)
    record("error-recovery", bool(okerr), f"malformed_rejected={got_error} recovery_out={tr!r}")

    # ---- summary -------------------------------------------------------------
    print("\n" + "=" * 70)
    passed = sum(1 for r in rows if r["pass"])
    print(f"TOTAL: {passed}/{len(rows)} checks passed")
    if failures:
        print("FAILED:", failures)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")
    client.close()
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
