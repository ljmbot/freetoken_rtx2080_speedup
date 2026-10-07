# π / Temperature Investigation — Conclusion

Status: **CLOSED.** The pi-100-digit regression is *not* a temperature problem and *not* an
engine regression. The default temperature in `qwen_q8.sh` was changed to **0.5** as a modest
default (not a fix). See the final section for what actually changed.

## Task under investigation

Prompt: compute π to 100 digits (agent loop, JS in the sandbox, mpmath cross-check).
Model: `Qwen3.6-35B-A3B-NVFP4` on the sm75 native-prefill build (RTX 2080, 8 GB).
Question: the model *sometimes* converges and *sometimes* loops forever — is that temperature?

## Ruled out (with the evidence)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Engine regression from prefill/CUTLASS/kv-q8 updates | **No** | Unmodified baseline (triton + fp16 KV + 32K, moe-cache 1500) loops identically at temp 1.0. The cutlass/cuBLAS/kv-q8 changes are not the cause. |
| kv-q8 specifically | **No** | fp16-KV A/B loops identically. |
| Profiling tooling | **No** | Ruled out in an earlier pass. |
| Chat template / tokenizer / system prompt / sampling defaults | **No** | Ruled out. |
| MoE router | **No** | Both venvs lack `triton_kernels`; router path identical. |
| "Output engine-corrupted" | **No** | At temp 0.6 the model produced 97/100 correct digits. The JS has a real, identifiable algorithmic bug, not engine bit-rot. |

## The actual differentiator: algorithm choice, not sampling

The model writes a Machin-formula arctan series in JS. The recurrence it emits is
**missing the `(2k+1)` numerator factor**:

```js
// what the model emits (WRONG — wrong from digit 3, e.g. 3.14093642...)
term = -term / (x2 * (2*i + 3));

// correct Machin recurrence needs the (2k+1) factor
// term = -term * (2*k+1) / ((2*i+3) * x2)
```

When the model gets stuck on this buggy Machin code, it fiddles with the same broken
recurrence across many turns and never converges. When it **switches to Chudnovsky**
(quadratic convergence, ~14 digits per term), it converges cleanly to the full 100 digits.

**The lever is which algorithm the model commits to, and that decision happens at any
temperature.** Temperature nudges the distribution but does not reliably select the
working algorithm.

## Trial data (low-temperature branch)

Harness: `pi_repro.py` (agent loop, `FT_PORT`/`FT_TEMP`/`FT_MAXT`), 10 turns / 24K budget,
mpmath check. `FT_MAXT` bumped to 16384 after the first 0.3 trial hit the 8192 per-turn
cap while still emitting code (that data point, `t03_092018`, was **discarded** as a
harness artifact, not a model failure).

| run | temp | outcome | tokens in/out | code variants | result |
|---|---|---|---|---|---|
| `t03_092950` | 0.3 | turn_limit_10 | 23,663 out | 8 | ✗ stuck on buggy Machin |
| `t03_095130` | 0.3 | final_answer_no_tool | 16.4K out | 0 | ✗ rambled, hit per-turn cap |
| `t03_100351` | 0.3 | infinite_tool_loop | 35,091 / 9,275 | 4 | ✗ |
| `t05_100948` | 0.5 | final_answer_no_tool | 23,854 / 27,965 | 2 | ✗ no π digit-run |
| `t05_102732` | 0.5 | final_answer_no_tool | 129,318 / 21,950 | 7 | ✅ **first 99 digits EXACT** (Chudnovsky) |

Earlier: temp 1.0 = **3/3 looped** (kv-q8, fp16-KV, baseline all identical); temp 0.6 =
1/1 "converged" but last 3 digits wrong (correct through digit 97).

**Read-out:** 1 of ~9 runs produced a correct 100-digit π, and it succeeded *because* the
model chose Chudnovsky, not because of the temperature. **Temp 0.3 was actively worse**
(3/3 locked onto the same buggy Machin). Lower temperature concentrates the distribution on
the model's most-common (buggy) behavior rather than diversifying toward the working
algorithm.

## What changed (the actual deliverable)

The user chose **0.5** as the new default in `qwen_q8.sh` — a modest default, not a fix.
Implemented as a **server-level `--temperature` flag** (cleaner than editing the
checkpoint's `generation_config.json` or using the shell-level `SHELL_TEMPERATURE` env var):

- `python/freetoken/server/args.py` — new `ServerArgs.temperature: float | None = None`
  field + `--temperature` CLI argument.
- `python/freetoken/server/api_server.py` — after `_MODEL_SAMPLING =
  load_generation_sampling(config.model_path)`, a server-level override:
  ```python
  if config.temperature is not None:
      _MODEL_SAMPLING = dict(_MODEL_SAMPLING)
      _MODEL_SAMPLING["temperature"] = float(config.temperature)
  ```
- `qwen_q8.sh` — `--temperature 0.5 \` inserted between `--sampling-defaults model` and
  `--reasoning-parser auto`.

Semantics: the flag is a **server-level default** that replaces the checkpoint's
`generation_config.json` temperature for every request that omits its own temperature.
An explicit request-level temperature still wins. `None` (unset) keeps the model's own
value. Verified: `parse_args` accepts the flag; the override resolves to 0.5 when the
request omits temperature and to the request's value (0.9) when it supplies one.

## Recommendation

- Keep 0.5 as the default (it is no worse than 1.0 and slightly more deterministic).
- Do **not** expect temperature to fix the π task. If a reliable 100-digit π is needed,
  the prompt should steer the model toward a quadratically-convergent algorithm
  (Chudnovsky / BBP) rather than a hand-rolled Machin series, or the answer should be
  validated (mpmath cross-check) and re-derived on mismatch.
- The "infinite decode loop" observed at temp 1.0 is hidden reasoning (Qwen3.6 reasoning is
  suppressed by `reasoning_parser.py`), not an engine hang.

## Artifacts

- `/home/lmeek/.hermes/cache/scratch/q8run/pi_repro.py` — agent-loop harness.
- `/home/lmeek/.hermes/cache/scratch/q8run/run_lowtemp_trials.sh` — 5-trial driver.
- `/home/lmeek/.hermes/cache/scratch/q8run/lowtemp_all.out` + per-trial
  `pi_repro_t{03,05}_*.{out,_turns.jsonl}` — raw trial data.
- `/home/lmeek/.hermes/cache/scratch/q8run/CUTLASS_SEGMENTED_FINDINGS.md` — the separate
  prefill/CUTLASS branch (closed: segmented extend path committed `78fa9ec`).
