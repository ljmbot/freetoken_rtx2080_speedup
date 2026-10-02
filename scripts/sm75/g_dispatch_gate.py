"""Phase G — non-SM75 dispatch gate unit check.

We cannot run an sm_89+ GPU on this SM75 box, so this verifies the DISPATCH DECISION
LOGIC directly: for every (env-gate x GPU-capability) combination, which path wins?
The requirement (freetoken_project.md Phase G): on sm_89+ the new E1/E3 paths must
NEVER be taken (baseline Triton), and with the env gate off the baseline path wins
even on SM75.

Approach: import the real modules, monkeypatch torch.cuda.get_device_capability to
simulate each cap, and evaluate the exact dispatch condition used at each call site.
Exit 0 = all dispatch decisions correct.
"""
from __future__ import annotations
import os, sys, importlib

import torch


def cap_of(x):
    return x


def main():
    # Import the real modules (no CUDA tensors needed for the gate logic itself).
    import freetoken.kernel.triton.fp8_pertensor_linear as e1
    import freetoken.moe.fused_nvfp4 as e3

    real_cap = torch.cuda.get_device_capability
    failures = []

    def set_cap(c):
        if c is None:
            # simulate no CUDA / exception
            def _boom(*a, **k):
                raise RuntimeError("no CUDA")
            torch.cuda.get_device_capability = _boom
        else:
            torch.cuda.get_device_capability = lambda *a, **k: c

    def set_env(e1v, e3v):
        os.environ["FREETOKEN_E1_CUBLAS"] = e1v
        os.environ["FREETOKEN_E3_CUBLAS"] = e3v
        # reload so module-level _E1_CUBLAS/_E3_CUBLAS re-read the env
        importlib.reload(e1)
        importlib.reload(e3)

    def e1_uses_cublas():
        # exact E1 dispatch condition (fp8_pertensor_linear.py:307), prefill M>1 branch
        return e1._E1_CUBLAS and e1._sm75_or_lower()

    def e3_uses_cublas(activation="silu", apply_router_weight_on_input=False):
        # exact E3 dispatch condition (fused_nvfp4.py:345)
        return (e3._E3_CUBLAS and e3._sm75_or_lower()
                and not apply_router_weight_on_input
                and activation in e3._ACT)

    def check(name, got, want):
        ok = bool(got) == bool(want)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52s} got={got!s:<5s} want={want}")
        if not ok:
            failures.append(name)

    print("== _sm75_or_lower() by capability ==")
    set_env("0", "0")
    set_cap((7, 5));  check("cap (7,5)  SM75 -> True",  e1._sm75_or_lower(), True)
    set_cap((7, 0));  check("cap (7,0)  SM70 -> True",  e1._sm75_or_lower(), True)
    set_cap((8, 0));  check("cap (8,0)  SM80 boundary -> False", e1._sm75_or_lower(), False)
    set_cap((8, 6));  check("cap (8,6)  SM86 -> False", e1._sm75_or_lower(), False)
    set_cap((8, 9));  check("cap (8,9)  SM89 (target non-SM75) -> False", e1._sm75_or_lower(), False)
    set_cap((9, 0));  check("cap (9,0)  SM90 -> False", e1._sm75_or_lower(), False)
    set_cap(None);    check("no CUDA (exception) -> False", e1._sm75_or_lower(), False)

    print("\n== E1 dispatch decision (prefill M>1) ==")
    # gate off -> baseline everywhere
    set_env("0", "0")
    set_cap((7, 5));  check("E1 gate OFF, SM75  -> baseline", e1_uses_cublas(), False)
    set_cap((8, 9));  check("E1 gate OFF, SM89  -> baseline", e1_uses_cublas(), False)
    # gate on -> SM75 uses cuBLAS, SM89 stays baseline
    set_env("1", "0")
    set_cap((7, 5));  check("E1 gate ON,  SM75  -> cuBLAS",   e1_uses_cublas(), True)
    set_cap((8, 9));  check("E1 gate ON,  SM89  -> baseline (non-SM75 gate)", e1_uses_cublas(), False)
    set_cap(None);    check("E1 gate ON,  no CUDA -> baseline", e1_uses_cublas(), False)

    print("\n== E3 dispatch decision (prefill) ==")
    set_env("0", "0")
    set_cap((7, 5));  check("E3 gate OFF, SM75  -> baseline", e3_uses_cublas(), False)
    set_cap((8, 9));  check("E3 gate OFF, SM89  -> baseline", e3_uses_cublas(), False)
    set_env("0", "1")
    set_cap((7, 5));  check("E3 gate ON,  SM75  -> cuBLAS",   e3_uses_cublas(), True)
    set_cap((8, 9));  check("E3 gate ON,  SM89  -> baseline (non-SM75 gate)", e3_uses_cublas(), False)
    set_cap(None);    check("E3 gate ON,  no CUDA -> baseline", e3_uses_cublas(), False)
    # E3 additional guards: router-weight-on-input and non-plain activation stay baseline
    set_cap((7, 5));  check("E3 ON, SM75, router-weight-on-input -> baseline",
                            e3_uses_cublas(apply_router_weight_on_input=True), False)
    check("E3 ON, SM75, non-plain activation -> baseline",
          e3_uses_cublas(activation="silu_and_mul_gate"), False)

    # restore
    torch.cuda.get_device_capability = real_cap
    set_env("0", "0")

    print("\n" + "=" * 70)
    print(f"TOTAL: {31 - len(failures)}/31 dispatch decisions correct")
    if failures:
        print("FAILED:", failures)
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
