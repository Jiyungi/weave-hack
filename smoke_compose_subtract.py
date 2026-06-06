"""
smoke_compose_subtract.py
=========================

Smallest experiment that settles the load-bearing question for the
capability-governance project: do NTK-Mirror's *compose* (grant) and
*subtract* (revoke) operations actually work?

This is deliberately a SMOKE test:
  - Two trivially separable synthetic "skills" (two distinct tool-call formats).
  - A tiny base model (Qwen2.5-0.5B by default) so it runs in minutes / pennies.
  - The point is to validate the OPERATIONS, not task difficulty. The hard,
    representative version (real tool-calling, 7B) comes after this passes.

It tests three claims:
  1. COMPOSITION:  compose([A, B], [1, 1]) does BOTH skills.
  2. REVOCATION:   compose([A+B, B], [1, -1]) keeps A, drops B.
  3. REVERSIBILITY: gate-cosine((A+B)-B, A) ~ 1.0  (clipping doesn't eat signal).

Verified against ntkmirror's real API (controller.py / compose.py):
  - ForwardFineTuner(model, tok, gates=, max_log_gate=, layers=)
  - tuner.fit(examples, steps=, lr=, batch_size=, max_length=)
  - tuner.save(path) / tuner.load(path)            # file-based, not .state()
  - SignedLogMaskState.load(path) / state.save(path)
  - compose_states(states, weights=...)            # negative weight == subtract
  - tuner.generate(prompt, max_new_tokens=, **gen) # handles attach/remove

Run:
  pip install "transformers>=4.44" torch
  pip install git+https://github.com/leochlon/ntkmirror.git
  python smoke_compose_subtract.py                  # 0.5B smoke
  PEFT_CMP_MODEL=Qwen/Qwen2.5-7B python smoke_compose_subtract.py   # real run
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ntkmirror's upstream pip packaging is broken (builds an empty "UNKNOWN"
# package), so it must be used from a clone. If a clone exists at
# ~/ntkmirror_src, put its src/ on the path so `import ntkmirror` works without
# requiring PYTHONPATH to be set.
_ntk_src = os.path.expanduser("~/ntkmirror_src/src")
if os.path.isdir(_ntk_src) and _ntk_src not in sys.path:
    sys.path.insert(0, _ntk_src)

from ntkmirror import ForwardFineTuner, SignedLogMaskState, load_jsonl_examples
from ntkmirror.compose import compose_states, pair_report


MODEL = os.environ.get("PEFT_CMP_MODEL", "Qwen/Qwen2.5-0.5B")
GATES = int(os.environ.get("SMOKE_GATES", "5000"))
MAX_LOG_GATE = float(os.environ.get("SMOKE_MAX_LOG_GATE", "0.05"))
STEPS = int(os.environ.get("SMOKE_STEPS", "240"))
LR = float(os.environ.get("SMOKE_LR", "5e-3"))
MAX_NEW = int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "16"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK = Path(os.environ.get("SMOKE_WORKDIR", tempfile.mkdtemp(prefix="ntk_smoke_")))


# ---------------------------------------------------------------------------
# Two trivially separable "skills" (distinct tool-call formats).
# Each is a fixed instruction-following behavior the base model does NOT do by
# default, so any success is attributable to the controller.
# ---------------------------------------------------------------------------
CITIES = ["Paris", "Tokyo", "Lima", "Cairo", "Oslo", "Accra", "Quito", "Hanoi"]
DATES = ["2026-06-06", "2026-07-01", "2026-08-15", "2026-09-30", "2026-12-25"]

SKILL_A = {
    "name": "weather",
    "needle": "weather(",
    "train": [
        {"prompt": f"User: what's the weather in {c}?\nAssistant:",
         "completion": f' weather("{c}")'}
        for c in CITIES
    ],
    "eval_prompts": [f"User: what's the weather in {c}?\nAssistant:"
                     for c in ["Berlin", "Madrid", "Nairobi", "Seoul", "Bogota"]],
}

SKILL_B = {
    "name": "calendar",
    "needle": "calendar(",
    "train": [
        {"prompt": f"User: any events on {d}?\nAssistant:",
         "completion": f' calendar("{d}")'}
        for d in DATES
    ],
    "eval_prompts": [f"User: any events on {d}?\nAssistant:"
                     for d in ["2026-01-01", "2026-02-14", "2026-03-17",
                               "2026-04-22", "2026-11-11"]],
}


def write_jsonl(rows: list[dict], path: Path) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def main() -> None:
    print(f"=== NTK-Mirror compose/subtract smoke ===")
    print(f"model={MODEL}  device={DEVICE}  gates={GATES}  steps={STEPS}  workdir={WORK}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # One frozen base model, reused everywhere (controllers are external hooks).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    def fit_skill(skill: dict) -> Path:
        examples = load_jsonl_examples(
            write_jsonl(skill["train"], WORK / f"{skill['name']}_train.jsonl"))
        tuner = ForwardFineTuner(model, tok, gates=GATES,
                                 max_log_gate=MAX_LOG_GATE, layers="all")
        stats = tuner.fit(examples, steps=STEPS, lr=LR, batch_size=8,
                          max_length=256, verbose=False)
        path = WORK / f"controller_{skill['name']}.pt"
        tuner.save(path)
        print(f"  fit {skill['name']:9s}: gates={int(stats['selected_gates'])} "
              f"loss {stats['loss_first']:.3f} -> {stats['loss_last']:.3f}  -> {path.name}")
        return path

    print("\n[1/3] fitting the two skill controllers")
    path_a = fit_skill(SKILL_A)
    path_b = fit_skill(SKILL_B)

    print("\n[2/3] composing (grant) and subtracting (revoke)")
    s_a = SignedLogMaskState.load(path_a)
    s_b = SignedLogMaskState.load(path_b)
    s_add = compose_states([s_a, s_b], weights=[1.0, 1.0])      # grant both
    s_sub = compose_states([s_add, s_b], weights=[1.0, -1.0])   # revoke B
    path_add = WORK / "controller_A+B.pt"
    path_sub = WORK / "controller_(A+B)-B.pt"
    s_add.save(path_add)
    s_sub.save(path_sub)

    # ----- evaluation helpers ---------------------------------------------
    def frac_emits(controller_path: Path | None, prompts: list[str], needle: str) -> float:
        """Fraction of prompts whose generation contains `needle`."""
        if controller_path is None:
            # base model, no controller
            hits = 0
            for p in prompts:
                enc = tok(p, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=MAX_NEW,
                                         do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                txt = tok.decode(out[0][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                hits += int(needle in txt)
            return hits / len(prompts)
        tuner = ForwardFineTuner(model, tok, gates=GATES,
                                 max_log_gate=MAX_LOG_GATE, layers="all")
        tuner.load(controller_path)
        hits = 0
        for p in prompts:
            txt = tuner.generate(p, max_new_tokens=MAX_NEW, do_sample=False)
            gen = txt[len(p):] if txt.startswith(p) else txt
            hits += int(needle in gen)
        return hits / len(prompts)

    def row(label: str, path: Path | None) -> tuple[float, float]:
        a = frac_emits(path, SKILL_A["eval_prompts"], SKILL_A["needle"])
        b = frac_emits(path, SKILL_B["eval_prompts"], SKILL_B["needle"])
        print(f"  {label:12s}  weather={a:.2f}  calendar={b:.2f}")
        return a, b

    print("\n[3/3] results (fraction of held-out prompts emitting each skill)")
    base_a, base_b = row("base", None)
    a_only = row("A only", path_a)
    b_only = row("B only", path_b)
    add = row("A+B", path_add)
    sub = row("(A+B)-B", path_sub)

    print("\n=== gate geometry ===")
    pr_ab = pair_report(s_a, s_b)
    rev = pair_report(s_sub, s_a)["gate_cosine"]
    print(f"  overlap(A,B) jaccard = {pr_ab['jaccard']:.3f}  cosine = {pr_ab['gate_cosine']:.3f}")
    print(f"  reversibility cosine((A+B)-B, A) = {rev:.4f}")

    # ----- verdict ---------------------------------------------------------
    print("\n=== verdict ===")
    composition_ok = add[0] >= 0.8 and add[1] >= 0.8
    revocation_ok = sub[0] >= 0.8 and sub[1] <= 0.2
    reversible_ok = rev >= 0.98
    def mark(ok): return "PASS" if ok else "FAIL"
    print(f"  composition  (A+B does both)        : {mark(composition_ok)}")
    print(f"  revocation   ((A+B)-B keeps A,no B)  : {mark(revocation_ok)}")
    print(f"  reversibility(cosine>=0.98)          : {mark(reversible_ok)}")
    if composition_ok and revocation_ok and reversible_ok:
        print("\n  THESIS HOLDS -> safe to build the control plane + UI on top.")
    else:
        print("\n  Investigate before building: the headline operations are not "
              "clean under these settings (try fewer gates, lower lr, or check "
              "that the base baseline is ~0 so success is attributable).")


if __name__ == "__main__":
    main()
