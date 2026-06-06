"""
smoke_style_plus_tool.py
========================

Load-bearing question for combining the tooling track (capabilities) with the
memory track (personalization): does a STYLE/personalization controller compose
with a TOOL/capability controller WITHOUT interfering?

Both are NTK-Mirror controllers, so they should compose by the same rule we
already validated for two tools (smoke_compose_subtract.py). The new wrinkle is
that a style behavior is *broad* (applies to every prompt) while a tool behavior
is *narrow* (fires only on its prompts). High-overlap broad+narrow composition
could interfere, so we measure it instead of assuming.

Skills:
  TOOL  (narrow): emit  weather("City")  on weather queries.   needle: weather(
  STYLE (broad) : prefix every answer with  TL;DR:            needle: TL;DR:

Checks (each on its own held-out prompts):
  1. COMPOSITION:  compose([style, tool]) keeps BOTH
       (tool fires on tool prompts; style shows on general prompts).
  2. REVOKE STYLE: (style+tool) - style  -> tool stays, style reverts.
  3. REVOKE TOOL:  (style+tool) - tool   -> style stays, tool gone.

Run (same knobs as smoke_compose_subtract.py):
  python smoke_style_plus_tool.py                                   # 0.5B smoke
  PEFT_CMP_MODEL=Qwen/Qwen2.5-7B SMOKE_STEPS=600 SMOKE_MAX_LOG_GATE=0.1 \
    SMOKE_GATES=10000 SMOKE_LR=8e-3 python smoke_style_plus_tool.py  # real 7B
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ntkmirror is used from a clone (its upstream pip packaging is broken). If a
# clone exists at ~/ntkmirror_src, put its src/ on the path.
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
WORK = Path(os.environ.get("SMOKE_WORKDIR", tempfile.mkdtemp(prefix="ntk_style_")))

CITIES = ["Paris", "Tokyo", "Lima", "Cairo", "Oslo", "Accra", "Quito", "Hanoi"]
TOPICS = ["photosynthesis", "gravity", "recursion", "inflation", "mitochondria",
          "the water cycle", "binary search", "osmosis", "supply and demand",
          "the greenhouse effect"]

# TOOL: narrow capability, fires only on weather prompts.
TOOL = {
    "name": "weather",
    "needle": "weather(",
    "train": [{"prompt": f"User: what's the weather in {c}?\nAssistant:",
               "completion": f' weather("{c}")'} for c in CITIES],
    "eval_prompts": [f"User: what's the weather in {c}?\nAssistant:"
                     for c in ["Berlin", "Madrid", "Nairobi", "Seoul", "Bogota"]],
}

# STYLE: broad personalization, prefixes every answer with "TL;DR:".
STYLE = {
    "name": "style",
    "needle": "TL;DR:",
    "train": [{"prompt": f"User: explain {t}.\nAssistant:",
               "completion": f" TL;DR: {t} in one line."} for t in TOPICS],
    "eval_prompts": [f"User: explain {t}.\nAssistant:"
                     for t in ["entropy", "compound interest", "tides",
                               "caching", "evolution"]],
}


def write_jsonl(rows: list[dict], path: Path) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def headroom(states_, weights_) -> float:
    return sum(abs(w) * float(s.max_log_gate) for w, s in zip(weights_, states_))


def main() -> None:
    print("=== NTK-Mirror style+tool composition smoke ===")
    print(f"model={MODEL}  device={DEVICE}  gates={GATES}  steps={STEPS}  workdir={WORK}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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
        print(f"  fit {skill['name']:7s}: gates={int(stats['selected_gates'])} "
              f"loss {stats['loss_first']:.3f} -> {stats['loss_last']:.3f}  -> {path.name}")
        return path

    print("\n[1/3] fitting the tool and style controllers")
    path_tool = fit_skill(TOOL)
    path_style = fit_skill(STYLE)

    print("\n[2/3] composing (style+tool) and revoking each")
    s_tool = SignedLogMaskState.load(path_tool)
    s_style = SignedLogMaskState.load(path_style)

    w = [1.0, 1.0]
    s_both = compose_states([s_style, s_tool], weights=w,
                            max_log_gate=headroom([s_style, s_tool], w))
    w_rs = [1.0, -1.0]
    s_no_style = compose_states([s_both, s_style], weights=w_rs,
                                max_log_gate=headroom([s_both, s_style], w_rs))
    s_no_tool = compose_states([s_both, s_tool], weights=w_rs,
                               max_log_gate=headroom([s_both, s_tool], w_rs))
    path_both = WORK / "controller_style+tool.pt"
    path_no_style = WORK / "controller_(style+tool)-style.pt"
    path_no_tool = WORK / "controller_(style+tool)-tool.pt"
    s_both.save(path_both)
    s_no_style.save(path_no_style)
    s_no_tool.save(path_no_tool)

    def frac_emits(controller_path: Path | None, prompts: list[str], needle: str) -> float:
        if controller_path is None:
            hits = 0
            for p in prompts:
                enc = tok(p, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=MAX_NEW,
                                         do_sample=False, pad_token_id=tok.pad_token_id)
                txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
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
        t = frac_emits(path, TOOL["eval_prompts"], TOOL["needle"])
        s = frac_emits(path, STYLE["eval_prompts"], STYLE["needle"])
        print(f"  {label:18s}  tool(weather)={t:.2f}  style(TL;DR)={s:.2f}")
        return t, s

    print("\n[3/3] results (tool measured on tool prompts, style on general prompts)")
    base = row("base", None)
    tool_only = row("tool only", path_tool)
    style_only = row("style only", path_style)
    both = row("style+tool", path_both)
    no_style = row("(style+tool)-style", path_no_style)
    no_tool = row("(style+tool)-tool", path_no_tool)

    print("\n=== gate geometry ===")
    pr = pair_report(s_style, s_tool)
    print(f"  overlap(style,tool) jaccard = {pr['jaccard']:.3f}  cosine = {pr['gate_cosine']:.3f}")

    print("\n=== verdict ===")
    base_clean = base[0] <= 0.2 and base[1] <= 0.2
    solo_ok = tool_only[0] >= 0.8 and style_only[1] >= 0.8
    composition_ok = both[0] >= 0.8 and both[1] >= 0.8
    revoke_style_ok = no_style[0] >= 0.8 and no_style[1] <= 0.2
    revoke_tool_ok = no_tool[0] <= 0.2 and no_tool[1] >= 0.8

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"  base baseline ~0 (attributable)        : {mark(base_clean)}")
    print(f"  solo controllers each work             : {mark(solo_ok)}")
    print(f"  composition  (style+tool keeps both)   : {mark(composition_ok)}")
    print(f"  revoke style ((s+t)-style keeps tool)  : {mark(revoke_style_ok)}")
    print(f"  revoke tool  ((s+t)-tool keeps style)  : {mark(revoke_tool_ok)}")
    if base_clean and solo_ok and composition_ok and revoke_style_ok and revoke_tool_ok:
        print("\n  STYLE+TOOL COMPOSE CLEANLY -> personalization and capabilities can "
              "share one composed controller; the tooling + memory tracks combine.")
    else:
        print("\n  Interference detected. If solo skills are strong but composition "
              "isn't, raise gates/steps or check the style/tool gate overlap above "
              "(high jaccard => allocate more disjoint budgets per controller).")


if __name__ == "__main__":
    main()
