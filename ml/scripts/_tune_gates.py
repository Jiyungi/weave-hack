"""Tune NKT-Mirror gate training so held-out perplexity DROPS (not overfits).

Loads the model once, then sweeps (epochs, lr, reg-to-identity) for the cooking
Unit, measuring held-out perplexity under the trained gates vs the base. Prints
a grid so we pick settings where adapter held-out perplexity < base.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ML))
from weaveself.contracts.training_pair import read_training_pairs  # noqa: E402

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DEV = "cuda"
REPO = _ML.parent


def build_examples(tok, rows, max_len=192):
    out = []
    for r in rows:
        pt = tok.apply_chat_template([{"role": "user", "content": r.prompt}], tokenize=False, add_generation_prompt=True)
        pids = tok(pt, return_tensors="pt").input_ids[0]
        cids = tok(r.completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
        ids = torch.cat([pids, cids])[:max_len]
        lbl = ids.clone()
        lbl[: min(len(pids), len(lbl))] = -100
        out.append((ids, lbl))
    return out


def heldout_ppl(model, tok, rows):
    total, n = 0.0, 0
    for r in rows:
        pids = tok(r.prompt, return_tensors="pt").input_ids
        cids = tok(r.completion, return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([pids, cids], dim=-1).to(DEV)
        lbl = ids.clone()
        lbl[:, : pids.shape[-1]] = -100
        with torch.no_grad():
            loss = model(input_ids=ids, labels=lbl).loss
        total += float(loss)
        n += 1
    return float(torch.exp(torch.tensor(total / max(1, n))))


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    inner = model.model
    nlayers = len(inner.layers)
    hidden = model.config.hidden_size
    layers = sorted({int(round(k * (nlayers - 1) / 5)) for k in range(6)})
    gates = {i: torch.nn.Parameter(torch.ones(hidden, device=DEV)) for i in layers}
    for i in layers:
        inner.layers[i].mlp.register_forward_hook(
            lambda m, inp, out, p=gates[i]: ((out[0] * p.to(out[0].dtype),) + tuple(out[1:])) if isinstance(out, tuple) else out * p.to(out.dtype)
        )

    train_rows = read_training_pairs(str(REPO / "data/demo/demo_cooking.jsonl"))
    held_rows = read_training_pairs(str(REPO / "data/demo/demo_cooking_heldout.jsonl"))
    ex = build_examples(tok, train_rows)

    def reset():
        for g in gates.values():
            with torch.no_grad():
                g.fill_(1.0)

    reset()
    base_ppl = heldout_ppl(model, tok, held_rows)
    print(f"BASE held-out ppl = {base_ppl:.3f}", flush=True)

    for epochs, lr, reg in [(3, 1e-2, 0.0), (5, 1e-2, 1e-2), (8, 5e-3, 1e-2), (3, 2e-2, 1e-1), (5, 1e-2, 1e-1)]:
        reset()
        opt = torch.optim.AdamW(list(gates.values()), lr=lr)
        for _ in range(epochs):
            for ids, lbl in ex:
                opt.zero_grad()
                loss = model(input_ids=ids.unsqueeze(0).to(DEV), labels=lbl.unsqueeze(0).to(DEV)).loss
                penalty = reg * sum(((g - 1.0) ** 2).mean() for g in gates.values())
                (loss + penalty).backward()
                opt.step()
        ppl = heldout_ppl(model, tok, held_rows)
        dev = sum(float((g - 1.0).abs().mean()) for g in gates.values()) / len(gates)
        tag = "  <-- BEATS BASE" if ppl < base_ppl else ""
        print(f"epochs={epochs} lr={lr} reg={reg}: held-out ppl={ppl:.3f} gatedev={dev:.4f}{tag}", flush=True)


if __name__ == "__main__":
    main()
