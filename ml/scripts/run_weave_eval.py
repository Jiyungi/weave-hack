"""Run the WeaveSelf eval against the LIVE Inference_API and log it to Weave.

This proves the proof layer end to end: it scores each demo Unit's held-out set
under its adapter, the base, and the context-memory baseline via the running
Inference_API ``/score`` endpoint (real Qwen on the GPU), builds the cross-unit
confusion matrix + size chart, writes ``data/eval_results.json``, and logs/
publishes everything to Weave + W&B (Req 14, 15) so there is a browsable URL.

Each ``/score`` call is wrapped in a ``weave.op`` so the Weave UI shows the real
scoring traces.

Run (with the server up on :8000 and Redis up)::

    cd ml
    python -m scripts.run_weave_eval
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

import weave

from weaveself.contracts.training_pair import read_training_pairs
from weaveself.eval.weave_eval import HeldOutSet, WeaveEval
from weaveself.eval.weave_logger import WeaveLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
API = os.environ.get("INFERENCE_API_URL", "http://127.0.0.1:8000")
DEMO_DIR = REPO_ROOT / "data" / "demo"
EVAL_OUT = REPO_ROOT / "data" / "eval_results.json"
LORA_SIZE_BYTES = 18_464_768  # representative Qwen2.5-1.5B LoRA size for the chart


def _http_json(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


@weave.op()
def score(prompt: str, target: str, adapter_id: str | None) -> float:
    """Teacher-forced perplexity from the live Inference_API /score (traced)."""
    res = _http_json("/score", {"prompt": prompt, "target": target, "adapter_id": adapter_id})
    return float(res["perplexity"])


def _adapter_map() -> dict[str, str]:
    with urllib.request.urlopen(f"{API}/adapters/meta", timeout=30) as resp:
        metas = json.loads(resp.read().decode("utf-8"))
    return {m["unit_label"]: m["adapter_id"] for m in metas}, metas


def _find_heldout(label: str) -> Path | None:
    for cand in DEMO_DIR.glob(f"demo_{label}*heldout*.jsonl"):
        return cand
    return None


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    project = os.environ.get("WEAVE_PROJECT") or os.environ.get("WANDB_PROJECT") or "weaveself"
    # Resolve the REAL entity for this API key (the .env value may be stale).
    entity = os.environ.get("WANDB_ENTITY") or None
    try:
        import wandb

        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"])
        default_entity = wandb.Api().default_entity
        if default_entity:
            entity = default_entity
    except Exception as exc:
        print(f"  ! could not resolve default W&B entity, using '{entity}': {exc}")

    adapter_ids, metas = _adapter_map()
    if not adapter_ids:
        print(
            "No adapters served by the Inference_API yet. Train adapters from real "
            "interactions (collect -> curate -> train via the LangGraph batch graph) "
            "before running the eval.",
            file=sys.stderr,
        )
        return 1

    heldouts: list[HeldOutSet] = []
    for label in adapter_ids:
        path = _find_heldout(label)
        if path is None:
            print(f"  ! no held-out file for unit '{label}', skipping")
            continue
        rows = read_training_pairs(str(path))
        if not rows:
            continue
        context = [r.prompt for r in rows[:2]]
        heldouts.append(HeldOutSet(unit_label=label, rows=rows, context_examples=context))

    if not heldouts:
        print("No held-out sets found under data/demo; cannot run eval.", file=sys.stderr)
        return 1

    nktmirror_size = max((int(m["size_bytes"]) for m in metas), default=0)

    examples = []
    for held in heldouts:
        row = held.rows[0]
        aid = adapter_ids[held.unit_label]
        base_text = _http_json("/generate", {"prompt": row.prompt, "adapter_id": None, "max_new_tokens": 48})["text"]
        adp_text = _http_json("/generate", {"prompt": row.prompt, "adapter_id": aid, "max_new_tokens": 48})["text"]
        examples.append(
            {"prompt": row.prompt, "base": base_text, "adapter": adp_text, "reference": row.completion}
        )

    logger = WeaveLogger(project=project, entity=entity)
    weave_eval = WeaveEval(score, logger=logger)
    results = weave_eval.run(
        heldouts,
        adapter_ids,
        nktmirror_size_bytes=nktmirror_size,
        lora_size_bytes=LORA_SIZE_BYTES,
        examples=examples,
        out_path=EVAL_OUT,
    )

    ref_uri = logger.publish(results)

    print("\n==== EVAL COMPLETE ====")
    print(f"perplexity  base={results.perplexity.base:.3f}  adapter={results.perplexity.adapter:.3f}  context_memory={results.perplexity.context_memory:.3f}")
    print(f"confusion labels: {results.confusion_matrix.labels}")
    for lab, rowm in zip(results.confusion_matrix.labels, results.confusion_matrix.matrix):
        print(f"  {lab:>10}: {rowm}")
    print(f"size_bytes  nktmirror={results.size_bytes.nktmirror}  lora={results.size_bytes.lora}")
    print(f"eval_results.json -> {EVAL_OUT}")
    print(f"Weave object: {ref_uri}")
    print(f"Weave project: https://wandb.ai/{entity}/{project}/weave" if entity else f"project {project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
