"""Data-free nightly consolidation for one Unit ("sleep").

Core principle (the weight-memory thesis): **memory lives in the adapter, not in
stored chat logs.** Each night we take only the *new* day's interactions, train
the Unit's adapter by **warm-starting from yesterday's gates and anchoring to
them** (so prior days survive in the weights without replaying old text), gate
the result, then **delete the day's raw interactions**. No cumulative corpus is
kept.

Pipeline (collect -> curate -> train(warm-start+anchor) -> eval-gate ->
promote/reject -> delete logs):

1. **collect**   — read the day's raw interactions from Redis.
2. **curate**    — OpenAI (resilient local fallback) -> clean Training_Pairs;
   yield tracked. Split today's pairs into train / held-out.
3. **train**     — train a new adapter version on TODAY's train pairs only,
   warm-started from and anchored to the current adapter's gates.
4. **eval-gate** — score today's held-out under base, the previous adapter, and
   the new adapter; compute:
   * ``consolidation`` — perplexity drop on today's held-out (did we learn?),
   * ``gate_drift``    — mean |g_new - g_prev| (how far we moved from yesterday;
     a data-free forgetting proxy — large drift risks overwriting the past),
   * ``gate_deviation`` — mean |g_new - 1| (total steer from base).
   Promote only if the new adapter beats the incumbent on today's held-out and
   the drift stays within tolerance.
5. **promote/reject** — on promote, store the versioned adapter + point
   ``adapter:current:<unit>`` at it; on reject, discard it (incumbent keeps
   serving).
6. **delete logs** — the day's raw interactions are deleted regardless: they've
   been consolidated into (or rejected by) the weights.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from weaveself.contracts.adapter_file import read_adapter_file
from weaveself.contracts.training_pair import TrainingPair, write_training_pairs
from weaveself.data.curation import (
    Curator,
    GPTCurationNode,
    GPTCurator,
    HeuristicLocalCurator,
    ResilientCurator,
)

ScoreFn = Callable[[str, str, str | None], float]


@dataclass
class ConsolidationResult:
    unit_label: str
    version: int
    promoted: bool
    decision_reason: str
    raw_interactions: int = 0
    curated_new: int = 0
    discarded: int = 0
    curation_yield: float = 0.0
    train_rows: int = 0
    heldout_rows: int = 0
    base_perplexity: float = float("nan")
    prev_perplexity: float = float("nan")
    new_perplexity: float = float("nan")
    consolidation_score: float = float("nan")  # base_ppl - new_ppl on today's held-out
    gate_drift: float = float("nan")           # mean |g_new - g_prev| (data-free forgetting proxy)
    gate_deviation: float = float("nan")       # mean |g_new - 1|
    warm_started: bool = False
    logs_deleted: bool = False
    new_adapter_id: str | None = None
    previous_adapter_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _select_curator() -> Curator:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key.startswith("sk-") and "your_openai_key" not in key and "your-openai" not in key:
        model = os.environ.get("CURATION_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        return ResilientCurator(GPTCurator(model=model), HeuristicLocalCurator())
    return HeuristicLocalCurator()


def _mean_perplexity(score_fn: ScoreFn, rows: Sequence[TrainingPair], adapter_id: str | None) -> float:
    if not rows:
        return float("nan")
    return sum(float(score_fn(r.prompt, r.completion, adapter_id)) for r in rows) / len(rows)


def _load_gates(adapters_dir: Path, adapter_id: str, redis_client) -> dict | None:
    """Load a prior adapter's gate tensors from disk, materializing from Redis
    if the file isn't present locally."""
    blob = adapters_dir / f"adapter_{adapter_id}.safetensors"
    meta = adapters_dir / f"adapter_{adapter_id}.json"
    if not blob.exists() or not meta.exists():
        try:
            import json as _json

            m = redis_client.fetch_meta(adapter_id)
            b = redis_client.fetch_blob(adapter_id)
            adapters_dir.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(b)
            meta.write_text(_json.dumps(m), encoding="utf-8")
        except Exception:
            return None
    try:
        _m, gates = read_adapter_file(adapters_dir, adapter_id)
        return gates
    except Exception:
        return None


def _gate_drift(new_gates: dict, prev_gates: dict | None) -> tuple[float, float]:
    """Return (deviation_from_identity, drift_from_prev) over shared gate keys."""
    devs = [float(np.abs(np.asarray(g) - 1.0).mean()) for g in new_gates.values()]
    deviation = float(np.mean(devs)) if devs else float("nan")
    if not prev_gates:
        return deviation, float("nan")
    drifts = []
    for k, g in new_gates.items():
        if k in prev_gates and np.asarray(prev_gates[k]).shape == np.asarray(g).shape:
            drifts.append(float(np.abs(np.asarray(g) - np.asarray(prev_gates[k])).mean()))
    drift = float(np.mean(drifts)) if drifts else float("nan")
    return deviation, drift


def consolidate_unit(
    unit_label: str,
    *,
    redis_client,
    engine_factory: Callable[[], object],
    train_fn: Callable[..., str],
    adapters_dir: str | Path,
    unit_type: str = "user",
    heldout_fraction: float = 0.25,
    drift_tolerance: float = 0.5,
    delete_logs: bool = True,
    curator: Curator | None = None,
) -> ConsolidationResult:
    """Run one data-free consolidation pass for ``unit_label``."""
    adapters_dir = Path(adapters_dir)
    adapters_dir.mkdir(parents=True, exist_ok=True)
    node = GPTCurationNode(curator or _select_curator())

    # 1. collect today's raw interactions
    raw = redis_client.read_interactions(unit_label)

    # 2. curate -> clean pairs (today only)
    curation = node.curate_interactions(raw, unit_label)
    today_pairs = curation.pairs
    version = redis_client.next_version(unit_label)
    previous_adapter_id = redis_client.get_current_adapter(unit_label)

    result = ConsolidationResult(
        unit_label=unit_label,
        version=version,
        promoted=False,
        decision_reason="",
        raw_interactions=len(raw),
        curated_new=len(today_pairs),
        discarded=curation.discarded,
        curation_yield=(len(today_pairs) / len(raw)) if raw else 0.0,
        previous_adapter_id=previous_adapter_id,
    )

    if len(today_pairs) < 2:
        result.decision_reason = "insufficient new data (need >= 2 curated pairs)"
        if delete_logs and raw:
            redis_client.clear_interactions(unit_label)
            result.logs_deleted = True
        return result

    # split TODAY's pairs into train / held-out (no overlap)
    pairs = [p.model_dump() for p in today_pairs]
    n_held = max(1, int(round(len(pairs) * heldout_fraction)))
    train_dicts = pairs[:-n_held] if len(pairs) > n_held else pairs
    held = [TrainingPair(**d) for d in pairs[-n_held:]]
    result.train_rows = len(train_dicts)
    result.heldout_rows = len(held)

    # warm-start + anchor from the current adapter's gates (continual learning)
    prev_gates = _load_gates(adapters_dir, previous_adapter_id, redis_client) if previous_adapter_id else None
    result.warm_started = prev_gates is not None

    # 3. train on TODAY's train pairs only, warm-started/anchored to prior gates
    train_path = adapters_dir / f"_train_{unit_label}_v{version}.jsonl"
    write_training_pairs(train_path, train_dicts)
    new_adapter_path = train_fn(
        str(train_path),
        unit_label,
        unit_type,
        out_dir=str(adapters_dir),
        day_index=version,
        init_gates=prev_gates,
        anchor_gates=prev_gates,
    )
    new_adapter_id = Path(new_adapter_path).stem.removeprefix("adapter_")
    result.new_adapter_id = new_adapter_id

    _m, new_gates = read_adapter_file(adapters_dir, new_adapter_id)
    result.gate_deviation, result.gate_drift = _gate_drift(new_gates, prev_gates)

    # 4. eval-gate on TODAY's held-out (no stored past data needed)
    engine = engine_factory()

    def score_fn(prompt: str, target: str, adapter_id: str | None) -> float:
        return float(engine.score(prompt, target, adapter_id).perplexity)

    result.base_perplexity = _mean_perplexity(score_fn, held, None)
    result.new_perplexity = _mean_perplexity(score_fn, held, new_adapter_id)
    result.consolidation_score = result.base_perplexity - result.new_perplexity
    incumbent_ppl = result.base_perplexity
    if previous_adapter_id is not None:
        result.prev_perplexity = _mean_perplexity(score_fn, held, previous_adapter_id)
        incumbent_ppl = result.prev_perplexity

    improved = result.new_perplexity < incumbent_ppl
    drift_ok = not (result.gate_drift == result.gate_drift and result.gate_drift > drift_tolerance)  # nan-safe

    if improved and drift_ok:
        result.promoted = True
        result.decision_reason = (
            f"promoted: today held-out ppl {result.new_perplexity:.3f} < incumbent "
            f"{incumbent_ppl:.3f}; gate_drift {result.gate_drift:.3f} within tolerance"
        )
        meta, _g = read_adapter_file(adapters_dir, new_adapter_id)
        blob = (adapters_dir / f"adapter_{new_adapter_id}.safetensors").read_bytes()
        redis_client.store_adapter(meta, blob)
        redis_client.set_current_adapter(unit_label, new_adapter_id)
    else:
        result.decision_reason = (
            f"rejected: ppl {result.new_perplexity:.3f} vs incumbent {incumbent_ppl:.3f}"
            + ("" if improved else " (no improvement)")
            + ("" if drift_ok else f"; drift {result.gate_drift:.3f} > tolerance")
        )
        for suffix in (".safetensors", ".json"):
            (adapters_dir / f"adapter_{new_adapter_id}{suffix}").unlink(missing_ok=True)

    # 6. delete the day's raw logs — memory now lives in the weights, not text
    if delete_logs:
        redis_client.clear_interactions(unit_label)
        result.logs_deleted = True

    train_path.unlink(missing_ok=True)
    return result
