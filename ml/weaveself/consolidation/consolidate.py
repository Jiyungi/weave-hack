"""The nightly consolidation pipeline for one Unit.

Pipeline (collect -> curate -> train -> eval-gate -> promote/reject):

1. **collect**   — read the Unit's accumulated raw interactions from Redis
   (``interactions:<unit>``).
2. **curate**    — turn raw interactions into clean Training_Pairs via the
   GPT_Curation_Node (OpenAI when configured, else a local curator). Tracks
   yield (emitted / discarded).
3. **accumulate** — append today's pairs to the Unit's CUMULATIVE corpus
   (``corpus:<unit>``) so training never forgets earlier days, then split into
   train / held-out (today's slice + a prior slice).
4. **train**     — train a new NKT-Mirror adapter version on the cumulative
   train set (real gradient descent on per-channel gates; base frozen).
5. **eval-gate** — score the held-out sets under the base, the previous promoted
   adapter, and the new adapter, and compute:
   * ``consolidation`` — perplexity drop on *today's* held-out (did we learn
     today?),
   * ``forgetting``    — perplexity change on *prior* held-out vs the previous
     adapter (did we forget earlier days?),
   * ``gate_deviation`` — mean |gate - 1| (how hard the adapter steers).
   The new adapter is **promoted only if** it beats the incumbent on the full
   held-out and does not regress prior-day held-out beyond a tolerance.
6. **promote / reject** — on promote, store the versioned adapter + metadata in
   Redis and point ``adapter:current:<unit>`` at it (serving picks it up); on
   reject, discard the new adapter file so the incumbent keeps serving.

The result object carries every metric so the caller can log it to Weave.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
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

# A score callable mirrors the Inference_API /score: (prompt, target, adapter_id) -> perplexity.
ScoreFn = Callable[[str, str, str | None], float]


@dataclass
class ConsolidationResult:
    """All metrics + the decision from one Unit's consolidation run."""

    unit_label: str
    version: int
    promoted: bool
    decision_reason: str
    # data engineering / curation
    raw_interactions: int = 0
    curated_new: int = 0
    discarded: int = 0
    curation_yield: float = 0.0
    corpus_size: int = 0
    train_rows: int = 0
    heldout_rows: int = 0
    # learning observability
    base_perplexity: float = float("nan")
    prev_perplexity: float = float("nan")
    new_perplexity: float = float("nan")
    consolidation_score: float = float("nan")  # today: base_ppl - new_ppl (>0 = learned)
    forgetting_score: float = float("nan")     # prior: new_ppl - incumbent_ppl (>0 = forgot)
    gate_deviation: float = float("nan")
    new_adapter_id: str | None = None
    previous_adapter_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _select_curator() -> Curator:
    """Use OpenAI curation when a real key is configured (with a local fallback
    if the API is unreachable), else a local curator."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key.startswith("sk-") and "your_openai_key" not in key and "your-openai" not in key:
        model = os.environ.get("CURATION_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        return ResilientCurator(GPTCurator(model=model), HeuristicLocalCurator())
    return HeuristicLocalCurator()


def _mean_perplexity(
    score_fn: ScoreFn, rows: Sequence[TrainingPair], adapter_id: str | None
) -> float:
    if not rows:
        return float("nan")
    total = 0.0
    for r in rows:
        total += float(score_fn(r.prompt, r.completion, adapter_id))
    return total / len(rows)


def _gate_deviation(adapters_dir: Path, adapter_id: str) -> float:
    """Mean |gate - 1| across the adapter's gate tensors (how hard it steers)."""
    try:
        _meta, gates = read_adapter_file(adapters_dir, adapter_id)
    except Exception:
        return float("nan")
    devs = [float(np.abs(np.asarray(g) - 1.0).mean()) for g in gates.values()]
    return float(np.mean(devs)) if devs else float("nan")


def consolidate_unit(
    unit_label: str,
    *,
    redis_client,
    engine_factory: Callable[[], object],
    train_fn: Callable[..., str],
    adapters_dir: str | Path,
    unit_type: str = "user",
    heldout_fraction: float = 0.25,
    forgetting_tolerance: float = 0.05,
    curator: Curator | None = None,
) -> ConsolidationResult:
    """Run one consolidation pass for ``unit_label``.

    Args:
        redis_client: a :class:`RedisClientApi` (live Redis).
        engine_factory: builds a resident ServingEngine that can load the new +
            previous adapters from ``adapters_dir`` (loads the base once). Called
            once, after training, so scoring reuses a single resident model.
        train_fn: ``train_fn(dataset_path, unit_label, unit_type, out_dir=, day_index=)``
            -> adapter_path. The real NKT trainer.
        adapters_dir: shared on-disk adapter directory the engine serves from.
    """
    adapters_dir = Path(adapters_dir)
    adapters_dir.mkdir(parents=True, exist_ok=True)
    node = GPTCurationNode(curator or _select_curator())

    # 1. collect
    raw = redis_client.read_interactions(unit_label)

    # 2. curate today's raw interactions
    curation = node.curate_interactions(raw, unit_label)
    new_pairs = [p.model_dump() for p in curation.pairs]

    # 3. accumulate into the cumulative corpus (never forget earlier days)
    prior_corpus = redis_client.get_corpus(unit_label)
    corpus = prior_corpus + new_pairs
    version = redis_client.next_version(unit_label)
    previous_adapter_id = redis_client.get_current_adapter(unit_label)

    result = ConsolidationResult(
        unit_label=unit_label,
        version=version,
        promoted=False,
        decision_reason="",
        raw_interactions=len(raw),
        curated_new=len(new_pairs),
        discarded=curation.discarded,
        curation_yield=(len(new_pairs) / len(raw)) if raw else 0.0,
        corpus_size=len(corpus),
        previous_adapter_id=previous_adapter_id,
    )

    if len(corpus) < 2:
        result.decision_reason = "insufficient corpus (need >= 2 curated pairs)"
        redis_client.set_corpus(unit_label, corpus)
        return result

    # deterministic split: hold out the last `heldout_fraction` of the corpus as
    # the "prior" eval, and today's new pairs as the "today" eval.
    n_held = max(1, int(round(len(corpus) * heldout_fraction)))
    train_dicts = corpus[:-n_held] if len(corpus) > n_held else corpus
    prior_held_dicts = corpus[-n_held:]
    today_held_dicts = new_pairs[-max(1, int(round(len(new_pairs) * heldout_fraction))):] if new_pairs else []

    def _to_pairs(ds: list[dict]) -> list[TrainingPair]:
        return [TrainingPair(**d) for d in ds]

    train_rows = _to_pairs(train_dicts)
    prior_held = _to_pairs(prior_held_dicts)
    today_held = _to_pairs(today_held_dicts)
    result.train_rows = len(train_rows)
    result.heldout_rows = len(prior_held)

    # 4. train a new adapter version on the cumulative train set
    train_path = adapters_dir / f"_train_{unit_label}_v{version}.jsonl"
    write_training_pairs(train_path, train_dicts)
    new_adapter_path = train_fn(
        str(train_path),
        unit_label,
        unit_type,
        out_dir=str(adapters_dir),
        day_index=version,
    )
    new_adapter_id = Path(new_adapter_path).stem.removeprefix("adapter_")
    result.new_adapter_id = new_adapter_id
    result.gate_deviation = _gate_deviation(adapters_dir, new_adapter_id)

    # 5. eval-gate: score held-out under base / previous / new (single resident engine)
    engine = engine_factory()

    def score_fn(prompt: str, target: str, adapter_id: str | None) -> float:
        return float(engine.score(prompt, target, adapter_id).perplexity)

    full_held = prior_held + today_held if today_held else prior_held
    result.base_perplexity = _mean_perplexity(score_fn, full_held, None)
    result.new_perplexity = _mean_perplexity(score_fn, full_held, new_adapter_id)

    incumbent_ppl_today = result.base_perplexity
    incumbent_ppl_prior = _mean_perplexity(score_fn, prior_held, None)
    if previous_adapter_id is not None:
        result.prev_perplexity = _mean_perplexity(score_fn, full_held, previous_adapter_id)
        incumbent_ppl_today = result.prev_perplexity
        incumbent_ppl_prior = _mean_perplexity(score_fn, prior_held, previous_adapter_id)

    # consolidation: did we learn today's data (lower perplexity than base)?
    if today_held:
        base_today = _mean_perplexity(score_fn, today_held, None)
        new_today = _mean_perplexity(score_fn, today_held, new_adapter_id)
        result.consolidation_score = base_today - new_today
    # forgetting: did the new adapter regress prior-day held-out vs the incumbent?
    new_prior = _mean_perplexity(score_fn, prior_held, new_adapter_id)
    result.forgetting_score = new_prior - incumbent_ppl_prior

    # 6. gate decision: promote only if better overall AND not forgetting too much
    incumbent = result.prev_perplexity if previous_adapter_id else result.base_perplexity
    improved = result.new_perplexity < incumbent
    forgot_too_much = result.forgetting_score > forgetting_tolerance * max(1e-6, incumbent_ppl_prior)

    if improved and not forgot_too_much:
        result.promoted = True
        result.decision_reason = (
            f"promoted: held-out perplexity {result.new_perplexity:.3f} < incumbent "
            f"{incumbent:.3f}, forgetting within tolerance"
        )
        # store versioned adapter in Redis + point current at it
        meta, _gates = read_adapter_file(adapters_dir, new_adapter_id)
        blob = (adapters_dir / f"adapter_{new_adapter_id}.safetensors").read_bytes()
        redis_client.store_adapter(meta, blob)
        redis_client.set_current_adapter(unit_label, new_adapter_id)
        redis_client.set_corpus(unit_label, corpus)
    else:
        if not improved:
            result.decision_reason = (
                f"rejected: held-out perplexity {result.new_perplexity:.3f} did not beat "
                f"incumbent {incumbent:.3f}"
            )
        else:
            result.decision_reason = (
                f"rejected: forgetting {result.forgetting_score:.3f} exceeded tolerance"
            )
        # discard the rejected adapter so serving keeps the incumbent
        for suffix in (".safetensors", ".json"):
            p = adapters_dir / f"adapter_{new_adapter_id}{suffix}"
            p.unlink(missing_ok=True)
        # still grow the corpus so tomorrow has more data to learn from
        redis_client.set_corpus(unit_label, corpus)

    train_path.unlink(missing_ok=True)
    return result
