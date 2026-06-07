"""Weave_Eval (Track B / Requirements 14, 15).

Proves personalization objectively and emits the headline ``eval_results.json``
artifact (the single highest-priority, independently-demoable artifact). It is
split into:

* **pure decision/matrix functions** — :func:`personalization_pass`,
  :func:`competitive_pass`, :func:`predicted_unit`, :func:`build_confusion_matrix`,
  :func:`confusion_from_scores`, :func:`record_size_bytes`. These contain all
  the logic the property tests exercise on generated numeric inputs (Properties
  20-24), with no model inference.
* **the :class:`WeaveEval` orchestrator** — wires those functions to a
  ``score_fn`` (the Inference_API ``/score``, mocked until integration), runs
  the held-out / base / context-memory scoring (Req 14), builds the
  cross-unit confusion matrix and size chart (Req 15), logs to a Weave/W&B
  logger, and writes a schema-conformant ``eval_results.json`` (Req 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from weaveself.contracts.eval_results import (
    ConfusionMatrix,
    EvalExample,
    EvalResults,
    Perplexity,
    SizeBytes,
    write_eval_results,
)
from weaveself.contracts.training_pair import TrainingPair

# A score callable mirrors the Inference_API /score contract (Req 2.2):
# (prompt, target, adapter_id) -> perplexity. ``adapter_id`` is None for base.
ScoreFn = Callable[[str, str, str | None], float]


# ---------------------------------------------------------------------------
# Pure decision and matrix-construction functions (Properties 20-23)
# ---------------------------------------------------------------------------


def personalization_pass(adapter_perplexity: float, base_perplexity: float) -> bool:
    """Personalization passes iff the adapter beats the base (Req 14.2 / Property 20).

    Strictly: passes if and only if ``adapter_perplexity < base_perplexity``.
    """
    return adapter_perplexity < base_perplexity


def competitive_pass(
    adapter_perplexity: float, context_memory_perplexity: float
) -> bool:
    """Competitive comparison passes iff the adapter is at least as good as the
    context-memory baseline (Req 14.4 / Property 21).

    Passes if and only if ``adapter_perplexity <= context_memory_perplexity``.
    """
    return adapter_perplexity <= context_memory_perplexity


def predicted_unit(adapter_perplexities: Mapping[str, float]) -> str:
    """Return the adapter label with the lowest perplexity (Req 15.1 / Property 22).

    Ties are broken deterministically by the order of ``adapter_perplexities``
    (the first label achieving the minimum wins).
    """
    if not adapter_perplexities:
        raise ValueError("adapter_perplexities must be non-empty")
    best_label: str | None = None
    best_ppl = math.inf
    for label, ppl in adapter_perplexities.items():
        if ppl < best_ppl:
            best_ppl = ppl
            best_label = label
    assert best_label is not None
    return best_label


def build_confusion_matrix(
    labels: Sequence[str],
    predictions: Sequence[tuple[str, str]],
) -> ConfusionMatrix:
    """Build a square Confusion_Matrix from (true_label, predicted_label) pairs.

    Rows are the true Unit and columns are the predicted Unit (Req 15.2). Each
    prediction contributes exactly one count to cell ``[true][predicted]`` so
    the sum of all entries equals ``len(predictions)`` (Property 23).
    """
    label_list = list(labels)
    index = {label: i for i, label in enumerate(label_list)}
    n = len(label_list)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for true_label, pred_label in predictions:
        if true_label not in index:
            raise KeyError(f"true label {true_label!r} not in labels")
        if pred_label not in index:
            raise KeyError(f"predicted label {pred_label!r} not in labels")
        matrix[index[true_label]][index[pred_label]] += 1.0
    return ConfusionMatrix(labels=label_list, matrix=matrix)


def confusion_from_scores(
    labels: Sequence[str],
    score_rows: Mapping[str, Mapping[str, float]],
) -> ConfusionMatrix:
    """Build the Confusion_Matrix from per-true-unit adapter perplexities.

    ``score_rows[true_label][adapter_label]`` is the perplexity of the true
    Unit's held-out set under that adapter; the predicted Unit is the
    lowest-perplexity adapter (Property 22), and one count is recorded per true
    Unit (Property 23).
    """
    predictions = [
        (true_label, predicted_unit(score_rows[true_label])) for true_label in labels
    ]
    return build_confusion_matrix(labels, predictions)


def record_size_bytes(nktmirror: int, lora: int) -> SizeBytes:
    """Record the NKT-Mirror vs LoRA size comparison (Req 15.3 / 5.4)."""
    return SizeBytes(nktmirror=int(nktmirror), lora=int(lora))


# ---------------------------------------------------------------------------
# Weave/W&B logging hook (Req 14.5)
# ---------------------------------------------------------------------------


class PerplexityLogger(Protocol):
    """Sink for perplexity results; the real impl logs to Weave/W&B (Req 14.5)."""

    def log_perplexity(self, payload: Mapping[str, object]) -> None: ...


class NullLogger:
    """A no-op logger used for standalone runs without Weave/W&B configured."""

    def log_perplexity(self, payload: Mapping[str, object]) -> None:  # noqa: D401
        return None


# ---------------------------------------------------------------------------
# Per-unit eval results (Req 14)
# ---------------------------------------------------------------------------


@dataclass
class UnitEval:
    """The scored eval outcome for a single Unit (Req 14.1-14.4)."""

    unit_label: str
    adapter_perplexity: float
    base_perplexity: float
    context_memory_perplexity: float

    @property
    def personalization_passed(self) -> bool:
        return personalization_pass(self.adapter_perplexity, self.base_perplexity)

    @property
    def competitive_passed(self) -> bool:
        return competitive_pass(
            self.adapter_perplexity, self.context_memory_perplexity
        )


@dataclass
class HeldOutSet:
    """A Unit's held-out rows plus the example text injected for the baseline."""

    unit_label: str
    rows: Sequence[TrainingPair]
    context_examples: Sequence[str] = field(default_factory=list)


def _mean_perplexity(
    score_fn: ScoreFn,
    rows: Sequence[TrainingPair],
    adapter_id: str | None,
    prompt_prefix: str = "",
) -> float:
    """Mean per-row perplexity of ``rows`` under ``adapter_id`` (None = base)."""
    if not rows:
        return math.inf
    total = 0.0
    for row in rows:
        prompt = f"{prompt_prefix}{row.prompt}" if prompt_prefix else row.prompt
        total += float(score_fn(prompt, row.completion, adapter_id))
    return total / len(rows)


class WeaveEval:
    """Runs the WeaveSelf evaluation and emits ``eval_results.json``.

    Args:
        score_fn: the Inference_API ``/score`` callable (mocked until integration).
        logger: a Weave/W&B perplexity logger (defaults to a no-op).
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        logger: PerplexityLogger | None = None,
    ) -> None:
        self._score_fn = score_fn
        self._logger = logger or NullLogger()

    # --- Requirement 14: perplexity + context-memory baseline -------------

    def evaluate_unit(
        self,
        heldout: HeldOutSet,
        adapter_id: str,
    ) -> UnitEval:
        """Score a Unit's held-out set under its adapter, the base, and the
        context-memory baseline (Req 14.1, 14.3) and record both/all values."""
        adapter_ppl = _mean_perplexity(self._score_fn, heldout.rows, adapter_id)
        base_ppl = _mean_perplexity(self._score_fn, heldout.rows, None)
        # Context-memory baseline: inject the Unit's examples into the prompt and
        # score the SAME held-out set with no adapter (zero extra weight cost).
        prefix = ""
        if heldout.context_examples:
            prefix = "\n".join(heldout.context_examples) + "\n"
        context_ppl = _mean_perplexity(
            self._score_fn, heldout.rows, None, prompt_prefix=prefix
        )
        unit_eval = UnitEval(
            unit_label=heldout.unit_label,
            adapter_perplexity=adapter_ppl,
            base_perplexity=base_ppl,
            context_memory_perplexity=context_ppl,
        )
        self._logger.log_perplexity(
            {
                "unit_label": heldout.unit_label,
                "adapter": adapter_ppl,
                "base": base_ppl,
                "context_memory": context_ppl,
                "personalization_passed": unit_eval.personalization_passed,
                "competitive_passed": unit_eval.competitive_passed,
            }
        )
        return unit_eval

    # --- Requirement 15: confusion matrix + size chart + artifact ---------

    def cross_unit_confusion(
        self,
        heldouts: Sequence[HeldOutSet],
        adapter_ids: Mapping[str, str],
    ) -> ConfusionMatrix:
        """Score each Unit's held-out set under *every* trained adapter and
        build the confusion matrix; the predicted Unit is the lowest-perplexity
        adapter (Req 15.1, 15.2 / Properties 22, 23)."""
        labels = [h.unit_label for h in heldouts]
        score_rows: dict[str, dict[str, float]] = {}
        for held in heldouts:
            row: dict[str, float] = {}
            for unit_label, adapter_id in adapter_ids.items():
                row[unit_label] = _mean_perplexity(
                    self._score_fn, held.rows, adapter_id
                )
            score_rows[held.unit_label] = row
        return confusion_from_scores(labels, score_rows)

    def run(
        self,
        heldouts: Sequence[HeldOutSet],
        adapter_ids: Mapping[str, str],
        *,
        nktmirror_size_bytes: int,
        lora_size_bytes: int,
        examples: Sequence[Mapping[str, str]] | None = None,
        out_path: str | Path | None = None,
    ) -> EvalResults:
        """Run the full eval and (optionally) write ``eval_results.json``.

        Produces a schema-conformant :class:`EvalResults` (Req 5, 15.4 /
        Property 24): aggregate perplexity, the cross-unit confusion matrix, the
        size chart, and base-vs-adapter example pairs.
        """
        unit_evals = [
            self.evaluate_unit(held, adapter_ids[held.unit_label])
            for held in heldouts
        ]
        confusion = self.cross_unit_confusion(heldouts, adapter_ids)

        # Aggregate perplexity across Units for the top-level summary.
        n = max(1, len(unit_evals))
        perplexity = Perplexity(
            base=sum(u.base_perplexity for u in unit_evals) / n,
            adapter=sum(u.adapter_perplexity for u in unit_evals) / n,
            context_memory=sum(u.context_memory_perplexity for u in unit_evals) / n,
        )
        size = record_size_bytes(nktmirror_size_bytes, lora_size_bytes)
        example_models = [
            EvalExample(
                prompt=str(ex["prompt"]),
                base=str(ex["base"]),
                adapter=str(ex["adapter"]),
                reference=str(ex["reference"]),
            )
            for ex in (examples or [])
        ]
        results = EvalResults(
            perplexity=perplexity,
            confusion_matrix=confusion,
            size_bytes=size,
            examples=example_models,
        )
        if out_path is not None:
            write_eval_results(out_path, results)
        return results
