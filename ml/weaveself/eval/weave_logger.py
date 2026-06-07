"""Real Weave / W&B logger for Weave_Eval (Requirement 14.5).

This replaces the no-op :class:`~weaveself.eval.weave_eval.NullLogger` with a
genuine Weights & Biases / Weave integration so the proof layer (perplexity,
confusion matrix, size chart) is actually logged and browsable in Weave — the
project's designated "Best Use of Weave" surface.

Usage::

    logger = WeaveLogger(project="weaveself", entity="narwhals")
    weave_eval = WeaveEval(score_fn, logger=logger)
    results = weave_eval.run(...)
    url = logger.publish(results)   # versioned Weave object + run URL

``weave``/``wandb`` are optional heavy deps; importing this module does not
import them. Authentication uses ``WANDB_API_KEY`` from the environment
(loaded from ``.env``).
"""

from __future__ import annotations

import os
from typing import Any, Mapping


class WeaveLogger:
    """Logs perplexity results to Weave/W&B and publishes the eval artifact.

    On construction it calls ``weave.init("<entity>/<project>")`` which creates
    (or reuses) the Weave project and prints the browsable project URL. Each
    ``log_perplexity`` payload is buffered and also logged to a W&B run so the
    perplexity table and pass/fail flags show up in the W&B UI; :meth:`publish`
    versions the final ``eval_results`` object in Weave and logs the confusion
    matrix + size chart to W&B.
    """

    def __init__(self, project: str, entity: str | None = None) -> None:
        import weave  # lazy heavy import

        self._weave = weave
        self.project = project
        self.entity = entity
        target = f"{entity}/{project}" if entity else project
        # weave.init authenticates via WANDB_API_KEY and returns a client whose
        # repr/URL points at the Weave project.
        self.client = weave.init(target)
        self._rows: list[dict[str, Any]] = []

        # Optional parallel W&B run for charts (perplexity table, matrix, size).
        self._wandb = None
        self._run = None
        try:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                entity=entity,
                job_type="weave_eval",
                reinit=True,
            )
        except Exception:
            self._run = None

    # PerplexityLogger protocol ------------------------------------------------

    def log_perplexity(self, payload: Mapping[str, object]) -> None:
        row = dict(payload)
        self._rows.append(row)
        if self._run is not None:
            try:
                self._wandb.log(
                    {
                        f"perplexity/{row.get('unit_label','unit')}/adapter": row.get("adapter"),
                        f"perplexity/{row.get('unit_label','unit')}/base": row.get("base"),
                        f"perplexity/{row.get('unit_label','unit')}/context_memory": row.get("context_memory"),
                    }
                )
            except Exception:
                pass

    # Publish the final artifact ----------------------------------------------

    def publish(self, results: Any) -> str:
        """Version the eval results in Weave and log charts to W&B.

        Returns the published Weave object reference URI (a browsable link).
        """
        payload = results.model_dump() if hasattr(results, "model_dump") else dict(results)
        ref_uri = ""
        try:
            ref = self._weave.publish(payload, name="eval_results")
            ref_uri = getattr(ref, "uri", lambda: str(ref))()
        except Exception as exc:  # pragma: no cover - network dependent
            ref_uri = f"(weave publish failed: {exc})"

        if self._run is not None:
            try:
                cm = payload.get("confusion_matrix", {})
                labels = cm.get("labels", [])
                matrix = cm.get("matrix", [])
                ppl = payload.get("perplexity", {})
                size = payload.get("size_bytes", {})
                self._wandb.log(
                    {
                        "summary/perplexity_base": ppl.get("base"),
                        "summary/perplexity_adapter": ppl.get("adapter"),
                        "summary/perplexity_context_memory": ppl.get("context_memory"),
                        "summary/size_nktmirror": size.get("nktmirror"),
                        "summary/size_lora": size.get("lora"),
                    }
                )
                if labels and matrix:
                    self._wandb.log(
                        {
                            "confusion_matrix": self._wandb.Table(
                                columns=["true"] + list(labels),
                                data=[
                                    [labels[i]] + list(matrix[i])
                                    for i in range(len(labels))
                                ],
                            )
                        }
                    )
            except Exception:
                pass
            try:
                self._run.finish()
            except Exception:
                pass

        return ref_uri
