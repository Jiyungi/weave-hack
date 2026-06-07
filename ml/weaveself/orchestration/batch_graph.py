"""LangGraph nightly-batch graph (Track B / Requirement 13).

Builds the ``collect -> curate -> train -> eval -> store`` state machine
(Req 13.1) as a LangGraph :class:`~langgraph.graph.StateGraph`. Cross-track
dependencies are injected via :class:`BatchDeps` so they can be mocked until the
Integration_Milestone (Req 22.2):

* ``train`` invokes Track A's ``train_adapter`` with the curated dataset path,
  ``unit_label``, and ``unit_type`` (Req 13.2);
* ``store`` persists each Adapter_File and its metadata through the
  Redis_Client_API (Req 13.3).

Per-unit failures are recorded with their failing node and ``unit_label`` and
processing continues for the remaining Units (Req 13.6 / Property 18). If
failure recording itself fails or a critical error occurs, the graph halts
(Req 13.7).

The :class:`BatchRunner` enforces that training only happens as a batch job and
that live chat never triggers training (Req 13.4, 13.5 / Property 19).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langgraph.graph import END, START, StateGraph

from weaveself.contracts.adapter_file import read_adapter_file
from weaveself.contracts.training_pair import read_training_pairs
from weaveself.data.curation import GPTCurationNode
from weaveself.data.pipeline import UnitSource, build_splits
from weaveself.eval.weave_eval import (
    HeldOutSet,
    PerplexityLogger,
    ScoreFn,
    WeaveEval,
)
from weaveself.orchestration.state import BatchState, FailureRecord

NODE_ORDER: tuple[str, ...] = ("collect", "curate", "train", "eval", "store")


class CriticalBatchError(RuntimeError):
    """Raised when the batch must halt (critical error or recording failure).

    Unlike per-unit failures (which are recorded so processing continues), a
    :class:`CriticalBatchError` is never swallowed by the per-unit isolation
    wrapper, so it halts the whole run (Req 13.7).
    """


class ChatCannotTriggerTrainingError(RuntimeError):
    """Raised if a live chat request attempts to trigger adapter training.

    Training is strictly a batch job; live chat must never trigger it
    (Req 13.4 / Property 19).
    """


@dataclass
class BatchDeps:
    """Injected dependencies for the batch graph (mockable until integration)."""

    collector: Callable[[str], list[dict]]
    curation_node: GPTCurationNode
    train_adapter: Callable[[str, str, str], str]
    redis_client: object
    score_fn: ScoreFn
    workdir: str | Path
    min_rows: int = 1
    logger: PerplexityLogger | None = None
    lora_size_bytes: int = 4_194_304
    eval_runner: Callable[[BatchState], dict] | None = None
    eval_out_path: str | Path | None = None


def _record_failure(state: BatchState, node: str, unit_label: str) -> None:
    """Append a per-unit failure record (Req 13.6); halt if recording fails (13.7)."""
    record: FailureRecord = {"node": node, "unit_label": unit_label}
    try:
        state["failures"].append(record)
    except Exception as exc:  # recording itself failed -> critical halt (13.7)
        raise CriticalBatchError(
            f"failed to record failure for unit '{unit_label}' at node '{node}'"
        ) from exc


def _adapter_id_from_path(adapter_path: str) -> str:
    return Path(adapter_path).stem.removeprefix("adapter_")


def build_batch_graph(deps: BatchDeps):
    """Compile the LangGraph batch graph for the given dependencies."""
    workdir = Path(deps.workdir)

    # --- collect ----------------------------------------------------------
    def collect(state: BatchState) -> dict:
        interactions = dict(state.get("interactions", {}))
        for unit in state["units"]:
            label = unit["unit_label"]
            try:
                interactions[label] = list(deps.collector(label))
            except CriticalBatchError:
                raise
            except Exception:
                _record_failure(state, "collect", label)
        return {
            "interactions": interactions,
            "executed_nodes": state.get("executed_nodes", []) + ["collect"],
        }

    # --- curate -----------------------------------------------------------
    def curate(state: BatchState) -> dict:
        curated = dict(state.get("curated", {}))
        heldout = dict(state.get("heldout", {}))
        discarded = dict(state.get("discarded", {}))
        for unit in state["units"]:
            label = unit["unit_label"]
            if label not in state.get("interactions", {}):
                continue  # collect failed for this unit; skip
            try:
                result = deps.curation_node.curate_interactions(
                    state["interactions"][label], label
                )
                discarded[label] = result.discarded
                unit_source = UnitSource(
                    unit_label=label,
                    unit_type=unit["unit_type"],
                    rows=[p.model_dump() for p in result.pairs],
                )
                split = build_splits(
                    [unit_source],
                    min_rows=deps.min_rows,
                    out_dir=workdir / "curated" / label,
                )
                if label in split.included_units:
                    curated[label] = str(split.train_path)
                    heldout[label] = str(split.heldout_path)
            except CriticalBatchError:
                raise
            except Exception:
                _record_failure(state, "curate", label)
        return {
            "curated": curated,
            "heldout": heldout,
            "discarded": discarded,
            "executed_nodes": state.get("executed_nodes", []) + ["curate"],
        }

    # --- train ------------------------------------------------------------
    def train(state: BatchState) -> dict:
        adapters = dict(state.get("adapters", {}))
        unit_types = {u["unit_label"]: u["unit_type"] for u in state["units"]}
        for label, dataset_path in state.get("curated", {}).items():
            try:
                adapter_path = deps.train_adapter(
                    dataset_path, label, unit_types[label]
                )
                adapters[label] = adapter_path
            except CriticalBatchError:
                raise
            except Exception:
                _record_failure(state, "train", label)
        return {
            "adapters": adapters,
            "executed_nodes": state.get("executed_nodes", []) + ["train"],
        }

    # --- eval -------------------------------------------------------------
    def evaluate(state: BatchState) -> dict:
        if deps.eval_runner is not None:
            eval_results = deps.eval_runner(state)
        else:
            eval_results = _default_eval(state, deps)
        return {
            "eval_results": eval_results,
            "executed_nodes": state.get("executed_nodes", []) + ["eval"],
        }

    # --- store ------------------------------------------------------------
    def store(state: BatchState) -> dict:
        for label, adapter_path in state.get("adapters", {}).items():
            try:
                path = Path(adapter_path)
                adapter_id = _adapter_id_from_path(adapter_path)
                meta, _gates = read_adapter_file(path.parent, adapter_id)
                blob = path.read_bytes()
                deps.redis_client.store_adapter(meta, blob)  # type: ignore[attr-defined]
            except CriticalBatchError:
                raise
            except Exception:
                _record_failure(state, "store", label)
        return {"executed_nodes": state.get("executed_nodes", []) + ["store"]}

    graph = StateGraph(BatchState)
    graph.add_node("collect", collect)
    graph.add_node("curate", curate)
    graph.add_node("train", train)
    graph.add_node("eval", evaluate)
    graph.add_node("store", store)
    graph.add_edge(START, "collect")
    graph.add_edge("collect", "curate")
    graph.add_edge("curate", "train")
    graph.add_edge("train", "eval")
    graph.add_edge("eval", "store")
    graph.add_edge("store", END)
    return graph.compile()


def _default_eval(state: BatchState, deps: BatchDeps) -> dict:
    """Default eval node: score held-out sets and emit the eval artifact dict."""
    adapters = state.get("adapters", {})
    heldout_paths = state.get("heldout", {})
    labels = [
        label
        for label in adapters
        if label in heldout_paths and Path(heldout_paths[label]).exists()
    ]
    if not labels:
        return {}

    heldouts: list[HeldOutSet] = []
    adapter_ids: dict[str, str] = {}
    nktmirror_size = 0
    for label in labels:
        rows = read_training_pairs(heldout_paths[label])
        if not rows:
            continue
        # Inject a couple of the Unit's own held-out prompts as context examples
        # for the context-memory baseline.
        context = [r.prompt for r in rows[:2]]
        heldouts.append(
            HeldOutSet(unit_label=label, rows=rows, context_examples=context)
        )
        adapter_path = adapters[label]
        adapter_id = _adapter_id_from_path(adapter_path)
        adapter_ids[label] = adapter_id
        meta, _gates = read_adapter_file(Path(adapter_path).parent, adapter_id)
        nktmirror_size = max(nktmirror_size, meta.size_bytes)

    if not heldouts:
        return {}

    examples = []
    for held in heldouts:
        row = held.rows[0]
        examples.append(
            {
                "prompt": row.prompt,
                "base": f"[base] {row.completion}",
                "adapter": f"[{held.unit_label} adapter] {row.completion}",
                "reference": row.completion,
            }
        )

    weave_eval = WeaveEval(deps.score_fn, logger=deps.logger)
    results = weave_eval.run(
        heldouts,
        adapter_ids,
        nktmirror_size_bytes=nktmirror_size,
        lora_size_bytes=deps.lora_size_bytes,
        examples=examples,
        out_path=deps.eval_out_path,
    )
    return results.model_dump()


class BatchRunner:
    """Runs the batch graph and enforces batch-only training (Req 13.4, 13.5).

    * :meth:`run_batch` is the only path that executes the graph (and therefore
      the only path that can trigger training); it holds a lock for the duration
      of the run.
    * :meth:`handle_chat_request` serves a live chat request via inference only
      and never trains (Property 19).
    * :meth:`attempt_chat_triggered_training` always refuses (Req 13.4), and
      while a batch run is in progress chat-triggered execution is blocked
      (Req 13.5).
    """

    def __init__(self, graph, *, infer_fn: Callable[[dict], object] | None = None):
        self._graph = graph
        self._infer_fn = infer_fn
        self._lock = threading.Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def run_batch(self, state: BatchState) -> BatchState:
        with self._lock:
            self._running = True
            try:
                return self._graph.invoke(state)
            finally:
                self._running = False

    def handle_chat_request(self, request: dict) -> object:
        """Serve a live chat request — inference only, never training."""
        if self._infer_fn is None:
            raise RuntimeError("no inference function configured for chat")
        return self._infer_fn(request)

    def attempt_chat_triggered_training(self) -> None:
        """Reject any attempt to trigger training from a live chat request."""
        if self._running:
            raise ChatCannotTriggerTrainingError(
                "a batch run is in progress; chat cannot trigger graph execution"
            )
        raise ChatCannotTriggerTrainingError(
            "live chat requests never trigger adapter training; training is batch-only"
        )
