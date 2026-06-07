"""Shared state types for the LangGraph nightly-batch graph (Track B / Req 13).

The batch is a linear state machine ``collect -> curate -> train -> eval ->
store`` (Req 13.1). :class:`BatchState` is the TypedDict threaded through the
graph; per-unit failures accumulate in ``failures`` (Req 13.6) and the executed
node names accumulate in ``executed_nodes`` so the fixed order is verifiable
(Property 17).
"""

from __future__ import annotations

from typing import TypedDict


class UnitSpec(TypedDict):
    """One Unit to process in a batch run."""

    unit_label: str
    unit_type: str  # "category" | "user"


class FailureRecord(TypedDict):
    """A recorded per-unit failure: the failing node and the Unit (Req 13.6)."""

    node: str
    unit_label: str


class BatchState(TypedDict, total=False):
    """State threaded through the batch graph.

    Distinct nodes own distinct keys; ``failures`` and ``executed_nodes`` are
    append-only across nodes.
    """

    units: list[UnitSpec]
    interactions: dict[str, list[dict]]  # unit_label -> raw interactions (collect)
    curated: dict[str, str]              # unit_label -> train dataset path (curate)
    heldout: dict[str, str]              # unit_label -> held-out dataset path (curate)
    discarded: dict[str, int]            # unit_label -> discarded interaction count
    adapters: dict[str, str]             # unit_label -> adapter_path (train)
    eval_results: dict                   # eval artifact dict (eval)
    failures: list[FailureRecord]        # per-unit failures (Req 13.6)
    executed_nodes: list[str]            # node execution trace (Property 17)


def initial_state(units: list[UnitSpec]) -> BatchState:
    """Build a fresh :class:`BatchState` for the given Units."""
    return BatchState(
        units=list(units),
        interactions={},
        curated={},
        heldout={},
        discarded={},
        adapters={},
        eval_results={},
        failures=[],
        executed_nodes=[],
    )
