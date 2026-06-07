"""Track B standalone test (task 7.8 / Requirement 16).

Runs the full LangGraph batch graph end to end against Mock_Dependencies for
Track A's ``train_adapter`` and inference API (``/score``) and Track C's Redis
client, and asserts that the run emits an ``eval_results.json`` containing a
well-formed Confusion_Matrix (Req 16.1) with no dependency on the Frontend_App
(Req 16.2). This is the single highest-priority, independently-demoable
artifact (Req 23.4).
"""

from __future__ import annotations

import sys

from weaveself.contracts.eval_results import EvalResults, read_eval_results
from weaveself.data.curation import GPTCurationNode
from weaveself.orchestration import (
    NODE_ORDER,
    BatchDeps,
    build_batch_graph,
    initial_state,
)
from weaveself.orchestration.mocks import (
    MockInference,
    MockRedisClient,
    MockTrainAdapter,
)


def _demo_units():
    return [
        {"unit_label": "alice", "unit_type": "user"},
        {"unit_label": "bob", "unit_type": "user"},
        {"unit_label": "carol", "unit_type": "user"},
    ]


def _demo_interactions(labels, n=10):
    data = {
        label: [
            {"prompt": f"{label} question {i}", "completion": f"{label} answer {i}"}
            for i in range(n)
        ]
        for label in labels
    }
    return lambda label: data.get(label, [])


def test_track_b_standalone_emits_confusion_matrix(tmp_path):
    units = _demo_units()
    labels = [u["unit_label"] for u in units]
    eval_path = tmp_path / "eval_results.json"

    redis = MockRedisClient()
    deps = BatchDeps(
        collector=_demo_interactions(labels),
        curation_node=GPTCurationNode(),  # default local (GPT-free) curator
        train_adapter=MockTrainAdapter(tmp_path / "adapters"),
        redis_client=redis,
        score_fn=MockInference(),
        workdir=tmp_path,
        min_rows=2,
        eval_out_path=eval_path,
    )

    graph = build_batch_graph(deps)
    final = graph.invoke(initial_state(units))

    # Full pipeline ran in the fixed order, no failures.
    assert tuple(final["executed_nodes"]) == NODE_ORDER
    assert final["failures"] == []
    assert set(final["adapters"]) == set(labels)

    # An eval_results.json artifact was emitted to disk (Req 16.1).
    assert eval_path.exists()
    results = read_eval_results(eval_path)
    assert isinstance(results, EvalResults)

    # It contains a well-formed Confusion_Matrix (Req 16.1, 5.3).
    cm = results.confusion_matrix
    n = len(cm.labels)
    assert n == len(labels)
    assert len(cm.matrix) == n
    assert all(len(row) == n for row in cm.matrix)
    # Each Unit's held-out set contributed exactly one prediction.
    assert sum(sum(row) for row in cm.matrix) == float(n)

    # The mock scores make each Unit identify itself: a perfect diagonal.
    index = {label: i for i, label in enumerate(cm.labels)}
    for label in labels:
        i = index[label]
        assert cm.matrix[i][i] == 1.0

    # Personalization passes overall (adapter perplexity below base).
    assert results.perplexity.adapter < results.perplexity.base

    # Adapters were persisted through the Redis_Client_API (Req 13.3).
    assert set(redis.meta.keys()) == {f"{label}-d0" for label in labels}


def test_track_b_standalone_has_no_frontend_dependency():
    """Req 16.2: the Track B track imports no Frontend_App / Node-TS modules."""
    # Importing the whole Track B surface must not require any frontend package.
    import weaveself.data  # noqa: F401
    import weaveself.eval  # noqa: F401
    import weaveself.orchestration  # noqa: F401

    forbidden = [m for m in sys.modules if m.startswith(("copilotkit", "react"))]
    assert forbidden == []
