"""Property-based and example tests for the LangGraph batch graph (Track B / Req 13).

Property tests (min 100 generated cases via Hypothesis):

* task 6.4 — Property 17: batch graph executes nodes in the fixed order
* task 6.5 — Property 18: per-unit failures are isolated and recorded
* task 6.6 — Property 19: live chat never triggers training

Example/unit tests cover the remaining Requirement 13 wiring: train-node calls
``train_adapter`` with the curated path + labels (13.2), store-node persists via
the Redis_Client_API (13.3), the batch lock blocks chat-triggered execution
(13.5), and the recording-failure halt path (13.7).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from weaveself.data.curation import GPTCurationNode
from weaveself.orchestration import (
    NODE_ORDER,
    BatchDeps,
    BatchRunner,
    ChatCannotTriggerTrainingError,
    CriticalBatchError,
    build_batch_graph,
    initial_state,
)
from weaveself.orchestration.mocks import (
    MockInference,
    MockRedisClient,
    MockTrainAdapter,
)

_unit_labels = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6
)


def _make_units(labels):
    return [{"unit_label": label, "unit_type": "user"} for label in labels]


def _collector_for(labels, rows_per_unit=8):
    data = {
        label: [
            {"prompt": f"{label} q{i}", "completion": f"{label} ans {i}"}
            for i in range(rows_per_unit)
        ]
        for label in labels
    }
    return lambda label: data.get(label, [])


def _make_deps(tmp_path, labels, *, train_adapter=None, redis_client=None):
    return BatchDeps(
        collector=_collector_for(labels),
        curation_node=GPTCurationNode(),
        train_adapter=train_adapter or MockTrainAdapter(tmp_path / "adapters"),
        redis_client=redis_client or MockRedisClient(),
        score_fn=MockInference(),
        workdir=tmp_path,
        min_rows=2,
    )


# Feature: weaveself, Property 17: Batch graph executes nodes in the fixed order
@settings(max_examples=100, deadline=None)
@given(labels=st.lists(_unit_labels, min_size=1, max_size=4, unique=True))
def test_property_17_fixed_node_order(tmp_path_factory, labels):
    tmp_path = tmp_path_factory.mktemp("batch")
    deps = _make_deps(tmp_path, labels)
    graph = build_batch_graph(deps)
    final = graph.invoke(initial_state(_make_units(labels)))
    # Nodes always run collect -> curate -> train -> eval -> store (Req 13.1).
    assert tuple(final["executed_nodes"]) == NODE_ORDER


# Feature: weaveself, Property 18: Per-unit failures are isolated and recorded
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_property_18_failures_isolated_and_recorded(tmp_path_factory, data):
    labels = data.draw(
        st.lists(_unit_labels, min_size=2, max_size=5, unique=True)
    )
    # Choose a non-empty proper subset of units to fail at the train node.
    failing = set(
        data.draw(
            st.lists(st.sampled_from(labels), min_size=1, max_size=len(labels) - 1)
        )
    )
    tmp_path = tmp_path_factory.mktemp("batch")

    real_train = MockTrainAdapter(tmp_path / "adapters")

    def flaky_train(dataset_path, unit_label, unit_type):
        if unit_label in failing:
            raise RuntimeError(f"injected train failure for {unit_label}")
        return real_train(dataset_path, unit_label, unit_type)

    deps = _make_deps(tmp_path, labels, train_adapter=flaky_train)
    graph = build_batch_graph(deps)
    final = graph.invoke(initial_state(_make_units(labels)))

    recorded = {(f["node"], f["unit_label"]) for f in final["failures"]}
    # Each failing Unit is recorded with its failing node + label (Req 13.6).
    for label in failing:
        assert ("train", label) in recorded
    # Every non-failing Unit still completes (produces an adapter) (Req 13.6).
    for label in set(labels) - failing:
        assert label in final["adapters"]
    # Failing units produced no adapter.
    for label in failing:
        assert label not in final["adapters"]


# Feature: weaveself, Property 19: Live chat never triggers training
@settings(max_examples=100, deadline=None)
@given(n_chats=st.integers(min_value=0, max_value=30))
def test_property_19_chat_never_trains(tmp_path_factory, n_chats):
    tmp_path = tmp_path_factory.mktemp("batch")
    train_adapter = MockTrainAdapter(tmp_path / "adapters")
    infer = MockInference()
    deps = _make_deps(tmp_path, ["alice"], train_adapter=train_adapter)
    graph = build_batch_graph(deps)
    runner = BatchRunner(
        graph,
        infer_fn=lambda req: infer.score(req["prompt"], req["target"], req.get("adapter_id")),
    )

    for i in range(n_chats):
        runner.handle_chat_request(
            {"prompt": f"hi {i}", "target": "there", "adapter_id": "alice-d0"}
        )

    # No chat request triggered any training invocation (Req 13.4 / Property 19).
    assert train_adapter.call_count == 0
    # And an explicit attempt to train from chat is always refused.
    with pytest.raises(ChatCannotTriggerTrainingError):
        runner.attempt_chat_triggered_training()


# --- Example / wiring tests for Requirement 13 -----------------------------


def test_train_node_invokes_train_adapter_with_curated_path_and_labels(tmp_path):
    """Req 13.2: train node calls train_adapter with curated path + labels."""
    labels = ["alice", "bob"]
    train_adapter = MockTrainAdapter(tmp_path / "adapters")
    deps = _make_deps(tmp_path, labels, train_adapter=train_adapter)
    graph = build_batch_graph(deps)
    final = graph.invoke(initial_state(_make_units(labels)))

    called_labels = {label for (_path, label, _type) in train_adapter.calls}
    assert called_labels == set(labels)
    for dataset_path, label, unit_type in train_adapter.calls:
        assert dataset_path == final["curated"][label]
        assert unit_type == "user"


def test_store_node_persists_through_redis_client(tmp_path):
    """Req 13.3: store node persists each Adapter_File + metadata via Redis API."""
    labels = ["alice", "bob"]
    redis = MockRedisClient()
    deps = _make_deps(tmp_path, labels, redis_client=redis)
    graph = build_batch_graph(deps)
    graph.invoke(initial_state(_make_units(labels)))

    assert set(redis.meta.keys()) == {"alice-d0", "bob-d0"}
    for adapter_id, meta in redis.meta.items():
        assert meta["adapter_id"] == adapter_id
        # The blob round-trips as bytes through the client.
        assert isinstance(redis.fetch_blob(adapter_id), bytes)
        assert len(redis.fetch_blob(adapter_id)) > 0


def test_running_batch_blocks_chat_triggered_execution(tmp_path):
    """Req 13.5: while a batch run is in progress, chat-triggered execution is blocked."""
    deps = _make_deps(tmp_path, ["alice"])
    graph = build_batch_graph(deps)
    runner = BatchRunner(graph)
    runner._running = True  # simulate an in-progress batch run
    with pytest.raises(ChatCannotTriggerTrainingError):
        runner.attempt_chat_triggered_training()


def test_recording_failure_halts_the_batch(tmp_path):
    """Req 13.7: if failure recording itself fails, the batch halts."""
    labels = ["alice"]

    def always_fail_train(dataset_path, unit_label, unit_type):
        raise RuntimeError("boom")

    deps = _make_deps(tmp_path, labels, train_adapter=always_fail_train)
    graph = build_batch_graph(deps)

    state = initial_state(_make_units(labels))

    class _Unappendable(list):
        def append(self, _item):
            raise RuntimeError("cannot record failure")

    state["failures"] = _Unappendable()

    # The train node hits a per-unit failure, tries to record it, recording
    # fails, and the run halts with a CriticalBatchError (Req 13.7).
    with pytest.raises(CriticalBatchError):
        graph.invoke(state)
