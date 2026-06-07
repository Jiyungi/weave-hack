"""Critical-path governance + blocked-state fallback tests (task 12.3).

Cover Requirements 22.1, 22.2, 22.3, 23.2, 23.4:

* When Track A serving is reported verified, the governed batch deps wire the
  REAL cross-track dependencies (Req 22.1).
* While serving is NOT verified, the Mock_Dependency stays wired AND the
  blocked-integration state is recorded (Req 22.2, 22.3).
* The Track B confusion-matrix fallback demo emits a valid eval_results.json
  with a square Confusion_Matrix independently of every other track (Req 23.2,
  23.4).

All tests run on the dependency-free defaults (StubBackend + mocks), so they
need no GPU, no Qwen download, and no live Redis.
"""

from __future__ import annotations

import json
from pathlib import Path

from weaveself.contracts.eval_results import EvalResults
from weaveself.integration import (
    CriticalPathGovernor,
    DependencyMode,
    IntegrationState,
    RedisClientApi,
    ServingVerification,
    build_governed_batch_deps,
    build_mock_batch_deps,
    fallback_demo_entry_points,
    run_track_a_compare_demo,
    run_track_b_confusion_matrix_demo,
    verify_track_a_serving,
)
from weaveself.orchestration import NODE_ORDER
from weaveself.orchestration.mocks import (
    MockInference,
    MockRedisClient,
    MockTrainAdapter,
)
from weaveself.serving.engine import ServingEngine


def _demo_collector(labels, n=6):
    data = {
        label: [
            {"prompt": f"{label} q{i}", "completion": f"{label} a{i}"}
            for i in range(n)
        ]
        for label in labels
    }
    return lambda label: list(data.get(label, []))


# --- Req 22.1: serving verified -> REAL dependencies wired -----------------


def test_serving_verified_wires_real_dependencies(tmp_path):
    governed = build_governed_batch_deps(
        serving_verified=True,
        collector=_demo_collector(["alice", "bob"]),
        workdir=tmp_path,
        base_model="stub-base",
        min_rows=2,
        status_path=tmp_path / "integration_status.json",
    )

    # Real deps wired (Req 22.1): real RedisClientApi, real bound train_adapter,
    # and the engine-backed score_fn — NOT the mocks.
    assert governed.status.state is IntegrationState.INTEGRATED
    assert governed.status.dependency_mode is DependencyMode.REAL
    assert isinstance(governed.deps.redis_client, RedisClientApi)
    assert not isinstance(governed.deps.train_adapter, MockTrainAdapter)
    assert not isinstance(governed.deps.score_fn, MockInference)
    # The real bound train_adapter exposes its adapters_dir (build_real_train_adapter).
    assert hasattr(governed.deps.train_adapter, "adapters_dir")

    # Status artifact records the integrated state.
    recorded = json.loads((tmp_path / "integration_status.json").read_text("utf-8"))
    assert recorded["state"] == "integrated"
    assert recorded["serving_verified"] is True


# --- Req 22.2 + 22.3: not verified -> mocks stay wired + blocked recorded --


def test_serving_unverified_keeps_mocks_and_records_blocked(tmp_path):
    status_path = tmp_path / "integration_status.json"
    governed = build_governed_batch_deps(
        serving_verified=False,
        collector=_demo_collector(["alice", "bob"]),
        workdir=tmp_path,
        min_rows=2,
        status_path=status_path,
    )

    # Mock_Dependency stays wired (Req 22.2).
    assert governed.status.dependency_mode is DependencyMode.MOCK
    assert isinstance(governed.deps.train_adapter, MockTrainAdapter)
    assert isinstance(governed.deps.score_fn, MockInference)
    assert isinstance(governed.deps.redis_client, MockRedisClient)

    # Blocked-integration state recorded (Req 22.3) with the per-track fallbacks.
    assert governed.status.state is IntegrationState.BLOCKED
    assert governed.status.blocked is True
    assert governed.status.fallback_demos  # references Req 23 standalone demos

    assert status_path.exists()
    recorded = json.loads(status_path.read_text("utf-8"))
    assert recorded["state"] == "blocked"
    assert recorded["serving_verified"] is False
    assert recorded["dependency_mode"] == "mock"
    assert len(recorded["fallback_demos"]) == 3


def test_governed_mock_deps_still_produce_confusion_matrix(tmp_path):
    """The mock-wired deps remain fully runnable (Req 22.2): the batch graph
    still emits a valid eval_results.json so Track B keeps moving."""
    from weaveself.contracts.eval_results import read_eval_results
    from weaveself.orchestration import build_batch_graph, initial_state

    labels = ["alice", "bob", "carol"]
    eval_path = tmp_path / "eval_results.json"
    deps = build_mock_batch_deps(
        collector=_demo_collector(labels),
        workdir=tmp_path,
        min_rows=2,
        eval_out_path=eval_path,
    )
    graph = build_batch_graph(deps)
    final = graph.invoke(initial_state([{"unit_label": l, "unit_type": "user"} for l in labels]))

    assert tuple(final["executed_nodes"]) == NODE_ORDER
    assert eval_path.exists()
    results = read_eval_results(eval_path)
    assert len(results.confusion_matrix.labels) == len(labels)


# --- Serving-verification gate -------------------------------------------


def test_verify_track_a_serving_passes_on_resident_engine(tmp_path):
    engine = ServingEngine("stub-base")
    verification = verify_track_a_serving(engine)
    assert isinstance(verification, ServingVerification)
    assert verification.verified is True
    assert verification.checks["base_loaded_once"] is True
    assert verification.checks["base_generates"] is True
    assert verification.checks["score_non_negative"] is True


def test_verify_track_a_serving_detects_adapter_differential(tmp_path):
    # Train a mock adapter the engine can load, then verify the differential.
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    from weaveself.contracts.training_pair import write_training_pairs

    ds = tmp_path / "t.jsonl"
    write_training_pairs(
        ds,
        [{"prompt": "p", "completion": "c", "unit_label": "alice"}],
    )
    adapter_path = MockTrainAdapter(adapters_dir)(str(ds), "alice", "user")
    adapter_id = Path(adapter_path).stem.removeprefix("adapter_")

    engine = ServingEngine("stub-base", adapters_dir=str(adapters_dir))
    verification = verify_track_a_serving(engine, adapter_id=adapter_id)
    assert verification.verified is True
    assert verification.checks["adapter_differs"] is True


def test_governor_accepts_callable_and_verification_inputs(tmp_path):
    # Callable producing a bool.
    assert CriticalPathGovernor(lambda: True).verify().verified is True
    # Pre-computed ServingVerification passes straight through.
    v = ServingVerification(verified=False, reason="manual", checks={})
    status = CriticalPathGovernor(v).status()
    assert status.state is IntegrationState.BLOCKED
    assert status.verification is v


# --- Req 23.2 + 23.4: Track B confusion-matrix fallback demo --------------


def test_track_b_confusion_matrix_fallback_demo_is_independent(tmp_path):
    results, eval_path = run_track_b_confusion_matrix_demo(tmp_path)

    assert isinstance(results, EvalResults)
    assert eval_path.exists()

    # Square Confusion_Matrix (Req 5.3 / 23.4).
    cm = results.confusion_matrix
    n = len(cm.labels)
    assert n >= 1
    assert len(cm.matrix) == n
    assert all(len(row) == n for row in cm.matrix)
    # Exactly one prediction per true Unit -> entries sum to n.
    assert sum(sum(row) for row in cm.matrix) == float(n)

    # The mock scores make each Unit identify itself: perfect diagonal.
    index = {label: i for i, label in enumerate(cm.labels)}
    for label in cm.labels:
        i = index[label]
        assert cm.matrix[i][i] == 1.0

    # Personalization passes overall (adapter perplexity below base).
    assert results.perplexity.adapter < results.perplexity.base


def test_track_a_compare_fallback_demo_differs(tmp_path):
    out = run_track_a_compare_demo(tmp_path)
    assert out["base"] != out["adapter"]
    assert out["differ"] is True


def test_fallback_demo_entry_points_cover_all_tracks():
    demos = fallback_demo_entry_points()
    assert set(demos) == {"track_a", "track_b", "track_c"}
    # Track B confusion matrix is the single highest-priority artifact (Req 23.4).
    assert demos["track_b"].highest_priority is True
    assert demos["track_a"].highest_priority is False
    assert demos["track_c"].highest_priority is False
    # Track A and Track B fallbacks are runnable in-process; Track C lives in Node/TS.
    assert callable(demos["track_a"].runnable)
    assert callable(demos["track_b"].runnable)
    assert demos["track_c"].runnable is None
