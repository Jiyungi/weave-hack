"""Demo pre-bake + Unit-selection routing-to-proof-visuals tests (task 12.2).

Covers Requirements 21.3 and 21.4:

* 21.3 — Adapter_Files are pre-baked for the demo ``day_index`` values so the
  demo is time-compressed: each demo day yields a distinct Adapter_File whose
  ``metadata.day_index`` matches the day it was baked for, all stored through
  the Redis_Client_API and servable by ``adapter_id``.
* 21.4 — selecting a Unit routes to the correct Adapter via the Redis_Client_API,
  generates a non-empty response through the Inference_API, and yields the
  corresponding ``eval_results.json`` proof-visual payload.

The serving Base_Model uses the dependency-free ``StubBackend`` default and the
Redis layer falls back to a file/in-memory backend, so the whole flow runs with
no GPU, no Qwen download, and no live Redis server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from weaveself.contracts.eval_results import EvalResults
from weaveself.integration import (
    DEMO_DAY_INDICES,
    DEMO_UNITS,
    DemoEnvironment,
    DemoSelection,
    RedisClientApi,
    make_demo_collector,
    prebake_demo_adapters,
)


@pytest.fixture
def demo_env(tmp_path) -> DemoEnvironment:
    return prebake_demo_adapters(workdir=tmp_path)


def _labels() -> list[str]:
    return [u["unit_label"] for u in DEMO_UNITS]


# --- Req 21.3: pre-baked adapters per demo day_index -----------------------


def test_prebake_produces_adapter_per_unit_per_day(demo_env):
    # Every demo day has a pre-baked adapter for every demo Unit, and pre-baking
    # never recorded a per-unit failure.
    assert demo_env.failures == []
    assert sorted(demo_env.catalog) == sorted(DEMO_DAY_INDICES)
    for day in DEMO_DAY_INDICES:
        assert set(demo_env.catalog[day]) == set(_labels())


def test_prebaked_adapter_metadata_day_index_matches(demo_env):
    # Each pre-baked adapter is stored through the Redis_Client_API and its
    # metadata.day_index equals the day it was baked for (Req 21.3 / Req 1.2).
    for day in DEMO_DAY_INDICES:
        for label in _labels():
            adapter_id = demo_env.catalog[day][label]
            meta = demo_env.redis_client.fetch_meta(adapter_id)
            assert meta["day_index"] == day
            assert meta["unit_label"] == label
            # Servable by adapter_id: the blob round-trips out of Redis.
            assert demo_env.redis_client.fetch_blob(adapter_id)


def test_prebaked_adapters_distinct_per_day(demo_env):
    # Time-compression: the same Unit yields a *distinct* Adapter_File on each
    # demo day (distinct adapter_id), so the days are genuinely different.
    for label in _labels():
        ids = {demo_env.catalog[day][label] for day in DEMO_DAY_INDICES}
        assert len(ids) == len(DEMO_DAY_INDICES)


def test_prebake_emits_eval_results_per_day(demo_env):
    for day in DEMO_DAY_INDICES:
        assert day in demo_env.eval_paths
        assert Path(demo_env.eval_paths[day]).exists()
        assert isinstance(demo_env.proof_visuals(day), EvalResults)


def test_base_model_loaded_once(demo_env):
    # One resident engine across every pre-bake day and the serving flow (Req 7.1).
    assert demo_env.engine.base_model_load_count == 1


# --- Req 21.4: Unit selection -> route -> generate -> proof visuals ---------


def test_select_unit_routes_generates_and_yields_proof_visuals(demo_env):
    active = demo_env.active_day
    for label in _labels():
        selection = demo_env.select_unit(label, prompt=f"As {label}, summarize today")

        assert isinstance(selection, DemoSelection)
        # Routed via the Redis_Client_API to the active day's adapter for this Unit.
        assert selection.adapter_id == demo_env.redis_client.route(label)
        assert selection.adapter_id == demo_env.catalog[active][label]
        # Generated a non-empty response through the Inference_API.
        assert isinstance(selection.text, str) and selection.text
        # Surfaced the corresponding eval_results.json proof-visual payload.
        assert isinstance(selection.eval_results, EvalResults)
        n = len(selection.eval_results.confusion_matrix.labels)
        assert n == len(selection.eval_results.confusion_matrix.matrix)


def test_select_unit_distinct_units_route_to_distinct_adapters(demo_env):
    routed = {
        label: demo_env.select_unit(label, prompt="hi").adapter_id
        for label in _labels()
    }
    assert len(set(routed.values())) == len(_labels())


def test_select_unit_round_trips_adapter_bytes_through_redis(demo_env):
    label = _labels()[0]
    selection = demo_env.select_unit(label, prompt="hello", from_redis_bytes=True)
    served = demo_env.serve_dir / f"adapter_{selection.adapter_id}.safetensors"
    assert served.exists()
    # The bytes materialized from Redis match what the engine loads from disk.
    assert served.read_bytes() == (
        demo_env.adapters_dir / f"adapter_{selection.adapter_id}.safetensors"
    ).read_bytes()


def test_activate_day_repoints_routing_to_that_day(demo_env):
    label = _labels()[0]
    # Selecting on an earlier day routes to THAT day's pre-baked adapter and
    # surfaces that day's proof visuals (Req 21.4 across the time-compressed days).
    for day in DEMO_DAY_INDICES:
        selection = demo_env.select_unit(label, prompt="hi", day_index=day)
        assert selection.day_index == day
        assert selection.adapter_id == demo_env.catalog[day][label]
        assert demo_env.redis_client.route(label) == demo_env.catalog[day][label]


def test_prebake_with_explicit_active_day(tmp_path):
    env = prebake_demo_adapters(workdir=tmp_path, active_day=0)
    assert env.active_day == 0
    label = _labels()[0]
    assert env.redis_client.route(label) == env.catalog[0][label]


def test_make_demo_collector_is_cumulative():
    collector = make_demo_collector(base_rows=4, rows_per_day=2)
    day0 = collector("alice", 0)
    day1 = collector("alice", 1)
    assert len(day0) == 4
    assert len(day1) == 6
    # Earlier rows are a prefix of later days' (accumulation).
    assert day1[: len(day0)] == day0


def test_reindex_route_targets_only_changes_routing(tmp_path):
    # The reindex helper repoints routing but leaves blob/meta addressable.
    env = prebake_demo_adapters(workdir=tmp_path)
    label = _labels()[0]
    day0_id = env.catalog[0][label]
    env.redis_client.reindex_route_targets([env.catalog[0][lbl] for lbl in _labels()])
    assert env.redis_client.route(label) == day0_id
    # A different day's blob is still fetchable by adapter_id.
    latest_id = env.catalog[DEMO_DAY_INDICES[-1]][label]
    assert env.redis_client.fetch_blob(latest_id)
    assert isinstance(env.redis_client, RedisClientApi)
