"""Integration-milestone wiring tests (task 12.1 / Requirements 21.1, 21.2).

These exercise the *real* cross-track wiring that replaces the Track B
``Mock_Dependency`` fixtures (Req 21.1) and the end-to-end loop that runs the
batch graph, stores each produced Adapter_File through the Redis_Client_API, and
serves each by ``adapter_id`` retrieved through the Redis_Client_API via the
Inference_API (Req 21.2).

The serving Base_Model uses the dependency-free ``StubBackend`` default, so the
real serving contract (single base load, per-request gate swap, teacher-forced
score) is exercised without a GPU or Qwen download.
"""

from __future__ import annotations

import json

from weaveself.contracts.eval_results import EvalResults, read_eval_results
from weaveself.integration import (
    RedisClientApi,
    adapter_blob_key,
    adapter_index_key,
    adapter_meta_key,
    build_end_to_end_loop,
    create_redis_client,
    embed_text,
    interactions_key,
)
from weaveself.integration.redis_client import FileKvBackend, InMemoryKvBackend
from weaveself.orchestration import NODE_ORDER


def _demo_units():
    return [
        {"unit_label": "alice", "unit_type": "user"},
        {"unit_label": "bob", "unit_type": "user"},
        {"unit_label": "carol", "unit_type": "user"},
    ]


def _demo_collector(labels, n=8):
    data = {
        label: [
            {"prompt": f"{label} question {i}", "completion": f"{label} answer {i}"}
            for i in range(n)
        ]
        for label in labels
    }
    return lambda label: list(data.get(label, []))


# --- Redis_Client_API contract (same key layout as Track C) ----------------


def test_redis_client_key_layout_and_roundtrip(tmp_path):
    client = RedisClientApi(FileKvBackend(tmp_path / "store.json"))
    meta = {
        "adapter_id": "abc123",
        "base_model": "stub-base",
        "unit_type": "user",
        "unit_label": "alice",
        "train_rows": 4,
        "trained_at": "2024-01-01T00:00:00Z",
        "day_index": 0,
        "size_bytes": 10,
    }
    blob = bytes(range(256)) * 4  # 1 KB of arbitrary bytes

    client.store_adapter(meta, blob)

    # Metadata round-trips independently of the blob (Req 19.2) ...
    assert client.fetch_meta("abc123")["unit_label"] == "alice"
    # ... and the blob is byte-identical (Req 19.4).
    assert client.fetch_blob("abc123") == blob

    # The persisted file uses the canonical Track C key layout (Req 3.1-3.3).
    raw = json.loads((tmp_path / "store.json").read_text(encoding="utf-8"))
    assert adapter_blob_key("abc123") in raw["strings"]
    assert adapter_meta_key("abc123") in raw["strings"]
    assert adapter_index_key() in raw["strings"]


def test_redis_client_route_and_interactions():
    client = RedisClientApi(InMemoryKvBackend())
    for label in ("alice", "bob", "carol"):
        client.store_adapter(
            {
                "adapter_id": f"id-{label}",
                "base_model": "stub-base",
                "unit_type": "user",
                "unit_label": label,
                "train_rows": 2,
                "trained_at": "2024-01-01T00:00:00Z",
                "day_index": 0,
                "size_bytes": 1,
            },
            b"\x00",
        )
    # route() returns each Unit's own adapter via vector search (Req 3.5/19.3).
    for label in ("alice", "bob", "carol"):
        assert client.route(label) == f"id-{label}"

    # append/read interactions under interactions:<unit_label> (Req 3.4/19.5).
    client.append_interaction("alice", {"prompt": "hi", "completion": "hello"})
    assert client.read_interactions("alice") == [
        {"prompt": "hi", "completion": "hello"}
    ]


def test_embed_text_matches_track_c_shape():
    # Deterministic 64-d embedding; identical labels embed identically.
    vec = embed_text("alice")
    assert len(vec) == 64
    assert embed_text("alice") == vec
    assert embed_text("alice") != embed_text("bob")


def test_interactions_key_builder():
    assert interactions_key("alice") == "interactions:alice"


# --- End-to-end loop: graph -> Redis -> Inference_API (Req 21.2) -----------


def test_end_to_end_loop_real_dependencies(tmp_path):
    units = _demo_units()
    labels = [u["unit_label"] for u in units]

    loop = build_end_to_end_loop(
        collector=_demo_collector(labels),
        workdir=tmp_path,
        base_model="stub-base",
        min_rows=2,
    )

    # The Redis client falls back to a real file-backed layer here (no live
    # redis package in this environment) but honors the same layout.
    assert isinstance(loop.redis_client, RedisClientApi)

    state = loop.run(units)

    # Real graph ran in the fixed order with no failures and trained an adapter
    # per Unit (Req 13.1, Property 17/18).
    assert tuple(state["executed_nodes"]) == NODE_ORDER
    assert state["failures"] == []
    assert set(state["adapters"]) == set(labels)

    # Real eval artifact emitted with a well-formed Confusion_Matrix (Req 5/15).
    eval_path = tmp_path / "eval_results.json"
    assert eval_path.exists()
    results = read_eval_results(eval_path)
    assert isinstance(results, EvalResults)
    n = len(results.confusion_matrix.labels)
    assert n == len(labels)
    assert sum(sum(row) for row in results.confusion_matrix.matrix) == float(n)

    # Each produced Adapter_File was stored through the Redis_Client_API and is
    # retrievable by the adapter_id the graph recorded (Req 13.3, 21.1).
    for label in labels:
        adapter_path = state["adapters"][label]
        adapter_id = loop.redis_client.route(label)
        stored_meta = loop.redis_client.fetch_meta(adapter_id)
        assert stored_meta["unit_label"] == label
        # The Redis blob is byte-identical to the on-disk Adapter_File.
        from pathlib import Path

        assert loop.redis_client.fetch_blob(adapter_id) == Path(
            adapter_path
        ).read_bytes()

    # Base_Model is loaded exactly once across eval + serving (Req 7.1).
    assert loop.engine.base_model_load_count == 1

    # Serve each Unit by adapter_id retrieved THROUGH the Redis_Client_API and
    # generated THROUGH the Inference_API engine (Req 21.2).
    served_ids = set()
    for label in labels:
        result = loop.serve_unit(label, prompt=f"As {label}, summarize today")
        served_ids.add(result.adapter_id)
        assert result.adapter_id == loop.redis_client.route(label)
        assert isinstance(result.text, str) and result.text
        # The Redis round-trip reproduced the Adapter_File the engine serves.
        served_blob = (loop.serve_dir / f"adapter_{result.adapter_id}.safetensors")
        assert served_blob.exists()

    # Distinct Units routed to distinct adapters.
    assert len(served_ids) == len(labels)
    # Single base load held after serving too (Req 7.1).
    assert loop.engine.base_model_load_count == 1


def test_create_redis_client_falls_back_without_live_redis(tmp_path):
    # No `redis` package / server in this environment -> file-backed fallback,
    # honoring the same key layout/interface (documented limitation).
    client = create_redis_client(file_path=tmp_path / "fallback.json")
    assert isinstance(client, RedisClientApi)
    client.store_adapter(
        {
            "adapter_id": "x1",
            "base_model": "stub-base",
            "unit_type": "category",
            "unit_label": "cooking",
            "train_rows": 1,
            "trained_at": "2024-01-01T00:00:00Z",
            "day_index": 0,
            "size_bytes": 1,
        },
        b"hello",
    )
    assert client.fetch_blob("x1") == b"hello"
    assert (tmp_path / "fallback.json").exists()
