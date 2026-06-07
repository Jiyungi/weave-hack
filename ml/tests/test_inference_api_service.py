"""Example/integration tests for the FastAPI Inference_API service (task 3.7).

These exercise the real FastAPI app built by
:func:`weaveself.serving.create_app` through FastAPI's in-process
``TestClient`` (no live server, no network). The engine uses the dependency-free
:class:`StubBackend`, and one Adapter_File is written into a temp adapters dir so
``/adapters`` and adapter-routed ``/generate`` have a loadable adapter.

Covered:

* ``POST /generate`` returns 200 with non-empty text, integer ``tokens`` and a
  present, non-negative ``latency_ms``; null ``adapter_id`` works (Req 8.1, 2.5).
* ``POST /score`` returns non-negative ``perplexity`` and ``nll`` (Req 8.2).
* ``GET /adapters`` returns the loadable set (Req 8.3).
* ``POST /train`` returns an ``adapter_path`` equal to a direct ``train_adapter``
  call with the same inputs (Req 9.3).
* A malformed ``/generate`` body returns 422 naming the offending field (Req 8.4).
* An unknown ``adapter_id`` on ``/generate`` returns an error naming the
  ``adapter_id`` (Req 7.5 surfaced through the API).
"""

from __future__ import annotations

import numpy as np
import pytest

# fastapi powers create_app and TestClient; skip the whole module gracefully if
# it is not installed in this environment (the module itself still imports).
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from weaveself.contracts import write_adapter_file  # noqa: E402
from weaveself.serving import ServingEngine, StubBackend, create_app  # noqa: E402
from weaveself.training import train_adapter  # noqa: E402


_ADAPTER_ID = "feedface1234"


def _write_loadable_adapter(adapters_dir) -> str:
    """Write one valid Adapter_File pair into ``adapters_dir`` and return its id."""
    rng = np.random.default_rng(0)
    gates = {
        "model.layers.0.mlp.gate": rng.standard_normal(64).astype(np.float32),
        "model.layers.1.mlp.gate": rng.standard_normal(64).astype(np.float32),
    }
    metadata = {
        "adapter_id": _ADAPTER_ID,
        "base_model": "stub-base",
        "unit_type": "user",
        "unit_label": "alice",
        "train_rows": 10,
        "trained_at": "2024-01-01T00:00:00+00:00",
        "day_index": 0,
        "size_bytes": 0,
    }
    write_adapter_file(adapters_dir, metadata, gates)
    return _ADAPTER_ID


@pytest.fixture
def client(tmp_path) -> TestClient:
    """A TestClient over an app whose StubBackend engine has one adapter."""
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    _write_loadable_adapter(adapters_dir)
    engine = ServingEngine(
        "stub-base", backend=StubBackend(), adapters_dir=str(adapters_dir)
    )
    app = create_app(engine=engine)
    return TestClient(app)


# ---------------------------------------------------------------------------
# /generate (Req 8.1, 2.5)
# ---------------------------------------------------------------------------


def test_generate_null_adapter_returns_populated_response(client: TestClient):
    resp = client.post(
        "/generate",
        json={"prompt": "hello world", "adapter_id": None, "max_new_tokens": 16},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["text"], str) and body["text"] != ""
    assert isinstance(body["tokens"], int)
    assert "latency_ms" in body
    assert isinstance(body["latency_ms"], int) and body["latency_ms"] >= 0


def test_generate_omitted_adapter_id_defaults_to_base(client: TestClient):
    # adapter_id omitted -> defaults to None -> pure base (Req 2.5).
    resp = client.post("/generate", json={"prompt": "hi", "max_new_tokens": 8})
    assert resp.status_code == 200
    assert resp.json()["text"] != ""


def test_generate_with_loadable_adapter(client: TestClient):
    resp = client.post(
        "/generate",
        json={"prompt": "hello", "adapter_id": _ADAPTER_ID, "max_new_tokens": 16},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] != ""


# ---------------------------------------------------------------------------
# /score (Req 8.2)
# ---------------------------------------------------------------------------


def test_score_returns_non_negative_perplexity_and_nll(client: TestClient):
    resp = client.post(
        "/score",
        json={"prompt": "hello", "target": "world", "adapter_id": None},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["perplexity"] >= 0
    assert body["nll"] >= 0


# ---------------------------------------------------------------------------
# /adapters (Req 8.3)
# ---------------------------------------------------------------------------


def test_adapters_returns_loadable_set(client: TestClient):
    resp = client.get("/adapters")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert _ADAPTER_ID in body


# ---------------------------------------------------------------------------
# /train equivalence to the direct call under a fixed seed (Req 9.3)
# ---------------------------------------------------------------------------


def test_train_returns_same_adapter_path_as_direct_call(tmp_path):
    # Build a tiny training-pair dataset.
    dataset = tmp_path / "train.jsonl"
    rows = [
        {"prompt": "p1", "completion": "c1", "unit_label": "alice"},
        {"prompt": "p2", "completion": "c2", "unit_label": "alice"},
        {"prompt": "p3", "completion": "c3", "unit_label": "alice"},
    ]
    dataset.write_text(
        "\n".join(__import__("json").dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )

    # Direct call (deterministic for fixed inputs).
    direct_path = train_adapter(str(dataset), "alice", "user")

    # HTTP call with the same inputs MUST return the same adapter_path (Req 9.3).
    app = create_app(
        engine=ServingEngine("stub-base", backend=StubBackend())
    )
    http_client = TestClient(app)
    resp = http_client.post(
        "/train",
        json={"dataset_path": str(dataset), "unit_label": "alice", "unit_type": "user"},
    )
    assert resp.status_code == 200
    assert resp.json()["adapter_path"] == direct_path


# ---------------------------------------------------------------------------
# Malformed body -> 422 naming the offending field (Req 8.4)
# ---------------------------------------------------------------------------


def test_generate_missing_prompt_returns_422_naming_field(client: TestClient):
    resp = client.post("/generate", json={"max_new_tokens": 16})
    assert resp.status_code == 422
    assert "prompt" in resp.text


def test_generate_wrong_type_max_new_tokens_returns_422_naming_field(
    client: TestClient,
):
    resp = client.post(
        "/generate",
        json={"prompt": "hello", "max_new_tokens": "not-an-int"},
    )
    assert resp.status_code == 422
    assert "max_new_tokens" in resp.text


# ---------------------------------------------------------------------------
# Unknown adapter_id -> error naming the adapter_id (Req 7.5 via the API)
# ---------------------------------------------------------------------------


def test_generate_unknown_adapter_id_errors_and_names_it(client: TestClient):
    missing = "no-such-adapter-xyz"
    resp = client.post(
        "/generate",
        json={"prompt": "hello", "adapter_id": missing, "max_new_tokens": 16},
    )
    assert resp.status_code >= 400
    assert missing in resp.text
