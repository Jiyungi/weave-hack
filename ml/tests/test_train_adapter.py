"""Unit tests for the NKT-Mirror ``train_adapter`` training loop (Task 3.1).

Covers the happy-path Adapter_File production contract from Requirements 6.2,
6.3, 6.4, 9.1, and 9.2:

* metadata ``train_rows`` equals the consumed row count and ``unit_label`` /
  ``unit_type`` equal the supplied arguments (Req 9.1, 9.2);
* the serialized safetensors is <= 200,000 bytes and metadata ``size_bytes``
  equals the real on-disk size (Req 6.4);
* the gate tensors round-trip back through ``read_adapter_file`` (Req 6.3 — only
  gate tensors are produced);
* the total number of gate parameters is ~5,000.

These tests use only numpy + the contracts (no torch / GPU / model download).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from weaveself.contracts import read_adapter_file, write_training_pairs
from weaveself.training import TOTAL_GATE_PARAMS, train_adapter


def _write_fixture_dataset(path: Path, unit_label: str, n: int) -> int:
    """Write a small JSONL fixture of Training_Pairs and return the row count."""
    rows = [
        {
            "prompt": f"Question {i} for {unit_label}?",
            "completion": f"Answer {i} in {unit_label}'s preferred style.",
            "unit_label": unit_label,
        }
        for i in range(n)
    ]
    write_training_pairs(path, rows)
    return len(rows)


def _adapter_id_from_path(adapter_path: str) -> str:
    # adapter_<id>.safetensors -> <id>
    return Path(adapter_path).name[len("adapter_") : -len(".safetensors")]


def test_train_adapter_metadata_reflects_inputs(tmp_path):
    dataset = tmp_path / "train.jsonl"
    n = _write_fixture_dataset(dataset, unit_label="alice", n=7)

    out_dir = tmp_path / "adapters"
    adapter_path = train_adapter(
        str(dataset),
        unit_label="alice",
        unit_type="user",
        out_dir=str(out_dir),
        day_index=3,
    )

    adapter_id = _adapter_id_from_path(adapter_path)
    meta, _gates = read_adapter_file(out_dir, adapter_id)

    # Req 9.1: train_rows equals consumed rows; unit_label equals argument.
    assert meta.train_rows == n
    assert meta.unit_label == "alice"
    # Req 9.2 / 9.1: unit_type equals the supplied argument.
    assert meta.unit_type == "user"
    assert meta.day_index == 3


def test_train_adapter_size_is_bounded_and_accurate(tmp_path):
    dataset = tmp_path / "train.jsonl"
    _write_fixture_dataset(dataset, unit_label="bravo", n=12)

    out_dir = tmp_path / "adapters"
    adapter_path = train_adapter(
        str(dataset),
        unit_label="bravo",
        unit_type="category",
        out_dir=str(out_dir),
    )

    blob = Path(adapter_path)
    actual_size = blob.stat().st_size

    # Req 6.4: serialized size bounded at 200,000 bytes.
    assert actual_size <= 200_000

    adapter_id = _adapter_id_from_path(adapter_path)
    meta, _gates = read_adapter_file(out_dir, adapter_id)

    # Req 6.4: metadata size_bytes equals the real serialized file size.
    assert meta.size_bytes == actual_size > 0


def test_train_adapter_gates_round_trip_and_param_count(tmp_path):
    dataset = tmp_path / "train.jsonl"
    _write_fixture_dataset(dataset, unit_label="carol", n=5)

    out_dir = tmp_path / "adapters"
    adapter_path = train_adapter(
        str(dataset),
        unit_label="carol",
        unit_type="user",
        out_dir=str(out_dir),
    )

    adapter_id = _adapter_id_from_path(adapter_path)
    _meta, gates = read_adapter_file(out_dir, adapter_id)

    # Req 6.3: only gate tensors are produced (no base-model weight tensors).
    assert len(gates) > 0
    total_params = sum(int(arr.size) for arr in gates.values())

    # ~5,000 activation-gating parameters.
    assert total_params == TOTAL_GATE_PARAMS
    assert 4_000 <= total_params <= 6_000

    # Gates are finite multipliers centered near 1.0 (steer, not overwrite).
    for arr in gates.values():
        assert np.all(np.isfinite(arr))
        assert np.all(np.abs(arr - 1.0) <= 0.5)


def test_train_adapter_is_deterministic_for_fixed_dataset(tmp_path):
    dataset = tmp_path / "train.jsonl"
    _write_fixture_dataset(dataset, unit_label="dave", n=6)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    path_a = train_adapter(str(dataset), "dave", "user", out_dir=str(out_a))
    path_b = train_adapter(str(dataset), "dave", "user", out_dir=str(out_b))

    # Deterministic adapter_id for a fixed dataset + arguments.
    assert _adapter_id_from_path(path_a) == _adapter_id_from_path(path_b)

    _meta_a, gates_a = read_adapter_file(out_a, _adapter_id_from_path(path_a))
    _meta_b, gates_b = read_adapter_file(out_b, _adapter_id_from_path(path_b))

    assert set(gates_a) == set(gates_b)
    for name in gates_a:
        np.testing.assert_array_equal(gates_a[name], gates_b[name])
