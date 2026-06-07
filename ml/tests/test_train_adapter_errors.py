"""Tests for ``train_adapter`` input error handling (Track A / Req 9.4, 9.5).

These cover the typed training-input errors from design.md "Training errors
(Track A)":

* An unreadable / non-existent / malformed ``dataset_path`` raises
  ``DatasetNotReadable`` naming the path and writes NO Adapter_File (Req 9.4).
* A readable dataset with zero rows raises ``InsufficientTrainingData`` and
  writes NO Adapter_File (Req 9.5).
* The happy path still produces an Adapter_File (smoke).

numpy only — no torch / model download required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from weaveself.contracts import write_training_pairs
from weaveself.training import (
    DatasetNotReadable,
    InsufficientTrainingData,
    train_adapter,
)


def _adapter_files(out_dir: Path) -> list[Path]:
    """Return any adapter artifacts written under ``out_dir`` (if it exists)."""
    if not out_dir.exists():
        return []
    return [
        p
        for p in out_dir.iterdir()
        if p.name.startswith("adapter_")
        and p.suffix in {".safetensors", ".json"}
    ]


def test_nonexistent_dataset_path_raises_named_error_and_writes_nothing(tmp_path):
    """Req 9.4: a missing path raises DatasetNotReadable naming it, no output."""
    missing = tmp_path / "does_not_exist.jsonl"
    out_dir = tmp_path / "adapters"

    with pytest.raises(DatasetNotReadable) as excinfo:
        train_adapter(
            str(missing),
            unit_label="alice",
            unit_type="user",
            out_dir=str(out_dir),
        )

    # The error message must name the offending path (Req 9.4).
    assert str(missing) in str(excinfo.value)
    assert excinfo.value.dataset_path == str(missing)

    # No Adapter_File was written.
    assert _adapter_files(out_dir) == []
    assert not list(tmp_path.glob("**/adapter_*.safetensors"))


def test_empty_dataset_raises_insufficient_and_writes_no_adapter(tmp_path):
    """Req 9.5: a zero-row dataset raises InsufficientTrainingData, no output."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")  # readable, zero training pairs
    out_dir = tmp_path / "adapters"

    with pytest.raises(InsufficientTrainingData) as excinfo:
        train_adapter(
            str(empty),
            unit_label="alice",
            unit_type="user",
            out_dir=str(out_dir),
        )

    assert excinfo.value.dataset_path == str(empty)

    # No Adapter_File was written.
    assert _adapter_files(out_dir) == []
    assert not list(tmp_path.glob("**/adapter_*.safetensors"))


def test_malformed_dataset_raises_named_not_readable_error(tmp_path):
    """Req 9.4: a garbage/unparseable file raises DatasetNotReadable naming it."""
    garbage = tmp_path / "garbage.jsonl"
    garbage.write_text("this is not json at all {{{\n", encoding="utf-8")
    out_dir = tmp_path / "adapters"

    with pytest.raises(DatasetNotReadable) as excinfo:
        train_adapter(
            str(garbage),
            unit_label="alice",
            unit_type="user",
            out_dir=str(out_dir),
        )

    assert str(garbage) in str(excinfo.value)
    assert _adapter_files(out_dir) == []


def test_happy_path_still_produces_adapter_file(tmp_path):
    """Smoke: a valid dataset still produces an Adapter_File (.safetensors)."""
    dataset = tmp_path / "train.jsonl"
    write_training_pairs(
        dataset,
        [
            {
                "prompt": "What's a good weekend project?",
                "completion": "Try building a small CLI tool in Rust.",
                "unit_label": "alice",
            },
            {
                "prompt": "Recommend a book.",
                "completion": "Give 'The Pragmatic Programmer' a read.",
                "unit_label": "alice",
            },
        ],
    )
    out_dir = tmp_path / "adapters"

    adapter_path = train_adapter(
        str(dataset),
        unit_label="alice",
        unit_type="user",
        out_dir=str(out_dir),
    )

    path = Path(adapter_path)
    assert path.exists()
    assert path.suffix == ".safetensors"
    assert path.name.startswith("adapter_")
    # Sidecar metadata exists alongside the blob.
    assert path.with_suffix(".json").exists()
