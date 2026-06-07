"""Adapter_File format contract (Track 0 / Requirement 1, owned by Track A).

An Adapter_File is a pair of files:

* ``adapter_<id>.safetensors`` — the NKT-Mirror gate tensors.
* ``adapter_<id>.json``        — the eight-field metadata sidecar.

This module provides:

* :class:`AdapterMetadata` — the eight-field Pydantic metadata model (Req 1.2).
* :func:`validate_metadata` — raises :class:`MissingFieldError` naming any
  missing required field (Req 1.4).
* :func:`write_adapter_file` / :func:`read_adapter_file` — the writer/reader
  pair whose round-trip preserves all metadata fields and gate tensors
  (Req 1.1, 1.3 / Property 1).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict
from safetensors.numpy import load_file as _st_load_file
from safetensors.numpy import save_file as _st_save_file

from weaveself.contracts.errors import MissingFieldError

UnitType = Literal["category", "user"]

# The eight required metadata fields, in contract order (Requirement 1.2).
ADAPTER_METADATA_FIELDS: tuple[str, ...] = (
    "adapter_id",
    "base_model",
    "unit_type",
    "unit_label",
    "train_rows",
    "trained_at",
    "day_index",
    "size_bytes",
)


class AdapterMetadata(BaseModel):
    """The eight-field Adapter_File metadata model. All fields are required."""

    model_config = ConfigDict(extra="forbid")

    adapter_id: str
    base_model: str
    unit_type: UnitType
    unit_label: str
    train_rows: int
    trained_at: str
    day_index: int
    size_bytes: int


def validate_metadata(data: Mapping[str, object] | AdapterMetadata) -> AdapterMetadata:
    """Validate adapter metadata and return a typed :class:`AdapterMetadata`.

    If any required field is missing, raises :class:`MissingFieldError` whose
    message names the missing field (Requirement 1.4). Other validation issues
    (wrong type, bad ``unit_type`` literal) surface as Pydantic ``ValidationError``.
    """
    if isinstance(data, AdapterMetadata):
        return data

    if not isinstance(data, Mapping):
        raise TypeError(
            f"adapter metadata must be a mapping or AdapterMetadata, got {type(data).__name__}"
        )

    # Explicit missing-field detection so we can raise a typed, field-named
    # error. A field that is absent OR present but null counts as missing.
    for field in ADAPTER_METADATA_FIELDS:
        if field not in data or data[field] is None:
            raise MissingFieldError(field, context="Adapter_File metadata")

    return AdapterMetadata(**dict(data))


def adapter_blob_filename(adapter_id: str) -> str:
    """Return the safetensors filename for an adapter id."""
    return f"adapter_{adapter_id}.safetensors"


def adapter_meta_filename(adapter_id: str) -> str:
    """Return the JSON sidecar filename for an adapter id."""
    return f"adapter_{adapter_id}.json"


def _normalize_gates(gate_tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for name, tensor in gate_tensors.items():
        arr = np.asarray(tensor)
        # safetensors requires contiguous arrays.
        normalized[name] = np.ascontiguousarray(arr)
    return normalized


def write_adapter_file(
    directory: str | os.PathLike[str],
    metadata: Mapping[str, object] | AdapterMetadata,
    gate_tensors: Mapping[str, np.ndarray],
) -> tuple[Path, Path]:
    """Write the Adapter_File pair and return ``(blob_path, meta_path)``.

    The gate tensors are serialized with ``safetensors``; metadata is written as
    JSON. ``size_bytes`` in the written metadata is set to the actual serialized
    size of the safetensors blob so the sidecar reflects reality (Req 1.1, 1.2).
    """
    meta = validate_metadata(metadata)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    blob_path = directory / adapter_blob_filename(meta.adapter_id)
    meta_path = directory / adapter_meta_filename(meta.adapter_id)

    gates = _normalize_gates(gate_tensors)
    _st_save_file(gates, str(blob_path))

    # Reflect the real serialized size in the metadata sidecar.
    meta = meta.model_copy(update={"size_bytes": blob_path.stat().st_size})

    meta_path.write_text(json.dumps(meta.model_dump(), indent=2), encoding="utf-8")
    return blob_path, meta_path


def read_adapter_file(
    directory: str | os.PathLike[str],
    adapter_id: str,
) -> tuple[AdapterMetadata, dict[str, np.ndarray]]:
    """Read and validate an Adapter_File pair.

    Returns the validated :class:`AdapterMetadata` and the gate tensors as a
    dict of numpy arrays identical to those written (Req 1.3 / Property 1).
    Raises :class:`MissingFieldError` if the sidecar omits a required field.
    """
    directory = Path(directory)
    meta_path = directory / adapter_meta_filename(adapter_id)
    blob_path = directory / adapter_blob_filename(adapter_id)

    if not meta_path.exists():
        raise FileNotFoundError(f"adapter metadata not found: {meta_path}")
    if not blob_path.exists():
        raise FileNotFoundError(f"adapter blob not found: {blob_path}")

    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = validate_metadata(raw_meta)

    gate_tensors = _st_load_file(str(blob_path))
    return meta, gate_tensors
