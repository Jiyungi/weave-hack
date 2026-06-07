"""Unit tests for the Adapter_File contract (Requirement 1).

These are example/edge-case unit tests. The universal round-trip property test
(Property 1) is implemented separately in task 1.2.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from weaveself.contracts import (
    AdapterMetadata,
    MissingFieldError,
    ADAPTER_METADATA_FIELDS,
    adapter_blob_filename,
    adapter_meta_filename,
    read_adapter_file,
    validate_metadata,
    write_adapter_file,
)


def test_round_trip_preserves_metadata_and_gates(tmp_path, sample_metadata, sample_gates):
    blob_path, meta_path = write_adapter_file(tmp_path, sample_metadata, sample_gates)

    assert blob_path.name == adapter_blob_filename(sample_metadata["adapter_id"])
    assert meta_path.name == adapter_meta_filename(sample_metadata["adapter_id"])
    assert blob_path.exists() and meta_path.exists()

    meta, gates = read_adapter_file(tmp_path, sample_metadata["adapter_id"])

    # All eight fields preserved with values + types.
    assert isinstance(meta, AdapterMetadata)
    for field in ADAPTER_METADATA_FIELDS:
        if field == "size_bytes":
            continue  # set to the real serialized size on write
        assert getattr(meta, field) == sample_metadata[field]

    # size_bytes reflects the actual serialized blob size.
    assert meta.size_bytes == blob_path.stat().st_size > 0

    # Gate tensors identical to those written.
    assert set(gates) == set(sample_gates)
    for name, original in sample_gates.items():
        np.testing.assert_array_equal(gates[name], original)
        assert gates[name].dtype == original.dtype


@pytest.mark.parametrize("missing", list(ADAPTER_METADATA_FIELDS))
def test_validate_metadata_reports_missing_field(sample_metadata, missing):
    incomplete = {k: v for k, v in sample_metadata.items() if k != missing}
    with pytest.raises(MissingFieldError) as exc:
        validate_metadata(incomplete)
    assert exc.value.field_name == missing
    assert missing in str(exc.value)


def test_none_value_treated_as_missing(sample_metadata):
    sample_metadata["unit_label"] = None
    with pytest.raises(MissingFieldError) as exc:
        validate_metadata(sample_metadata)
    assert exc.value.field_name == "unit_label"


def test_read_rejects_sidecar_missing_field(tmp_path, sample_metadata, sample_gates):
    write_adapter_file(tmp_path, sample_metadata, sample_gates)
    meta_path = tmp_path / adapter_meta_filename(sample_metadata["adapter_id"])
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    del data["base_model"]
    meta_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MissingFieldError) as exc:
        read_adapter_file(tmp_path, sample_metadata["adapter_id"])
    assert exc.value.field_name == "base_model"


def test_invalid_unit_type_raises_validation_error(sample_metadata):
    sample_metadata["unit_type"] = "team"  # not in {"category", "user"}
    with pytest.raises(Exception) as exc:
        validate_metadata(sample_metadata)
    # Not a MissingFieldError — it's a value/validation error.
    assert not isinstance(exc.value, MissingFieldError)
