"""Unit tests for the Eval_Results contract (Requirement 5).

These are example/edge-case unit tests covering a valid full round-trip, the
square-matrix invariant (accept square, reject non-square), and missing
top-level field reporting. The universal eval-artifact property test
(Property 24) is implemented separately in task 7.7.
"""

from __future__ import annotations

import json

import pytest

from weaveself.contracts import (
    ConfusionMatrix,
    EvalResults,
    EVAL_RESULTS_FIELDS,
    InvalidConfusionMatrixError,
    MissingFieldError,
    read_eval_results,
    validate_eval_results,
    write_eval_results,
)


@pytest.fixture
def sample_eval_results() -> dict:
    """A complete, valid eval_results.json artifact with two labels."""
    return {
        "perplexity": {"base": 12.5, "adapter": 9.1, "context_memory": 10.0},
        "confusion_matrix": {
            "labels": ["alice", "bob"],
            "matrix": [[3.0, 1.0], [0.0, 4.0]],
        },
        "size_bytes": {"nktmirror": 102400, "lora": 4194304},
        "examples": [
            {
                "prompt": "What's a good weekend project?",
                "base": "Build a website.",
                "adapter": "Try a small CLI tool in Rust.",
                "reference": "A Rust CLI tool.",
            }
        ],
    }


def test_valid_full_round_trip(tmp_path, sample_eval_results):
    out_path = tmp_path / "eval_results.json"
    written = write_eval_results(out_path, sample_eval_results)
    assert written == out_path
    assert out_path.exists()

    results = read_eval_results(out_path)

    assert isinstance(results, EvalResults)
    assert results.perplexity.base == 12.5
    assert results.perplexity.adapter == 9.1
    assert results.perplexity.context_memory == 10.0
    assert results.confusion_matrix.labels == ["alice", "bob"]
    assert results.confusion_matrix.matrix == [[3.0, 1.0], [0.0, 4.0]]
    assert results.size_bytes.nktmirror == 102400
    assert results.size_bytes.lora == 4194304
    assert len(results.examples) == 1
    assert results.examples[0].prompt == "What's a good weekend project?"
    assert results.examples[0].reference == "A Rust CLI tool."


def test_square_confusion_matrix_accepted():
    cm = ConfusionMatrix(
        labels=["a", "b", "c"],
        matrix=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
    )
    assert len(cm.matrix) == len(cm.labels) == 3


def test_non_square_matrix_wrong_row_count_rejected():
    with pytest.raises(InvalidConfusionMatrixError) as exc:
        ConfusionMatrix(labels=["a", "b"], matrix=[[1.0, 0.0]])
    # Mentions the expected row count.
    assert "2" in str(exc.value)


def test_non_square_matrix_wrong_column_width_rejected():
    with pytest.raises(InvalidConfusionMatrixError) as exc:
        ConfusionMatrix(labels=["a", "b"], matrix=[[1.0, 0.0], [0.0]])
    assert "column" in str(exc.value)


@pytest.mark.parametrize("missing", list(EVAL_RESULTS_FIELDS))
def test_validate_reports_missing_top_level_field(sample_eval_results, missing):
    incomplete = {k: v for k, v in sample_eval_results.items() if k != missing}
    with pytest.raises(MissingFieldError) as exc:
        validate_eval_results(incomplete)
    assert exc.value.field_name == missing
    assert missing in str(exc.value)


def test_read_rejects_file_missing_top_level_field(tmp_path, sample_eval_results):
    out_path = tmp_path / "eval_results.json"
    write_eval_results(out_path, sample_eval_results)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    del data["size_bytes"]
    out_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MissingFieldError) as exc:
        read_eval_results(out_path)
    assert exc.value.field_name == "size_bytes"
