"""Unit tests for the Inference_API schema contract (Requirement 2).

These are example/edge-case unit tests covering a valid request for each model,
the nullable-with-default ``adapter_id`` behaviour (Req 2.5), missing-required-
field rejection naming the field, and wrong-type rejection (supports Req 8.4).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from weaveself.contracts import (
    GenerateRequest,
    GenerateResponse,
    ScoreRequest,
    ScoreResponse,
    AdaptersResponse,
    TrainRequest,
    TrainResponse,
)


# ---------------------------------------------------------------------------
# Valid request/response for each model (Req 2.1–2.4)
# ---------------------------------------------------------------------------


def test_generate_request_valid():
    req = GenerateRequest(prompt="hello", adapter_id="abc123", max_new_tokens=64)
    assert req.prompt == "hello"
    assert req.adapter_id == "abc123"
    assert req.max_new_tokens == 64


def test_generate_response_valid():
    res = GenerateResponse(text="hi there", tokens=3, latency_ms=120)
    assert res.text == "hi there"
    assert res.tokens == 3
    assert res.latency_ms == 120


def test_score_request_valid():
    req = ScoreRequest(prompt="hello", target="world", adapter_id="abc123")
    assert req.prompt == "hello"
    assert req.target == "world"
    assert req.adapter_id == "abc123"


def test_score_response_valid():
    res = ScoreResponse(perplexity=12.5, nll=2.5)
    assert res.perplexity == 12.5
    assert res.nll == 2.5


def test_adapters_response_is_list_of_strings():
    res = AdaptersResponse(["abc123", "def456"])
    assert res.root == ["abc123", "def456"]
    assert AdaptersResponse([]).root == []


def test_train_request_valid():
    req = TrainRequest(
        dataset_path="/data/train.jsonl", unit_label="alice", unit_type="user"
    )
    assert req.dataset_path == "/data/train.jsonl"
    assert req.unit_label == "alice"
    assert req.unit_type == "user"


def test_train_response_valid():
    res = TrainResponse(adapter_path="/adapters/adapter_abc123.safetensors")
    assert res.adapter_path == "/adapters/adapter_abc123.safetensors"


# ---------------------------------------------------------------------------
# adapter_id is nullable with default None and routes to Base_Model (Req 2.5)
# ---------------------------------------------------------------------------


def test_generate_adapter_id_defaults_to_none():
    req = GenerateRequest(prompt="hello", max_new_tokens=32)
    assert req.adapter_id is None


def test_generate_adapter_id_accepts_explicit_null():
    req = GenerateRequest(prompt="hello", adapter_id=None, max_new_tokens=32)
    assert req.adapter_id is None


def test_score_adapter_id_defaults_to_none():
    req = ScoreRequest(prompt="hello", target="world")
    assert req.adapter_id is None


def test_score_adapter_id_accepts_explicit_null():
    req = ScoreRequest(prompt="hello", target="world", adapter_id=None)
    assert req.adapter_id is None


# ---------------------------------------------------------------------------
# Missing required field raises ValidationError naming the field (-> Req 8.4)
# ---------------------------------------------------------------------------


def test_generate_request_missing_prompt_names_field():
    with pytest.raises(ValidationError) as exc:
        GenerateRequest(max_new_tokens=32)
    assert "prompt" in str(exc.value)


def test_generate_request_missing_max_new_tokens_names_field():
    with pytest.raises(ValidationError) as exc:
        GenerateRequest(prompt="hello")
    assert "max_new_tokens" in str(exc.value)


def test_score_request_missing_target_names_field():
    with pytest.raises(ValidationError) as exc:
        ScoreRequest(prompt="hello")
    assert "target" in str(exc.value)


def test_train_request_missing_unit_type_names_field():
    with pytest.raises(ValidationError) as exc:
        TrainRequest(dataset_path="/data/train.jsonl", unit_label="alice")
    assert "unit_type" in str(exc.value)


# ---------------------------------------------------------------------------
# Wrong-type field is rejected (-> Req 8.4)
# ---------------------------------------------------------------------------


def test_generate_request_wrong_type_rejected():
    with pytest.raises(ValidationError) as exc:
        GenerateRequest(prompt="hello", max_new_tokens="not-an-int")
    assert "max_new_tokens" in str(exc.value)


def test_train_request_invalid_unit_type_literal_rejected():
    with pytest.raises(ValidationError) as exc:
        TrainRequest(
            dataset_path="/data/train.jsonl", unit_label="alice", unit_type="team"
        )
    assert "unit_type" in str(exc.value)


# ---------------------------------------------------------------------------
# extra="forbid": unknown fields are rejected
# ---------------------------------------------------------------------------


def test_generate_request_rejects_unknown_field():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hello", max_new_tokens=32, unexpected="x")
