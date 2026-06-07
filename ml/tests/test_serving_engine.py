"""Unit tests for the resident Serving_Engine (task 2.1).

Covers Base_Model residency (Req 7.1) and the per-adapter gate cache (Req 6.1)
using the dependency-free :class:`StubBackend`, so these tests never hit the
network or require a GPU. The differential-serving and scoring property tests
(2.4-2.7) are implemented separately.
"""

from __future__ import annotations

import math

import pytest

from weaveself.contracts import write_adapter_file
from weaveself.serving import AdapterNotLoadable, ServingEngine, StubBackend


def _write_adapter(directory, sample_metadata, sample_gates, adapter_id):
    meta = dict(sample_metadata, adapter_id=adapter_id)
    write_adapter_file(directory, meta, sample_gates)
    return adapter_id


def test_base_model_loaded_exactly_once_on_construction():
    backend = StubBackend()
    assert backend.load_count == 0
    engine = ServingEngine("Qwen2.5-1.5B-Instruct", backend=backend)
    assert backend.load_count == 1
    assert engine.base_model_load_count == 1
    assert backend.is_loaded
    assert backend.base_model_id == "Qwen2.5-1.5B-Instruct"


def test_base_model_loaded_once_across_many_requests(tmp_path, sample_metadata, sample_gates):
    backend = StubBackend()
    _write_adapter(tmp_path, sample_metadata, sample_gates, "a1")
    _write_adapter(tmp_path, sample_metadata, sample_gates, "a2")
    engine = ServingEngine("base-x", backend=backend, adapters_dir=tmp_path)

    # Many adapter loads + generate/score requests across multiple adapters.
    for _ in range(5):
        engine.load_adapter("a1")
        engine.load_adapter("a2")
        engine.generate("hello", adapter_id="a1", max_new_tokens=8)
        engine.generate("hello", adapter_id=None, max_new_tokens=8)
        engine.score("hello", "world", adapter_id="a2")

    # Base_Model still loaded exactly once per process (Req 7.1).
    assert backend.load_count == 1
    assert engine.base_model_load_count == 1


def test_load_adapter_caches_and_returns_gate_tensors(tmp_path, sample_metadata, sample_gates):
    _write_adapter(tmp_path, sample_metadata, sample_gates, "cached")
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)

    first = engine.load_adapter("cached")
    assert set(first) == set(sample_gates)

    # A repeat load returns the exact same cached object (no re-read).
    second = engine.load_adapter("cached")
    assert second is first


def test_load_adapter_returns_equivalent_tensors(tmp_path, sample_metadata, sample_gates):
    import numpy as np

    _write_adapter(tmp_path, sample_metadata, sample_gates, "g1")
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)

    gates = engine.load_adapter("g1")
    for name, original in sample_gates.items():
        np.testing.assert_array_equal(gates[name], original)


def test_unknown_adapter_raises_named_error(tmp_path):
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)
    with pytest.raises(AdapterNotLoadable) as exc:
        engine.load_adapter("does-not-exist")
    assert exc.value.adapter_id == "does-not-exist"
    assert "does-not-exist" in str(exc.value)


def test_unknown_adapter_without_dir_raises_named_error():
    engine = ServingEngine("base-x", backend=StubBackend())
    with pytest.raises(AdapterNotLoadable) as exc:
        engine.load_adapter("nope")
    assert exc.value.adapter_id == "nope"
    assert "nope" in str(exc.value)


def test_list_adapters_reflects_loadable_set(tmp_path, sample_metadata, sample_gates):
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)
    assert engine.list_adapters() == []

    _write_adapter(tmp_path, sample_metadata, sample_gates, "alpha")
    _write_adapter(tmp_path, sample_metadata, sample_gates, "beta")

    assert engine.list_adapters() == ["alpha", "beta"]


def test_list_adapters_includes_cached_only_adapters(tmp_path, sample_metadata, sample_gates):
    _write_adapter(tmp_path, sample_metadata, sample_gates, "ondisk")
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)

    engine.load_adapter("ondisk")
    assert "ondisk" in engine.list_adapters()


def test_generate_with_adapter_differs_from_base(tmp_path, sample_metadata, sample_gates):
    _write_adapter(tmp_path, sample_metadata, sample_gates, "style")
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)

    base_out = engine.generate("hi there", adapter_id=None, max_new_tokens=16)
    adapter_out = engine.generate("hi there", adapter_id="style", max_new_tokens=16)
    assert base_out.text != adapter_out.text


def test_score_is_non_negative(tmp_path, sample_metadata, sample_gates):
    _write_adapter(tmp_path, sample_metadata, sample_gates, "s1")
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)

    result = engine.score("a prompt", "a target", adapter_id="s1")
    assert result.nll >= 0.0
    assert result.perplexity >= 0.0


# -- per-request gate application + clearing (task 2.2, Req 6.3/7.2/7.3/8.2) --


class _BoomBackend(StubBackend):
    """A StubBackend whose generate raises *while gates are applied*.

    Used to prove the per-request gate lifecycle clears applied gates even when
    the request raises mid-flight (Req 6.3, 7.2; protects Property 4).
    """

    def generate(self, prompt, gates, max_new_tokens):  # type: ignore[override]
        with self._apply_gates(gates):
            assert self.active_gate_signature is not None
            raise RuntimeError("boom during gated request")


def test_null_adapter_generate_equals_pure_base_and_clears_gates(
    tmp_path, sample_metadata, sample_gates
):
    backend = StubBackend()
    _write_adapter(tmp_path, sample_metadata, sample_gates, "style")
    engine = ServingEngine("base-x", backend=backend, adapters_dir=tmp_path)

    # A pure base reference computed directly on a fresh backend.
    reference = StubBackend()
    reference.load_base_model("base-x")
    pure = reference.generate("hi there", None, 16)

    via_engine = engine.generate("hi there", adapter_id=None, max_new_tokens=16)
    assert via_engine.text == pure.text
    assert via_engine.tokens == pure.tokens
    # No gates left applied after the request (Property 4).
    assert backend.active_gate_signature is None


def test_null_adapter_score_equals_pure_base_and_clears_gates(
    tmp_path, sample_metadata, sample_gates
):
    backend = StubBackend()
    _write_adapter(tmp_path, sample_metadata, sample_gates, "style")
    engine = ServingEngine("base-x", backend=backend, adapters_dir=tmp_path)

    reference = StubBackend()
    reference.load_base_model("base-x")
    pure = reference.score("a prompt", "the target text", None)

    via_engine = engine.score("a prompt", "the target text", adapter_id=None)
    assert via_engine.nll == pure.nll
    assert via_engine.perplexity == pure.perplexity
    assert backend.active_gate_signature is None


def test_adapter_request_leaves_no_gates_applied(tmp_path, sample_metadata, sample_gates):
    backend = StubBackend()
    _write_adapter(tmp_path, sample_metadata, sample_gates, "style")
    engine = ServingEngine("base-x", backend=backend, adapters_dir=tmp_path)

    engine.generate("hi", adapter_id="style", max_new_tokens=8)
    assert backend.active_gate_signature is None
    engine.score("hi", "there", adapter_id="style")
    assert backend.active_gate_signature is None


def test_gates_cleared_even_when_request_raises(sample_gates):
    backend = _BoomBackend()
    backend.load_base_model("base-x")

    with pytest.raises(RuntimeError):
        backend.generate("hi", sample_gates, max_new_tokens=8)

    # finally-clearing must have run despite the exception.
    assert backend.active_gate_signature is None


def test_score_perplexity_consistent_with_nll(tmp_path, sample_metadata, sample_gates):
    _write_adapter(tmp_path, sample_metadata, sample_gates, "s1")
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)

    target = "a multi word target string"
    for adapter_id in (None, "s1"):
        result = engine.score("a prompt", target, adapter_id=adapter_id)
        token_count = max(1, len(target.split()))
        assert result.nll >= 0.0
        assert result.perplexity >= 0.0
        # perplexity == exp(nll / token_count) exactly (Property 8).
        assert result.perplexity == pytest.approx(math.exp(result.nll / token_count))


def test_adapter_score_differs_from_base(tmp_path, sample_metadata, sample_gates):
    _write_adapter(tmp_path, sample_metadata, sample_gates, "s1")
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)

    base = engine.score("a prompt", "a target", adapter_id=None)
    adapter = engine.score("a prompt", "a target", adapter_id="s1")
    # Gate-sensitive scoring: the adapter changes the score relative to base.
    assert (adapter.nll, adapter.perplexity) != (base.nll, base.perplexity)


# -- adapter listing + unknown-adapter validation before mutation (task 2.3) --
# Requirements 2.3 (loadable set is the single source of truth) and 7.5 (an
# unknown adapter_id raises AdapterNotLoadable naming the id BEFORE any
# forward-pass mutation, so the resident model is never entered/mutated).


class _SpyBackend(StubBackend):
    """A StubBackend that records whether generate/score were ever entered.

    Used to prove that an unknown ``adapter_id`` is rejected by the engine
    BEFORE the backend's forward pass is reached (Req 7.5): the spy counters
    MUST remain zero and no gates are ever applied.
    """

    def __init__(self) -> None:
        super().__init__()
        self.generate_calls = 0
        self.score_calls = 0

    def generate(self, prompt, gates, max_new_tokens):  # type: ignore[override]
        self.generate_calls += 1
        return super().generate(prompt, gates, max_new_tokens)

    def score(self, prompt, target, gates):  # type: ignore[override]
        self.score_calls += 1
        return super().score(prompt, target, gates)


def test_generate_unknown_adapter_raises_before_backend_entered(tmp_path):
    backend = _SpyBackend()
    engine = ServingEngine("base-x", backend=backend, adapters_dir=tmp_path)

    with pytest.raises(AdapterNotLoadable) as exc:
        engine.generate("hello", adapter_id="ghost", max_new_tokens=8)

    # The error names the missing adapter (Req 7.5).
    assert exc.value.adapter_id == "ghost"
    assert "ghost" in str(exc.value)
    # The backend forward pass was never reached and no gates were applied,
    # so the resident model was never mutated.
    assert backend.generate_calls == 0
    assert backend.active_gate_signature is None


def test_score_unknown_adapter_raises_before_backend_entered(tmp_path):
    backend = _SpyBackend()
    engine = ServingEngine("base-x", backend=backend, adapters_dir=tmp_path)

    with pytest.raises(AdapterNotLoadable) as exc:
        engine.score("a prompt", "a target", adapter_id="ghost")

    assert exc.value.adapter_id == "ghost"
    assert "ghost" in str(exc.value)
    assert backend.score_calls == 0
    assert backend.active_gate_signature is None


def test_generate_unknown_adapter_without_dir_raises_before_backend_entered():
    backend = _SpyBackend()
    engine = ServingEngine("base-x", backend=backend)

    with pytest.raises(AdapterNotLoadable) as exc:
        engine.generate("hello", adapter_id="ghost", max_new_tokens=8)

    assert exc.value.adapter_id == "ghost"
    assert backend.generate_calls == 0
    assert backend.active_gate_signature is None


def test_list_adapters_returns_exactly_union_of_cached_and_on_disk(
    tmp_path, sample_metadata, sample_gates
):
    # One adapter exists only on disk; another is forced into the in-memory
    # cache without a sidecar on disk. list_adapters() is the single source of
    # truth and MUST return exactly the union (Req 2.3).
    _write_adapter(tmp_path, sample_metadata, sample_gates, "ondisk")
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)

    # Cache-only adapter (not written to disk) injected into the engine cache.
    engine._gate_cache["cacheonly"] = dict(sample_gates)

    assert engine.list_adapters() == ["cacheonly", "ondisk"]


def test_list_adapters_is_idempotent_and_reflects_new_writes(
    tmp_path, sample_metadata, sample_gates
):
    engine = ServingEngine("base-x", backend=StubBackend(), adapters_dir=tmp_path)
    assert engine.list_adapters() == []

    _write_adapter(tmp_path, sample_metadata, sample_gates, "alpha")
    assert engine.list_adapters() == ["alpha"]

    _write_adapter(tmp_path, sample_metadata, sample_gates, "beta")
    # Exactly the loadable set, sorted, no duplicates on repeat calls.
    assert engine.list_adapters() == ["alpha", "beta"]
    assert engine.list_adapters() == ["alpha", "beta"]
