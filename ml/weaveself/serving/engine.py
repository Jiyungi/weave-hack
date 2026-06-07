"""The resident Serving_Engine (Track A, critical path).

The :class:`ServingEngine` loads the Base_Model **exactly once** per process
(Requirement 7.1) and keeps it resident, then serves many tiny NKT-Mirror
adapters by swapping their ~100 KB gate tensors per request. Loaded gate tensors
are cached in memory keyed by ``adapter_id`` (Requirement 6.1) so repeated
requests for the same adapter never re-read the Adapter_File from disk.

Per request, the engine resolves the selected adapter's gate tensors (or
``None`` for the pure Base_Model) and hands them to the backend, which applies
them for the duration of that single request and ALWAYS clears them afterward —
including when the request raises — so the next request runs against the pure
Base_Model (Req 6.3, 7.2, 7.3). When ``adapter_id`` is ``None`` no gates are
applied (Req 7.3). :meth:`score` returns a teacher-forced NLL and a perplexity
consistent with it (Req 8.2).
"""

from __future__ import annotations

import os
from pathlib import Path

from weaveself.contracts import adapter_meta_filename, read_adapter_file
from weaveself.serving.backend import (
    GateTensors,
    Generation,
    ModelBackend,
    ScoreResult,
    StubBackend,
)
from weaveself.serving.errors import AdapterNotLoadable

# Prefix/suffix of the Adapter_File JSON sidecar: ``adapter_<id>.json``.
_META_PREFIX = "adapter_"
_META_SUFFIX = ".json"


class ServingEngine:
    """Loads the Base_Model once and serves adapters by swapping gate tensors.

    Args:
        base_model_id: The instruct Base_Model identifier (the ``base_model``
            metadata field, Req 6.1). The model is loaded exactly once on
            construction and kept resident (Req 7.1).
        backend: The :class:`ModelBackend` used to load the model and run
            forward passes. Defaults to :class:`StubBackend` so tests and CI
            never hit the network or require a GPU.
        adapters_dir: Directory holding ``adapter_<id>.safetensors`` /
            ``adapter_<id>.json`` pairs. The loadable adapter set is discovered
            from this directory.
    """

    def __init__(
        self,
        base_model_id: str,
        backend: ModelBackend | None = None,
        adapters_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.base_model_id = base_model_id
        self._backend: ModelBackend = backend if backend is not None else StubBackend()
        self._adapters_dir: Path | None = (
            Path(adapters_dir) if adapters_dir is not None else None
        )
        # In-memory cache of loaded gate tensors keyed by adapter_id (Req 6.1).
        self._gate_cache: dict[str, GateTensors] = {}

        # Load the Base_Model exactly once per process (Req 7.1). The engine
        # guarantees a single load by loading here and never again.
        self._backend.load_base_model(base_model_id)

    # -- residency / introspection -----------------------------------------

    @property
    def backend(self) -> ModelBackend:
        """The model backend (exposes ``load_count`` for the single-load check)."""
        return self._backend

    @property
    def base_model_load_count(self) -> int:
        """How many times the Base_Model has been loaded (MUST be 1, Req 7.1)."""
        return self._backend.load_count

    # -- adapter loading / caching -----------------------------------------

    def load_adapter(self, adapter_id: str) -> GateTensors:
        """Load and cache the gate tensors for ``adapter_id`` (Req 6.1).

        Returns the cached tensors on a repeat call without re-reading disk.
        Raises :class:`AdapterNotLoadable` naming ``adapter_id`` if the
        Adapter_File is absent or unreadable (Req 7.5) — validated before any
        forward-pass mutation.
        """
        cached = self._gate_cache.get(adapter_id)
        if cached is not None:
            return cached

        if self._adapters_dir is None:
            raise AdapterNotLoadable(
                adapter_id, reason="no adapters directory is configured"
            )

        try:
            _meta, gate_tensors = read_adapter_file(self._adapters_dir, adapter_id)
        except FileNotFoundError as exc:
            raise AdapterNotLoadable(adapter_id, reason=str(exc)) from exc

        self._gate_cache[adapter_id] = gate_tensors
        return gate_tensors

    def list_adapters(self) -> list[str]:
        """Return the currently loadable ``adapter_id`` values (Req 2.3).

        The loadable set is the union of adapters already cached in memory and
        the Adapter_File sidecars discovered in ``adapters_dir``.
        """
        ids: set[str] = set(self._gate_cache)
        if self._adapters_dir is not None and self._adapters_dir.is_dir():
            for path in self._adapters_dir.glob(f"{_META_PREFIX}*{_META_SUFFIX}"):
                name = path.name
                adapter_id = name[len(_META_PREFIX) : -len(_META_SUFFIX)]
                if adapter_id:
                    ids.add(adapter_id)
        return sorted(ids)

    def _resolve_gates(self, adapter_id: str | None) -> GateTensors | None:
        """Resolve gate tensors for a request: ``None`` adapter -> pure base."""
        if adapter_id is None:
            return None
        return self.load_adapter(adapter_id)

    # -- generation / scoring (per-request hook-based gating) ---------------

    def generate(
        self,
        prompt: str,
        adapter_id: str | None,
        max_new_tokens: int,
    ) -> Generation:
        """Generate text for ``prompt`` under an optional adapter.

        Resolves and applies the adapter's gate tensors for this request when
        ``adapter_id`` is set, or runs the pure resident Base_Model when it is
        ``None`` (Req 7.2, 7.3). The backend applies the gates via forward hooks
        and clears them in a ``finally`` block, so a request that raises never
        leaves gates applied for the next request (protects Property 4).

        The adapter is resolved (and thus validated) BEFORE the backend is
        entered: an unknown ``adapter_id`` raises :class:`AdapterNotLoadable`
        naming the missing id and the backend's forward pass is never reached,
        so the resident model is never mutated (Req 7.5).
        """
        gates = self._resolve_gates(adapter_id)
        return self._backend.generate(prompt, gates, max_new_tokens)

    def score(
        self,
        prompt: str,
        target: str,
        adapter_id: str | None,
    ) -> ScoreResult:
        """Teacher-forced NLL/perplexity of ``target`` under an optional adapter.

        Resolves the gate tensors (``None`` -> pure Base_Model) and delegates to
        the backend, which applies and clears them per request. The returned
        ``perplexity`` is consistent with the token-averaged ``nll`` and both are
        non-negative (Req 8.2).

        As with :meth:`generate`, the adapter is resolved (and thus validated)
        BEFORE the backend is entered: an unknown ``adapter_id`` raises
        :class:`AdapterNotLoadable` naming the missing id and the backend's
        forward pass is never reached (Req 7.5).
        """
        gates = self._resolve_gates(adapter_id)
        return self._backend.score(prompt, target, gates)
