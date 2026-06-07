"""Real cross-track wiring + end-to-end loop (Integration_Milestone / Req 21).

This module replaces the Track B ``Mock_Dependency`` fixtures with the real
cross-track dependencies (Req 21.1) and assembles the end-to-end loop (Req 21.2):

* ``train_adapter`` — the real Track A NKT-Mirror trainer
  (:func:`weaveself.training.train_adapter`), bound to write Adapter_Files into a
  shared adapters directory.
* ``score_fn`` — the real Track A serving/inference ``score`` exposed by a
  resident :class:`~weaveself.serving.engine.ServingEngine` (Base_Model loaded
  once, gates swapped per request — the hard serving constraint, Req 7.1).
* ``redis_client`` — the real :class:`~weaveself.integration.redis_client.RedisClientApi`
  speaking the canonical Track C Redis key layout.

The :class:`EndToEndLoop` runs the batch graph to produce real Adapter_Files,
stores each through the Redis_Client_API, then for each Unit retrieves its
``adapter_id`` **through the Redis_Client_API** and serves it through the
Inference_API (Req 21.2).

Serving backend note: the default :class:`ServingEngine` uses the
dependency-free :class:`~weaveself.serving.backend.StubBackend`, so the loop runs
without a GPU or a multi-GB Qwen download while still exercising the real
serving contract (single base load, per-request gate swap, teacher-forced
score). Inject an :class:`~weaveself.serving.backend.HFBackend` (``pip install
-e '.[serving]'``) to serve the real Qwen Base_Model.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from weaveself.contracts.adapter_file import (
    adapter_blob_filename,
    adapter_meta_filename,
)
from weaveself.data.curation import Curator, GPTCurationNode
from weaveself.eval.weave_eval import PerplexityLogger, ScoreFn
from weaveself.integration.redis_client import RedisClientApi, create_redis_client
from weaveself.orchestration import (
    BatchDeps,
    BatchState,
    build_batch_graph,
    initial_state,
)
from weaveself.orchestration.state import UnitSpec
from weaveself.serving.backend import ModelBackend
from weaveself.serving.engine import ServingEngine
from weaveself.training import train_adapter as _real_train_adapter

# A collector pulls a Unit's raw interactions by ``unit_label`` (Req 13.1).
Collector = Callable[[str], list[dict]]

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _engine_score_fn(engine: ServingEngine) -> ScoreFn:
    """Adapt the Inference_API ``score`` to the eval ``ScoreFn`` (perplexity)."""

    def score(prompt: str, target: str, adapter_id: str | None) -> float:
        return float(engine.score(prompt, target, adapter_id).perplexity)

    return score


def build_real_train_adapter(
    adapters_dir: str | Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    day_index: int = 0,
) -> Callable[[str, str, str], str]:
    """Bind the real Track A ``train_adapter`` to a shared adapters directory.

    The batch graph calls ``train_adapter(dataset_path, unit_label, unit_type)``;
    this wrapper fixes ``out_dir`` (so every Adapter_File lands in one directory
    the Serving_Engine can read) and the ``base_model``/``day_index`` while
    preserving the three-argument graph contract. Training stays batch-only
    (Req 13.4): this callable is only ever invoked from the graph's ``train``
    node inside :meth:`BatchRunner.run_batch`.
    """
    adapters_dir = str(adapters_dir)

    def train(dataset_path: str, unit_label: str, unit_type: str) -> str:
        return _real_train_adapter(
            dataset_path,
            unit_label,
            unit_type,
            base_model=base_model,
            out_dir=adapters_dir,
            day_index=day_index,
        )

    # Expose the bound directory for introspection/tests.
    train.adapters_dir = adapters_dir  # type: ignore[attr-defined]
    return train


def build_real_batch_deps(
    *,
    collector: Collector,
    redis_client: RedisClientApi,
    engine: ServingEngine,
    workdir: str | Path,
    adapters_dir: str | Path,
    curator: Curator | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    day_index: int = 0,
    min_rows: int = 1,
    logger: PerplexityLogger | None = None,
    eval_out_path: str | Path | None = None,
) -> BatchDeps:
    """Compose :class:`BatchDeps` from the real cross-track dependencies (Req 21.1).

    Replaces ``MockTrainAdapter`` with the real :func:`train_adapter`,
    ``MockInference`` with the resident :class:`ServingEngine`'s ``score``, and
    ``MockRedisClient`` with the real :class:`RedisClientApi`.
    """
    return BatchDeps(
        collector=collector,
        curation_node=GPTCurationNode(curator),  # real curation node (Req 12)
        train_adapter=build_real_train_adapter(
            adapters_dir, base_model=base_model, day_index=day_index
        ),
        redis_client=redis_client,
        score_fn=_engine_score_fn(engine),
        workdir=workdir,
        min_rows=min_rows,
        logger=logger,
        eval_out_path=eval_out_path,
    )


def materialize_adapter_from_redis(
    redis_client: RedisClientApi,
    adapter_id: str,
    target_dir: str | Path,
) -> Path:
    """Reconstruct an Adapter_File pair on disk from the Redis_Client_API.

    Fetches the metadata and blob bytes for ``adapter_id`` **through the
    Redis_Client_API** and writes the ``adapter_<id>.safetensors`` /
    ``adapter_<id>.json`` pair into ``target_dir`` so a Serving_Engine can load
    and serve bytes that round-tripped through the Redis_Layer. Returns the blob
    path.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    meta = redis_client.fetch_meta(adapter_id)
    blob = redis_client.fetch_blob(adapter_id)

    blob_path = target_dir / adapter_blob_filename(adapter_id)
    meta_path = target_dir / adapter_meta_filename(adapter_id)
    blob_path.write_bytes(blob)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return blob_path


@dataclass
class ServeResult:
    """The outcome of serving one Unit at the Integration_Milestone (Req 21.2)."""

    unit_label: str
    adapter_id: str  # retrieved THROUGH the Redis_Client_API (route)
    prompt: str
    text: str
    tokens: int


@dataclass
class EndToEndLoop:
    """The wired end-to-end loop: batch graph -> Redis -> Inference_API (Req 21.2).

    Holds the real dependencies and the resident serving engine. :meth:`run`
    executes the batch graph (which trains, evals, and stores adapters through
    the Redis_Client_API), then :meth:`serve_unit` retrieves an ``adapter_id``
    through the Redis_Client_API and serves it through the Inference_API.
    """

    deps: BatchDeps
    engine: ServingEngine
    redis_client: RedisClientApi
    adapters_dir: Path
    serve_dir: Path
    last_state: BatchState | None = field(default=None)

    def run(self, units: list[UnitSpec]) -> BatchState:
        """Run the batch graph for ``units`` and return the final state.

        Training only happens here (batch-only, Req 13.4/13.5). Each produced
        Adapter_File is persisted through the Redis_Client_API by the graph's
        ``store`` node.
        """
        graph = build_batch_graph(self.deps)
        state = graph.invoke(initial_state(units))
        self.last_state = state
        return state

    def serve_unit(
        self,
        query_or_unit: str,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        from_redis_bytes: bool = True,
    ) -> ServeResult:
        """Serve a Unit by ``adapter_id`` retrieved through the Redis_Client_API.

        1. Resolve the ``adapter_id`` via :meth:`RedisClientApi.route` (the
           retrieval goes through the Redis_Client_API, Req 21.2).
        2. When ``from_redis_bytes`` is set, fetch the blob/metadata back
           through the Redis_Client_API and write the Adapter_File into
           ``serve_dir``, proving the bytes round-trip through the Redis_Layer
           and reproduce the same Adapter_File the resident engine serves.
        3. Generate a response through the Inference_API: the single resident
           engine (Base_Model loaded once, Req 7.1) swaps in the adapter's gate
           tensors for this request.
        """
        adapter_id = self.redis_client.route(query_or_unit)

        if from_redis_bytes:
            materialize_adapter_from_redis(
                self.redis_client, adapter_id, self.serve_dir
            )

        generation = self.engine.generate(prompt, adapter_id, max_new_tokens)
        return ServeResult(
            unit_label=query_or_unit,
            adapter_id=adapter_id,
            prompt=prompt,
            text=generation.text,
            tokens=generation.tokens,
        )


def build_end_to_end_loop(
    *,
    collector: Collector,
    workdir: str | Path,
    redis_client: RedisClientApi | None = None,
    redis_url: str | None = None,
    redis_file_path: str | Path | None = None,
    backend: ModelBackend | None = None,
    curator: Curator | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    day_index: int = 0,
    min_rows: int = 1,
    logger: PerplexityLogger | None = None,
    eval_out_path: str | Path | None = None,
) -> EndToEndLoop:
    """Construct the fully-wired :class:`EndToEndLoop` with real dependencies.

    Args:
        collector: pulls a Unit's raw interactions by ``unit_label``. At the
            Integration_Milestone this is typically
            ``redis_client.read_interactions`` (interactions persisted by the
            Frontend_App), but any real source satisfies the contract.
        workdir: scratch directory for curated splits and serving materials.
        redis_client: an explicit Redis_Client_API; when ``None`` one is built
            via :func:`create_redis_client` (live Redis when available, else a
            file/in-memory fallback honoring the same layout).
        backend: the serving :class:`ModelBackend`; defaults to the
            dependency-free ``StubBackend`` (inject ``HFBackend`` for real Qwen).
        eval_out_path: where the ``eval_results.json`` artifact is written
            (defaults to ``<workdir>/eval_results.json``).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    adapters_dir = workdir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)
    serve_dir = workdir / "serve_adapters"
    serve_dir.mkdir(parents=True, exist_ok=True)

    if redis_client is None:
        redis_client = create_redis_client(
            url=redis_url,
            file_path=redis_file_path or (workdir / "redis_store.json"),
        )

    if eval_out_path is None:
        eval_out_path = workdir / "eval_results.json"

    # One resident Serving_Engine: Base_Model loaded exactly once, adapters
    # served by swapping gate tensors read from ``adapters_dir`` (Req 7.1).
    engine = ServingEngine(
        base_model,
        backend=backend,
        adapters_dir=str(adapters_dir),
    )

    deps = build_real_batch_deps(
        collector=collector,
        redis_client=redis_client,
        engine=engine,
        workdir=workdir,
        adapters_dir=adapters_dir,
        curator=curator,
        base_model=base_model,
        day_index=day_index,
        min_rows=min_rows,
        logger=logger,
        eval_out_path=eval_out_path,
    )

    return EndToEndLoop(
        deps=deps,
        engine=engine,
        redis_client=redis_client,
        adapters_dir=adapters_dir,
        serve_dir=serve_dir,
    )
