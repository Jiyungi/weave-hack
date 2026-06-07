"""Cross-track integration wiring (Integration_Milestone / Requirement 21).

Until task 12 the three tracks ran against ``Mock_Dependency`` fixtures
(:mod:`weaveself.orchestration.mocks`). This package provides the *real*
cross-track wiring that replaces those mocks at the Integration_Milestone
(Req 21.1):

* :mod:`weaveself.integration.redis_client` — a Python Redis_Client_API adapter
  that speaks the SAME Redis key layout as the canonical Track C Node/TS layer
  (``adapter:blob:<id>``, ``adapter:meta:<id>``, ``adapter:index``,
  ``interactions:<unit_label>``). It talks to a live Redis when the ``redis``
  package and a reachable server are available, and otherwise falls back to a
  file/in-memory backend that honors the identical layout and interface.
* :mod:`weaveself.integration.wiring` — composes :class:`BatchDeps` from the
  real Track A ``train_adapter`` and Inference_API ``score``, the real
  Redis_Client_API, and the real curation node, and builds the end-to-end loop
  that runs the batch graph, stores each produced Adapter_File through the
  Redis_Client_API, and serves each by ``adapter_id`` retrieved through the
  Redis_Client_API via the Inference_API (Req 21.2).

The :mod:`weaveself.orchestration.mocks` module is intentionally left in place
(task 12.3 keeps the mocks wired while serving is unverified, Req 22.2).
"""

from weaveself.integration.redis_client import (
    REDIS_KEY_PREFIXES,
    FileKvBackend,
    InMemoryKvBackend,
    KvBackend,
    RedisClientApi,
    RedisKvBackend,
    adapter_blob_key,
    adapter_index_key,
    adapter_meta_key,
    create_redis_client,
    embed_text,
    interactions_key,
)
from weaveself.integration.wiring import (
    EndToEndLoop,
    ServeResult,
    build_end_to_end_loop,
    build_real_batch_deps,
    materialize_adapter_from_redis,
)
from weaveself.integration.demo import (
    DEMO_DAY_INDICES,
    DEMO_UNITS,
    DayCollector,
    DemoEnvironment,
    DemoSelection,
    make_demo_collector,
    prebake_demo_adapters,
)
from weaveself.integration.governance import (
    CriticalPathGovernor,
    DependencyMode,
    FallbackDemo,
    GovernedBatch,
    IntegrationState,
    IntegrationStatus,
    ServingVerification,
    build_governed_batch_deps,
    build_governed_end_to_end_loop,
    build_mock_batch_deps,
    fallback_demo_entry_points,
    run_track_a_compare_demo,
    run_track_b_confusion_matrix_demo,
    verify_track_a_serving,
)

__all__ = [
    # redis_client
    "REDIS_KEY_PREFIXES",
    "KvBackend",
    "InMemoryKvBackend",
    "FileKvBackend",
    "RedisKvBackend",
    "RedisClientApi",
    "create_redis_client",
    "adapter_blob_key",
    "adapter_meta_key",
    "adapter_index_key",
    "interactions_key",
    "embed_text",
    # wiring
    "build_real_batch_deps",
    "build_end_to_end_loop",
    "materialize_adapter_from_redis",
    "EndToEndLoop",
    "ServeResult",
    # demo (task 12.2 — pre-bake + Unit-selection routing to proof visuals)
    "DEMO_UNITS",
    "DEMO_DAY_INDICES",
    "DayCollector",
    "DemoEnvironment",
    "DemoSelection",
    "make_demo_collector",
    "prebake_demo_adapters",
    # governance (critical-path + blocked-state fallback, Req 22/23)
    "CriticalPathGovernor",
    "DependencyMode",
    "FallbackDemo",
    "GovernedBatch",
    "IntegrationState",
    "IntegrationStatus",
    "ServingVerification",
    "build_governed_batch_deps",
    "build_governed_end_to_end_loop",
    "build_mock_batch_deps",
    "fallback_demo_entry_points",
    "run_track_a_compare_demo",
    "run_track_b_confusion_matrix_demo",
    "verify_track_a_serving",
]
