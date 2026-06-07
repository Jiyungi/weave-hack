"""Cross-track integration wiring (Integration_Milestone / Requirement 21).

Real, production wiring only:

* :mod:`weaveself.integration.redis_client` — a Python Redis_Client_API adapter
  speaking the canonical Redis key layout (``adapter:blob:<id>``,
  ``adapter:meta:<id>``, ``adapter:index``, ``interactions:<unit_label>``)
  against a live Redis server.
* :mod:`weaveself.integration.wiring` — composes :class:`BatchDeps` from the
  real Track A ``train_adapter`` and Inference_API ``score``, the real
  Redis_Client_API, and the real curation node, and builds the end-to-end loop
  that runs the batch graph, stores each produced Adapter_File through the
  Redis_Client_API, and serves each by ``adapter_id`` retrieved through the
  Redis_Client_API via the Inference_API (Req 21.2).
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
]
