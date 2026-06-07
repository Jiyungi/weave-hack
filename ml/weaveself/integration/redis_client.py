"""Python Redis_Client_API adapter (Integration_Milestone / Requirement 21).

The canonical Redis_Layer is owned by Track C (Node/TS — see
``app/src/redis/``). The Python batch graph stores adapters through a
``redis_client.store_adapter(meta, blob)`` interface, so this module provides a
Python client that speaks the **same Redis key layout** as Track C so both
languages address one keyspace (design.md "Redis Layout", Requirement 3):

    adapter:blob:<adapter_id>     -> adapter bytes (base64 string)        (Req 3.1)
    adapter:meta:<adapter_id>     -> adapter metadata JSON                (Req 3.2)
    adapter:index                 -> vector index of unit_label embeds    (Req 3.3)
    interactions:<unit_label>     -> raw daily interactions for a Unit    (Req 3.4)

Serialization matches the Track C ``RedisBackedClient`` byte-for-byte (blobs are
base64, metadata is JSON, the index is a JSON array of
``{adapterId, unitLabel, embedding}`` records, interactions are JSON strings
pushed onto a list) so a live Redis written by this client is readable by the
Node client and vice versa.

Backends:

* :class:`RedisKvBackend` — wraps the ``redis`` Python package against a live
  server. Used automatically by :func:`create_redis_client` when ``redis`` is
  importable and a server is reachable.
* :class:`FileKvBackend` — a JSON-file-backed key/value store honoring the
  identical key layout and interface; used as the fallback when no live Redis is
  available (e.g. CI / this environment, where the ``redis`` package is not
  installed). The persisted file IS the keyspace, so it can be inspected or
  re-loaded across processes.
* :class:`InMemoryKvBackend` — process-local fallback for tests.
"""

from __future__ import annotations

import base64
import json
import math
import threading
from pathlib import Path
from typing import Mapping, Sequence

from weaveself.contracts.adapter_file import AdapterMetadata

# --- Redis key layout (mirrors app/src/redis/keys.ts) ----------------------

REDIS_KEY_PREFIXES = {
    "adapter_blob": "adapter:blob:",
    "adapter_meta": "adapter:meta:",
    "adapter_index": "adapter:index",
    "interactions": "interactions:",
}


def _assert_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def adapter_blob_key(adapter_id: str) -> str:
    """Key for an adapter's blob bytes (Req 3.1)."""
    _assert_non_empty(adapter_id, "adapter_id")
    return f"{REDIS_KEY_PREFIXES['adapter_blob']}{adapter_id}"


def adapter_meta_key(adapter_id: str) -> str:
    """Key for an adapter's metadata JSON (Req 3.2)."""
    _assert_non_empty(adapter_id, "adapter_id")
    return f"{REDIS_KEY_PREFIXES['adapter_meta']}{adapter_id}"


def adapter_index_key() -> str:
    """Fixed key for the vector index of ``unit_label`` embeddings (Req 3.3)."""
    return REDIS_KEY_PREFIXES["adapter_index"]


def interactions_key(unit_label: str) -> str:
    """Key for a Unit's raw daily interactions (Req 3.4)."""
    _assert_non_empty(unit_label, "unit_label")
    return f"{REDIS_KEY_PREFIXES['interactions']}{unit_label}"


# --- Embedding / routing (mirrors app/src/redis/embedding.ts) --------------

_EMBEDDING_DIMENSIONS = 64
_UINT32 = 0xFFFFFFFF


def _token_weight(token: str) -> int:
    """FNV-1a-style 32-bit token hash, matching the Track C ``tokenWeight``."""
    h = 2166136261
    for ch in token:
        h ^= ord(ch) & _UINT32
        h = (h * 16777619) & _UINT32  # emulate Math.imul mod 2**32
    return h & _UINT32


def embed_text(text: str) -> list[float]:
    """Deterministic bag-of-tokens embedding matching Track C ``embedText``.

    Replicating the Node embedding exactly means the ``adapter:index`` this
    client writes is identical to one Track C would write, so routing is
    consistent across the two languages.
    """
    vector = [0.0] * _EMBEDDING_DIMENSIONS
    normalized = "".join(
        ch if (ch.isascii() and ch.isalnum()) else " " for ch in text.lower()
    ).strip()
    if normalized:
        tokens = normalized.split()
    else:
        tokens = [text.lower()]
    for token in tokens:
        h = _token_weight(token)
        vector[h % _EMBEDDING_DIMENSIONS] += 1.0
        vector[(h >> 8) % _EMBEDDING_DIMENSIONS] += 0.5
    return vector


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    length = max(len(left), len(right))
    dot = left_mag = right_mag = 0.0
    for i in range(length):
        lv = left[i] if i < len(left) else 0.0
        rv = right[i] if i < len(right) else 0.0
        dot += lv * rv
        left_mag += lv * lv
        right_mag += rv * rv
    if left_mag == 0 or right_mag == 0:
        return 0.0
    return dot / (math.sqrt(left_mag) * math.sqrt(right_mag))


# --- Key/value backends ----------------------------------------------------


class KvBackend:
    """Minimal Redis-shaped key/value interface used by :class:`RedisClientApi`.

    Implementations provide string ``get``/``set`` plus list ``rpush``/``lrange``
    so the client logic can mirror the Track C ``RedisBackedClient`` exactly,
    independent of whether the backing store is a live Redis or a local file.
    """

    def get(self, key: str) -> str | None:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def rpush(self, key: str, value: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def lrange(self, key: str, start: int, stop: int) -> list[str]:  # pragma: no cover
        raise NotImplementedError


class InMemoryKvBackend(KvBackend):
    """Process-local key/value backend (no persistence)."""

    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._lists: dict[str, list[str]] = {}

    def get(self, key: str) -> str | None:
        return self._strings.get(key)

    def set(self, key: str, value: str) -> None:
        self._strings[key] = value

    def rpush(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).append(value)

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        values = self._lists.get(key, [])
        if stop == -1:
            return list(values[start:])
        return list(values[start : stop + 1])


class FileKvBackend(KvBackend):
    """JSON-file-backed key/value store honoring the Redis key layout.

    The whole keyspace is persisted to a single JSON document so it survives
    across processes and can be inspected. String keys and list keys live in
    separate top-level maps. Access is guarded by a lock so concurrent batch
    writes are safe within a process.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({"strings": {}, "lists": {}})

    def _read(self) -> dict:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"strings": {}, "lists": {}}

    def _write(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._read().get("strings", {}).get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            data = self._read()
            data.setdefault("strings", {})[key] = value
            self._write(data)

    def rpush(self, key: str, value: str) -> None:
        with self._lock:
            data = self._read()
            data.setdefault("lists", {}).setdefault(key, []).append(value)
            self._write(data)

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        with self._lock:
            values = self._read().get("lists", {}).get(key, [])
        if stop == -1:
            return list(values[start:])
        return list(values[start : stop + 1])


class RedisKvBackend(KvBackend):
    """Live-Redis backend wrapping a ``redis.Redis`` client.

    Uses ``decode_responses=True`` semantics: values are stored and returned as
    UTF-8 strings, matching the Track C client which stores base64/JSON strings.
    """

    def __init__(self, client: object) -> None:
        self._client = client

    @staticmethod
    def _to_str(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def get(self, key: str) -> str | None:
        return self._to_str(self._client.get(key))  # type: ignore[attr-defined]

    def set(self, key: str, value: str) -> None:
        self._client.set(key, value)  # type: ignore[attr-defined]

    def rpush(self, key: str, value: str) -> None:
        self._client.rpush(key, value)  # type: ignore[attr-defined]

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        raw = self._client.lrange(key, start, stop)  # type: ignore[attr-defined]
        return [self._to_str(v) or "" for v in raw]


# --- Redis_Client_API ------------------------------------------------------


def _bytes_to_base64(blob: bytes) -> str:
    return base64.b64encode(bytes(blob)).decode("ascii")


def _base64_to_bytes(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _meta_to_dict(meta: Mapping[str, object] | AdapterMetadata) -> dict:
    if isinstance(meta, AdapterMetadata):
        return meta.model_dump()
    return dict(meta)


class RedisClientApi:
    """Python Redis_Client_API over a :class:`KvBackend` (Req 3, 19).

    Mirrors the Track C ``RedisBackedClient`` contract so a Python batch run and
    a Node frontend share one keyspace:

    * :meth:`store_adapter` persists metadata under ``adapter:meta:<id>``, the
      blob (base64) under ``adapter:blob:<id>``, and upserts an index record
      under ``adapter:index`` (Req 3.1, 3.2, 19.1).
    * :meth:`fetch_meta` returns stored metadata without needing the blob
      (Req 19.2); :meth:`fetch_blob` round-trips bytes identically (Req 19.4).
    * :meth:`route` returns the top-1 ``adapter_id`` by cosine similarity over
      the unit-label embedding index (Req 3.5, 19.3).
    * :meth:`append_interaction` appends a raw interaction under
      ``interactions:<unit_label>`` (Req 3.4, 19.5).
    """

    def __init__(self, backend: KvBackend) -> None:
        self._backend = backend

    # -- store / fetch ------------------------------------------------------

    def store_adapter(
        self, meta: Mapping[str, object] | AdapterMetadata, blob: bytes
    ) -> None:
        meta_dict = _meta_to_dict(meta)
        adapter_id = str(meta_dict["adapter_id"])
        unit_label = str(meta_dict["unit_label"])

        self._backend.set(adapter_blob_key(adapter_id), _bytes_to_base64(blob))
        self._backend.set(adapter_meta_key(adapter_id), json.dumps(meta_dict))

        records = self._read_index()
        records = [r for r in records if r.get("adapterId") != adapter_id]
        records.append(
            {
                "adapterId": adapter_id,
                "unitLabel": unit_label,
                "embedding": embed_text(unit_label),
            }
        )
        self._backend.set(adapter_index_key(), json.dumps(records))

    def fetch_meta(self, adapter_id: str) -> dict:
        raw = self._backend.get(adapter_meta_key(adapter_id))
        if raw is None:
            raise KeyError(f"Adapter metadata not found: {adapter_id}")
        return json.loads(raw)

    def fetch_blob(self, adapter_id: str) -> bytes:
        raw = self._backend.get(adapter_blob_key(adapter_id))
        if raw is None:
            raise KeyError(f"Adapter blob not found: {adapter_id}")
        return _base64_to_bytes(raw)

    # -- routing ------------------------------------------------------------

    def route(self, query_or_user: str) -> str:
        records = self._read_index()
        if not records:
            raise KeyError(f"{adapter_index_key()} is empty")
        query = embed_text(query_or_user)
        best = records[0]
        best_score = _cosine_similarity(query, best["embedding"])
        for record in records[1:]:
            score = _cosine_similarity(query, record["embedding"])
            if score > best_score or (
                score == best_score and record["adapterId"] < best["adapterId"]
            ):
                best = record
                best_score = score
        return str(best["adapterId"])

    def reindex_route_targets(self, adapter_ids: Sequence[str]) -> None:
        """Rebuild ``adapter:index`` to contain exactly ``adapter_ids`` (Req 21.4).

        The blob and metadata for every stored adapter remain addressable by
        ``adapter_id`` (those keys are untouched); only the routing index is
        rewritten so :meth:`route` resolves against just this set. This backs the
        time-compressed demo (task 12.2): many ``day_index`` adapters can share a
        ``unit_label`` in the keyspace, but the route index is pointed at a single
        active demo day so ``route(unit_label)`` returns that day's adapter rather
        than an arbitrary tie-break across days.

        Each id's ``unit_label`` is read from its stored metadata, so the rebuilt
        index records match exactly what :meth:`store_adapter` would have written.
        Raises :class:`KeyError` if any ``adapter_id`` has no stored metadata.
        """
        records = []
        for adapter_id in adapter_ids:
            meta = self.fetch_meta(adapter_id)
            unit_label = str(meta["unit_label"])
            records.append(
                {
                    "adapterId": adapter_id,
                    "unitLabel": unit_label,
                    "embedding": embed_text(unit_label),
                }
            )
        self._backend.set(adapter_index_key(), json.dumps(records))

    # -- interactions -------------------------------------------------------

    def append_interaction(self, unit_label: str, interaction: Mapping[str, object]) -> None:
        self._backend.rpush(interactions_key(unit_label), json.dumps(dict(interaction)))

    def read_interactions(self, unit_label: str) -> list[dict]:
        values = self._backend.lrange(interactions_key(unit_label), 0, -1)
        return [json.loads(v) for v in values]

    # -- helpers ------------------------------------------------------------

    def _read_index(self) -> list[dict]:
        raw = self._backend.get(adapter_index_key())
        if raw is None:
            return []
        parsed = json.loads(raw)
        return list(parsed) if isinstance(parsed, list) else []


# --- Factory ---------------------------------------------------------------


def create_redis_client(
    *,
    url: str | None = None,
    file_path: str | Path | None = None,
    backend: KvBackend | None = None,
) -> RedisClientApi:
    """Build a :class:`RedisClientApi`, preferring a live Redis when available.

    Resolution order:

    1. An explicit ``backend`` is used as-is.
    2. Otherwise, if a ``url`` is explicitly provided, the ``redis`` package is
       importable, and the server at ``url`` is reachable, a
       :class:`RedisKvBackend` is used (the canonical live Redis_Layer). A short
       connect/socket timeout keeps an unreachable URL from hanging the caller.
    3. Otherwise (no ``url``, ``redis`` missing, or server unreachable) fall back
       to a :class:`FileKvBackend` at ``file_path`` (or an
       :class:`InMemoryKvBackend` when no path is given). The fallback honors the
       identical key layout and interface, so the only difference from the live
       path is durability/visibility — not behavior.

    A ``url`` of ``None`` never triggers a live-connection attempt: callers that
    want a live Redis pass an explicit URL (e.g. ``REDIS_URL``), and callers that
    only pass ``file_path`` deterministically get the file fallback regardless of
    whether some unrelated Redis happens to be listening on a default port.
    """
    if backend is not None:
        return RedisClientApi(backend)

    if url:
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            client.ping()  # verify the server is actually reachable
            return RedisClientApi(RedisKvBackend(client))
        except Exception:
            # Server unreachable or ``redis`` missing: fall through to a backend
            # that honors the same key layout/interface.
            pass

    if file_path is not None:
        return RedisClientApi(FileKvBackend(file_path))
    return RedisClientApi(InMemoryKvBackend())
