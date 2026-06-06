"""Append-only audit trail.

Every governance action (register, policy, session open, act, revoke) is recorded
so the demo can show *who* was granted/revoked *what*, *when*, and what actually
fired. Backends, in priority order:
  1. Redis stream (XADD) if REDIS_URL is set and reachable  [sponsor integration]
  2. a JSONL file on disk
  3. an in-memory ring buffer (always, for fast /audit reads)
Redis and file writes are best-effort: a backend failure never breaks a request.
"""
from __future__ import annotations

import json
import time
from collections import deque

from . import config


class Audit:
    def __init__(self) -> None:
        self._mem: deque[dict] = deque(maxlen=2000)
        self._redis = None
        self.backend = "memory+file"
        if config.REDIS_URL:
            try:
                import redis  # optional dependency
                client = redis.from_url(config.REDIS_URL)
                client.ping()
                self._redis = client
                self.backend = "redis+memory+file"
            except Exception:
                # No server / no redis-py: silently use memory+file.
                self._redis = None

    def record(self, event: str, **fields) -> dict:
        entry = {"ts": round(time.time(), 3), "event": event, **fields}
        self._mem.append(entry)
        try:
            with config.AUDIT_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        if self._redis is not None:
            try:
                self._redis.xadd(config.AUDIT_STREAM, {"json": json.dumps(entry)})
            except Exception:
                pass
        return entry

    def tail(self, n: int = 50) -> list[dict]:
        return list(self._mem)[-n:]


audit = Audit()
