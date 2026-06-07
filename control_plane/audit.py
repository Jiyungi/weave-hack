"""Append-only audit trail (Redis stream + JSONL backup).

Every governance action is recorded in Redis stream ``cp:audit`` (primary, sponsor
integration). A JSONL file is written as backup. The UI reads from Redis via
``tail()``.
"""
from __future__ import annotations

import json
import time

from . import config
from .redis_client import get_redis


class Audit:
    def __init__(self) -> None:
        self._redis = get_redis()
        self.backend = "redis+file"

    def record(self, event: str, **fields) -> dict:
        entry = {"ts": round(time.time(), 3), "event": event, **fields}
        try:
            with config.AUDIT_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
        try:
            self._redis.xadd(config.AUDIT_STREAM, {"json": json.dumps(entry)})
        except Exception:
            pass
        return entry

    def tail(self, n: int = 50) -> list[dict]:
        try:
            raw = self._redis.xrevrange(config.AUDIT_STREAM, count=n)
            out: list[dict] = []
            for _id, fields in reversed(raw):
                payload = fields.get("json")
                if payload:
                    out.append(json.loads(payload))
            return out
        except Exception:
            return []


audit = Audit()
