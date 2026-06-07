"""Required Redis connection — sponsor integration for state + audit."""
from __future__ import annotations

from . import config


class RedisRequiredError(RuntimeError):
    """Raised when REDIS_URL is missing or Redis is unreachable."""


_client = None


def get_redis():
    """Return a shared Redis client. Fails fast if Redis is not available."""
    global _client
    if _client is not None:
        return _client

    url = (config.REDIS_URL or "").strip()
    if not url:
        raise RedisRequiredError(
            "REDIS_URL is required (e.g. redis://localhost:6379/0). "
            "Start redis-server and export REDIS_URL before starting the control plane."
        )
    try:
        import redis
    except ImportError as exc:
        raise RedisRequiredError("redis package required: pip install redis") from exc

    try:
        client = redis.from_url(url, decode_responses=True)
        client.ping()
    except Exception as exc:
        raise RedisRequiredError(f"Redis unreachable at {url}: {exc}") from exc

    _client = client
    return _client
