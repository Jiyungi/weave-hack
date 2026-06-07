"""Required Redis connection — sponsor integration for state + audit."""
from __future__ import annotations

from . import config


class RedisRequiredError(RuntimeError):
    """Raised when REDIS_URL is missing or Redis is unreachable."""


_client = None


def _host_label(url: str) -> str:
    return url.split("@")[-1] if "@" in url else url


def _connect(url: str):
    import ssl

    import redis

    def ping_url(connect_url: str, *, use_ssl: bool) -> redis.Redis:
        kwargs: dict = {"decode_responses": True}
        if use_ssl:
            kwargs["ssl_cert_reqs"] = ssl.CERT_NONE
        client = redis.from_url(connect_url, **kwargs)
        client.ping()
        return client

    if url.startswith("rediss://"):
        try:
            return ping_url(url, use_ssl=True)
        except Exception as exc:
            msg = str(exc).lower()
            # Redis Cloud: some ports are plain TCP — rediss:// causes WRONG_VERSION_NUMBER.
            if "wrong version number" in msg:
                plain = "redis://" + url[len("rediss://") :]
                try:
                    return ping_url(plain, use_ssl=False)
                except Exception as plain_exc:
                    raise RedisRequiredError(
                        f"Redis unreachable at {_host_label(url)} "
                        f"(tried TLS and plain): {plain_exc}"
                    ) from plain_exc
            raise

    return ping_url(url, use_ssl=False)


def get_redis():
    """Return a shared Redis client. Fails fast if Redis is not available."""
    global _client
    if _client is not None:
        return _client

    url = (config.REDIS_URL or "").strip()
    if not url:
        raise RedisRequiredError(
            "REDIS_URL is required (e.g. redis://localhost:6379/0). "
            "Set REDIS_URL in .env before starting the control plane."
        )
    try:
        import redis  # noqa: F401 — ensure package present
    except ImportError as exc:
        raise RedisRequiredError("redis package required: pip install redis") from exc

    try:
        client = _connect(url)
    except RedisRequiredError:
        raise
    except Exception as exc:
        raise RedisRequiredError(f"Redis unreachable at {_host_label(url)}: {exc}") from exc

    _client = client
    return _client
