"""Shared HTTP connection pool.

One AsyncClient for the app's lifetime: connections to Upstash and the
providers are established once and reused, instead of paying a TCP+TLS
handshake on every call. On Render's free 0.1-CPU instance that handshake
tax dominated cache-hit latency (p50 ~3.8s before pooling).
"""

from typing import Optional

import httpx

_client: Optional[httpx.AsyncClient] = None


async def startup() -> None:
    global _client
    _client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )


async def shutdown() -> None:
    if _client is not None:
        await _client.aclose()


def client() -> httpx.AsyncClient:
    assert _client is not None, "http.startup() not called"
    return _client
