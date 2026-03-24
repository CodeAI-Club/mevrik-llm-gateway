"""
Async HTTP client pool for proxying requests to vLLM backends.

Concurrency model (Python equivalent of Go goroutines + buffered channels):
- One shared httpx.AsyncClient with a large connection pool.
- Per-backend asyncio.Semaphore acts like a Go buffered channel —
  caps in-flight requests so one slow backend can't starve others.
- All I/O is non-blocking via asyncio + httpx (like goroutines on netpoll).
- uvloop (if installed) replaces the default event loop for ~2x throughput.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import httpx

from app.config import settings
from app.registry import ModelEntry

logger = logging.getLogger("llm-gateway")

_client: httpx.AsyncClient | None = None
_semaphores: dict[str, asyncio.Semaphore] = {}

# Per-backend concurrency cap (like a Go buffered channel size)
_MAX_INFLIGHT_PER_BACKEND = 200


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(settings.request_timeout),
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=settings.max_connections,
                max_keepalive_connections=200,
                keepalive_expiry=60,
            ),
            follow_redirects=True,
            http2=True,
        )
    return _client


async def close_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


def _sem(backend_url: str) -> asyncio.Semaphore:
    """Get or create a per-backend semaphore for backpressure."""
    if backend_url not in _semaphores:
        _semaphores[backend_url] = asyncio.Semaphore(_MAX_INFLIGHT_PER_BACKEND)
    return _semaphores[backend_url]


async def forward(model: ModelEntry, path: str, body: dict) -> httpx.Response:
    """Proxy a JSON request to the upstream vLLM backend."""
    client = await get_client()
    url = f"{model.backend_url.rstrip('/')}{path}"
    async with _sem(model.backend_url):
        return await client.post(url, json=body)


async def forward_stream(model: ModelEntry, path: str, body: dict) -> AsyncIterator[bytes]:
    """Proxy a streaming request, yielding raw SSE chunks."""
    client = await get_client()
    url = f"{model.backend_url.rstrip('/')}{path}"
    async with _sem(model.backend_url):
        async with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk
