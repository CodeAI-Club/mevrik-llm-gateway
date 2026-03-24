"""Async HTTP client for proxying requests to vLLM backends."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx

from app.config import settings
from app.registry import ModelEntry

logger = logging.getLogger("llm-gateway")

_client: httpx.AsyncClient | None = None


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
                max_keepalive_connections=100,
                keepalive_expiry=30,
            ),
            follow_redirects=True,
        )
    return _client


async def close_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


async def forward(model: ModelEntry, path: str, body: dict) -> httpx.Response:
    """Proxy a JSON request to the upstream vLLM backend."""
    client = await get_client()
    url = f"{model.backend_url.rstrip('/')}{path}"
    return await client.post(url, json=body)


async def forward_stream(model: ModelEntry, path: str, body: dict) -> AsyncIterator[bytes]:
    """Proxy a streaming request, yielding raw SSE chunks."""
    client = await get_client()
    url = f"{model.backend_url.rstrip('/')}{path}"
    async with client.stream("POST", url, json=body) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            yield chunk
