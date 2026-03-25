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
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from app.config import settings
from app.registry import ModelEntry

logger = logging.getLogger("llm-gateway")

_client: httpx.AsyncClient | None = None
_semaphores: dict[str, asyncio.Semaphore] = {}

# Per-backend concurrency cap (like a Go buffered channel size)
_MAX_INFLIGHT_PER_BACKEND = 200


# ---------------------------------------------------------------------------
# Structured proxy result (avoids raw httpx.Response leaking into routers)
# ---------------------------------------------------------------------------
@dataclass
class ProxyResult:
    """Encapsulates a proxied response with pre-parsed data."""

    status_code: int
    body: Optional[Dict[str, Any]]  # parsed JSON, or None if not JSON
    raw_text: str  # raw response text for debugging
    error: Optional[str] = None  # set if the proxy call itself failed

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300

    @property
    def is_backend_error(self) -> bool:
        """True when the backend responded but with a non-2xx status."""
        return self.error is None and self.status_code >= 400


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------
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
            trust_env=False,
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


# ---------------------------------------------------------------------------
# Core proxy functions
# ---------------------------------------------------------------------------
async def forward(model: ModelEntry, path: str, body: dict) -> httpx.Response:
    """
    Proxy a JSON request to the upstream vLLM backend.
    Returns the raw httpx.Response.  Callers that need safer handling
    should prefer `forward_safe()` instead.
    """
    client = await get_client()
    url = f"{model.backend_url.rstrip('/')}{path}"
    async with _sem(model.backend_url):
        return await client.post(url, json=body)


async def forward_safe(model: ModelEntry, path: str, body: dict) -> ProxyResult:
    """
    Proxy a JSON request and return a structured ProxyResult.

    - Catches network/timeout errors and wraps them.
    - Safely parses JSON (returns raw text if parsing fails).
    - Always returns a ProxyResult — never raises.
    """
    url = f"{model.backend_url.rstrip('/')}{path}"
    logger.debug("→ %s  %s  model=%s", path, url, body.get("model", "?"))

    try:
        client = await get_client()
        async with _sem(model.backend_url):
            resp = await client.post(url, json=body)

    except httpx.ConnectError as exc:
        logger.error("Connection refused [%s] %s: %s", model.id, url, exc)
        return ProxyResult(
            status_code=502,
            body=None,
            raw_text="",
            error=f"Cannot connect to backend at {model.backend_url} — is vLLM running?",
        )

    except httpx.TimeoutException as exc:
        logger.error("Timeout [%s] %s: %s", model.id, url, exc)
        return ProxyResult(
            status_code=504,
            body=None,
            raw_text="",
            error=f"Backend timed out ({settings.request_timeout}s) for model '{model.id}'",
        )

    except httpx.HTTPError as exc:
        logger.error("HTTP error [%s] %s: %s", model.id, url, exc)
        return ProxyResult(
            status_code=502,
            body=None,
            raw_text="",
            error=f"Backend HTTP error: {exc}",
        )

    except Exception as exc:
        logger.exception("Unexpected proxy error [%s] %s", model.id, url)
        return ProxyResult(
            status_code=500,
            body=None,
            raw_text="",
            error=f"Unexpected proxy error: {type(exc).__name__}: {exc}",
        )

    # --- Parse response ---
    raw_text = resp.text
    parsed: Optional[Dict[str, Any]] = None

    try:
        parsed = resp.json()
    except Exception:
        logger.warning(
            "Non-JSON response from [%s] %s (status %d): %.200s",
            model.id, url, resp.status_code, raw_text,
        )

    if resp.status_code >= 400:
        # Extract vLLM error message if available
        detail = ""
        if parsed and isinstance(parsed, dict):
            detail = (
                    parsed.get("message")
                    or parsed.get("detail")
                    or (parsed.get("error", {}).get("message") if isinstance(parsed.get("error"), dict) else "")
                    or raw_text[:500]
            )
        else:
            detail = raw_text[:500]

        logger.warning(
            "Backend error [%s] %s → %d: %s",
            model.id, url, resp.status_code, detail,
        )

    return ProxyResult(
        status_code=resp.status_code,
        body=parsed,
        raw_text=raw_text,
    )


async def forward_stream(model: ModelEntry, path: str, body: dict) -> AsyncIterator[bytes]:
    """Proxy a streaming request, yielding raw SSE chunks."""
    client = await get_client()
    url = f"{model.backend_url.rstrip('/')}{path}"
    async with _sem(model.backend_url):
        async with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk
