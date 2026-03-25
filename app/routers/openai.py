"""OpenAI-compatible proxy endpoints — forwards to vLLM backends.

Pure pass-through proxy. Every endpoint:
 1. Resolves the gateway model-id → backend entry + rewritten payload.
 2. Validates the model type matches the endpoint.
 3. Proxies the request via forward_safe (non-streaming) or
    forward_stream (streaming SSE).
 4. Records per-user stats (latency, tokens, throughput).
 5. Returns the backend response as-is — errors included.

Note: /rerank and /score are handled by app.routers.rerank.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.deps import resolve_model, verify_api_key
from app.proxy import ProxyResult, forward_safe, forward_stream
from app.registry import ModelEntry
from app.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
)
from app.stats import RequestStat, tracker

logger = logging.getLogger("llm-gateway")

router = APIRouter(prefix="/v1", tags=["openai"], dependencies=[Depends(verify_api_key)])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _user_key(request: Request) -> str:
    """Extract a user identifier from the Authorization header for stats."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        return f"key_...{token[-8:]}" if len(token) > 8 else token
    return "__anonymous__"


# Keywords that indicate the caller sent an input too large for the backend
_BATCH_SIZE_HINTS = ("too large", "batch size", "max_num_batched_tokens", "too long", "exceed")


def _error_response(result: ProxyResult) -> JSONResponse:
    """Build a JSON error response from a failed ProxyResult."""
    raw_msg = ""
    if result.body and isinstance(result.body, dict):
        err_obj = result.body.get("error", result.body)
        if isinstance(err_obj, dict):
            raw_msg = err_obj.get("message", "")
        elif isinstance(err_obj, str):
            raw_msg = err_obj
        if not raw_msg:
            raw_msg = result.body.get("message", "") or result.body.get("detail", "")
    if not raw_msg:
        raw_msg = result.error or result.raw_text[:500]

    msg_lower = raw_msg.lower()
    if any(hint in msg_lower for hint in _BATCH_SIZE_HINTS):
        enriched = {
            "error": {
                "message": raw_msg,
                "type": "input_too_large",
                "code": result.status_code,
                "hint": (
                    "The input exceeds the backend's physical batch size limit. "
                    "Reduce your chunk/split size before sending to the gateway, "
                    "or increase --max-num-batched-tokens on the vLLM backend."
                ),
            }
        }
        return JSONResponse(content=enriched, status_code=result.status_code)

    if result.body is not None:
        return JSONResponse(content=result.body, status_code=result.status_code)

    return JSONResponse(
        content={
            "error": {
                "message": result.error or f"Backend returned {result.status_code}",
                "type": "proxy_error",
                "code": result.status_code,
            }
        },
        status_code=result.status_code,
    )


def _extract_usage(data: dict | None) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from a response body."""
    if not data or not isinstance(data, dict):
        return 0, 0
    usage = data.get("usage") or {}
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


async def _record_stat(
        user_key: str,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        ttfb_ms: float = 0.0,
):
    """Fire-and-forget stat recording."""
    try:
        stat = RequestStat(
            model_id=model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=round(latency_ms, 2),
            ttfb_ms=round(ttfb_ms or latency_ms, 2),
        )
        await tracker.record(user_key, stat)
    except Exception as exc:
        logger.debug("Stats record failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Non-streaming proxy
# ---------------------------------------------------------------------------
async def _proxy_json(
        request: Request,
        model: ModelEntry,
        path: str,
        payload: dict,
) -> JSONResponse:
    """Proxy a non-streaming request, record stats, return response as-is."""
    user = _user_key(request)
    t0 = time.perf_counter()

    result = await forward_safe(model, path, payload)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not result.ok:
        logger.warning(
            "Proxy error [%s] %s → %d (%s) %.0fms",
            model.id, path, result.status_code,
            result.error or "backend error", elapsed_ms,
        )
        return _error_response(result)

    tokens_in, tokens_out = _extract_usage(result.body)
    await _record_stat(user, model.id, tokens_in, tokens_out, elapsed_ms)

    return JSONResponse(content=result.body, status_code=result.status_code)


# ---------------------------------------------------------------------------
# Streaming proxy
# ---------------------------------------------------------------------------
async def _proxy_stream(
        request: Request,
        model: ModelEntry,
        path: str,
        payload: dict,
):
    """Proxy a streaming SSE request, wrap errors into SSE events."""
    user = _user_key(request)
    t0 = time.perf_counter()
    first_chunk = True
    ttfb_ms = 0.0
    chunk_count = 0

    try:
        async for chunk in forward_stream(model, path, payload):
            if first_chunk:
                ttfb_ms = (time.perf_counter() - t0) * 1000
                first_chunk = False
            chunk_count += 1
            yield chunk

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "Stream error [%s] %s after %.0fms (%d chunks): %s",
            model.id, path, elapsed_ms, chunk_count, exc,
        )
        err = json.dumps({
            "error": {
                "message": str(exc),
                "type": "proxy_error",
                "code": 502,
            }
        })
        yield f"data: {err}\n\n".encode()
        yield b"data: [DONE]\n\n"
        return

    elapsed_ms = (time.perf_counter() - t0) * 1000
    await _record_stat(user, model.id, 0, chunk_count, elapsed_ms, ttfb_ms)


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------
@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    model, payload = resolve_model(
        body.model_dump(exclude_none=True),
        allowed_types={"chat"},
    )

    if payload.get("stream"):
        return StreamingResponse(
            _proxy_stream(request, model, "/chat/completions", payload),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    return await _proxy_json(request, model, "/chat/completions", payload)


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------
@router.post("/completions")
async def completions(body: CompletionRequest, request: Request):
    model, payload = resolve_model(
        body.model_dump(exclude_none=True),
        allowed_types={"completion", "chat"},
    )

    if payload.get("stream"):
        return StreamingResponse(
            _proxy_stream(request, model, "/completions", payload),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    return await _proxy_json(request, model, "/completions", payload)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
@router.post("/embeddings")
async def embeddings(body: EmbeddingRequest, request: Request):
    model, payload = resolve_model(
        body.model_dump(exclude_none=True),
        allowed_types={"embedding"},
    )
    return await _proxy_json(request, model, "/embeddings", payload)
