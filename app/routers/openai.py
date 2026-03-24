"""OpenAI-compatible proxy endpoints — forwards to vLLM backends."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from app.deps import resolve_model, verify_api_key
from app.proxy import forward, forward_stream
from app.registry import ModelEntry
from app.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    RerankRequest,
    ScoreRequest,
)

logger = logging.getLogger("llm-gateway")

router = APIRouter(prefix="/v1", tags=["openai"], dependencies=[Depends(verify_api_key)])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------
@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    model, payload = resolve_model(body.model_dump(exclude_none=True))

    if payload.get("stream"):
        return StreamingResponse(
            _safe_stream(model, "/chat/completions", payload),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    resp = await forward(model, "/chat/completions", payload)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------
@router.post("/completions")
async def completions(body: CompletionRequest):
    model, payload = resolve_model(body.model_dump(exclude_none=True))

    if payload.get("stream"):
        return StreamingResponse(
            _safe_stream(model, "/completions", payload),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    resp = await forward(model, "/completions", payload)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
@router.post("/embeddings")
async def embeddings(body: EmbeddingRequest):
    model, payload = resolve_model(body.model_dump(exclude_none=True))

    resp = await forward(model, "/embeddings", payload)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Rerank / Score
# ---------------------------------------------------------------------------
@router.post("/rerank")
async def rerank(body: RerankRequest):
    model, payload = resolve_model(body.model_dump(exclude_none=True))

    resp = await forward(model, "/rerank", payload)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@router.post("/score")
async def score(body: ScoreRequest):
    model, payload = resolve_model(body.model_dump(exclude_none=True))

    resp = await forward(model, "/score", payload)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------
async def _safe_stream(model: ModelEntry, path: str, body: dict):
    try:
        async for chunk in forward_stream(model, path, body):
            yield chunk
    except Exception as exc:
        logger.error("Stream error [%s]: %s", model.id, exc)
        err = json.dumps({"error": {"message": str(exc), "type": "proxy_error"}})
        yield f"data: {err}\n\n".encode()
        yield b"data: [DONE]\n\n"
