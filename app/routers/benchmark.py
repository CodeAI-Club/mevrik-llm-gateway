"""Benchmark & stats endpoints for model performance testing."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.deps import verify_api_key
from app.proxy import forward, forward_stream
from app.registry import ModelEntry, registry
from app.stats import RequestStat, tracker

logger = logging.getLogger("llm-gateway")

router = APIRouter(prefix="/v1", tags=["benchmark"], dependencies=[Depends(verify_api_key)])

_TEST_PROMPT = "Say 'hello' in one sentence."


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TestRequest(BaseModel):
    prompt: str = Field(_TEST_PROMPT, description="Test prompt to send")
    max_tokens: int = Field(50, ge=1, le=4096)


class LoadTestRequest(BaseModel):
    prompt: str = Field(_TEST_PROMPT, description="Prompt for each request")
    max_tokens: int = Field(50, ge=1, le=4096)
    concurrency: int = Field(5, ge=1, le=50, description="Parallel requests")
    total_requests: int = Field(10, ge=1, le=200, description="Total requests to send")


# ---------------------------------------------------------------------------
# 1. Test single model performance
# ---------------------------------------------------------------------------
@router.post("/models/{model_id}/test")
async def test_model(model_id: str, body: TestRequest):
    """Send a single test request and measure latency, TTFB, tokens/sec."""
    entry = registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    payload = {
        "model": entry.backend_model,
        "messages": [{"role": "user", "content": body.prompt}],
        "max_tokens": body.max_tokens,
        "stream": True,
    }

    ttfb_ms = 0.0
    chunks: list[bytes] = []
    t_start = time.perf_counter()
    first_chunk = True

    try:
        async for chunk in forward_stream(entry, "/chat/completions", payload):
            if first_chunk:
                ttfb_ms = (time.perf_counter() - t_start) * 1000
                first_chunk = False
            chunks.append(chunk)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Backend error: {exc}")

    total_ms = (time.perf_counter() - t_start) * 1000
    raw = b"".join(chunks).decode(errors="replace")

    # Parse token count from SSE chunks
    tokens_out = _count_tokens_from_sse(raw)
    tps = (tokens_out / total_ms) * 1000 if total_ms > 0 and tokens_out > 0 else 0

    # Record stat
    stat = RequestStat(
        model_id=model_id,
        tokens_in=len(body.prompt.split()),
        tokens_out=tokens_out,
        latency_ms=round(total_ms, 2),
        ttfb_ms=round(ttfb_ms, 2),
    )
    await tracker.record("__benchmark__", stat)

    return {
        "model": model_id,
        "status": "ok",
        "latency_ms": round(total_ms, 2),
        "ttfb_ms": round(ttfb_ms, 2),
        "tokens_generated": tokens_out,
        "tokens_per_sec": round(tps, 2),
    }


# ---------------------------------------------------------------------------
# 2. Load test model with concurrency
# ---------------------------------------------------------------------------
@router.post("/models/{model_id}/loadtest")
async def load_test_model(model_id: str, body: LoadTestRequest):
    """Fire concurrent requests to stress-test a model backend."""
    entry = registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    sem = asyncio.Semaphore(body.concurrency)
    results: list[dict] = []

    async def _single(idx: int):
        async with sem:
            payload = {
                "model": entry.backend_model,
                "messages": [{"role": "user", "content": body.prompt}],
                "max_tokens": body.max_tokens,
            }
            t0 = time.perf_counter()
            try:
                resp = await forward(entry, "/chat/completions", payload)
                elapsed = (time.perf_counter() - t0) * 1000
                data = resp.json()
                usage = data.get("usage", {})
                tokens_out = usage.get("completion_tokens", 0)

                stat = RequestStat(
                    model_id=model_id,
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=tokens_out,
                    latency_ms=round(elapsed, 2),
                    ttfb_ms=round(elapsed, 2),
                )
                await tracker.record("__loadtest__", stat)

                return {
                    "index": idx,
                    "status": resp.status_code,
                    "latency_ms": round(elapsed, 2),
                    "tokens_out": tokens_out,
                }
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                return {
                    "index": idx,
                    "status": "error",
                    "latency_ms": round(elapsed, 2),
                    "error": str(exc),
                }

    tasks = [_single(i) for i in range(body.total_requests)]
    t_total_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_wall_ms = (time.perf_counter() - t_total_start) * 1000

    successes = [r for r in results if r.get("status") == 200]
    failures = [r for r in results if r.get("status") != 200]
    latencies = sorted(r["latency_ms"] for r in successes) if successes else [0]
    n = len(successes)

    return {
        "model": model_id,
        "config": {
            "concurrency": body.concurrency,
            "total_requests": body.total_requests,
        },
        "wall_time_ms": round(total_wall_ms, 2),
        "succeeded": n,
        "failed": len(failures),
        "avg_latency_ms": round(sum(latencies) / n, 2) if n else 0,
        "min_latency_ms": round(latencies[0], 2) if n else 0,
        "max_latency_ms": round(latencies[-1], 2) if n else 0,
        "p50_latency_ms": round(latencies[n // 2], 2) if n else 0,
        "p95_latency_ms": round(latencies[int(n * 0.95)], 2) if n else 0,
        "requests_per_sec": round((n / total_wall_ms) * 1000, 2) if total_wall_ms > 0 else 0,
        "results": results,
    }


# ---------------------------------------------------------------------------
# 3. Per-user token speed stats
# ---------------------------------------------------------------------------
@router.get("/stats")
async def get_stats(
        user: str = Query("__benchmark__", description="User key to query stats for"),
        model_id: Optional[str] = Query(None, description="Filter by model ID"),
):
    """Get average token speed and latency stats for a user."""
    return {
        "user": user,
        "model_filter": model_id,
        **tracker.summary(user, model_id),
    }


@router.get("/stats/users")
async def list_stat_users():
    """List all user keys that have recorded stats."""
    return {"users": tracker.all_users()}


@router.delete("/stats")
async def clear_stats(
        user: Optional[str] = Query(None, description="Clear stats for specific user, or all if omitted"),
):
    """Clear recorded stats."""
    await tracker.clear(user)
    return {"message": f"Stats cleared for {'all users' if not user else user}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_tokens_from_sse(raw: str) -> int:
    """Best-effort token count from SSE stream chunks."""
    import json as _json

    count = 0
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            obj = _json.loads(data_str)
            # Check usage field first (vLLM includes it in final chunk)
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens"):
                return usage["completion_tokens"]
            # Otherwise count content deltas
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content:
                    count += 1  # approximate: 1 chunk ≈ 1 token
        except (_json.JSONDecodeError, TypeError):
            continue
    return count
