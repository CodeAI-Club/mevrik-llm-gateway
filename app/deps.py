"""Shared FastAPI dependencies."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.registry import ModelEntry, registry

logger = logging.getLogger("llm-gateway")

# auto_error=False → missing header returns None instead of 403
_bearer = HTTPBearer(auto_error=False)

# Map endpoint paths to expected model_type(s)
_ENDPOINT_TYPES: dict[str, set[str]] = {
    "/chat/completions": {"chat"},
    "/completions": {"completion", "chat"},
    "/embeddings": {"embedding"},
    "/rerank": {"rerank"},
    "/score": {"rerank", "score"},
}


def verify_api_key(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
):
    """Validate Bearer token against API_KEY from .env."""
    if not settings.api_key:
        return  # no key configured = open access
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def resolve_model(body: dict) -> tuple[ModelEntry, dict]:
    """
    Look up model by id, rewrite model field to backend name.

    Returns (ModelEntry, rewritten_body) where body["model"] is replaced
    with the actual model name the vLLM backend expects.
    """
    model_id = body.get("model", "")
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail="Request body must include a 'model' field",
        )

    entry = registry.get(model_id)
    if entry is None:
        available = [m.id for m in registry.list_all()]
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_id}' not found",
                    "type": "invalid_request_error",
                    "available_models": available,
                }
            },
        )

    # Rewrite model to the backend's expected name
    rewritten = {**body, "model": entry.backend_model}
    return entry, rewritten
