"""Shared FastAPI dependencies."""

from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from app.registry import ModelEntry, registry

# auto_error=False → missing header returns None instead of 403
_bearer = HTTPBearer(auto_error=False)


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
    """Look up model by id, rewrite model field to backend name."""
    model_id = body.get("model", "")
    entry = registry.get(model_id)
    if entry is None:
        available = [m.id for m in registry.list_all()]
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: {available}",
        )
    return entry, {**body, "model": entry.backend_model}
