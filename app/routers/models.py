"""Model management — CRUD endpoints that persist to models.json."""

import time

from fastapi import APIRouter, Depends, HTTPException

from app.deps import verify_api_key
from app.registry import ModelCreate, ModelUpdate, registry

router = APIRouter(
    prefix="/v1/models", tags=["models"], dependencies=[Depends(verify_api_key)]
)


@router.get("")
async def list_models():
    """OpenAI-compatible GET /v1/models."""
    data = []
    for m in registry.list_all():
        data.append(
            {
                "id": m.id,
                "object": "model",
                "created": m.created_at,
                "owned_by": m.owned_by,
                "name": m.name,
                "model_type": m.model_type,
                "backend_url": m.backend_url,
            }
        )
    return {"object": "list", "data": data}


@router.get("/{model_id}")
async def get_model(model_id: str):
    entry = registry.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {
        "id": entry.id,
        "object": "model",
        "created": entry.created_at,
        "owned_by": entry.owned_by,
        "name": entry.name,
        "model_type": entry.model_type,
        "backend_url": entry.backend_url,
    }


@router.post("", status_code=201)
async def add_model(body: ModelCreate):
    """Add a new model at runtime — saved to models.json immediately."""
    try:
        entry = await registry.add(body)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"message": f"Model '{entry.id}' added", "model": entry.model_dump()}


@router.patch("/{model_id}")
async def update_model(model_id: str, body: ModelUpdate):
    """Update an existing model's fields."""
    try:
        entry = await registry.update(model_id, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"message": f"Model '{entry.id}' updated", "model": entry.model_dump()}


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Remove a model — takes effect immediately."""
    try:
        entry = await registry.delete(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"message": f"Model '{entry.id}' deleted"}
