"""Health check router."""

from fastapi import APIRouter

from app.registry import registry

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": len(registry.list_all()),
    }
