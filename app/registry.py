"""
Model registry — loads from models.json, supports runtime CRUD.

Thread-safe via asyncio.Lock. All mutations persist to disk immediately.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger("llm-gateway")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class ModelEntry(BaseModel):
    id: str = Field(..., description="Unique model identifier clients will use")
    name: str = Field("", description="Human-readable display name")
    backend_url: str = Field(..., description="vLLM base URL, e.g. http://host:8010/v1")
    backend_model: str = Field(..., description="Actual model name on vLLM")
    model_type: str = Field("chat", description="chat | completion | embedding | rerank")
    owned_by: str = Field("organization")
    created_at: int = Field(default_factory=lambda: int(time.time()))


class ModelCreate(BaseModel):
    id: str
    name: str = ""
    backend_url: str
    backend_model: str
    model_type: str = "chat"
    owned_by: str = "organization"


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    backend_url: Optional[str] = None
    backend_model: Optional[str] = None
    model_type: Optional[str] = None
    owned_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry singleton
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self, path: str):
        self._path = Path(path)
        self._models: dict[str, ModelEntry] = {}
        self._lock = asyncio.Lock()
        self._load()

    # -- persistence --

    def _load(self):
        if not self._path.exists():
            logger.info("models.json not found at %s — creating empty file", self._path)
            self._path.write_text("[]", encoding="utf-8")
            self._models = {}
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        for item in raw:
            entry = ModelEntry(**item)
            self._models[entry.id] = entry
        logger.info("Loaded %d model(s) from %s", len(self._models), self._path)

    def _save(self):
        data = [m.model_dump() for m in self._models.values()]
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- read --

    def list_all(self) -> list[ModelEntry]:
        return list(self._models.values())

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.get(model_id)

    # -- write (all require the lock from the caller) --

    async def add(self, data: ModelCreate) -> ModelEntry:
        async with self._lock:
            if data.id in self._models:
                raise ValueError(f"Model '{data.id}' already exists")
            entry = ModelEntry(**data.model_dump())
            self._models[entry.id] = entry
            self._save()
            logger.info("Added model: %s → %s", entry.id, entry.backend_url)
            return entry

    async def update(self, model_id: str, data: ModelUpdate) -> ModelEntry:
        async with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not found")
            current = self._models[model_id]
            updates = data.model_dump(exclude_none=True)
            updated = current.model_copy(update=updates)
            self._models[model_id] = updated
            self._save()
            logger.info("Updated model: %s", model_id)
            return updated

    async def delete(self, model_id: str) -> ModelEntry:
        async with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not found")
            entry = self._models.pop(model_id)
            self._save()
            logger.info("Deleted model: %s", model_id)
            return entry


# Global instance
registry = ModelRegistry(settings.models_path)
