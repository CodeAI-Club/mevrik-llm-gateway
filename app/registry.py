"""Model registry for managing LLM configurations.

Thread-safe via asyncio.Lock. All mutations persist to disk immediately
using atomic file replacements to support multiple Uvicorn workers securely.
"""

from __future__ import annotations

import json
import logging
import os
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Dict

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger("llm-gateway.registry")


# ---------------------------------------------------------------------------
# Schema (Restored inside registry.py)
# ---------------------------------------------------------------------------
class ModelEntry(BaseModel):
    id: str = Field(..., description="Unique model identifier clients will use")
    name: str = Field("", description="Human-readable display name")
    backend_url: str = Field(..., description="vLLM base URL, e.g. http://host:8010/v1")
    backend_model: str = Field(..., description="Actual model name on vLLM")
    model_type: str = Field(
        "chat", description="chat | completion | embedding | rerank"
    )
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
    def __init__(self, data_dir: str = "data", filename: str = "models.json"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._data_dir / filename

        # Async lock for thread/task safety within the current worker
        self._lock = asyncio.Lock()

        # In-memory cache
        self._models: Dict[str, ModelEntry] = {}
        self._last_mtime: float = 0.0

        # Initial boot load
        self._load()

    def _load(self, force: bool = False) -> None:
        """
        Loads models from disk.
        Uses file modification time to skip unnecessary reads unless forced.
        """
        if not self._filepath.exists():
            return

        try:
            mtime = self._filepath.stat().st_mtime
            if force or mtime > self._last_mtime:
                with open(self._filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._models = {item["id"]: ModelEntry(**item) for item in data}
                self._last_mtime = mtime
                logger.debug("Loaded %d models from disk.", len(self._models))
        except Exception as e:
            logger.error("Failed to load models from disk: %s", e)

    def _save(self) -> None:
        """
        Saves models to disk using an atomic write to prevent file corruption.
        """
        temp_filepath = self._filepath.with_suffix(".tmp")
        try:
            # Dump to a temporary file first
            data = [
                model.model_dump(exclude_none=True) for model in self._models.values()
            ]
            with open(temp_filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Atomic swap (replaces the old file with the new temp file instantly)
            os.replace(temp_filepath, self._filepath)

            # Update the last modified time so this worker doesn't unnecessarily reload
            self._last_mtime = self._filepath.stat().st_mtime

        except Exception as e:
            logger.error("Failed to save models to disk: %s", e)
            if temp_filepath.exists():
                os.remove(temp_filepath)
            raise

    # --- Read Operations ---

    def get(self, model_id: str) -> Optional[ModelEntry]:
        """Retrieve a specific model by ID."""
        self._load()  # Sync cache if file was modified by another worker
        return self._models.get(model_id)

    def list_all(self) -> List[ModelEntry]:
        """List all available models."""
        self._load()  # Sync cache if file was modified by another worker
        return list(self._models.values())

    # --- Write Operations ---

    async def add(self, data: ModelCreate) -> ModelEntry:
        """Add a new model to the registry."""
        async with self._lock:
            self._load(
                force=True
            )  # Strictly sync to prevent overwriting other worker's changes

            if data.id in self._models:
                raise ValueError(f"Model '{data.id}' already exists")

            entry = ModelEntry(**data.model_dump())
            self._models[entry.id] = entry
            self._save()

            logger.info("Added model: %s -> %s", entry.id, entry.backend_url)
            return entry

    async def update(self, model_id: str, data: ModelUpdate) -> ModelEntry:
        """Update an existing model's properties."""
        async with self._lock:
            self._load(force=True)

            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not found")

            current = self._models[model_id]

            # Use exclude_unset=True to only update provided fields
            updates = data.model_dump(exclude_unset=True)
            updated = current.model_copy(update=updates)

            self._models[model_id] = updated
            self._save()

            logger.info("Updated model: %s", model_id)
            return updated

    async def delete(self, model_id: str) -> ModelEntry:
        """Remove a model from the registry."""
        async with self._lock:
            self._load(force=True)

            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not found")

            entry = self._models.pop(model_id)
            self._save()

            logger.info("Deleted model: %s", model_id)
            return entry


# Global singleton instance
# Defaulting to "data" folder in case settings.data_dir isn't explicitly defined
registry = ModelRegistry(
    data_dir=getattr(settings, "data_dir", "data"), filename="models.json"
)
