"""Application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.proxy import close_client
from app.registry import registry
from app.routers import health, models, openai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("llm-gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LLM Gateway starting — %d model(s)", len(registry.list_all()))
    for m in registry.list_all():
        logger.info("  %-25s → %s  [%s]", m.id, m.backend_url, m.model_type)
    yield
    await close_client()
    logger.info("LLM Gateway stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Gateway",
        version="1.0.0",
        description="OpenAI-compatible gateway for vLLM backends",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(openai.router)

    @app.get("/")
    async def root():
        return {
            "service": "LLM Gateway",
            "version": "1.0.0",
            "docs": "/docs",
        }

    return app


app = create_app()
