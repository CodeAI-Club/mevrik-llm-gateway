"""Application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.proxy import close_client
from app.registry import registry
from app.routers import benchmark, health, models, openai
from app.stats import tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("llm-gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "%s v%s starting — %d model(s)",
        settings.service_name,
        settings.service_version,
        len(registry.list_all()),
    )
    for m in registry.list_all():
        logger.info("  %-25s → %s  [%s]", m.id, m.backend_url, m.model_type)
    tracker.start_flush_loop()
    yield
    await tracker.save()
    await close_client()
    logger.info("%s stopped", settings.service_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        description=settings.service_description,
        root_path=settings.root_path,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(openai.router)
    app.include_router(benchmark.router)

    @app.get("/")
    async def root():
        return {
            "service": settings.service_name,
            "version": settings.service_version,
            "docs": "/docs",
        }

    return app


app = create_app()
