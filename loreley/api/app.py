"""FastAPI application factory for Loreley read-only UI API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from loreley.api.routers.health import router as health_router
from loreley.api.routers.archive import router as archive_router
from loreley.api.routers.instance import router as instance_router
from loreley.api.routers.jobs import router as jobs_router
from loreley.api.routers.commits import router as commits_router
from loreley.api.routers.logs import router as logs_router
from loreley.api.routers.graphs import router as graphs_router
from loreley.db.base import ensure_database_schema

API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """FastAPI lifespan that validates DB schema/instance marker on startup."""
    ensure_database_schema()
    yield


def create_app() -> FastAPI:
    """Create the FastAPI application instance."""

    app = FastAPI(
        title="Loreley UI API",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.include_router(health_router, prefix=API_V1_PREFIX, tags=["health"])
    app.include_router(instance_router, prefix=API_V1_PREFIX, tags=["instance"])
    app.include_router(archive_router, prefix=API_V1_PREFIX, tags=["archive"])
    app.include_router(jobs_router, prefix=API_V1_PREFIX, tags=["jobs"])
    app.include_router(commits_router, prefix=API_V1_PREFIX, tags=["commits"])
    app.include_router(logs_router, prefix=API_V1_PREFIX, tags=["logs"])
    app.include_router(graphs_router, prefix=API_V1_PREFIX, tags=["graphs"])
    return app


# Uvicorn default import target: `uvicorn loreley.api.app:app`
app = create_app()


