"""FastAPI application factory for the Loreley UI API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from loreley.api.agent_errors import (
    AgentAPIError,
    agent_api_error_handler,
    agent_validation_exception_handler,
)
from loreley.api.routers.agent import router as agent_router
from loreley.api.routers.health import router as health_router
from loreley.api.routers.archive import router as archive_router
from loreley.api.routers.instance import router as instance_router
from loreley.api.routers.jobs import router as jobs_router
from loreley.api.routers.commits import router as commits_router
from loreley.api.routers.logs import router as logs_router
from loreley.api.routers.graphs import router as graphs_router
from loreley.api.routers.operator import router as operator_router
from loreley.api.routers.repair import router as repair_router
from loreley.api.routers.usage import router as usage_router
from loreley.api.services.operator import mark_stale_baseline_ensure_tasks_failed
from loreley.db.base import INSTANCE_SCHEMA_VERSION, ensure_database_schema, session_scope
from loreley.db.instance import validate_instance_marker_schema

API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """FastAPI lifespan that validates DB schema/instance marker on startup."""
    ensure_database_schema(validate_marker=False)
    with session_scope() as session:
        validate_instance_marker_schema(
            session=session,
            schema_version=INSTANCE_SCHEMA_VERSION,
        )
    mark_stale_baseline_ensure_tasks_failed()
    yield


def create_app() -> FastAPI:
    """Create the FastAPI application instance."""

    app = FastAPI(
        title="Loreley UI API",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.add_exception_handler(AgentAPIError, agent_api_error_handler)
    app.add_exception_handler(RequestValidationError, agent_validation_exception_handler)

    app.include_router(health_router, prefix=API_V1_PREFIX, tags=["health"])
    app.include_router(instance_router, prefix=API_V1_PREFIX, tags=["instance"])
    app.include_router(archive_router, prefix=API_V1_PREFIX, tags=["archive"])
    app.include_router(jobs_router, prefix=API_V1_PREFIX, tags=["jobs"])
    app.include_router(commits_router, prefix=API_V1_PREFIX, tags=["commits"])
    app.include_router(logs_router, prefix=API_V1_PREFIX, tags=["logs"])
    app.include_router(graphs_router, prefix=API_V1_PREFIX, tags=["graphs"])
    app.include_router(operator_router, prefix=API_V1_PREFIX, tags=["operator"])
    app.include_router(repair_router, prefix=API_V1_PREFIX, tags=["repair"])
    app.include_router(agent_router, prefix=API_V1_PREFIX, tags=["agent"])
    app.include_router(usage_router, prefix=API_V1_PREFIX, tags=["usage"])
    return app


# Uvicorn default import target: `uvicorn loreley.api.app:app`
app = create_app()
