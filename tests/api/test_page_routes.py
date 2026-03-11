from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

import loreley.api.routers.archive as archive_router
import loreley.api.routers.commits as commits_router
import loreley.api.routers.jobs as jobs_router
from loreley.api.routers.archive import router as archive_api_router
from loreley.api.routers.commits import router as commits_api_router
from loreley.api.routers.jobs import router as jobs_api_router
from loreley.api.services.archive import ArchiveRecordPage
from loreley.api.services.commits import CommitPage
from loreley.api.services.jobs import JobPage
from loreley.config import Settings
from loreley.db.models import JobStatus


def _build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(jobs_api_router, prefix="/api/v1")
    app.include_router(commits_api_router, prefix="/api/v1")
    app.include_router(archive_api_router, prefix="/api/v1")
    return TestClient(app)


def test_jobs_page_route_returns_items_and_next_cursor(monkeypatch) -> None:
    row = SimpleNamespace(
        id=uuid4(),
        status=JobStatus.SUCCEEDED,
        priority=0,
        island_id="main",
        base_commit_hash="abc",
        scheduled_at=None,
        started_at=None,
        completed_at=datetime(2026, 3, 11, tzinfo=timezone.utc),
        last_error=None,
        is_seed_job=False,
        result_commit_hash="def",
        ingestion_status=None,
    )
    monkeypatch.setattr(
        jobs_router,
        "list_jobs_page",
        lambda **_kwargs: JobPage(items=[row], next_cursor="next-job"),
    )

    client = _build_test_client()
    response = client.get("/api/v1/jobs/page")

    assert response.status_code == 200
    assert response.json()["next_cursor"] == "next-job"
    assert response.json()["items"][0]["id"] == str(row.id)


def test_commits_page_route_returns_items_and_next_cursor(monkeypatch) -> None:
    row = SimpleNamespace(
        id=uuid4(),
        commit_hash="abc",
        parent_commit_hash=None,
        island_id="main",
        job_id=None,
        author="bot",
        subject="Subject",
        change_summary="Summary",
        evaluation_summary=None,
        tags=[],
        key_files=[],
        highlights=[],
        created_at=datetime(2026, 3, 11, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        commits_router,
        "list_commits_page",
        lambda **_kwargs: CommitPage(items=[row], next_cursor="next-commit"),
    )

    client = _build_test_client()
    response = client.get("/api/v1/commits/page")

    assert response.status_code == 200
    assert response.json()["next_cursor"] == "next-commit"
    assert response.json()["items"][0]["commit_hash"] == "abc"


def test_archive_records_page_route_returns_items_and_next_cursor(monkeypatch) -> None:
    settings = Settings.model_validate({"mapelites_code_embedding_dimensions": 8})
    monkeypatch.setattr(archive_router, "get_settings", lambda: settings)
    monkeypatch.setattr(archive_router, "resolve_default_island_id", lambda _settings: "main")
    monkeypatch.setattr(
        archive_router,
        "list_records_page",
        lambda **_kwargs: ArchiveRecordPage(
            items=[
                archive_router.ArchiveRecordOut(
                    commit_hash="abc",
                    island_id="main",
                    cell_index=1,
                    fitness=1.0,
                    objective=1.0,
                    metric_value=1.0,
                    metric_name="score",
                    higher_is_better=True,
                    measures=[0.1, 0.2],
                    solution=[0.1, 0.2],
                    timestamp=10.0,
                )
            ],
            next_cursor="next-record",
        ),
    )

    client = _build_test_client()
    response = client.get("/api/v1/archive/records/page")

    assert response.status_code == 200
    assert response.json()["next_cursor"] == "next-record"
    assert response.json()["items"][0]["commit_hash"] == "abc"
