from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import loreley.api.routers.archive as archive_router
import loreley.api.routers.commits as commits_router
import loreley.api.routers.jobs as jobs_router
from loreley.api.pagination import PaginationCursorError
from loreley.api.routers.archive import router as archive_api_router
from loreley.api.routers.commits import router as commits_api_router
from loreley.api.routers.jobs import router as jobs_api_router
from loreley.api.services.archive import ArchiveRecordPage
from loreley.api.services.commits import CommitPage
from loreley.api.services.evidence import EvidenceIndicator
from loreley.api.services.jobs import JobPage
from loreley.config import Settings
from loreley.core.candidate_fate import CandidateFate
from loreley.db.models import JobStatus


def _build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(jobs_api_router, prefix="/api/v1")
    app.include_router(commits_api_router, prefix="/api/v1")
    app.include_router(archive_api_router, prefix="/api/v1")
    return TestClient(app)


def _patch_no_evidence(monkeypatch) -> None:
    monkeypatch.setattr(jobs_router, "load_evidence_indicators_by_commit_hash", lambda _hashes: {})
    monkeypatch.setattr(commits_router, "load_evidence_indicators_by_commit_hash", lambda _hashes: {})
    monkeypatch.setattr(archive_router, "load_evidence_indicators_by_commit_hash", lambda _hashes: {})
    monkeypatch.setattr(jobs_router, "load_candidate_fates_for_jobs", lambda _rows: {})
    monkeypatch.setattr(commits_router, "load_candidate_fates_for_commits", lambda _rows: {})


def _commit_row(commit_hash: str = "abc") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        commit_hash=commit_hash,
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


def test_jobs_page_route_returns_items_and_next_cursor(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
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


def test_jobs_page_route_passes_job_kind_filter(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
    captured: dict[str, object] = {}
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
        job_kind="repair",
        result_commit_hash="def",
        ingestion_status=None,
    )

    def _fake_list_jobs_page(**kwargs):
        captured.update(kwargs)
        return JobPage(items=[row], next_cursor=None)

    monkeypatch.setattr(jobs_router, "list_jobs_page", _fake_list_jobs_page)

    client = _build_test_client()
    response = client.get("/api/v1/jobs/page?job_kind=repair")

    assert response.status_code == 200
    assert captured["job_kind"] == "repair"
    assert response.json()["items"][0]["job_kind"] == "repair"


def test_jobs_routes_pass_candidate_fate_and_evidence_filters(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
    captured_page: dict[str, object] = {}
    captured_list: dict[str, object] = {}
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
        job_kind="evolution",
        result_commit_hash="def",
        ingestion_status=None,
    )

    def _fake_list_jobs_page(**kwargs):
        captured_page.update(kwargs)
        return JobPage(items=[row], next_cursor=None)

    def _fake_list_jobs(**kwargs):
        captured_list.update(kwargs)
        return [row]

    monkeypatch.setattr(jobs_router, "list_jobs_page", _fake_list_jobs_page)
    monkeypatch.setattr(jobs_router, "list_jobs", _fake_list_jobs)

    client = _build_test_client()
    page_response = client.get(
        "/api/v1/jobs/page?candidate_fate=elite_inserted&evidence=agent_visible"
    )
    list_response = client.get(
        "/api/v1/jobs?candidate_fate=candidate_failed&evidence=none"
    )

    assert page_response.status_code == 200
    assert captured_page["candidate_fate"] == "elite_inserted"
    assert captured_page["evidence"] == "agent_visible"
    assert list_response.status_code == 200
    assert captured_list["candidate_fate"] == "candidate_failed"
    assert captured_list["evidence"] == "none"


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/jobs/page?candidate_fate=not_a_fate",
        "/api/v1/jobs/page?evidence=hidden",
        "/api/v1/jobs?candidate_fate=not_a_fate",
        "/api/v1/jobs?evidence=hidden",
    ],
)
def test_jobs_routes_reject_bad_filter_values(path: str) -> None:
    response = _build_test_client().get(path)

    assert response.status_code == 422


def test_jobs_page_route_serializes_candidate_fate(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
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
        ingestion_status="succeeded",
    )
    monkeypatch.setattr(
        jobs_router,
        "list_jobs_page",
        lambda **_kwargs: JobPage(items=[row], next_cursor=None),
    )
    monkeypatch.setattr(
        jobs_router,
        "load_candidate_fates_for_jobs",
        lambda _rows: {
            str(row.id): CandidateFate(
                label="elite_inserted",
                reason="Candidate entered an empty archive niche.",
            )
        },
    )

    client = _build_test_client()
    response = client.get("/api/v1/jobs/page")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["candidate_fate_label"] == "elite_inserted"
    assert item["candidate_fate_reason"] == "Candidate entered an empty archive niche."


def test_commits_page_route_returns_items_and_next_cursor(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
    row = _commit_row()
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


def test_commits_page_route_passes_query_filter(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
    captured: dict[str, object] = {}
    row = _commit_row()

    def _fake_list_commits_page(**kwargs):
        captured.update(kwargs)
        return CommitPage(items=[row], next_cursor=None)

    monkeypatch.setattr(commits_router, "list_commits_page", _fake_list_commits_page)

    client = _build_test_client()
    response = client.get("/api/v1/commits/page?query=bugfix")

    assert response.status_code == 200
    assert captured["query"] == "bugfix"


def test_commit_routes_serialize_evidence_indicators(monkeypatch) -> None:
    row = _commit_row()
    indicator = EvidenceIndicator(
        has_evaluation_evidence=True,
        agent_visible_evidence_count=2,
        top_evaluation_diagnosis="p95 latency regressed",
    )

    monkeypatch.setattr(commits_router, "list_commits", lambda **_kwargs: [row])
    monkeypatch.setattr(
        commits_router,
        "list_commits_page",
        lambda **_kwargs: CommitPage(items=[row], next_cursor=None),
    )
    monkeypatch.setattr(commits_router, "get_commit", lambda **_kwargs: row)
    monkeypatch.setattr(commits_router, "list_metrics", lambda **_kwargs: [])
    monkeypatch.setattr(commits_router, "list_evaluation_artifacts_for_commit", lambda **_kwargs: [])
    monkeypatch.setattr(commits_router, "build_agent_feedback_payload", lambda _rows: None)
    monkeypatch.setattr(commits_router, "load_candidate_fates_for_commits", lambda _rows: {})
    monkeypatch.setattr(
        commits_router,
        "load_evidence_indicators_by_commit_hash",
        lambda _hashes: {"abc": indicator},
    )

    client = _build_test_client()

    list_payload = client.get("/api/v1/commits").json()[0]
    page_payload = client.get("/api/v1/commits/page").json()["items"][0]
    detail_payload = client.get("/api/v1/commits/abc").json()

    for payload in (list_payload, page_payload, detail_payload):
        assert payload["has_evaluation_evidence"] is True
        assert payload["agent_visible_evidence_count"] == 2
        assert payload["top_evaluation_diagnosis"] == "p95 latency regressed"


def test_commit_routes_serialize_candidate_fate(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
    row = _commit_row()
    monkeypatch.setattr(commits_router, "list_commits", lambda **_kwargs: [row])
    monkeypatch.setattr(
        commits_router,
        "list_commits_page",
        lambda **_kwargs: CommitPage(items=[row], next_cursor=None),
    )
    monkeypatch.setattr(commits_router, "get_commit", lambda **_kwargs: row)
    monkeypatch.setattr(commits_router, "list_metrics", lambda **_kwargs: [])
    monkeypatch.setattr(commits_router, "list_evaluation_artifacts_for_commit", lambda **_kwargs: [])
    monkeypatch.setattr(commits_router, "build_agent_feedback_payload", lambda _rows: None)
    monkeypatch.setattr(
        commits_router,
        "load_candidate_fates_for_commits",
        lambda _rows: {
            "abc": CandidateFate(
                label="valid_not_elite",
                reason="Candidate passed evaluation but did not enter the archive.",
            )
        },
    )

    client = _build_test_client()

    list_payload = client.get("/api/v1/commits").json()[0]
    page_payload = client.get("/api/v1/commits/page").json()["items"][0]
    detail_payload = client.get("/api/v1/commits/abc").json()

    for payload in (list_payload, page_payload, detail_payload):
        assert payload["candidate_fate_label"] == "valid_not_elite"
        assert payload["candidate_fate_reason"] == "Candidate passed evaluation but did not enter the archive."


def test_archive_records_page_route_returns_items_and_next_cursor(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
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


def test_jobs_page_route_rejects_invalid_cursor(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
    monkeypatch.setattr(
        jobs_router,
        "list_jobs_page",
        lambda **_kwargs: (_ for _ in ()).throw(PaginationCursorError("Jobs cursor is invalid.")),
    )

    client = _build_test_client()
    response = client.get("/api/v1/jobs/page?cursor=bad")

    assert response.status_code == 400
    assert response.json() == {"detail": "Jobs cursor is invalid."}


def test_commits_page_route_rejects_invalid_cursor(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
    monkeypatch.setattr(
        commits_router,
        "list_commits_page",
        lambda **_kwargs: (_ for _ in ()).throw(PaginationCursorError("Commits cursor is invalid.")),
    )

    client = _build_test_client()
    response = client.get("/api/v1/commits/page?cursor=bad")

    assert response.status_code == 400
    assert response.json() == {"detail": "Commits cursor is invalid."}


def test_archive_records_page_route_rejects_invalid_cursor(monkeypatch) -> None:
    _patch_no_evidence(monkeypatch)
    settings = Settings.model_validate({"mapelites_code_embedding_dimensions": 8})
    monkeypatch.setattr(archive_router, "get_settings", lambda: settings)
    monkeypatch.setattr(archive_router, "resolve_default_island_id", lambda _settings: "main")
    monkeypatch.setattr(
        archive_router,
        "list_records_page",
        lambda **_kwargs: (_ for _ in ()).throw(PaginationCursorError("Archive records cursor is invalid.")),
    )

    client = _build_test_client()
    response = client.get("/api/v1/archive/records/page?cursor=bad")

    assert response.status_code == 400
    assert response.json() == {"detail": "Archive records cursor is invalid."}


def test_job_evaluation_artifact_download_returns_404_when_hidden_or_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        jobs_router,
        "get_downloadable_evaluation_artifact",
        lambda **_kwargs: None,
    )

    client = _build_test_client()
    response = client.get(f"/api/v1/jobs/{uuid4()}/evaluation-artifacts/hidden")

    assert response.status_code == 404
    assert response.json() == {"detail": "Evaluation artifact not found."}
