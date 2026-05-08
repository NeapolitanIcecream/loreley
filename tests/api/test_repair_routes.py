from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import loreley.api.routers.repair as repair_router
import loreley.api.services.repair as repair_service
from loreley.api.routers.repair import router as repair_api_router
from loreley.api.services.repair import (
    RepairConflictError,
    RepairNotFoundError,
    RepairPoolPage,
)
from loreley.db.models import CandidateCommit


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(repair_api_router, prefix="/api/v1")
    return TestClient(app)


def _candidate_payload(candidate_id=None) -> dict[str, object]:
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    return {
        "id": candidate_id or uuid4(),
        "commit_hash": "c" * 40,
        "git_parent_commit_hash": "p" * 40,
        "nearest_viable_ancestor_hash": "v" * 40,
        "island_id": "main",
        "produced_by_job_id": uuid4(),
        "job_kind": "evolution",
        "repair_source_candidate_id": None,
        "campaign_program_hash": "a" * 64,
        "publication_status": "published",
        "evaluation_status": "candidate_failed",
        "archive_status": "not_considered",
        "lifecycle_status": "active",
        "failure_stage": "evaluation",
        "failure_kind": "test_failed",
        "failure_summary": "tests failed",
        "repair_state": "eligible",
        "failed_depth": 0,
        "repair_attempts": 0,
        "last_repair_job_id": None,
        "last_repair_job_status": None,
        "active_repair_job_id": None,
        "active_repair_job_status": None,
        "diagnostic_policy_passed": True,
        "diagnostic_summary": "pytest failed",
        "diagnostic_omitted_reasons": [],
        "created_at": now,
        "updated_at": now,
    }


def test_repair_pool_route_returns_candidates(monkeypatch) -> None:
    candidate_id = uuid4()

    def _list_repair_pool_page(**kwargs):
        assert kwargs["repair_state"] == "eligible"
        return RepairPoolPage(
            items=[_candidate_payload(candidate_id)],
            next_cursor="next",
            summary={
                "total_failed_candidates": 1,
                "active_repair_jobs": 0,
                "by_repair_state": {"eligible": 1},
                "by_lifecycle_status": {"active": 1},
                "by_failure_kind": {"test_failed": 1},
            },
        )

    monkeypatch.setattr(repair_router, "list_repair_pool_page", _list_repair_pool_page)

    response = _client().get("/api/v1/repair/pool?repair_state=eligible")

    assert response.status_code == 200
    assert response.json()["items"][0]["id"] == str(candidate_id)
    assert response.json()["summary"]["by_repair_state"] == {"eligible": 1}


def test_repair_schedule_one_route_returns_scheduled_job(monkeypatch) -> None:
    job_id = uuid4()
    source_id = uuid4()
    monkeypatch.setattr(
        repair_router,
        "schedule_one_repair",
        lambda: {
            "scheduled": True,
            "job_id": job_id,
            "repair_source_candidate_id": source_id,
            "base_commit_hash": "base",
            "message": "Repair job scheduled.",
        },
    )

    response = _client().post("/api/v1/repair/schedule-one", json={})

    assert response.status_code == 200
    assert response.json()["job_id"] == str(job_id)


@pytest.mark.parametrize(
    ("field_name", "blocked_value"),
    [
        ("failed_candidate_repair_enabled", False),
        ("failed_candidate_repair_max_jobs_per_tick", 0),
        ("failed_candidate_repair_max_active_jobs", 0),
        ("failed_candidate_repair_max_tokens", 0),
    ],
)
def test_repair_schedule_one_api_guard_blocks_disabled_repair_caps(
    monkeypatch,
    settings,
    field_name: str,
    blocked_value: object,
) -> None:
    """Regression: schedule-one must not bypass disabled scheduler repair caps."""

    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    settings.failed_candidate_repair_max_active_jobs = 1
    setattr(settings, field_name, blocked_value)

    class _SamplerShouldNotBeCreated:
        def __init__(self, **_kwargs):
            raise AssertionError("API guard should block before constructing the sampler")

    monkeypatch.setattr(
        repair_service,
        "FailedCandidateRepairSampler",
        _SamplerShouldNotBeCreated,
    )

    payload = repair_service.schedule_one_repair(settings=settings)

    assert payload["scheduled"] is False
    assert payload["job_id"] is None


def test_repair_schedule_one_api_guard_blocks_exhausted_token_budget(
    monkeypatch,
    settings,
) -> None:
    """Regression: manual repair scheduling must not bypass repair tokens."""

    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    settings.failed_candidate_repair_max_active_jobs = 1
    settings.failed_candidate_repair_max_tokens = 1

    class _SamplerShouldNotSchedule:
        def count_active_repair_jobs(self) -> int:
            return 0

        def schedule_one(self):
            raise AssertionError("API guard should not schedule without a repair token")

    monkeypatch.setattr(
        repair_service,
        "FailedCandidateRepairSampler",
        lambda **_kwargs: _SamplerShouldNotSchedule(),
    )
    monkeypatch.setattr(
        repair_service,
        "_with_manual_repair_schedule_lock",
        lambda **kwargs: kwargs["callback"](),
    )
    monkeypatch.setattr(repair_service, "_manual_repair_tokens_available", lambda **_kwargs: 0)

    payload = repair_service.schedule_one_repair(settings=settings)

    assert payload["scheduled"] is False
    assert payload["job_id"] is None
    assert "token" in str(payload["message"]).lower()


def test_repair_schedule_one_api_guard_serializes_checks_and_schedule(
    monkeypatch,
    settings,
) -> None:
    """Regression: cap/token checks and schedule-one must share one API lock."""

    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    settings.failed_candidate_repair_max_active_jobs = 1
    settings.failed_candidate_repair_max_tokens = 1
    events: list[str] = []
    job_id = uuid4()
    source_id = uuid4()

    def _lock(**kwargs):
        events.append("lock.enter")
        try:
            return kwargs["callback"]()
        finally:
            events.append("lock.exit")

    class _Sampler:
        def count_active_repair_jobs(self) -> int:
            assert events == ["lock.enter"]
            events.append("active.count")
            return 0

        def schedule_one(self):
            assert events == ["lock.enter", "active.count", "tokens.count"]
            events.append("schedule.one")
            return SimpleNamespace(
                job_id=job_id,
                repair_source_candidate_id=source_id,
                base_commit_hash="base",
            )

    def _tokens(**_kwargs):
        assert events == ["lock.enter", "active.count"]
        events.append("tokens.count")
        return 1

    monkeypatch.setattr(repair_service, "_with_manual_repair_schedule_lock", _lock)
    monkeypatch.setattr(repair_service, "_manual_repair_tokens_available", _tokens)
    monkeypatch.setattr(
        repair_service,
        "FailedCandidateRepairSampler",
        lambda **_kwargs: _Sampler(),
    )

    payload = repair_service.schedule_one_repair(settings=settings)

    assert payload["scheduled"] is True
    assert payload["job_id"] == job_id
    assert events == [
        "lock.enter",
        "active.count",
        "tokens.count",
        "schedule.one",
        "lock.exit",
    ]


def test_repair_schedule_one_api_guard_blocks_when_active_jobs_at_cap(
    monkeypatch,
    settings,
) -> None:
    """Regression: repeated API calls must respect the scheduler active repair cap."""

    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    settings.failed_candidate_repair_max_active_jobs = 1

    class _CappedSampler:
        def count_active_repair_jobs(self) -> int:
            return 1

        def schedule_one(self):
            raise AssertionError("API guard should not schedule when active jobs are capped")

    monkeypatch.setattr(
        repair_service,
        "FailedCandidateRepairSampler",
        lambda **_kwargs: _CappedSampler(),
    )
    monkeypatch.setattr(
        repair_service,
        "_with_manual_repair_schedule_lock",
        lambda **kwargs: kwargs["callback"](),
    )

    payload = repair_service.schedule_one_repair(settings=settings)

    assert payload["scheduled"] is False
    assert payload["job_id"] is None


def test_repair_candidate_action_conflict_returns_409(monkeypatch) -> None:
    def _conflict(**_kwargs):
        raise RepairConflictError("Candidate has an active repair job.")

    monkeypatch.setattr(repair_router, "update_candidate_operator_state", _conflict)

    response = _client().post(f"/api/v1/repair/candidates/{uuid4()}/discard", json={})

    assert response.status_code == 409


def test_restore_sets_candidate_active_audit_only(monkeypatch) -> None:
    candidate_id = uuid4()
    candidate = SimpleNamespace(
        id=candidate_id,
        commit_hash="c" * 40,
        git_parent_commit_hash="p" * 40,
        nearest_viable_ancestor_hash="v" * 40,
        island_id="main",
        produced_by_job_id=None,
        job_kind="evolution",
        repair_source_candidate_id=None,
        campaign_program_hash=None,
        publication_status="published",
        evaluation_status="candidate_failed",
        archive_status="not_considered",
        lifecycle_status="discarded",
        failure_stage="evaluation",
        failure_kind="test_failed",
        failure_summary="failed",
        repair_state="eligible",
        failed_depth=0,
        repair_attempts=0,
        last_repair_job_id=None,
        failure_evidence_id=None,
        created_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
    )

    class _ScalarRows:
        def __iter__(self):
            return iter([])

        def first(self):
            return None

    class _ExecResult:
        def scalars(self):
            return _ScalarRows()

        def all(self):
            return []

    class _Session:
        def get(self, model, key):
            if model is CandidateCommit and key == candidate_id:
                return candidate
            return None

        def execute(self, _stmt):
            return _ExecResult()

        def flush(self):
            return None

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(repair_service, "session_scope", _scope)

    payload = repair_service.update_candidate_operator_state(
        candidate_id=candidate_id,
        action="restore",
    )

    assert candidate.lifecycle_status == "active"
    assert candidate.repair_state == "audit_only"
    assert payload["repair_state"] == "audit_only"


@pytest.mark.parametrize("action", ["quarantine", "discard", "restore"])
def test_repair_candidate_actions_reject_non_failed_candidates_without_mutating(
    monkeypatch,
    action: str,
) -> None:
    """Regression: direct actions must not mutate rows outside the failed repair pool."""

    candidate_id = uuid4()
    candidate = SimpleNamespace(
        id=candidate_id,
        commit_hash="c" * 40,
        git_parent_commit_hash="p" * 40,
        nearest_viable_ancestor_hash="v" * 40,
        island_id="main",
        produced_by_job_id=None,
        job_kind="evolution",
        repair_source_candidate_id=None,
        campaign_program_hash=None,
        publication_status="published",
        evaluation_status="passed",
        archive_status="not_considered",
        lifecycle_status="discarded",
        failure_stage=None,
        failure_kind=None,
        failure_summary=None,
        repair_state="discarded",
        failed_depth=0,
        repair_attempts=0,
        last_repair_job_id=None,
        failure_evidence_id=None,
        created_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 8, tzinfo=timezone.utc),
    )

    class _Session:
        def get(self, model, key):
            if model is CandidateCommit and key == candidate_id:
                return candidate
            return None

        def execute(self, _stmt):
            raise AssertionError("non-failed rows should be rejected before active-job lookup")

        def flush(self):
            raise AssertionError("non-failed rows must not be flushed")

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(repair_service, "session_scope", _scope)

    with pytest.raises(RepairNotFoundError):
        repair_service.update_candidate_operator_state(candidate_id=candidate_id, action=action)

    assert candidate.lifecycle_status == "discarded"
    assert candidate.repair_state == "discarded"
