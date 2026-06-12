from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.dialects import postgresql

import loreley.api.auth as api_auth
from loreley.api.auth import require_write_auth
import loreley.api.routers.repair as repair_router
import loreley.api.services.repair as repair_service
from loreley.api.routers.repair import router as repair_api_router
from loreley.api.services.repair import (
    RepairConflictError,
    RepairNotFoundError,
    RepairPoolPage,
)
from loreley.db.models import CandidateCommit
from tests.support import TestSettings


def _client(*, authenticated: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(repair_api_router, prefix="/api/v1")
    if authenticated:
        app.dependency_overrides[require_write_auth] = lambda: "test-operator"
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


class _EmptyScalarRows:
    def __iter__(self):
        return iter([])

    def first(self):
        return None


class _EmptyExecResult:
    def scalars(self):
        return _EmptyScalarRows()

    def all(self):
        return []


class _CandidateLockResult:
    def __init__(self, candidate) -> None:
        self._candidate = candidate

    def scalar_one_or_none(self):
        return self._candidate


class _RepairActionSession:
    def __init__(
        self,
        candidate,
        *,
        assert_candidate_locked: bool = False,
        reject_after_candidate_lookup: bool = False,
    ) -> None:
        self.candidate = candidate
        self.assert_candidate_locked = assert_candidate_locked
        self.reject_after_candidate_lookup = reject_after_candidate_lookup
        self.statements = []
        self.added = []

    def execute(self, stmt):
        self.statements.append(stmt)
        if len(self.statements) == 1:
            if self.assert_candidate_locked:
                compiled = str(stmt.compile(dialect=postgresql.dialect())).upper()
                assert "FOR UPDATE" in compiled
            return _CandidateLockResult(self.candidate)
        if self.reject_after_candidate_lookup:
            raise AssertionError("non-failed rows should be rejected before active-job lookup")
        return _EmptyExecResult()

    def flush(self):
        return None

    def add(self, row):
        self.added.append(row)


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


def test_repair_schedule_one_route_returns_deprecated_noop(monkeypatch) -> None:
    monkeypatch.setattr(
        repair_router,
        "schedule_one_repair",
        lambda: {
            "scheduled": False,
            "job_id": None,
            "repair_source_candidate_id": None,
            "base_commit_hash": None,
            "message": "Repair pool scheduling is deprecated.",
        },
    )

    response = _client().post("/api/v1/repair/schedule-one", json={})

    assert response.status_code == 200
    assert response.json()["scheduled"] is False
    assert response.json()["job_id"] is None
    assert "deprecated" in response.json()["message"].lower()


def test_repair_schedule_one_route_requires_write_auth_configuration(monkeypatch) -> None:
    """Regression: direct repair writes must not run without a write token."""

    monkeypatch.setattr(api_auth, "get_settings", lambda: TestSettings(LORELEY_API_WRITE_TOKEN=None))
    monkeypatch.setattr(
        repair_router,
        "schedule_one_repair",
        lambda: (_ for _ in ()).throw(AssertionError("unauthenticated write executed")),
    )

    response = _client(authenticated=False).post("/api/v1/repair/schedule-one", json={})

    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "write_auth_not_configured"


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

    payload = repair_service.schedule_one_repair(settings=settings)

    assert payload["scheduled"] is False
    assert payload["job_id"] is None
    assert "deprecated" in str(payload["message"]).lower()


def test_repair_schedule_one_api_guard_blocks_exhausted_token_budget(
    monkeypatch,
    settings,
) -> None:
    """Regression: manual repair scheduling must not bypass repair tokens."""

    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    settings.failed_candidate_repair_max_active_jobs = 1
    settings.failed_candidate_repair_max_tokens = 1

    payload = repair_service.schedule_one_repair(settings=settings)

    assert payload["scheduled"] is False
    assert payload["job_id"] is None
    assert "deprecated" in str(payload["message"]).lower()


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
    payload = repair_service.schedule_one_repair(settings=settings)

    assert payload["scheduled"] is False
    assert payload["job_id"] is None
    assert "deprecated" in str(payload["message"]).lower()
    assert events == []


def test_repair_schedule_one_api_guard_blocks_when_active_jobs_at_cap(
    monkeypatch,
    settings,
) -> None:
    """Regression: repeated API calls must respect the scheduler active repair cap."""

    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    settings.failed_candidate_repair_max_active_jobs = 1

    payload = repair_service.schedule_one_repair(settings=settings)

    assert payload["scheduled"] is False
    assert payload["job_id"] is None
    assert "deprecated" in str(payload["message"]).lower()


def test_repair_candidate_action_conflict_returns_409(monkeypatch) -> None:
    def _conflict(**_kwargs):
        raise RepairConflictError("Candidate has an active repair job.")

    monkeypatch.setattr(repair_router, "update_candidate_operator_state", _conflict)

    response = _client().post(f"/api/v1/repair/candidates/{uuid4()}/discard", json={})

    assert response.status_code == 409


def test_repair_candidate_action_route_passes_reason_and_returns_audit_id(monkeypatch) -> None:
    """Regression: repair candidate action bodies accepted `reason` but ignored it."""

    candidate_id = uuid4()
    audit_id = uuid4()
    captured: dict[str, object] = {}

    def _update_candidate_operator_state(**kwargs):
        captured.update(kwargs)
        payload = _candidate_payload(candidate_id)
        payload["operator_reason"] = "operator reviewed flaky failure"
        payload["operator_audit_task_id"] = audit_id
        return payload

    monkeypatch.setattr(
        repair_router,
        "update_candidate_operator_state",
        _update_candidate_operator_state,
    )

    response = _client().post(
        f"/api/v1/repair/candidates/{candidate_id}/quarantine",
        json={"reason": "operator reviewed flaky failure"},
    )

    assert response.status_code == 200
    assert captured["candidate_id"] == candidate_id
    assert captured["action"] == "quarantine"
    assert captured["reason"] == "operator reviewed flaky failure"
    assert captured["actor"] == "test-operator"
    assert response.json()["reason"] == "operator reviewed flaky failure"
    assert response.json()["operator_audit_task_id"] == str(audit_id)


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

    session = _RepairActionSession(candidate, assert_candidate_locked=True)

    @contextmanager
    def _scope():
        yield session

    monkeypatch.setattr(repair_service, "session_scope", _scope)

    payload = repair_service.update_candidate_operator_state(
        candidate_id=candidate_id,
        action="restore",
        reason="operator restored after manual audit",
        actor="operator-test",
    )

    assert candidate.lifecycle_status == "active"
    assert candidate.repair_state == "audit_only"
    assert payload["repair_state"] == "audit_only"
    assert payload["operator_reason"] == "operator restored after manual audit"
    assert len(session.added) == 1
    audit = session.added[0]
    assert audit.kind == "repair_candidate_action"
    assert audit.status == "succeeded"
    assert audit.request_payload == {
        "action": "restore",
        "actor": "operator-test",
        "candidate_id": str(candidate_id),
        "reason": "operator restored after manual audit",
    }
    assert audit.result_payload["previous_state"] == {
        "lifecycle_status": "discarded",
        "repair_state": "eligible",
    }
    assert audit.result_payload["current_state"] == {
        "lifecycle_status": "active",
        "repair_state": "audit_only",
    }
    assert payload["operator_audit_task_id"] == audit.id
    assert session.statements


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

    session = _RepairActionSession(candidate, reject_after_candidate_lookup=True)

    @contextmanager
    def _scope():
        yield session

    monkeypatch.setattr(repair_service, "session_scope", _scope)

    with pytest.raises(RepairNotFoundError):
        repair_service.update_candidate_operator_state(candidate_id=candidate_id, action=action)

    assert candidate.lifecycle_status == "discarded"
    assert candidate.repair_state == "discarded"
    assert len(session.statements) == 1
