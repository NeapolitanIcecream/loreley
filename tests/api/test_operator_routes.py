from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

import loreley.api.auth as api_auth
from loreley.api.auth import require_write_auth
import loreley.api.routers.operator as operator_router
import loreley.api.services.operator as operator_service
from loreley.api.routers.operator import router as operator_api_router
from loreley.api.services.operator import (
    OperatorTaskAlreadyActiveError,
    OperatorTaskNotFoundError,
)
from loreley.db.models import OperatorTask, OperatorTaskStatus
from tests.support import TestSettings


def _client(*, authenticated: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(operator_api_router, prefix="/api/v1")
    if authenticated:
        app.dependency_overrides[require_write_auth] = lambda: "test-operator"
    return TestClient(app)


def _status_payload() -> dict[str, object]:
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    return {
        "campaign_program": {
            "current_file": {
                "found": True,
                "source_path": "loreley.program.md",
                "hash": "a" * 64,
                "normalized_hash": "b" * 64,
                "title": "Campaign",
                "recognized_sections": ["Goal"],
                "parse_warnings": [],
                "sections": {"goal": "Ship operator console."},
            },
            "scheduler": {
                "active_hash": "a" * 64,
                "active_source": "database:evolution_jobs",
                "persisted_hash": "a" * 64,
                "persisted_source": "database:evolution_jobs",
                "current_hash": "a" * 64,
                "current_matches_active": True,
                "change_policy": "locked",
            },
        },
        "baseline": None,
        "repair_pool": {
            "total_failed_candidates": 0,
            "active_repair_jobs": 0,
            "by_repair_state": {},
            "by_lifecycle_status": {},
            "by_failure_kind": {},
        },
        "job_health": {
            "jobs": {"unfinished": 0, "pending_ingestion": 0},
            "job_leases": {"running": 0},
            "by_status": {},
            "by_job_kind": {},
        },
        "generated_at": now,
    }


def test_operator_status_route_serializes_status(monkeypatch) -> None:
    monkeypatch.setattr(operator_router, "operator_status", _status_payload)

    response = _client().get("/api/v1/operator/status")

    assert response.status_code == 200
    assert response.json()["campaign_program"]["current_file"]["hash"] == "a" * 64


def test_operator_task_create_uses_background_task(monkeypatch) -> None:
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    task_id = uuid4()
    task = SimpleNamespace(
        id=task_id,
        kind="baseline_ensure",
        status="pending",
        request_payload={},
        result_payload={},
        error_summary=None,
        started_at=None,
        completed_at=None,
        created_at=now,
        updated_at=now,
    )
    calls: list[object] = []
    monkeypatch.setattr(operator_router, "create_baseline_ensure_task", lambda: task)
    monkeypatch.setattr(operator_router, "run_baseline_ensure_task", lambda task_id: calls.append(task_id))

    response = _client().post("/api/v1/operator/tasks/baseline-ensure", json={})

    assert response.status_code == 200
    assert response.json()["id"] == str(task_id)
    assert calls == [task_id]


def test_operator_task_create_requires_write_auth_configuration(monkeypatch, captured_logs) -> None:
    """Regression: direct operator writes must not run without a write token."""

    monkeypatch.setattr(api_auth, "get_settings", lambda: TestSettings(LORELEY_API_WRITE_TOKEN=None))
    monkeypatch.setattr(
        operator_router,
        "create_baseline_ensure_task",
        lambda: (_ for _ in ()).throw(AssertionError("unauthenticated write executed")),
    )

    response = _client(authenticated=False).post("/api/v1/operator/tasks/baseline-ensure", json={})

    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "write_auth_not_configured"
    assert any(
        record["module"] == "api.auth"
        and record["message"] == "UI API write request rejected by auth"
        and record["extra"].get("reason") == "token_unconfigured"
        for record in captured_logs
    )


def test_operator_task_create_rejects_active_baseline_task(monkeypatch) -> None:
    task_id = uuid4()
    calls: list[object] = []

    def _active():
        raise OperatorTaskAlreadyActiveError(f"Baseline ensure task already active: {task_id}.")

    monkeypatch.setattr(operator_router, "create_baseline_ensure_task", _active)
    monkeypatch.setattr(operator_router, "run_baseline_ensure_task", lambda task_id: calls.append(task_id))

    response = _client().post("/api/v1/operator/tasks/baseline-ensure", json={})

    assert response.status_code == 409
    assert str(task_id) in response.json()["detail"]
    assert calls == []


def test_create_baseline_ensure_task_rejects_existing_active_task(monkeypatch, settings) -> None:
    task_id = uuid4()
    active = SimpleNamespace(id=task_id)

    class _Scalar:
        def first(self):
            return active

    class _Result:
        def scalars(self):
            return _Scalar()

    class _Session:
        def execute(self, _stmt):
            return _Result()

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(operator_service, "session_scope", _scope)

    with pytest.raises(OperatorTaskAlreadyActiveError, match=str(task_id)):
        operator_service.create_baseline_ensure_task(settings=settings)


def test_create_baseline_ensure_task_rejects_atomic_active_conflict(monkeypatch, settings) -> None:
    """Regression: concurrent creates must rely on the DB unique constraint."""

    class _Scalar:
        def first(self):
            return None

    class _Result:
        def scalars(self):
            return _Scalar()

    class _Session:
        def execute(self, _stmt):
            return _Result()

        def add(self, _row):
            pass

        def flush(self):
            raise IntegrityError("INSERT", {}, RuntimeError("duplicate active task"))

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(operator_service, "session_scope", _scope)

    with pytest.raises(OperatorTaskAlreadyActiveError, match="Baseline ensure task already active"):
        operator_service.create_baseline_ensure_task(settings=settings)


def test_create_baseline_ensure_task_replaces_stale_pending_task(monkeypatch, settings) -> None:
    """Regression: orphaned pending background tasks must not block forever."""

    task_id = uuid4()
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    stale = SimpleNamespace(
        id=task_id,
        status=OperatorTaskStatus.PENDING.value,
        created_at=now - timedelta(minutes=31),
        completed_at=None,
        error_summary=None,
    )
    added: list[object] = []

    class _Session:
        def add(self, row):
            added.append(row)

        def flush(self):
            return None

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(operator_service, "session_scope", _scope)
    monkeypatch.setattr(
        operator_service,
        "_active_baseline_ensure_task",
        lambda *_args, **_kwargs: stale,
    )
    monkeypatch.setattr(operator_service, "_operator_now", lambda: now)

    task = operator_service.create_baseline_ensure_task(settings=settings)

    assert stale.status == OperatorTaskStatus.FAILED.value
    assert stale.completed_at == now
    assert "stale pending" in str(stale.error_summary).lower()
    assert task in added


def test_create_baseline_ensure_task_replaces_stale_running_task(monkeypatch, settings) -> None:
    """Regression: orphaned running background tasks must not block forever."""

    task_id = uuid4()
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    stale = SimpleNamespace(
        id=task_id,
        status=OperatorTaskStatus.RUNNING.value,
        created_at=now - timedelta(hours=7),
        started_at=now - timedelta(hours=7),
        completed_at=None,
        error_summary=None,
    )
    added: list[object] = []

    class _Session:
        def add(self, row):
            added.append(row)

        def flush(self):
            return None

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(operator_service, "session_scope", _scope)
    monkeypatch.setattr(
        operator_service,
        "_active_baseline_ensure_task",
        lambda *_args, **_kwargs: stale,
    )
    monkeypatch.setattr(operator_service, "_operator_now", lambda: now)

    task = operator_service.create_baseline_ensure_task(settings=settings)

    assert stale.status == OperatorTaskStatus.FAILED.value
    assert stale.completed_at == now
    assert "stale running" in str(stale.error_summary).lower()
    assert task in added


def test_create_baseline_ensure_task_rejects_recent_running_task(monkeypatch, settings) -> None:
    task_id = uuid4()
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    active = SimpleNamespace(
        id=task_id,
        status=OperatorTaskStatus.RUNNING.value,
        created_at=now - timedelta(minutes=5),
        started_at=now - timedelta(minutes=5),
    )

    class _Session:
        def add(self, _row):
            raise AssertionError("recent running task should block before insert")

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(operator_service, "session_scope", _scope)
    monkeypatch.setattr(
        operator_service,
        "_active_baseline_ensure_task",
        lambda *_args, **_kwargs: active,
    )
    monkeypatch.setattr(operator_service, "_operator_now", lambda: now)

    with pytest.raises(OperatorTaskAlreadyActiveError, match=str(task_id)):
        operator_service.create_baseline_ensure_task(settings=settings)


def test_operator_task_model_has_active_baseline_unique_index() -> None:
    indexes = {index.name: index for index in OperatorTask.__table__.indexes}
    index = indexes["uq_operator_tasks_active_baseline_ensure"]

    where = index.dialect_options["postgresql"]["where"]
    compiled_where = str(
        where.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert index.unique is True
    assert [column.name for column in index.columns] == ["kind"]
    assert "baseline_ensure" in compiled_where
    assert "pending" in compiled_where
    assert "running" in compiled_where


def test_operator_task_detail_returns_404(monkeypatch) -> None:
    def _missing(**_kwargs):
        raise OperatorTaskNotFoundError("Operator task not found.")

    monkeypatch.setattr(operator_router, "get_operator_task", _missing)

    response = _client().get(f"/api/v1/operator/tasks/{uuid4()}")

    assert response.status_code == 404


def test_baseline_task_failure_is_persisted_and_logged(monkeypatch, settings, captured_logs) -> None:
    task_id = uuid4()
    now = datetime(2026, 5, 8, tzinfo=timezone.utc)
    task = SimpleNamespace(
        id=task_id,
        status=OperatorTaskStatus.PENDING.value,
        started_at=None,
        completed_at=None,
        error_summary=None,
        result_payload={},
    )

    class _Scalar:
        def scalar_one(self):
            return now

    class _Session:
        def get(self, _model, key):
            assert key == task_id
            return task

        def execute(self, _stmt):
            return _Scalar()

    @contextmanager
    def _scope():
        yield _Session()

    settings.mapelites_experiment_root_commit = ""
    monkeypatch.setattr(operator_service, "session_scope", _scope)
    monkeypatch.setattr(operator_service, "get_settings", lambda: settings)
    monkeypatch.setattr(
        operator_service,
        "load_campaign_program_from_repo",
        lambda _repo_root: SimpleNamespace(snapshot=None, raw_markdown=None, source_path=None),
    )

    operator_service.run_baseline_ensure_task(task_id)

    assert task.status == OperatorTaskStatus.FAILED.value
    assert "MAPELITES_EXPERIMENT_ROOT_COMMIT" in str(task.error_summary)
    assert any(
        record["module"] == "api.operator"
        and "Operator baseline ensure task failed" in record["message"]
        for record in captured_logs
    )


def test_baseline_task_start_failure_is_persisted_and_logged(monkeypatch, captured_logs) -> None:
    """Regression: _mark_task_running failures must not disappear from BackgroundTasks."""

    task_id = uuid4()
    persisted: list[tuple[object, str]] = []

    def _fail_start(_task_id):
        raise RuntimeError("database unavailable")

    def _persist_failure(task_id, *, error_summary: str) -> None:
        persisted.append((task_id, error_summary))

    monkeypatch.setattr(operator_service, "_mark_task_running", _fail_start)
    monkeypatch.setattr(operator_service, "_mark_task_failed", _persist_failure)
    monkeypatch.setattr(
        operator_service,
        "get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("task body should not run")),
    )

    operator_service.run_baseline_ensure_task(task_id)

    assert persisted == [(task_id, "Failed to start baseline ensure task: database unavailable")]
    assert any(
        record["module"] == "api.operator"
        and "Operator baseline ensure task failed before start" in record["message"]
        for record in captured_logs
    )


def test_startup_reconciliation_leaves_recent_active_baseline_tasks_running(
    monkeypatch,
    captured_logs,
) -> None:
    """Regression: one API process startup must not fail another process's task."""

    now = datetime(2026, 5, 9, tzinfo=timezone.utc)
    pending = SimpleNamespace(
        id=uuid4(),
        status=OperatorTaskStatus.PENDING.value,
        created_at=now - timedelta(minutes=1),
        started_at=None,
        completed_at=None,
        error_summary=None,
    )
    running = SimpleNamespace(
        id=uuid4(),
        status=OperatorTaskStatus.RUNNING.value,
        created_at=now - timedelta(minutes=30),
        started_at=now - timedelta(minutes=1),
        completed_at=None,
        error_summary=None,
    )

    class _NowResult:
        def scalar_one(self):
            return now

    class _RowsResult:
        def scalars(self):
            return [pending, running]

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _NowResult()
            return _RowsResult()

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(operator_service, "session_scope", _scope)

    count = operator_service.mark_stale_baseline_ensure_tasks_failed()

    assert count == 0
    assert pending.status == OperatorTaskStatus.PENDING.value
    assert running.status == OperatorTaskStatus.RUNNING.value
    assert pending.completed_at is None
    assert running.completed_at is None
    assert any(
        record["module"] == "api.operator"
        and record["message"] == "Active operator baseline ensure tasks left running at startup"
        and record["extra"].get("active_count") == 2
        for record in captured_logs
    )


def test_startup_reconciliation_marks_only_stale_baseline_tasks_failed(
    monkeypatch,
    captured_logs,
) -> None:
    now = datetime(2026, 5, 9, tzinfo=timezone.utc)
    stale_pending = SimpleNamespace(
        id=uuid4(),
        status=OperatorTaskStatus.PENDING.value,
        created_at=now - timedelta(minutes=11),
        started_at=None,
        completed_at=None,
        error_summary=None,
    )
    recent_running = SimpleNamespace(
        id=uuid4(),
        status=OperatorTaskStatus.RUNNING.value,
        created_at=now - timedelta(hours=1),
        started_at=now - timedelta(hours=1),
        completed_at=None,
        error_summary=None,
    )

    class _NowResult:
        def scalar_one(self):
            return now

    class _RowsResult:
        def scalars(self):
            return [stale_pending, recent_running]

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _NowResult()
            return _RowsResult()

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(operator_service, "session_scope", _scope)

    count = operator_service.mark_stale_baseline_ensure_tasks_failed()

    assert count == 1
    assert stale_pending.status == OperatorTaskStatus.FAILED.value
    assert stale_pending.completed_at == now
    assert "Stale baseline ensure task" in str(stale_pending.error_summary)
    assert recent_running.status == OperatorTaskStatus.RUNNING.value
    assert recent_running.completed_at is None
    assert any(
        record["module"] == "api.operator"
        and record["message"] == "Stale operator baseline ensure tasks marked failed at startup"
        and record["extra"].get("count") == 1
        and record["extra"].get("stale_counts") == {"pending": 1}
        for record in captured_logs
    )
