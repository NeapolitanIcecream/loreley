from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

import loreley.api.routers.operator as operator_router
import loreley.api.services.operator as operator_service
from loreley.api.routers.operator import router as operator_api_router
from loreley.api.services.operator import (
    OperatorTaskAlreadyActiveError,
    OperatorTaskNotFoundError,
)
from loreley.db.models import OperatorTask, OperatorTaskStatus


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(operator_api_router, prefix="/api/v1")
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
