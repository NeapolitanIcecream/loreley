from __future__ import annotations

from datetime import datetime, timezone
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.exc import SQLAlchemyError

import loreley.core.worker.job_store as job_store
from loreley.config import Settings
from loreley.core.worker.coding import (
    CodingAgentResponse,
    ExecutionReport,
)
from loreley.core.worker.evaluator import EvaluationMetric, EvaluationResult
from loreley.core.worker.evolution import JobContext
from loreley.core.worker.job_store import (
    EvolutionJobStore,
    JobLeaseLost,
    JobPreconditionError,
)
from loreley.core.worker.planning import PlanDocument, PlanningAgentResponse
from loreley.db.models import EvolutionJob, JobStatus


def test_is_lock_conflict_matches_pgcode_and_messages(settings: Settings) -> None:
    store = EvolutionJobStore(settings=settings)

    class DummyOrig:
        def __init__(self, pgcode: str | None, message: str) -> None:
            self.pgcode = pgcode
            self._message = message

        def __str__(self) -> str:  # pragma: no cover - trivial
            return self._message

    class DummyExc(SQLAlchemyError):
        def __init__(self, orig: Any) -> None:
            super().__init__()
            self.orig = orig

    assert store._is_lock_conflict(DummyExc(DummyOrig("55P03", "lock"))) is True  # type: ignore[attr-defined]
    assert store._is_lock_conflict(DummyExc(DummyOrig(None, "database is locked"))) is True  # type: ignore[attr-defined]
    assert store._is_lock_conflict(SQLAlchemyError()) is False  # type: ignore[attr-defined]


def test_start_job_marks_running_and_returns_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    monkeypatch.setenv("LORELEY_WORKER_INSTANCE_ID", "worker-01")
    settings.worker_job_lease_ttl_seconds = 600

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.base_commit_hash = "abc123"
            self.island_id = "island"
            self.inspiration_commit_hashes = ["i1", "i2"]
            self.goal = "value"
            self.constraints = []
            self.acceptance_criteria = []
            self.notes = []
            self.tags = []
            self.iteration_hint = None
            self.is_seed_job = False
            self.sampling_strategy = None
            self.sampling_initial_radius = None
            self.sampling_radius_used = None
            self.sampling_fallback_inspirations = None
            self.status = JobStatus.PENDING
            self.started_at = None
            self.candidate_commit_hash = "oldcandidate"
            self.candidate_branch_name = "exp/old-branch"
            self.candidate_published_at = datetime.now(timezone.utc)
            self.heartbeat_at = None
            self.lease_expires_at = None
            self.run_token = None
            self.worker_id = None
            self.last_error = "previous"

    dummy_job = DummyJob()

    class DummyResult:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def scalar_one_or_none(self) -> Any:
            return self.obj

    class DummySession:
        def __init__(self) -> None:
            self.executed = False

        def execute(self, _stmt: Any) -> DummyResult:
            self.executed = True
            return DummyResult(dummy_job)

    @contextmanager
    def fake_scope() -> Any:
        session = DummySession()
        yield session

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    locked = store.start_job(job_id)
    assert dummy_job.status is JobStatus.RUNNING
    assert dummy_job.started_at is not None
    assert dummy_job.candidate_commit_hash is None
    assert dummy_job.candidate_branch_name is None
    assert dummy_job.candidate_published_at is None
    assert dummy_job.heartbeat_at is not None
    assert dummy_job.lease_expires_at is not None
    assert dummy_job.lease_expires_at > dummy_job.started_at
    assert dummy_job.run_token is not None
    assert dummy_job.worker_id == "worker-01"
    assert dummy_job.last_error is None
    assert locked.job_id == job_id
    assert locked.base_commit_hash == dummy_job.base_commit_hash
    assert locked.inspiration_commit_hashes == tuple(dummy_job.inspiration_commit_hashes)
    assert locked.run_token == dummy_job.run_token
    assert locked.worker_id == "worker-01"


def test_record_candidate_commit_updates_job_metadata(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.status = JobStatus.RUNNING
            self.run_token = run_token
            self.candidate_commit_hash = None
            self.candidate_branch_name = None
            self.candidate_published_at = None

    job_row = DummyJob()

    class DummySession:
        def get(self, model: Any, key: Any) -> Any:
            if model is EvolutionJob and key == job_id:
                return job_row
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    store.record_candidate_commit(
        job_id,
        "cand123",
        "exp/job-branch",
        run_token=run_token,
        published=False,
    )
    assert job_row.candidate_commit_hash == "cand123"
    assert job_row.candidate_branch_name == "exp/job-branch"
    assert job_row.candidate_published_at is None

    store.record_candidate_commit(
        job_id,
        "cand123",
        "exp/job-branch",
        run_token=run_token,
        published=True,
    )
    assert job_row.candidate_commit_hash == "cand123"
    assert job_row.candidate_branch_name == "exp/job-branch"
    assert job_row.candidate_published_at is not None


def test_record_candidate_commit_rejects_stale_run_token(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    active_run_token = uuid.uuid4()
    stale_run_token = uuid.uuid4()

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.status = JobStatus.RUNNING
            self.run_token = active_run_token
            self.candidate_commit_hash = "cand-old"
            self.candidate_branch_name = "exp/old-branch"
            self.candidate_published_at = datetime.now(timezone.utc)

    job_row = DummyJob()

    class DummySession:
        def get(self, model: Any, key: Any) -> Any:
            if model is EvolutionJob and key == job_id:
                return job_row
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    with pytest.raises(JobLeaseLost):
        store.record_candidate_commit(
            job_id,
            "cand123",
            "exp/job-branch",
            run_token=stale_run_token,
            published=True,
        )

    assert job_row.candidate_commit_hash == "cand-old"
    assert job_row.candidate_branch_name == "exp/old-branch"
    assert job_row.candidate_published_at is not None


def test_start_job_rejects_missing_or_invalid_jobs(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    class DummyResult:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def scalar_one_or_none(self) -> Any:
            return self.obj

    class DummySession:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult(self.obj)

    @contextmanager
    def missing_scope() -> Any:
        yield DummySession(None)

    monkeypatch.setattr(job_store, "session_scope", missing_scope)
    store = EvolutionJobStore(settings=settings)
    with pytest.raises(JobPreconditionError):
        store.start_job(uuid.uuid4())

    class DummyJob:
        def __init__(self) -> None:
            self.id = uuid.uuid4()
            self.base_commit_hash = "hash"
            self.island_id = None
            self.inspiration_commit_hashes = []
            self.goal = "g"
            self.constraints = []
            self.acceptance_criteria = []
            self.notes = []
            self.tags = []
            self.iteration_hint = None
            self.is_seed_job = False
            self.sampling_strategy = None
            self.sampling_initial_radius = None
            self.sampling_radius_used = None
            self.sampling_fallback_inspirations = None
            self.status = JobStatus.RUNNING

    @contextmanager
    def invalid_status_scope() -> Any:
        yield DummySession(DummyJob())

    monkeypatch.setattr(job_store, "session_scope", invalid_status_scope)
    with pytest.raises(JobPreconditionError):
        store.start_job(uuid.uuid4())


def test_renew_job_lease_raises_when_run_token_is_no_longer_active(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    executed: list[Any] = []

    class DummyResult:
        rowcount = 0

    class DummySession:
        def execute(self, stmt: Any) -> DummyResult:
            executed.append(stmt)
            return DummyResult()

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    with pytest.raises(JobLeaseLost):
        store.renew_job_lease(uuid.uuid4(), uuid.uuid4())

    assert len(executed) == 1


def test_persist_success_updates_job_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.status = JobStatus.RUNNING
            self.plan_summary: str | None = None
            self.completed_at = None
            self.candidate_commit_hash = None
            self.candidate_branch_name = None
            self.candidate_published_at = None
            self.run_token = run_token
            self.worker_id = "worker-01"
            self.heartbeat_at = datetime.now(timezone.utc)
            self.lease_expires_at = datetime.now(timezone.utc)
            self.last_error = "err"
            self.island_id = "island"
            self.base_commit_hash = "base"
            self.result_commit_hash = None
            self.ingestion_status = None
            self.ingestion_attempts = 0
            self.ingestion_delta = None
            self.ingestion_status_code = None
            self.ingestion_message = None
            self.ingestion_cell_index = None
            self.ingestion_last_attempt_at = None
            self.ingestion_reason = None

    job_row = DummyJob()
    added: list[Any] = []

    class DummySession:
        def __init__(self) -> None:
            self.added = added

        def get(self, model: Any, key: Any) -> Any:
            if model is EvolutionJob and key == job_id:
                return job_row
            return None

        def add(self, obj: Any) -> None:
            self.added.append(obj)

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    plan = PlanDocument(
        summary="plan",
        markdown="## Summary\n- plan\n",
        focus_metrics=("f",),
        guardrails=("g",),
    )
    plan_response = PlanningAgentResponse(
        plan=plan,
        raw_output="raw",
        prompt="prompt",
        command=("cmd",),
        stderr="",
        attempts=1,
        duration_seconds=1.0,
    )
    report = ExecutionReport(
        summary="impl",
        markdown="## Summary\n- impl\n",
    )
    coding_response = CodingAgentResponse(
        report=report,
        raw_output="raw",
        prompt="p",
        command=("cmd",),
        stderr="",
        attempts=1,
        duration_seconds=1.0,
    )
    evaluation = EvaluationResult(
        summary="eval",
        metrics=(EvaluationMetric(name="score", value=1.0),),
        tests_executed=("pytest -q",),
        logs=("log",),
        extra={},
    )
    job_ctx = JobContext(
        job_id=job_id,
        run_token=run_token,
        base_commit_hash="base",
        island_id="island",
        inspiration_commit_hashes=(),
        goal="goal",
        constraints=("c",),
        acceptance_criteria=("done",),
        iteration_hint=None,
        notes=(),
        tags=("tag",),
        is_seed_job=False,
        sampling_strategy=None,
        sampling_initial_radius=None,
        sampling_radius_used=None,
        sampling_fallback_inspirations=None,
    )

    store.persist_success(
        job_ctx=job_ctx,
        plan=plan_response,
        coding=coding_response,
        evaluation=evaluation,
        worktree=Path("."),  # dummy path; artifacts/git diff are best-effort in tests
        commit_hash="newcommit",
        commit_message="msg",
    )

    assert job_row.status is JobStatus.SUCCEEDED
    assert job_row.plan_summary == "plan"
    assert job_row.result_commit_hash == "newcommit"
    assert job_row.run_token is None
    assert job_row.worker_id is None
    assert job_row.heartbeat_at is None
    assert job_row.lease_expires_at is None
    metadata = [obj for obj in added if isinstance(obj, job_store.CommitCard)]
    metrics = [obj for obj in added if isinstance(obj, job_store.Metric)]
    assert len(metadata) == 1
    assert metadata[0].commit_hash == "newcommit"
    assert len(metrics) == 1
    assert metrics[0].name == "score"


def test_persist_success_rejects_stale_run_token(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    active_token = uuid.uuid4()
    stale_token = uuid.uuid4()

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.status = JobStatus.RUNNING
            self.plan_summary: str | None = None
            self.completed_at = None
            self.run_token = active_token
            self.worker_id = "worker-02"
            self.heartbeat_at = datetime.now(timezone.utc)
            self.lease_expires_at = datetime.now(timezone.utc)
            self.last_error = None
            self.island_id = "island"
            self.base_commit_hash = "base"
            self.result_commit_hash = None
            self.ingestion_status = None
            self.ingestion_attempts = 0
            self.ingestion_delta = None
            self.ingestion_status_code = None
            self.ingestion_message = None
            self.ingestion_cell_index = None
            self.ingestion_last_attempt_at = None
            self.ingestion_reason = None

    job_row = DummyJob()
    added: list[Any] = []

    class DummySession:
        def get(self, model: Any, key: Any) -> Any:
            if model is EvolutionJob and key == job_id:
                return job_row
            return None

        def add(self, obj: Any) -> None:
            added.append(obj)

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    monkeypatch.setattr(job_store, "write_job_artifacts", lambda **_kwargs: {})
    monkeypatch.setattr(
        job_store,
        "build_commit_card_from_git",
        lambda **_kwargs: type("Build", (), {"key_files": [], "highlights": []})(),
    )
    store = EvolutionJobStore(settings=settings)

    plan_response = PlanningAgentResponse(
        plan=PlanDocument(summary="plan", markdown="## Summary\n- plan\n"),
        raw_output="raw",
        prompt="prompt",
        command=("cmd",),
        stderr="",
        attempts=1,
        duration_seconds=1.0,
    )
    coding_response = CodingAgentResponse(
        report=ExecutionReport(summary="impl", markdown="## Summary\n- impl\n"),
        raw_output="raw",
        prompt="p",
        command=("cmd",),
        stderr="",
        attempts=1,
        duration_seconds=1.0,
    )
    evaluation = EvaluationResult(summary="eval")
    job_ctx = JobContext(
        job_id=job_id,
        run_token=stale_token,
        base_commit_hash="base",
        island_id="island",
        inspiration_commit_hashes=(),
        goal="goal",
        constraints=(),
        acceptance_criteria=(),
        iteration_hint=None,
        notes=(),
        tags=(),
        is_seed_job=False,
        sampling_strategy=None,
        sampling_initial_radius=None,
        sampling_radius_used=None,
        sampling_fallback_inspirations=None,
    )

    with pytest.raises(JobLeaseLost):
        store.persist_success(
            job_ctx=job_ctx,
            plan=plan_response,
            coding=coding_response,
            evaluation=evaluation,
            worktree=Path("."),
            commit_hash="newcommit",
            commit_message="msg",
        )

    assert added == []
