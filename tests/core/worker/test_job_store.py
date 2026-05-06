from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

import loreley.core.worker.job_store as job_store
from loreley.config import Settings
from loreley.core.worker.artifacts import (
    FixedJobArtifactPaths,
    JobArtifactWriteResult,
    MaterializedEvaluationArtifact,
)
from loreley.core.worker.coding import (
    CodingAgentResponse,
    ExecutionReport,
)
from loreley.core.worker.evaluator import (
    EvaluationArtifact,
    EvaluationDiagnostic,
    EvaluationFailureResult,
    EvaluationMetric,
    EvaluationOutcome,
    EvaluationResult,
)
from loreley.core.worker.evolution import JobContext
from loreley.core.worker.job_store import (
    EvolutionJobStore,
    EvolutionWorkerError,
    JobLeaseLost,
    JobPreconditionError,
)
from loreley.core.worker.planning import PlanDocument, PlanningAgentResponse
from loreley.db.models import JobStatus


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
    now = datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc)

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
    monkeypatch.setattr(job_store, "_db_utc_now", lambda _session: now)
    store = EvolutionJobStore(settings=settings)

    locked = store.start_job(job_id)
    assert dummy_job.status is JobStatus.RUNNING
    assert dummy_job.started_at == now
    assert dummy_job.candidate_commit_hash is None
    assert dummy_job.candidate_branch_name is None
    assert dummy_job.candidate_published_at is None
    assert dummy_job.heartbeat_at == now
    assert dummy_job.lease_expires_at == now + timedelta(seconds=600)
    assert dummy_job.run_token is not None
    assert dummy_job.worker_id == "worker-01"
    assert dummy_job.last_error is None
    assert locked.job_id == job_id
    assert locked.base_commit_hash == dummy_job.base_commit_hash
    assert locked.inspiration_commit_hashes == tuple(dummy_job.inspiration_commit_hashes)
    assert locked.run_token == dummy_job.run_token
    assert locked.worker_id == "worker-01"


def test_start_job_clamps_worker_id_to_column_budget(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    raw_worker_id = "worker-" + ("x" * 200)
    monkeypatch.setenv("LORELEY_WORKER_INSTANCE_ID", raw_worker_id)
    settings.worker_job_lease_ttl_seconds = 600
    now = datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc)

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.base_commit_hash = "abc123"
            self.island_id = "island"
            self.inspiration_commit_hashes = []
            self.goal = None
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
            self.candidate_commit_hash = None
            self.candidate_branch_name = None
            self.candidate_published_at = None
            self.heartbeat_at = None
            self.lease_expires_at = None
            self.run_token = None
            self.worker_id = None
            self.last_error = None

    dummy_job = DummyJob()

    class DummyResult:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def scalar_one_or_none(self) -> Any:
            return self.obj

    class DummySession:
        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult(dummy_job)

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    digest = hashlib.sha1(raw_worker_id.encode("utf-8")).hexdigest()[:12]
    expected_worker_id = f"{raw_worker_id[:115]}-{digest}"

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    monkeypatch.setattr(job_store, "_db_utc_now", lambda _session: now)
    store = EvolutionJobStore(settings=settings)

    locked = store.start_job(job_id)

    assert len(dummy_job.worker_id) == 128
    assert dummy_job.worker_id == expected_worker_id
    assert locked.worker_id == expected_worker_id


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
    added: list[Any] = []

    class DummyResult:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def scalar_one_or_none(self) -> Any:
            return self.obj

    class DummySession:
        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult(job_row)

        def add(self, obj: Any) -> None:
            added.append(obj)

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
    assert any(isinstance(obj, job_store.CandidateCommit) for obj in added)

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

    class DummyResult:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def scalar_one_or_none(self) -> Any:
            return self.obj

    class DummySession:
        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult(None)

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


def test_record_candidate_commit_without_run_token_preserves_failed_job(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: no-token candidate writes could mutate jobs already marked FAILED."""

    job_id = uuid.uuid4()
    published_at = datetime(2026, 3, 25, 8, 40, tzinfo=timezone.utc)

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.status = JobStatus.FAILED
            self.candidate_commit_hash = "cand-old"
            self.candidate_branch_name = "exp/old-branch"
            self.candidate_published_at = published_at

    job_row = DummyJob()

    class DummySession:
        def get(self, _model: Any, row_id: uuid.UUID) -> DummyJob | None:
            assert row_id == job_id
            return job_row

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    with pytest.raises(EvolutionWorkerError, match="cannot record a candidate"):
        store.record_candidate_commit(
            job_id,
            "cand-new",
            "exp/new-branch",
            published=True,
        )

    assert job_row.candidate_commit_hash == "cand-old"
    assert job_row.candidate_branch_name == "exp/old-branch"
    assert job_row.candidate_published_at == published_at


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
    now = datetime(2026, 3, 25, 8, 15, tzinfo=timezone.utc)

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
    monkeypatch.setattr(job_store, "_db_utc_now", lambda _session: now)
    store = EvolutionJobStore(settings=settings)

    with pytest.raises(JobLeaseLost):
        store.renew_job_lease(uuid.uuid4(), uuid.uuid4())

    assert len(executed) == 1


def test_renew_job_lease_uses_database_time_for_expiry(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    executed: list[Any] = []
    now = datetime(2026, 3, 25, 8, 20, tzinfo=timezone.utc)
    settings.worker_job_lease_ttl_seconds = 90

    class DummyResult:
        rowcount = 1

    class DummySession:
        def execute(self, stmt: Any) -> DummyResult:
            executed.append(stmt)
            return DummyResult()

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    monkeypatch.setattr(job_store, "_db_utc_now", lambda _session: now)
    store = EvolutionJobStore(settings=settings)

    lease_expires_at = store.renew_job_lease(uuid.uuid4(), uuid.uuid4())

    assert lease_expires_at == now + timedelta(seconds=90)
    assert len(executed) == 1


def test_persist_success_updates_job_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()
    added: list[Any] = []
    job_row = _PersistSuccessDummyJob(job_id=job_id, run_token=run_token)
    _install_persist_success_fakes(monkeypatch, job_row=job_row, added=added)
    store = EvolutionJobStore(settings=settings)

    store.persist_success(
        job_ctx=_sample_job_context(job_id=job_id, run_token=run_token),
        plan=_sample_plan_response(),
        coding=_sample_coding_response(),
        evaluation=_sample_evaluation_result(),
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
    artifacts = [obj for obj in added if isinstance(obj, job_store.EvaluationArtifactRecord)]
    assert len(metadata) == 1
    assert metadata[0].commit_hash == "newcommit"
    assert len(metrics) == 1
    assert metrics[0].name == "score"
    assert len(artifacts) == 1
    assert artifacts[0].job_id == job_id
    assert artifacts[0].commit_card_id == metadata[0].id
    assert artifacts[0].key == "benchmark_report"
    assert artifacts[0].diagnostics[0]["message"] == "throughput improved"


def test_persist_success_materializes_passed_outcome_artifact_records(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: ADR envelope artifact_records were dropped on successful evaluations."""

    job_id = uuid.uuid4()
    run_token = uuid.uuid4()
    added: list[Any] = []
    captured_artifacts: list[EvaluationArtifact] = []
    job_row = _PersistSuccessDummyJob(job_id=job_id, run_token=run_token)

    _install_persist_success_fakes(monkeypatch, job_row=job_row, added=added)

    def record_artifact_write(request: Any) -> JobArtifactWriteResult:
        captured_artifacts.extend(request.evaluation.artifacts)
        materialized = (
            _materialized_benchmark_artifact(key=captured_artifacts[0].key)
            if captured_artifacts
            else ()
        )
        return JobArtifactWriteResult(
            fixed=FixedJobArtifactPaths(evaluation_json_path="/worker/artifacts/evaluation.json"),
            evaluation_artifacts=materialized if isinstance(materialized, tuple) else (materialized,),
        )

    monkeypatch.setattr(job_store, "write_job_artifacts", record_artifact_write)
    store = EvolutionJobStore(settings=settings)
    evaluation = _sample_evaluation_result()
    outcome_artifact = EvaluationArtifact(
        key="adr-envelope-report",
        kind="benchmark_json",
        mime_type="application/json",
        inline_payload={"score": 1.0},
        summary="Declared only on EvaluationOutcome.artifact_records.",
        visibility="agent_visible",
    )
    outcome = EvaluationOutcome(
        evaluator_name="pytest",
        candidate_commit_hash="newcommit",
        outcome_kind="passed",
        result=evaluation,
        artifacts=(outcome_artifact,),
    )

    store.persist_success(
        job_ctx=_sample_job_context(job_id=job_id, run_token=run_token),
        plan=_sample_plan_response(),
        coding=_sample_coding_response(),
        evaluation=evaluation,
        evaluation_outcome=outcome,
        worktree=Path("."),
        commit_hash="newcommit",
        commit_message="msg",
    )

    artifacts = [obj for obj in added if isinstance(obj, job_store.EvaluationArtifactRecord)]
    assert [artifact.key for artifact in captured_artifacts] == ["adr-envelope-report"]
    assert len(artifacts) == 1
    assert artifacts[0].key == "adr-envelope-report"


class _PersistSuccessDummyJob:
    def __init__(self, *, job_id: uuid.UUID, run_token: uuid.UUID) -> None:
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


class _PersistSuccessResult:
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def scalar_one_or_none(self) -> Any:
        return self.obj


class _PersistSuccessSession:
    def __init__(self, *, job_row: _PersistSuccessDummyJob, added: list[Any]) -> None:
        self.job_row = job_row
        self.added = added

    def execute(self, _stmt: Any) -> _PersistSuccessResult:
        return _PersistSuccessResult(self.job_row)

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        for obj in self.added:
            if isinstance(obj, job_store.CommitCard) and obj.id is None:
                obj.id = uuid.uuid4()
            if isinstance(obj, job_store.CandidateCommit) and obj.id is None:
                obj.id = uuid.uuid4()
            if isinstance(obj, job_store.EvaluationAttempt) and obj.id is None:
                obj.id = uuid.uuid4()
            if isinstance(obj, job_store.DiagnosticCapsule) and obj.id is None:
                obj.id = uuid.uuid4()


def _install_persist_success_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    job_row: _PersistSuccessDummyJob,
    added: list[Any],
) -> None:
    @contextmanager
    def fake_scope() -> Any:
        yield _PersistSuccessSession(job_row=job_row, added=added)

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    monkeypatch.setattr(
        job_store,
        "build_commit_card_from_git",
        lambda **_kwargs: type("Build", (), {"key_files": ["file.py"], "highlights": ["changed"]})(),
    )
    monkeypatch.setattr(
        job_store,
        "write_job_artifacts",
        lambda _request: JobArtifactWriteResult(
            fixed=FixedJobArtifactPaths(evaluation_json_path="/worker/artifacts/evaluation.json"),
            evaluation_artifacts=(_materialized_benchmark_artifact(),),
        ),
    )


def _materialized_benchmark_artifact(*, key: str = "benchmark_report") -> MaterializedEvaluationArtifact:
    return MaterializedEvaluationArtifact(
        key=key,
        kind="benchmark_json",
        mime_type="application/json",
        label="Benchmark report",
        summary="Parser throughput improved.",
        visibility="agent_visible",
        agent_projection="summary",
        storage_path="/worker/artifacts/benchmark_report.json",
        size_bytes=42,
        sha256="a" * 64,
        diagnostics=(
            EvaluationDiagnostic(
                kind="improvement",
                message="throughput improved",
                severity="info",
            ),
        ),
        metadata={"source": "bench"},
    )


def _sample_plan_response() -> PlanningAgentResponse:
    return PlanningAgentResponse(
        plan=PlanDocument(
            summary="plan",
            markdown="## Summary\n- plan\n",
            focus_metrics=("f",),
            guardrails=("g",),
        ),
        raw_output="raw",
        prompt="prompt",
        command=("cmd",),
        stderr="",
        attempts=1,
        duration_seconds=1.0,
    )


def _sample_coding_response() -> CodingAgentResponse:
    return CodingAgentResponse(
        report=ExecutionReport(
            summary="impl",
            markdown="## Summary\n- impl\n",
        ),
        raw_output="raw",
        prompt="p",
        command=("cmd",),
        stderr="",
        attempts=1,
        duration_seconds=1.0,
    )


def _sample_evaluation_result() -> EvaluationResult:
    return EvaluationResult(
        summary="eval",
        metrics=(EvaluationMetric(name="score", value=1.0),),
        tests_executed=("pytest -q",),
        logs=("log",),
        extra={},
    )


def _sample_job_context(*, job_id: uuid.UUID, run_token: uuid.UUID) -> JobContext:
    return JobContext(
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


def test_persist_success_rejects_stale_run_token(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    stale_token = uuid.uuid4()

    added: list[Any] = []

    class DummyResult:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def scalar_one_or_none(self) -> Any:
            return self.obj

    class DummySession:
        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult(None)

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


def test_persist_success_validates_run_token_before_artifact_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: stale workers must not write success artifacts before lease validation."""

    job_id = uuid.uuid4()
    stale_token = uuid.uuid4()
    side_effects: list[str] = []

    class DummyResult:
        def scalar_one_or_none(self) -> Any:
            return None

    class DummySession:
        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult()

        def add(self, _obj: Any) -> None:
            side_effects.append("db_add")

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    def record_git_inspection(**_kwargs: Any) -> Any:
        side_effects.append("git_inspection")
        return type("Build", (), {"key_files": [], "highlights": []})()

    def record_artifact_write(_request: Any) -> JobArtifactWriteResult:
        side_effects.append("artifact_write")
        return JobArtifactWriteResult(
            fixed=FixedJobArtifactPaths(evaluation_json_path="/tmp/evaluation.json"),
        )

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    monkeypatch.setattr(job_store, "build_commit_card_from_git", record_git_inspection)
    monkeypatch.setattr(job_store, "write_job_artifacts", record_artifact_write)
    store = EvolutionJobStore(settings=settings)

    with pytest.raises(JobLeaseLost):
        store.persist_success(
            job_ctx=_sample_job_context(job_id=job_id, run_token=stale_token),
            plan=_sample_plan_response(),
            coding=_sample_coding_response(),
            evaluation=EvaluationResult(summary="eval"),
            worktree=Path("."),
            commit_hash="newcommit",
            commit_message="msg",
        )

    assert side_effects == []


def test_persist_failure_records_failed_candidate_without_commit_card_and_logs_eligibility(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    captured_logs: list[dict[str, Any]],
) -> None:
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()
    added: list[Any] = []
    job_row = _PersistSuccessDummyJob(job_id=job_id, run_token=run_token)
    job_row.candidate_commit_hash = "failedcommit"
    job_row.candidate_branch_name = "exp/job-branch"
    job_row.candidate_published_at = datetime.now(timezone.utc)

    class DummyResult:
        def __init__(self, obj: Any = None, *, first: Any = None) -> None:
            self.obj = obj
            self._first = first

        def scalar_one_or_none(self) -> Any:
            return self.obj

        def first(self) -> Any:
            return self._first

    class DummySession:
        def __init__(self) -> None:
            self.execute_calls = 0

        def execute(self, _stmt: Any) -> DummyResult:
            self.execute_calls += 1
            if self.execute_calls == 1:
                return DummyResult(job_row)
            return DummyResult(None)

        def get(self, _model: Any, _row_id: uuid.UUID) -> Any:
            return None

        def add(self, obj: Any) -> None:
            added.append(obj)

        def flush(self) -> None:
            for obj in added:
                if isinstance(obj, job_store.CandidateCommit) and obj.id is None:
                    obj.id = uuid.uuid4()
                if isinstance(obj, job_store.DiagnosticCapsule) and obj.id is None:
                    obj.id = uuid.uuid4()
                if isinstance(obj, job_store.EvaluationAttempt) and obj.id is None:
                    obj.id = uuid.uuid4()

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    monkeypatch.setattr(
        job_store,
        "write_failure_job_artifacts",
        lambda _request: JobArtifactWriteResult(fixed=FixedJobArtifactPaths()),
    )
    monkeypatch.setattr(
        EvolutionJobStore,
        "_ancestor_aggregate_ready",
        lambda _self, **_kwargs: True,
    )
    store = EvolutionJobStore(settings=settings)
    outcome = EvaluationOutcome(
        evaluator_name="pytest",
        candidate_commit_hash="failedcommit",
        outcome_kind="candidate_failed",
        failure=EvaluationFailureResult(
            failure_stage="evaluation",
            failure_kind="test_failed",
            repairability="repairable",
            safe_failure_summary="One focused regression failed.",
            agent_visible_evidence_refs=("pytest-summary",),
        ),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )

    recorded = store.persist_failure(
        job_ctx=_sample_job_context(job_id=job_id, run_token=run_token),
        message="failed",
        outcome=outcome,
        plan=_sample_plan_response(),
        coding=_sample_coding_response(),
        worktree=Path("."),
        candidate_commit_hash="failedcommit",
    )

    candidates = [obj for obj in added if isinstance(obj, job_store.CandidateCommit)]
    attempts = [obj for obj in added if isinstance(obj, job_store.EvaluationAttempt)]
    capsules = [obj for obj in added if isinstance(obj, job_store.DiagnosticCapsule)]
    cards = [obj for obj in added if isinstance(obj, job_store.CommitCard)]
    assert recorded is True
    assert job_row.status is JobStatus.FAILED
    assert cards == []
    assert len(candidates) == 1
    assert candidates[0].evaluation_status == "candidate_failed"
    assert candidates[0].repair_state == "eligible"
    assert candidates[0].commit_card_id is None
    assert len(attempts) == 1
    assert attempts[0].outcome_kind == "candidate_failed"
    assert len(capsules) == 1
    assert capsules[0].policy_passed is True
    assert any(
        record["module"] == "worker.job_store"
        and "Repair eligibility decided" in record["message"]
        for record in captured_logs
    )


def test_repair_eligibility_rejects_non_whitelisted_failure_kind(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    store = EvolutionJobStore(settings=settings)
    monkeypatch.setattr(
        EvolutionJobStore,
        "_ancestor_aggregate_ready",
        lambda _self, **_kwargs: True,
    )
    candidate = job_store.CandidateCommit(
        commit_hash="failed",
        git_parent_commit_hash="base",
        nearest_viable_ancestor_hash="base",
        publication_status="published",
        evaluation_status="candidate_failed",
        failure_stage="evaluation",
        failure_kind="evaluator_error",
        repair_state="audit_only",
        lifecycle_status="active",
        failed_depth=0,
        repair_attempts=0,
    )
    outcome = EvaluationOutcome(
        outcome_kind="candidate_failed",
        failure=EvaluationFailureResult(
            failure_stage="evaluation",
            failure_kind="evaluator_error",
            repairability="repairable",
            safe_failure_summary="Evaluator exploded.",
        ),
    )
    capsule = job_store.DiagnosticCapsule(policy_version="v", policy_passed=True, payload={})

    state = store._decide_repair_state(  # type: ignore[attr-defined]
        session=SimpleNamespace(),
        job=SimpleNamespace(job_kind="evolution", is_seed_job=False),
        job_ctx=_sample_job_context(job_id=uuid.uuid4(), run_token=uuid.uuid4()),
        candidate=candidate,
        outcome=outcome,
        capsule=capsule,
    )

    assert state == "ineligible"


def test_mark_job_failed_updates_running_job_when_run_token_matches(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.status = JobStatus.RUNNING
            self.completed_at = None
            self.run_token = run_token
            self.worker_id = "worker-01"
            self.heartbeat_at = datetime.now(timezone.utc)
            self.lease_expires_at = datetime.now(timezone.utc)
            self.last_error = None

    job_row = DummyJob()

    class DummyResult:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def scalar_one_or_none(self) -> Any:
            return self.obj

    class DummySession:
        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult(job_row)

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    recorded = store.mark_job_failed(job_id, "boom", run_token=run_token)

    assert recorded is True
    assert job_row.status is JobStatus.FAILED
    assert job_row.completed_at is not None
    assert job_row.run_token is None
    assert job_row.worker_id is None
    assert job_row.heartbeat_at is None
    assert job_row.lease_expires_at is None
    assert job_row.last_error == "boom"


def test_mark_job_failed_exhausts_repair_source_after_pre_candidate_repair_failure(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: repair jobs that failed before commit creation stranded sources in repairing."""

    settings.failed_candidate_repair_max_attempts = 1
    job_id = uuid.uuid4()
    run_token = uuid.uuid4()
    source_id = uuid.uuid4()
    source = SimpleNamespace(repair_state="repairing", repair_attempts=1)

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.status = JobStatus.RUNNING
            self.completed_at = None
            self.run_token = run_token
            self.worker_id = "worker-01"
            self.heartbeat_at = datetime.now(timezone.utc)
            self.lease_expires_at = datetime.now(timezone.utc)
            self.last_error = None
            self.job_kind = "repair"
            self.repair_source_candidate_id = source_id

    job_row = DummyJob()

    class DummyResult:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def scalar_one_or_none(self) -> Any:
            return self.obj

    class DummySession:
        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult(job_row)

        def get(self, model: Any, row_id: uuid.UUID) -> Any:
            if model is job_store.CandidateCommit and row_id == source_id:
                return source
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    recorded = store.mark_job_failed(job_id, "patch conflict", run_token=run_token)

    assert recorded is True
    assert job_row.status is JobStatus.FAILED
    assert source.repair_state == "exhausted"


def test_mark_job_failed_rejects_stale_run_token_without_overwriting_state(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()

    class DummyResult:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

        def scalar_one_or_none(self) -> Any:
            return self.obj

    class DummySession:
        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult(None)

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    recorded = store.mark_job_failed(job_id, "boom", run_token=uuid.uuid4())

    assert recorded is False


def test_mark_job_failed_without_run_token_preserves_existing_failed_job(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: unowned failure writes must not erase stale-lease failure signals."""

    job_id = uuid.uuid4()
    completed_at = datetime(2026, 3, 25, 8, 45, tzinfo=timezone.utc)
    stale_failure = "Lease expired after missing heartbeat; recovered by scheduler (attempt=4)."

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.status = JobStatus.FAILED
            self.completed_at = completed_at
            self.run_token = None
            self.worker_id = None
            self.heartbeat_at = None
            self.lease_expires_at = None
            self.last_error = stale_failure

    job_row = DummyJob()

    class DummySession:
        def get(self, _model: Any, row_id: uuid.UUID) -> DummyJob | None:
            assert row_id == job_id
            return job_row

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    recorded = store.mark_job_failed(job_id, "late worker startup failure")

    assert recorded is False
    assert job_row.status is JobStatus.FAILED
    assert job_row.completed_at == completed_at
    assert job_row.last_error == stale_failure
