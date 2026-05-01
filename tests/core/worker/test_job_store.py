from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
from loreley.core.worker.evaluator import EvaluationDiagnostic, EvaluationMetric, EvaluationResult
from loreley.core.worker.evolution import JobContext
from loreley.core.worker.job_store import (
    EvolutionJobStore,
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


def _materialized_benchmark_artifact() -> MaterializedEvaluationArtifact:
    return MaterializedEvaluationArtifact(
        key="benchmark_report",
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
