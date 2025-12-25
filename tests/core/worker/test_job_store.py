from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any

import pytest
from sqlalchemy.exc import SQLAlchemyError

import loreley.core.worker.job_store as job_store
from loreley.config import Settings
from loreley.core.worker.coding import (
    CodingAgentResponse,
    CodingPlanExecution,
    CodingStepReport,
    StepExecutionStatus,
)
from loreley.core.worker.evaluator import EvaluationMetric, EvaluationResult
from loreley.core.worker.evolution import JobContext
from loreley.core.worker.job_store import (
    EvolutionJobStore,
    JobPreconditionError,
)
from loreley.core.worker.planning import PlanStep, PlanningAgentResponse, PlanningPlan
from loreley.db.models import Experiment, EvolutionJob, JobStatus


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

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.base_commit_hash = "abc123"
            self.island_id = "island"
            self.experiment_id = uuid.uuid4()
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
            self.last_error = "previous"
            self.experiment = type(
                "Exp",
                (),
                {"repository_id": uuid.uuid4()},
            )()

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
    assert dummy_job.last_error is None
    assert locked.job_id == job_id
    assert locked.base_commit_hash == dummy_job.base_commit_hash
    assert locked.inspiration_commit_hashes == tuple(dummy_job.inspiration_commit_hashes)
    assert locked.repository_id == dummy_job.experiment.repository_id


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
            self.experiment_id = None
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
            self.experiment = None

    @contextmanager
    def invalid_status_scope() -> Any:
        yield DummySession(DummyJob())

    monkeypatch.setattr(job_store, "session_scope", invalid_status_scope)
    with pytest.raises(JobPreconditionError):
        store.start_job(uuid.uuid4())


def test_persist_success_updates_job_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    experiment_id = uuid.uuid4()
    repository_id = uuid.uuid4()

    class DummyJob:
        def __init__(self) -> None:
            self.id = job_id
            self.status = JobStatus.PENDING
            self.plan_summary: str | None = None
            self.completed_at = None
            self.last_error = "err"
            self.experiment_id = experiment_id
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
    experiment_row = Experiment(id=experiment_id, repository_id=repository_id, config_hash="hash")  # type: ignore[arg-type]
    added: list[Any] = []

    class DummySession:
        def __init__(self) -> None:
            self.added = added

        def get(self, model: Any, key: Any) -> Any:
            if model is EvolutionJob and key == job_id:
                return job_row
            if model is Experiment and key == experiment_id:
                return experiment_row
            return None

        def add(self, obj: Any) -> None:
            self.added.append(obj)

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_store, "session_scope", fake_scope)
    store = EvolutionJobStore(settings=settings)

    plan_step = PlanStep(
        step_id="s1",
        title="t",
        intent="i",
        actions=("a",),
        files=(),
        dependencies=(),
        validation=("v",),
        risks=(),
        references=(),
    )
    plan = PlanningPlan(
        summary="plan",
        rationale="r",
        focus_metrics=("f",),
        guardrails=("g",),
        risks=("r1",),
        validation=("v",),
        steps=(plan_step,),
        handoff_notes=(),
        fallback_plan=None,
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
    step_report = CodingStepReport(
        step_id="s1",
        status=StepExecutionStatus.COMPLETED,
        summary="ok",
        files=(),
        commands=(),
    )
    execution = CodingPlanExecution(
        implementation_summary="impl",
        commit_message="msg",
        step_results=(step_report,),
        tests_executed=(),
        tests_recommended=(),
        follow_up_items=(),
        notes=(),
    )
    coding_response = CodingAgentResponse(
        execution=execution,
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
        base_commit_hash="base",
        island_id="island",
        experiment_id=experiment_id,
        repository_id=repository_id,
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
        worktree=".",  # dummy path; artifacts/git diff are best-effort in tests
        commit_hash="newcommit",
        commit_message="msg",
    )

    assert job_row.status is JobStatus.SUCCEEDED
    assert job_row.plan_summary == "plan"
    assert job_row.result_commit_hash == "newcommit"
    metadata = [obj for obj in added if isinstance(obj, job_store.CommitCard)]
    metrics = [obj for obj in added if isinstance(obj, job_store.Metric)]
    assert len(metadata) == 1
    assert metadata[0].commit_hash == "newcommit"
    assert len(metrics) == 1
    assert metrics[0].name == "score"

