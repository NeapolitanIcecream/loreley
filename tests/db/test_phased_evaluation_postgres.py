from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import time
from typing import Any, cast
import uuid

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

import loreley.api.services.jobs as jobs_service
import loreley.core.worker.evaluation_runtime as runtime_module
import loreley.core.worker.job_store as job_store_module
from loreley.api.services.jobs import get_latest_evaluation_attempt_payload
from loreley.config import Settings
from loreley.core.progress import load_campaign_progress
from loreley.core.worker.coding import CodingAgentResponse, ExecutionReport
from loreley.core.worker.evaluation_runtime import (
    EvaluationRuntimeCoordinator,
    EvaluationRuntimeError,
)
from loreley.core.worker.evaluator import (
    EvalPass,
    EvaluationArtifact,
    EvaluationMeasurement,
    EvaluationPreparation,
    Evaluator,
    MeasurementEvidence,
)
from loreley.core.worker.evolution import EvolutionWorker, JobContext
from loreley.core.worker.job_store import CandidateCommitRecord, EvolutionJobStore
from loreley.core.worker.planning import PlanDocument, PlanningAgentResponse
from loreley.core.worker.repository import CheckoutContext
from loreley.db.base import Base
from loreley.db.models import (
    CandidateCommit,
    EvaluationArtifactRecord,
    EvaluationAttempt,
    EvaluationMeasurement as EvaluationMeasurementRow,
    EvolutionJob,
    JobStatus,
    MapElitesArchiveCell,
)
from tests.support import TestSettings


POSTGRES_TEST_DSN = os.getenv("LORELEY_TEST_DATABASE_URL") or os.getenv(
    "LORELEY_POSTGRES_TEST_DSN"
)


class _SharedIdentityPhasedEvaluator:
    evaluation_protocol = "phased-v1"
    evaluation_concurrency_scope = "measurement"

    def __init__(self, event_log: str) -> None:
        self.event_log = event_log

    def _record(self, event: str) -> None:
        with Path(self.event_log).open("a", encoding="utf-8") as handle:
            handle.write(event + "\n")

    def prepare(self, context: Any) -> EvaluationPreparation:
        commit = str(context.candidate_commit_hash)
        self._record(f"prepare:{commit}")
        return EvaluationPreparation(
            candidate_identity="release-binary:shared",
            measurement_contract_fingerprint="benchmark-corpus-v1",
            state={"candidate_commit": commit},
            artifacts=(
                EvaluationArtifact(
                    key=f"prepare-{commit[:12]}",
                    kind="source-gate",
                    mime_type="application/json",
                    inline_payload={"candidate_commit": commit},
                    visibility="human_only",
                ),
            ),
        )

    def measure(
        self,
        _context: Any,
        _preparation: EvaluationPreparation,
    ) -> EvaluationMeasurement:
        self._record("measure")
        time.sleep(0.2)
        evidence_payload = b"stable benchmark evidence\n"
        return EvaluationMeasurement(
            data={"score": 1.25},
            evidence=(
                MeasurementEvidence(
                    key="benchmark-report",
                    sha256=hashlib.sha256(evidence_payload).hexdigest(),
                    size_bytes=len(evidence_payload),
                ),
            ),
            artifacts=(
                EvaluationArtifact(
                    key="benchmark-report",
                    kind="benchmark",
                    mime_type="text/plain",
                    inline_payload=evidence_payload,
                    visibility="human_only",
                ),
            ),
            cacheable=True,
        )

    def finalize(
        self,
        context: Any,
        preparation: EvaluationPreparation,
        measurement: EvaluationMeasurement,
        _provenance: Any,
    ) -> EvalPass:
        commit = str(context.candidate_commit_hash)
        self._record(f"finalize:{commit}")
        return EvalPass(
            summary="Source gate passed and the shared release binary was measured.",
            candidate_identity=preparation.candidate_identity,
            metrics={"name": "score", "value": measurement.data["score"]},
            artifacts=(
                EvaluationArtifact(
                    key=f"final-{commit[:12]}",
                    kind="validation",
                    mime_type="application/json",
                    inline_payload={"candidate_commit": commit, "passed": True},
                    visibility="human_only",
                ),
            ),
        )


@pytest.fixture
def postgres_engine() -> Iterator[Engine]:
    if not POSTGRES_TEST_DSN:
        pytest.skip(
            "set LORELEY_TEST_DATABASE_URL or LORELEY_POSTGRES_TEST_DSN "
            "to run PostgreSQL evaluator tests"
        )
    schema_name = f"loreley_phased_test_{uuid.uuid4().hex}"
    admin_engine = create_engine(POSTGRES_TEST_DSN, future=True)
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
    engine = create_engine(
        POSTGRES_TEST_DSN,
        connect_args={"options": f"-csearch_path={schema_name}"},
        future=True,
    )
    try:
        yield engine
    finally:
        runtime_module._lock_engine.cache_clear()  # noqa: SLF001
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
        admin_engine.dispose()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _candidate_repo(repo: Path) -> tuple[str, tuple[tuple[str, str, str], ...]]:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "candidate.txt").write_text("root\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-m", "root")
    root = _git(repo, "rev-parse", "HEAD")

    candidates: list[tuple[str, str, str]] = []
    for index in (1, 2):
        branch = f"candidate-{index}"
        _git(repo, "checkout", "-B", branch, root)
        (repo / "candidate.txt").write_text(f"candidate {index}\n", encoding="utf-8")
        _git(repo, "add", "candidate.txt")
        _git(repo, "commit", "-m", f"candidate {index}")
        commit = _git(repo, "rev-parse", "HEAD")
        tree = _git(repo, "rev-parse", f"{commit}^{{tree}}")
        candidates.append((branch, commit, tree))
    return root, tuple(candidates)


def _plan() -> PlanningAgentResponse:
    return PlanningAgentResponse(
        plan=PlanDocument(summary="plan", markdown="# Plan\n"),
        raw_output="plan",
        prompt="plan",
        command=("zero-model",),
        stderr="",
        attempts=1,
        duration_seconds=0.0,
    )


def _coding() -> CodingAgentResponse:
    return CodingAgentResponse(
        report=ExecutionReport(summary="implementation", markdown="# Report\n"),
        raw_output="implementation",
        prompt="implementation",
        command=("zero-model",),
        stderr="",
        attempts=1,
        duration_seconds=0.0,
    )


def _job_context(*, locked: Any, campaign_hash: str) -> JobContext:
    return JobContext(
        job_id=locked.job_id,
        run_token=locked.run_token,
        base_commit_hash=locked.base_commit_hash,
        island_id=locked.island_id,
        inspiration_commit_hashes=(),
        goal="exercise phased evaluation",
        constraints=(),
        acceptance_criteria=(),
        iteration_hint=None,
        notes=(),
        tags=("zero-model",),
        is_seed_job=False,
        sampling_strategy=None,
        sampling_initial_radius=None,
        sampling_radius_used=None,
        sampling_fallback_inspirations=None,
        campaign_program_hash=campaign_hash,
    )


def _worker(
    *,
    settings: Settings,
    evaluator: Evaluator,
    store: EvolutionJobStore,
) -> EvolutionWorker:
    worker = cast(Any, EvolutionWorker.__new__(EvolutionWorker))
    worker.settings = settings
    worker.evaluator = evaluator
    worker.job_store = store
    worker.evaluation_runtime = EvaluationRuntimeCoordinator(settings)
    return cast(EvolutionWorker, worker)


def test_two_source_trees_share_one_measurement_with_public_provenance(
    postgres_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Core-only E2E: prepare twice, benchmark once, finalize twice, then stop by identity."""

    Base.metadata.create_all(postgres_engine)
    with postgres_engine.connect() as connection:
        schema = str(connection.execute(text("SELECT current_schema()")).scalar_one())
    scoped_dsn = postgres_engine.url.update_query_dict(
        {"options": f"-csearch_path={schema}"}
    ).render_as_string(hide_password=False)
    session_factory = sessionmaker(
        bind=postgres_engine,
        expire_on_commit=False,
        future=True,
    )

    @contextmanager
    def scoped_session() -> Iterator[Any]:
        with session_factory.begin() as session:
            yield session

    monkeypatch.setattr(runtime_module, "session_scope", scoped_session)
    monkeypatch.setattr(job_store_module, "session_scope", scoped_session)
    monkeypatch.setattr(jobs_service, "session_scope", scoped_session)
    runtime_module._lock_engine.cache_clear()  # noqa: SLF001

    repo = tmp_path / "repo"
    root, candidates = _candidate_repo(repo)
    campaign_hash = "c" * 64
    settings = TestSettings(
        DATABASE_URL=scoped_dsn,
        EXPERIMENT_ID="phased-e2e",
        LOGS_BASE_DIR=str(tmp_path / "logs"),
        MAPELITES_EXPERIMENT_ROOT_COMMIT=root,
        SCHEDULER_MAX_UNIQUE_EVALUATION_IDENTITIES=1,
        WORKER_EVALUATOR_VERSION="phased-e2e-v1",
        WORKER_EVALUATOR_TIMEOUT_SECONDS=20,
        WORKER_EVALUATOR_MAX_CONCURRENCY=2,
        WORKER_EVALUATOR_SLOT_POLL_SECONDS=0.02,
    )
    event_log = tmp_path / "evaluator-events.log"
    evaluator = Evaluator(
        settings,
        plugin=_SharedIdentityPhasedEvaluator(str(event_log)),  # type: ignore[arg-type]
    )
    store = EvolutionJobStore(settings=settings)
    worker = _worker(settings=settings, evaluator=evaluator, store=store)
    plan = _plan()
    coding = _coding()

    job_rows = []
    with session_factory.begin() as session:
        for _candidate in candidates:
            job = EvolutionJob(
                id=uuid.uuid4(),
                status=JobStatus.PENDING,
                base_commit_hash=root,
                island_id="island-1",
                goal="exercise phased evaluation",
                campaign_program_hash=campaign_hash,
            )
            session.add(job)
            job_rows.append(job.id)

    prepared_jobs: list[tuple[JobContext, CheckoutContext, str]] = []
    for job_id, (branch, commit, tree) in zip(job_rows, candidates, strict=True):
        locked = store.start_job(job_id)
        job_ctx = _job_context(locked=locked, campaign_hash=campaign_hash)
        store.record_candidate_commit(
            CandidateCommitRecord(
                job_id=job_id,
                run_token=locked.run_token,
                commit_hash=commit,
                branch_name=branch,
                source_tree_hash=tree,
            )
        )
        prepared_jobs.append(
            (
                job_ctx,
                CheckoutContext(
                    job_id=str(job_id),
                    branch_name=branch,
                    base_commit=root,
                    worktree=repo,
                ),
                commit,
            )
        )

    def evaluate_and_persist(item: tuple[JobContext, CheckoutContext, str]) -> str:
        job_ctx, checkout, commit = item
        outcome = worker._run_evaluation(  # noqa: SLF001 - full core phase boundary
            job_ctx=job_ctx,
            checkout=checkout,
            plan=plan,
            candidate_commit=commit,
        )
        try:
            attempt_id = store.record_evaluation_observation(
                job_ctx=job_ctx,
                candidate_commit_hash=commit,
                outcome=outcome,
            )
            outcome.persisted_attempt_id = str(attempt_id)
            assert outcome.result is not None
            store.persist_success(
                job_ctx=job_ctx,
                plan=plan,
                coding=coding,
                evaluation=outcome.result,
                evaluation_outcome=outcome,
                worktree=repo,
                commit_hash=commit,
                commit_message="zero-model phased evaluation",
            )
            return commit
        finally:
            worker._release_runtime_leases(  # noqa: SLF001 - mirrors worker terminal path
                outcome,
                reason="test_attempt_persisted",
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(evaluate_and_persist, prepared_jobs))
    assert set(completed) == {candidate[1] for candidate in candidates}

    with session_factory.begin() as session:
        first_commit = candidates[0][1]
        session.add(
            MapElitesArchiveCell(
                island_id="island-1",
                cell_index=0,
                commit_hash=first_commit,
                objective_values=[1.25],
                measures=[0.0],
                timestamp=1.0,
            )
        )

    lines = event_log.read_text(encoding="utf-8").splitlines()
    assert len([line for line in lines if line.startswith("prepare:")]) == 2
    assert lines.count("measure") == 1
    assert len([line for line in lines if line.startswith("finalize:")]) == 2

    with session_factory.begin() as session:
        attempts = list(
            session.execute(
                select(EvaluationAttempt).order_by(EvaluationAttempt.created_at)
            ).scalars()
        )
        measurements = list(session.execute(select(EvaluationMeasurementRow)).scalars())
        candidates_db = list(session.execute(select(CandidateCommit)).scalars())
        source_artifacts = session.execute(
            select(func.count())
            .select_from(EvaluationArtifactRecord)
            .where(EvaluationArtifactRecord.key.like("prepare-%"))
        ).scalar_one()
        progress = load_campaign_progress(session, settings)

    assert len(attempts) == 2
    assert len(measurements) == 1
    source_attempt = next(attempt for attempt in attempts if attempt.measurement_executed)
    reused_attempt = next(attempt for attempt in attempts if attempt.measurement_reused)
    measurement = measurements[0]
    assert source_attempt.measurement_id == measurement.id
    assert source_attempt.reuse_kind == "none"
    assert source_attempt.evaluator_slot_release_reason == "measurement_completed"
    assert reused_attempt.measurement_id == measurement.id
    assert reused_attempt.reuse_kind == "measurement"
    assert reused_attempt.reused_from_attempt_id == source_attempt.id
    assert measurement.source_evaluation_attempt_id == source_attempt.id
    assert source_artifacts == 2
    assert len({candidate.source_tree_hash for candidate in candidates_db}) == 2
    assert len({candidate.evaluation_identity_key for candidate in candidates_db}) == 1

    assert progress.terminal_jobs == 2
    assert progress.succeeded_jobs == 2
    assert progress.distinct_passed_source_trees == 2
    assert progress.real_measurements == 1
    assert progress.measurement_reuses == 1
    assert progress.distinct_passed_evaluation_identities == 1
    assert progress.archive_entries == 1
    assert progress.archive_unique_evaluation_identities == 1
    assert progress.occupied_coordinates == 1
    assert progress.identity_target_reached is True
    assert progress.identity_overshoot == 0
    assert progress.unfinished_jobs == 0

    public = get_latest_evaluation_attempt_payload(job_id=reused_attempt.job_id)
    assert public is not None
    assert public["measurement_reused"] is True
    assert public["measurement_executed"] is False
    assert public["reused_from_attempt_id"] == source_attempt.id
    assert public["measurement_payload_sha256"] == measurement.payload_sha256
    assert public["measurement_evidence"] == [
        {
            "key": "benchmark-report",
            "sha256": hashlib.sha256(b"stable benchmark evidence\n").hexdigest(),
            "size_bytes": len(b"stable benchmark evidence\n"),
        }
    ]
    assert str(tmp_path) not in repr(public)

    with session_factory.begin() as session:
        row = session.get(EvaluationMeasurementRow, measurement.id)
        assert row is not None
        row.source_evaluation_attempt_id = None
    with pytest.raises(EvaluationRuntimeError, match="no original evaluation attempt"):
        worker.evaluation_runtime.lookup_measurement(measurement.cache_key)
    with session_factory.begin() as session:
        row = session.get(EvaluationMeasurementRow, measurement.id)
        assert row is not None
        row.source_evaluation_attempt_id = source_attempt.id

    with session_factory.begin() as session:
        evidence_row = session.execute(
            select(EvaluationArtifactRecord).where(
                EvaluationArtifactRecord.evaluation_attempt_id == source_attempt.id,
                EvaluationArtifactRecord.key == "benchmark-report",
            )
        ).scalar_one()
        evidence_path = Path(str(evidence_row.storage_path))
    original_evidence = evidence_path.read_bytes()
    evidence_path.write_bytes(b"tampered benchmark evidence\n")
    with pytest.raises(EvaluationRuntimeError, match="payload (?:size|hash) drifted"):
        worker.evaluation_runtime.lookup_measurement(measurement.cache_key)
    evidence_path.write_bytes(original_evidence)

    with session_factory.begin() as session:
        row = session.get(EvaluationMeasurementRow, measurement.id)
        assert row is not None
        row.evidence_manifest = [{"key": "tampered", "sha256": "0" * 64}]
    with pytest.raises(EvaluationRuntimeError, match="evidence manifest"):
        worker.evaluation_runtime.lookup_measurement(measurement.cache_key)
