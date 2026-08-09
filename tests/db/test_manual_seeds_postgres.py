from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, cast
import uuid

import pytest
from git import Repo
from rich.console import Console
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import loreley.core.experiments as experiments_module
import loreley.core.manual_seeds as manual_seeds_module
import loreley.core.worker.evaluation_runtime as evaluation_runtime_module
import loreley.core.worker.evolution as evolution_module
import loreley.core.worker.job_store as job_store_module
import loreley.core.worker.repository as repository_module
import loreley.db.base as base_module
import loreley.scheduler.ingestion as ingestion_module
import loreley.scheduler.job_scheduler as job_scheduler_module
from loreley.core.progress import load_campaign_progress
from loreley.core.manual_seeds import ManualSeedError, import_manual_seed_manifest
from loreley.core.map_elites.sampler import MapElitesSampler
from loreley.core.map_elites.types import (
    CommitEmbeddingArtifacts,
    MapElitesInsertionResult,
    MapElitesRecord,
)
from loreley.core.worker.evaluator import (
    EvalPass,
    EvaluationArtifact,
    EvaluationMeasurement,
    EvaluationPreparation,
    Evaluator,
    MeasurementEvidence,
)
from loreley.core.worker.evolution import EvolutionWorker
from loreley.core.worker.job_store import EvolutionJobStore
from loreley.core.worker.repository import WorkerRepository
from loreley.db.models import (
    CandidateCommit,
    EvaluationArtifactRecord,
    EvaluationAttempt,
    EvaluationMeasurement as EvaluationMeasurementRow,
    EvolutionJob,
    JobArtifacts,
    JobStatus,
    MapElitesArchiveCell,
)
from loreley.scheduler.ingestion import MapElitesIngestion
from loreley.scheduler.job_scheduler import JobScheduler
from tests.support import TestSettings


POSTGRES_TEST_DSN = os.getenv("LORELEY_TEST_DATABASE_URL") or os.getenv(
    "LORELEY_POSTGRES_TEST_DSN"
)


@pytest.fixture
def postgres_engine() -> Iterator[Engine]:
    if not POSTGRES_TEST_DSN:
        pytest.skip(
            "set LORELEY_TEST_DATABASE_URL or LORELEY_POSTGRES_TEST_DSN "
            "to run PostgreSQL manual-seed tests"
        )
    schema_name = f"loreley_manual_seed_test_{uuid.uuid4().hex}"
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


def _seed_repository(tmp_path: Path, *, count: int) -> tuple[Path, Path, str, list[str]]:
    remote = tmp_path / "remote.git"
    repo = tmp_path / "source"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "remote", "add", "origin", str(remote))
    (repo / "src").mkdir()
    (repo / "src" / "root.txt").write_text("root\n", encoding="utf-8")
    (repo / "loreley.program.md").write_text(
        "# Test campaign\n\n"
        "## Goal\n\nEvaluate supplied candidates.\n\n"
        "## Editable scope\n\n- `src/**`\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "root")
    root = _git(repo, "rev-parse", "HEAD")
    _git(repo, "push", "origin", f"{root}:refs/heads/main")

    commits: list[str] = []
    for ordinal in range(1, count + 1):
        ref = f"refs/heads/loreley-seeds/seed-{ordinal:02d}"
        _git(repo, "checkout", "--detach", root)
        path = repo / "src" / f"seed_{ordinal:02d}.txt"
        path.write_text(f"seed {ordinal}\n", encoding="utf-8")
        _git(repo, "add", str(path.relative_to(repo)))
        _git(repo, "commit", "-m", f"seed {ordinal}")
        commit = _git(repo, "rev-parse", "HEAD")
        _git(repo, "push", "origin", f"{commit}:{ref}")
        commits.append(commit)
    return repo, remote, root, commits


def _write_manifest(path: Path, commits: list[str]) -> Path:
    lines = ["schema_version: 1", "seeds:"]
    for ordinal, commit in enumerate(commits, start=1):
        lines.extend(
            (
                f"  - key: seed-{ordinal:02d}",
                f"    commit: {commit}",
                f"    remote_ref: refs/heads/loreley-seeds/seed-{ordinal:02d}",
                f"    summary: Independent seed direction {ordinal}.",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _settings(*, repo: Path, remote: Path, root: str, endpoint: int) -> TestSettings:
    return TestSettings(
        experiment_id=f"manual-seed-pg-{uuid.uuid4().hex}",
        scheduler_repo_root=str(repo),
        worker_repo_worktree=str(repo),
        worker_repo_remote_url=str(remote),
        mapelites_experiment_root_commit=root,
        scheduler_max_total_jobs=endpoint,
        scheduler_max_unfinished_jobs=4,
        db_auto_migrate=True,
    )


def _patch_database(monkeypatch: pytest.MonkeyPatch, engine: Engine) -> None:
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    @contextmanager
    def scoped_session() -> Iterator[Session]:
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(base_module, "get_engine", lambda: engine)
    monkeypatch.setattr(base_module, "session_scope", scoped_session)
    monkeypatch.setattr(manual_seeds_module, "get_engine", lambda: engine)
    monkeypatch.setattr(manual_seeds_module, "session_scope", scoped_session)
    monkeypatch.setattr(experiments_module, "session_scope", scoped_session)
    monkeypatch.setattr(job_scheduler_module, "session_scope", scoped_session)
    monkeypatch.setattr(evaluation_runtime_module, "session_scope", scoped_session)
    monkeypatch.setattr(evolution_module, "session_scope", scoped_session)
    monkeypatch.setattr(job_store_module, "session_scope", scoped_session)
    monkeypatch.setattr(repository_module, "session_scope", scoped_session)
    monkeypatch.setattr(ingestion_module, "session_scope", scoped_session)
    evaluation_runtime_module._lock_engine.cache_clear()  # noqa: SLF001


class _ExplodingAgent:
    def run(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("manual seeds must not invoke planning or coding")


class _SharedBinaryEvaluator:
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
            measurement_contract_fingerprint="zero-model-corpus-v1",
            state={"commit": commit},
            artifacts=(
                EvaluationArtifact(
                    key=f"source-{commit[:12]}",
                    kind="source-gate",
                    mime_type="application/json",
                    inline_payload={"commit": commit},
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
        payload = b"one accepted measurement\n"
        return EvaluationMeasurement(
            data={"score": 1.1},
            evidence=(
                MeasurementEvidence(
                    key="benchmark-report",
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                ),
            ),
            artifacts=(
                EvaluationArtifact(
                    key="benchmark-report",
                    kind="benchmark",
                    mime_type="text/plain",
                    inline_payload=payload,
                    visibility="human_only",
                ),
            ),
            cacheable=True,
        )

    def finalize(
        self,
        _context: Any,
        preparation: EvaluationPreparation,
        measurement: EvaluationMeasurement,
        _provenance: Any,
    ) -> EvalPass:
        self._record("finalize")
        return EvalPass(
            summary="Candidate passed the shared zero-model evaluator.",
            candidate_identity=preparation.candidate_identity,
            metrics={"name": "score", "value": measurement.data["score"]},
        )


class _SingleAdmissionManager:
    def __init__(self) -> None:
        self.calls = 0

    def ingest(
        self,
        *,
        commit_hash: str,
        island_id: str | None,
        snapshot_session: Session | None,
        **_kwargs: Any,
    ) -> MapElitesInsertionResult:
        self.calls += 1
        assert self.calls == 1, "equivalent identity should bypass a second manager ingest"
        assert snapshot_session is not None
        effective_island = island_id or "default"
        record = MapElitesRecord(
            commit_hash=commit_hash,
            island_id=effective_island,
            cell_index=0,
            objective_values=(1.1,),
            objective_scores=(1.1,),
            measures=(0.0,),
            timestamp=1.0,
        )
        snapshot_session.add(
            MapElitesArchiveCell(
                island_id=effective_island,
                cell_index=0,
                commit_hash=commit_hash,
                objective_values=[1.1],
                measures=[0.0],
                timestamp=1.0,
            )
        )
        return MapElitesInsertionResult(
            status=1,
            delta=1.1,
            record=record,
            artifacts=CommitEmbeddingArtifacts(
                repo_state_stats=None,
                preprocessed_files=(),
                code_embedding=None,
                final_embedding=None,
            ),
            message="admitted",
        )


def test_public_manual_seed_import_is_atomic_idempotent_and_u_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
) -> None:
    repo, remote, root, commits = _seed_repository(tmp_path, count=8)
    manifest = _write_manifest(tmp_path / "seeds.yaml", commits)
    settings = _settings(repo=repo, remote=remote, root=root, endpoint=8)
    _patch_database(monkeypatch, postgres_engine)

    first = import_manual_seed_manifest(settings=settings, manifest_path=manifest)
    replay = import_manual_seed_manifest(settings=settings, manifest_path=manifest)

    assert first.created == 8
    assert first.existing == 0
    assert replay.created == 0
    assert replay.existing == 8
    with postgres_engine.connect() as connection:
        rows = list(
            connection.execute(
                select(
                    EvolutionJob.status,
                    EvolutionJob.input_candidate_commit_hash,
                    EvolutionJob.input_provenance,
                ).order_by(EvolutionJob.created_at, EvolutionJob.id)
            ).all()
        )
    assert [row.status for row in rows] == [JobStatus.STAGED] * 8
    assert [row.input_candidate_commit_hash for row in rows] == commits
    assert [row.input_provenance["seed_ordinal"] for row in rows] == list(range(1, 9))

    monkeypatch.setattr(
        job_scheduler_module,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: cast(Any, object()),
    )
    scheduler = JobScheduler(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    assert scheduler.promote_staged_jobs() == 4
    with postgres_engine.begin() as connection:
        counts = dict(
            connection.execute(
                select(EvolutionJob.status, func.count(EvolutionJob.id)).group_by(
                    EvolutionJob.status
                )
            ).all()
        )
        connection.execute(
            EvolutionJob.__table__.update()
            .where(EvolutionJob.status == JobStatus.PENDING)
            .values(status=JobStatus.SUCCEEDED)
        )
    assert counts == {JobStatus.PENDING: 4, JobStatus.STAGED: 4}
    assert scheduler.promote_staged_jobs() == 4

    wrong_experiment = settings.model_copy(update={"experiment_id": "wrong-experiment"})
    with pytest.raises(Exception, match="EXPERIMENT_ID.*database marker"):
        import_manual_seed_manifest(settings=wrong_experiment, manifest_path=manifest)
    with postgres_engine.connect() as connection:
        assert connection.execute(select(func.count(EvolutionJob.id))).scalar_one() == 8


def test_manual_seed_import_rolls_back_when_endpoint_cannot_fit_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
) -> None:
    repo, remote, root, commits = _seed_repository(tmp_path, count=8)
    manifest = _write_manifest(tmp_path / "too-many-seeds.yaml", commits)
    settings = _settings(repo=repo, remote=remote, root=root, endpoint=7)
    _patch_database(monkeypatch, postgres_engine)

    with pytest.raises(ManualSeedError, match="would exceed SCHEDULER_MAX_TOTAL_JOBS=7"):
        import_manual_seed_manifest(settings=settings, manifest_path=manifest)

    with postgres_engine.connect() as connection:
        assert connection.execute(select(func.count(EvolutionJob.id))).scalar_one() == 0


def test_manual_seeds_run_through_worker_cache_ingestion_and_identity_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    postgres_engine: Engine,
) -> None:
    """Public zero-model flow: import -> worker -> evaluator -> ingestion -> endpoint."""

    repo, remote, root, commits = _seed_repository(tmp_path, count=2)
    manifest = _write_manifest(tmp_path / "e2e-seeds.yaml", commits)
    with postgres_engine.connect() as connection:
        schema = str(
            connection.execute(text("SELECT current_schema()"))
            .scalar_one()
        ).strip()
    scoped_dsn = postgres_engine.url.update_query_dict(
        {"options": f"-csearch_path={schema}"}
    ).render_as_string(hide_password=False)
    worker_repo = tmp_path / "fresh-worker"
    settings = _settings(repo=repo, remote=remote, root=root, endpoint=2).model_copy(
        update={
            "database_url": scoped_dsn,
            "worker_repo_worktree": str(worker_repo),
            "worker_repo_branch": "main",
            "logs_base_dir": str(tmp_path / "logs"),
            "scheduler_max_unfinished_jobs": 2,
            "scheduler_max_unique_evaluation_identities": 1,
            "worker_evaluator_version": "manual-seed-e2e-v1",
            "worker_evaluator_timeout_seconds": 20,
            "worker_evaluator_max_concurrency": 1,
            "worker_evaluator_slot_poll_seconds": 0.01,
        }
    )
    _patch_database(monkeypatch, postgres_engine)
    monkeypatch.setattr(
        job_scheduler_module,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: cast(Any, object()),
    )

    imported = import_manual_seed_manifest(settings=settings, manifest_path=manifest)
    assert imported.created == 2
    scheduler = JobScheduler(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    assert scheduler.promote_staged_jobs() == 2

    event_log = tmp_path / "evaluator-events.log"
    worker = EvolutionWorker(
        settings=settings,
        repository=WorkerRepository(settings),
        planning_agent=cast(Any, _ExplodingAgent()),
        coding_agent=cast(Any, _ExplodingAgent()),
        evaluator=Evaluator(
            settings,
            plugin=_SharedBinaryEvaluator(str(event_log)),  # type: ignore[arg-type]
        ),
        job_store=EvolutionJobStore(settings=settings),
    )
    with postgres_engine.connect() as connection:
        job_ids = list(
            connection.execute(
                select(EvolutionJob.id).order_by(EvolutionJob.created_at, EvolutionJob.id)
            ).scalars()
        )
    results = [worker.run(job_id) for job_id in job_ids]
    assert all(result.plan is None and result.coding is None for result in results)

    events = event_log.read_text(encoding="utf-8").splitlines()
    assert len([event for event in events if event.startswith("prepare:")]) == 2
    assert events.count("measure") == 1
    assert events.count("finalize") == 2

    manager = _SingleAdmissionManager()
    ingestion = MapElitesIngestion(
        settings=settings,
        console=Console(record=True),
        repo_root=repo,
        repo=Repo(repo),
        manager=cast(Any, manager),
    )
    assert ingestion.ingest_completed_jobs() == 1
    assert manager.calls == 1

    with sessionmaker(bind=postgres_engine, expire_on_commit=False, future=True).begin() as session:
        progress = load_campaign_progress(session, settings)
        attempts = list(
            session.execute(
                select(EvaluationAttempt).order_by(
                    EvaluationAttempt.created_at,
                    EvaluationAttempt.id,
                )
            ).scalars()
        )
        measurements = list(session.execute(select(EvaluationMeasurementRow)).scalars())
        candidates = list(session.execute(select(CandidateCommit)).scalars())
        artifacts = list(session.execute(select(EvaluationArtifactRecord)).scalars())
        fixed = list(session.execute(select(JobArtifacts)).scalars())

    assert progress.terminal_jobs == 2
    assert progress.unfinished_jobs == 0
    assert progress.distinct_passed_source_trees == 2
    assert progress.distinct_passed_evaluation_identities == 1
    assert progress.real_measurements == 1
    assert progress.measurement_reuses == 1
    assert progress.archive_entries == 1
    assert progress.archive_unique_evaluation_identities == 1
    assert progress.identity_target_reached is True
    assert len(attempts) == 2
    assert {attempt.attempt_ordinal for attempt in attempts} == {1}
    assert all(attempt.run_token is not None for attempt in attempts)
    assert all(attempt.artifact_paths for attempt in attempts)
    assert len(measurements) == 1
    assert len(candidates) == 2
    assert len({candidate.source_tree_hash for candidate in candidates}) == 2
    assert len({candidate.evaluation_identity_key for candidate in candidates}) == 1
    assert artifacts
    assert all(artifact.evaluation_attempt_id is not None for artifact in artifacts)
    assert fixed
    assert all(row.planning_prompt_path is None for row in fixed)
    assert all(row.coding_prompt_path is None for row in fixed)

    restarted = JobScheduler(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    assert restarted.promote_staged_jobs() == 0
    assert restarted.cancel_pending_for_identity_endpoint() == 0
