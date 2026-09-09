from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from time import monotonic
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from rich.console import Console
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

import loreley.core.fresh_comparison as comparison
import loreley.core.map_elites.snapshot as snapshot_module
from loreley.config import resolve_objective_contract
from loreley.core.comparison_contract import ComparisonContractError
from loreley.core.evolution_events import (
    COMPARISON_DECIDED,
    COMPARISON_MEASURED,
    COMPARISON_PREPARED,
)
from loreley.core.worker.evaluation_runtime import EvaluationRuntimeError
from loreley.core.worker.evaluator import EvaluationMetric, EvaluationResult
from loreley.core.map_elites.code_embedding import CommitCodeEmbedding
from loreley.core.map_elites.dimension_reduction import PCAProjection
from loreley.core.map_elites.manager import MapElitesManager
from loreley.core.map_elites.objectives import ObjectiveSpec
from loreley.core.map_elites.snapshot import DatabaseSnapshotStore, SnapshotElite, SnapshotUpdate
from loreley.db.base import INSTANCE_SCHEMA_VERSION
from loreley.db.migrations.runner import ensure_schema_current
from loreley.db.models import (
    CandidateCommit,
    EvaluationAttempt,
    EvolutionEvent,
    EvolutionJob,
    JobStatus,
    MapElitesArchiveCell,
    MapElitesPcaHistory,
)
from loreley.scheduler.ingestion import JobSnapshot, MapElitesIngestion
from tests.support import TestSettings

POSTGRES_TEST_DSN = os.getenv("LORELEY_TEST_DATABASE_URL") or os.getenv(
    "LORELEY_POSTGRES_TEST_DSN"
)
COMMIT = "c" * 40
CAMPAIGN = "a" * 64


@pytest.fixture
def comparison_engine() -> Iterator[Engine]:
    if not POSTGRES_TEST_DSN:
        pytest.skip("set LORELEY_TEST_DATABASE_URL to run fresh comparison PostgreSQL tests")
    schema = f"loreley_fresh_comparison_test_{uuid4().hex}"
    admin = create_engine(POSTGRES_TEST_DSN, future=True)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(
        POSTGRES_TEST_DSN,
        connect_args={"options": f"-csearch_path={schema}"},
        future=True,
    )
    settings = TestSettings(
        EXPERIMENT_ID="fresh-comparison-test",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="a" * 40,
        DB_AUTO_MIGRATE=True,
    )
    ensure_schema_current(
        engine=engine,
        settings=settings,
        target_version=INSTANCE_SCHEMA_VERSION,
        auto_migrate=True,
    )
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def store(comparison_engine, monkeypatch):
    factory = sessionmaker(bind=comparison_engine, future=True)

    @contextmanager
    def scoped_session():
        with factory.begin() as session:
            yield session

    monkeypatch.setattr(comparison, "session_scope", scoped_session)
    settings = SimpleNamespace(
        experiment_id="fresh-comparison-test",
        mapelites_admission_policy="fresh_comparison",
        mapelites_comparison_confidence=0.95,
    )

    def prepare_comparison(**kwargs):
        return {
            "candidate_commit_hash": kwargs["commit_hash"],
            "island_id": kwargs["island_id"],
            "cell_index": 7,
            "measures": [0.5],
            "projection_fingerprint": "projection-v1",
            "incumbent_commit_hash": "b" * 40,
            "objective_name": "cpu",
            "higher_is_better": False,
        }

    manager = SimpleNamespace(prepare_comparison=prepare_comparison)
    monkeypatch.setattr(comparison, "MapElitesManager", lambda **kwargs: manager)
    return SimpleNamespace(factory=factory, settings=settings, manager=manager)


def _job(store, *, status=JobStatus.RUNNING, run_token=None, ingestion_status=None):
    job_id, token = uuid4(), run_token or uuid4()
    with store.factory.begin() as session:
        session.add(
            EvolutionJob(
                id=job_id,
                status=status,
                base_commit_hash="a" * 40,
                island_id="main",
                run_token=token,
                result_commit_hash=COMMIT if status == JobStatus.SUCCEEDED else None,
                campaign_program_hash=CAMPAIGN,
                ingestion_status=ingestion_status,
            )
        )
    return job_id, token


def _prepare(store, job_id, token, *, deadline=None):
    return comparison.prepare_context(
        settings=store.settings,
        request=comparison.ComparisonRequest(
            job_id=job_id,
            run_token=token,
            commit_hash=COMMIT,
            island_id="main",
            repo_root=Path("."),
            evaluator_name="paired",
            evaluator_version="1",
            campaign_program_hash=CAMPAIGN,
            candidate_identity="binary-v1",
            measurement_contract_fingerprint="benchmark-v1",
        ),
        deadline=deadline if deadline is not None else monotonic() + 2,
    )


def _metrics(value=99.9):
    return [{"name": "cpu", "value": value, "higher_is_better": False}]


def _result(context, **updates):
    payload = {"context_id": context["context_id"], "complete": True}
    if context["incumbent_commit_hash"] is not None:
        payload.update(
            incumbent_value=100.0,
            improvement_lower_bound=0.05,
            confidence_level=0.95,
            sample_count=12,
        )
    payload.update(updates)
    return EvaluationResult(
        summary="Fresh paired measurement",
        metrics=(EvaluationMetric(name="cpu", value=99.9, higher_is_better=False),),
        extra={"fresh_comparison": payload},
    )


def _measure(store, job_id, token, context, **updates):
    comparison.record_measurement(
        settings=store.settings,
        job_id=job_id,
        run_token=token,
        context=context,
        result=_result(context, **updates),
    )


def _succeed(store, job_id, token, **attempt_overrides):
    attempt_values = dict(
        job_id=job_id,
        run_token=token,
        attempt_ordinal=1,
        outcome_kind="passed",
        protocol="phased-v1",
        measurement_executed=True,
        measurement_reused=False,
        evaluator_name="paired",
        evaluator_version="1",
        campaign_program_hash=CAMPAIGN,
        candidate_identity="binary-v1",
        measurement_contract_fingerprint="benchmark-v1",
    )
    attempt_values.update(attempt_overrides)
    with store.factory.begin() as session:
        job = session.get(EvolutionJob, job_id)
        job.status = JobStatus.SUCCEEDED
        job.result_commit_hash = COMMIT
        job.run_token = None
        attempt = EvaluationAttempt(**attempt_values)
        session.add(attempt)
        session.flush()
        return attempt.id


def _load(store, job_id, *, metrics=None):
    with store.factory.begin() as session:
        return comparison.load_handoff(
            session=session,
            settings=store.settings,
            job_id=job_id,
            commit_hash=COMMIT,
            metrics=_metrics() if metrics is None else metrics,
        )


def _event_count(store, kind):
    with store.factory() as session:
        return session.scalar(
            select(func.count(EvolutionEvent.id)).where(EvolutionEvent.event_type == kind)
        )


def test_prepare_measure_load_and_ack_round_trip(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context)
    _succeed(store, job_id, token)

    issued, decision = _load(store, job_id)
    assert issued == context
    assert decision["allowed"] is True
    assert decision["candidate_value"] == 99.9
    assert decision["incumbent_value"] == 100
    with store.factory.begin() as session:
        comparison.acknowledge(session, job_id=job_id, context=issued, allowed=True)
        session.get(EvolutionJob, job_id).ingestion_status = "succeeded"
    assert _event_count(store, COMPARISON_PREPARED) == 1
    assert _event_count(store, COMPARISON_MEASURED) == 1
    assert _event_count(store, COMPARISON_DECIDED) == 1
    with store.factory() as session:
        assert comparison._pending_cycle(session, uuid4()) is False


def test_empty_cell_survives_event_sanitization_and_revalidation(store) -> None:
    prepare = store.manager.prepare_comparison

    def empty(**kwargs):
        return {**prepare(**kwargs), "incumbent_commit_hash": None}

    store.manager.prepare_comparison = empty
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context)
    _succeed(store, job_id, token)
    issued, decision = _load(store, job_id)
    assert issued["incumbent_commit_hash"] is None
    assert decision["allowed"] is True
    assert "improvement_lower_bound" not in decision


def test_uncertain_result_is_successful_measurement_but_not_admission(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context, improvement_lower_bound=-0.05)
    _succeed(store, job_id, token)
    assert _load(store, job_id)[1]["allowed"] is False


def test_result_cannot_invent_a_core_context(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    context["context_id"] = str(uuid4())
    with pytest.raises(EvaluationRuntimeError, match="core-issued"):
        _measure(store, job_id, token, context)
    assert _event_count(store, COMPARISON_MEASURED) == 0


def test_mutated_plugin_context_cannot_change_issued_objective_or_incumbent(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    altered = {**context, "objective_name": "other", "incumbent_commit_hash": None}
    with pytest.raises(ComparisonContractError, match="incumbent_value"):
        _measure(store, job_id, token, altered)
    assert _event_count(store, COMPARISON_MEASURED) == 0


@pytest.mark.parametrize(
    "updates", [{"complete": False}, {"improvement_lower_bound": float("nan")}, {"sample_count": 1}]
)
def test_invalid_or_incomplete_measurement_is_not_recorded_as_no_improvement(store, updates):
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    with pytest.raises(ComparisonContractError):
        _measure(store, job_id, token, context, **updates)
    assert _event_count(store, COMPARISON_MEASURED) == 0


def test_stale_worker_cannot_prepare_or_record_measurement(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    with store.factory.begin() as session:
        session.get(EvolutionJob, job_id).run_token = uuid4()
    with pytest.raises(EvaluationRuntimeError, match="inactive worker run"):
        _prepare(store, job_id, token)
    with pytest.raises(EvaluationRuntimeError, match="inactive worker run"):
        _measure(store, job_id, token, context)
    assert _event_count(store, COMPARISON_MEASURED) == 0


@pytest.mark.parametrize(
    "status,ingestion,current_run,expected",
    [
        (JobStatus.RUNNING, None, True, True),
        (JobStatus.RUNNING, None, False, False),
        (JobStatus.PENDING, None, True, False),
        (JobStatus.FAILED, None, True, False),
        (JobStatus.CANCELLED, None, True, False),
        (JobStatus.SUCCEEDED, None, True, True),
        (JobStatus.SUCCEEDED, "failed", True, True),
        (JobStatus.SUCCEEDED, "succeeded", True, False),
        (JobStatus.SUCCEEDED, "skipped", True, False),
    ],
)
def test_pending_cycle_tracks_worker_and_ingestion_lifecycle(
    store, status, ingestion, current_run, expected
):
    job_id, token = _job(store)
    _prepare(store, job_id, token)
    with store.factory.begin() as session:
        job = session.get(EvolutionJob, job_id)
        job.status = status
        job.ingestion_status = ingestion
        if status == JobStatus.SUCCEEDED:
            job.run_token = None
        elif not current_run:
            job.run_token = uuid4()
    with store.factory() as session:
        assert comparison._pending_cycle(session, uuid4()) is expected
        assert comparison._pending_cycle(session, job_id) is False


def test_next_cycle_waits_for_ingestion_not_just_measurement_completion(store) -> None:
    first, first_token = _job(store)
    context = _prepare(store, first, first_token)
    _measure(store, first, first_token, context)
    _succeed(store, first, first_token)
    next_job, next_token = _job(store)
    with pytest.raises(EvaluationRuntimeError, match="not completed ingestion"):
        _prepare(store, next_job, next_token, deadline=monotonic() + 0.025)
    with store.factory.begin() as session:
        comparison.acknowledge(session, job_id=first, context=context, allowed=True)
        session.get(EvolutionJob, first).ingestion_status = "succeeded"
    assert _prepare(store, next_job, next_token)["context_id"] != context["context_id"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"measurement_reused": True},
        {"measurement_executed": False},
        {"protocol": "one_shot"},
        {"run_token": None},
        {"run_token": UUID(int=1)},
        {"evaluator_name": "other"},
        {"evaluator_version": "other"},
        {"candidate_identity": "other"},
        {"measurement_contract_fingerprint": "other"},
        {"campaign_program_hash": "b" * 64},
    ],
)
def test_handoff_rejects_wrong_or_reused_successful_attempt(store, overrides) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context)
    _succeed(store, job_id, token, **overrides)
    with pytest.raises(EvaluationRuntimeError):
        _load(store, job_id)


def test_success_without_measurement_evidence_cannot_be_ingested(store) -> None:
    job_id, token = _job(store)
    _prepare(store, job_id, token)
    _succeed(store, job_id, token)
    with pytest.raises(EvaluationRuntimeError, match="no fresh comparison evidence"):
        _load(store, job_id)


def test_changed_persisted_candidate_metric_cannot_reuse_the_gate(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context)
    _succeed(store, job_id, token)
    with pytest.raises(ComparisonContractError, match="does not match"):
        _load(store, job_id, metrics=_metrics(99.8))


def test_changed_job_commit_cannot_reuse_comparison(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context)
    _succeed(store, job_id, token)
    with store.factory.begin() as session:
        session.get(EvolutionJob, job_id).result_commit_hash = "d" * 40
    with pytest.raises(EvaluationRuntimeError, match="changed before ingestion"):
        _load(store, job_id)


def test_ack_and_job_completion_roll_back_together(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context)
    _succeed(store, job_id, token)
    with pytest.raises(RuntimeError, match="injected transaction failure"):
        with store.factory.begin() as session:
            comparison.acknowledge(session, job_id=job_id, context=context, allowed=True)
            session.get(EvolutionJob, job_id).ingestion_status = "succeeded"
            session.flush()
            raise RuntimeError("injected transaction failure")
    assert _event_count(store, COMPARISON_DECIDED) == 0
    with store.factory() as session:
        assert session.get(EvolutionJob, job_id).ingestion_status is None
        assert comparison._pending_cycle(session, uuid4()) is True
    # Retrying from the persisted evidence remains possible after rollback.
    assert _load(store, job_id)[1]["allowed"] is True


def test_duplicate_measurement_and_ack_delivery_are_idempotent(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context)
    _measure(store, job_id, token, context)
    _succeed(store, job_id, token)
    with store.factory.begin() as session:
        comparison.acknowledge(session, job_id=job_id, context=context, allowed=True)
        comparison.acknowledge(session, job_id=job_id, context=context, allowed=True)
    assert _event_count(store, COMPARISON_MEASURED) == 1
    assert _event_count(store, COMPARISON_DECIDED) == 1


def test_issue_failure_rolls_back_the_prepared_cycle(store) -> None:
    job_id, token = _job(store)
    prepare = store.manager.prepare_comparison

    def broken(**kwargs):
        raise RuntimeError("injected descriptor failure")

    store.manager.prepare_comparison = broken
    with pytest.raises(RuntimeError, match="injected descriptor failure"):
        _prepare(store, job_id, token)
    assert _event_count(store, COMPARISON_PREPARED) == 0
    store.manager.prepare_comparison = prepare
    assert _prepare(store, job_id, token)["candidate_commit_hash"] == COMMIT


def test_context_issued_to_one_job_cannot_be_used_by_another(store) -> None:
    first, first_token = _job(store)
    second, second_token = _job(store)
    context = _prepare(store, first, first_token)
    with pytest.raises(EvaluationRuntimeError, match="core-issued"):
        _measure(store, second, second_token, context)
    assert _event_count(store, COMPARISON_MEASURED) == 0


def test_parallel_issue_transactions_create_only_one_active_cycle(store) -> None:
    jobs = [_job(store), _job(store)]
    barrier = Barrier(2)

    def issue(job):
        barrier.wait(timeout=5)
        try:
            return _prepare(store, *job, deadline=monotonic() + 0.2)
        except EvaluationRuntimeError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(issue, jobs))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, EvaluationRuntimeError) for result in results) == 1
    assert _event_count(store, COMPARISON_PREPARED) == 1


def test_bootstrap_seed_must_finish_ingestion_before_first_comparison(store) -> None:
    seed_id, _ = _job(store, status=JobStatus.PENDING)
    with store.factory.begin() as session:
        session.get(EvolutionJob, seed_id).is_seed_job = True
    job_id, token = _job(store)
    with pytest.raises(EvaluationRuntimeError):
        _prepare(store, job_id, token, deadline=monotonic() + 0.025)
    with store.factory.begin() as session:
        seed = session.get(EvolutionJob, seed_id)
        seed.status = JobStatus.SUCCEEDED
        seed.run_token = None
    with pytest.raises(EvaluationRuntimeError):
        _prepare(store, job_id, token, deadline=monotonic() + 0.025)
    assert _event_count(store, COMPARISON_PREPARED) == 0
    with store.factory.begin() as session:
        session.get(EvolutionJob, seed_id).ingestion_status = "succeeded"
    assert _prepare(store, job_id, token)["candidate_commit_hash"] == COMMIT


def test_ack_without_ingestion_completion_does_not_release_cycle(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context)
    _succeed(store, job_id, token)
    with store.factory.begin() as session:
        comparison.acknowledge(session, job_id=job_id, context=context, allowed=True)
    with store.factory() as session:
        assert comparison._pending_cycle(session, uuid4()) is True


def test_tampered_persisted_gate_cannot_override_recomputed_decision(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context, improvement_lower_bound=-0.05)
    _succeed(store, job_id, token)
    with store.factory.begin() as session:
        event = session.scalar(
            select(EvolutionEvent).where(EvolutionEvent.event_type == COMPARISON_MEASURED)
        )
        event.payload["allowed"] = True
    with pytest.raises(EvaluationRuntimeError, match="decision changed"):
        _load(store, job_id)


@pytest.fixture
def integrated_store(store, settings, monkeypatch, tmp_path):
    settings.experiment_id = "fresh-comparison-test"
    settings.mapelites_islands = ("main",)
    settings.mapelites_objectives = (ObjectiveSpec(name="cpu", direction="min"),)
    settings.mapelites_dimensionality_target_dims = 1
    settings.mapelites_archive_cells_per_dim = 2
    settings.mapelites_dimensionality_refit_interval = 0
    settings.mapelites_dimensionality_penultimate_normalize = False
    settings.mapelites_feature_truncation_k = 1.0
    settings.mapelites_admission_policy = "fresh_comparison"
    settings.mapelites_migration_interval_jobs = 0
    settings.worker_evaluator_max_concurrency = 1
    projection = PCAProjection(
        feature_count=1,
        components=((1.0,),),
        mean=(0.0,),
        explained_variance=(1.0,),
        explained_variance_ratio=(1.0,),
        sample_count=8,
        epoch=1,
        fitted_at=0.0,
        whiten=False,
    )
    snapshots = DatabaseSnapshotStore()
    with store.factory.begin() as session:
        snapshots.apply_update(
            "main",
            session=session,
            update=SnapshotUpdate(
                objective_contract=resolve_objective_contract(settings),
                lower_bounds=[0.0],
                upper_bounds=[1.0],
                projection=projection,
                samples_since_fit=8,
                archive_replace=[
                    SnapshotElite(
                        cell_index=0,
                        commit_hash="b" * 40,
                        objective_values=(1.0,),
                        measures=(0.25,),
                        timestamp=1.0,
                    )
                ],
            ),
        )
    manager = MapElitesManager(settings=settings, repo_root=tmp_path)
    monkeypatch.setattr(
        manager,
        "_embed_repo_state_for_ingest",
        lambda **kwargs: SimpleNamespace(
            code_embedding=CommitCodeEmbedding((), (-0.5,), "test-embedding", 1),
            stats=None,
        ),
    )
    monkeypatch.setattr(comparison, "MapElitesManager", lambda **kwargs: manager)
    monkeypatch.setattr(snapshot_module, "session_scope", comparison.session_scope)
    store.settings = settings
    store.manager = manager
    store.ingestion = MapElitesIngestion(
        settings=settings,
        console=Console(quiet=True),
        repo_root=tmp_path,
        repo=SimpleNamespace(),
        manager=manager,
    )
    return store


def _ready_for_ingestion(store, *, lower=0.05):
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context, improvement_lower_bound=lower)
    with store.factory.begin() as session:
        candidate = CandidateCommit(
            commit_hash=COMMIT,
            git_parent_commit_hash="b" * 40,
            island_id="main",
            produced_by_job_id=job_id,
            run_token=token,
            campaign_program_hash=CAMPAIGN,
            evaluation_status="passed",
            archive_status="not_considered",
        )
        session.add(candidate)
        session.flush()
        candidate_id = candidate.id
    attempt_id = _succeed(store, job_id, token, candidate_commit_id=candidate_id)
    with store.factory.begin() as session:
        session.get(CandidateCommit, candidate_id).latest_evaluation_attempt_id = attempt_id
    snapshot = JobSnapshot(
        job_id=job_id,
        base_commit_hash="b" * 40,
        island_id="main",
        result_commit_hash=COMMIT,
        completed_at=None,
    )
    return snapshot, context, candidate_id


@pytest.mark.parametrize("allowed", [True, False])
def test_real_ingestion_persists_archive_candidate_status_and_ack_atomically(
    integrated_store, allowed
) -> None:
    store = integrated_store
    snapshot, context, candidate_id = _ready_for_ingestion(
        store, lower=0.05 if allowed else -0.05
    )
    assert context["incumbent_commit_hash"] == "b" * 40
    assert store.manager.get_records("main")[0].objective_values == (1.0,)

    with store.factory.begin() as session:
        insertion = store.ingestion._ingest_with_manager(
            snapshot,
            commit_hash=COMMIT,
            metrics_payload=_metrics(),
            snapshot_session=session,
        )
        assert insertion.inserted is allowed
        # Another transaction cannot see the replacement or ACK prematurely.
        with store.factory() as reader:
            assert reader.scalar(select(MapElitesArchiveCell.commit_hash)) == "b" * 40
            assert reader.get(CandidateCommit, candidate_id).archive_status == "not_considered"
            assert reader.scalar(
                select(func.count(EvolutionEvent.id)).where(
                    EvolutionEvent.event_type == COMPARISON_DECIDED
                )
            ) == 0

    with store.factory() as session:
        archive = session.scalar(select(MapElitesArchiveCell))
        assert archive.commit_hash == (COMMIT if allowed else "b" * 40)
        assert archive.objective_values == ([99.9] if allowed else [1.0])
        candidate = session.get(CandidateCommit, candidate_id)
        assert candidate.archive_status == ("member" if allowed else "rejected")
        assert candidate.evaluation_status == "passed"
        assert session.get(EvolutionJob, snapshot.job_id).ingestion_status == (
            "succeeded" if allowed else "skipped"
        )
        assert comparison._pending_cycle(session, uuid4()) is False
        assert session.scalar(select(func.count(MapElitesPcaHistory.commit_hash))) == 0
        decision_event = session.scalar(
            select(EvolutionEvent).where(EvolutionEvent.event_type == COMPARISON_DECIDED)
        )
        assert decision_event.payload["allowed"] is allowed
    # A passing fresh comparison replaces even a historically much better score.
    assert store.manager.get_cell_fronts("main") == {
        0: (COMMIT if allowed else "b" * 40,)
    }


def test_real_ingestion_rolls_back_archive_ack_and_status_then_restores_memory(
    integrated_store, monkeypatch
) -> None:
    store = integrated_store
    snapshot, _, candidate_id = _ready_for_ingestion(store)
    original = MapElitesIngestion._record_successful_ingestion

    def fail_after_status(self, *args, **kwargs):
        original(self, *args, **kwargs)
        kwargs["session"].flush()
        assert self.manager.get_cell_fronts("main") == {0: (COMMIT,)}
        assert kwargs["session"].scalar(
            select(func.count(EvolutionEvent.id)).where(
                EvolutionEvent.event_type == COMPARISON_DECIDED
            )
        ) == 1
        raise RuntimeError("injected failure after archive, ACK and status")

    monkeypatch.setattr(MapElitesIngestion, "_record_successful_ingestion", fail_after_status)
    with store.factory.begin() as session:
        with pytest.raises(RuntimeError, match="injected failure"):
            store.ingestion._ingest_with_manager(
                snapshot,
                commit_hash=COMMIT,
                metrics_payload=_metrics(),
                snapshot_session=session,
            )
        # This is the scheduler's normal error path after the savepoint rolls back.
        store.ingestion._reload_island_after_ingest_error(
            snapshot, snapshot_session=session
        )
        assert store.manager.get_cell_fronts("main") == {0: ("b" * 40,)}

    with store.factory() as session:
        assert session.scalar(select(MapElitesArchiveCell.commit_hash)) == "b" * 40
        assert session.get(CandidateCommit, candidate_id).archive_status == "not_considered"
        job = session.get(EvolutionJob, snapshot.job_id)
        assert job.ingestion_status is None
        assert job.ingestion_attempts == 0
        assert comparison._pending_cycle(session, uuid4()) is True
    assert _event_count(store, COMPARISON_DECIDED) == 0

    monkeypatch.setattr(MapElitesIngestion, "_record_successful_ingestion", original)
    with store.factory.begin() as session:
        assert store.ingestion._ingest_with_manager(
            snapshot,
            commit_hash=COMMIT,
            metrics_payload=_metrics(),
            snapshot_session=session,
        ).inserted
    assert _event_count(store, COMPARISON_DECIDED) == 1
    assert store.manager.get_cell_fronts("main") == {0: (COMMIT,)}


def test_conflicting_measurement_replay_does_not_change_accepted_evidence(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    _measure(store, job_id, token, context)
    with pytest.raises(EvaluationRuntimeError, match="Conflicting replay"):
        _measure(store, job_id, token, context, improvement_lower_bound=-0.05)
    _succeed(store, job_id, token)
    assert _load(store, job_id)[1]["allowed"] is True
    assert _event_count(store, COMPARISON_MEASURED) == 1


def test_conflicting_ack_replay_does_not_change_original_decision(store) -> None:
    job_id, token = _job(store)
    context = _prepare(store, job_id, token)
    with store.factory.begin() as session:
        comparison.acknowledge(session, job_id=job_id, context=context, allowed=True)
    with pytest.raises(EvaluationRuntimeError, match="Conflicting replay"):
        with store.factory.begin() as session:
            comparison.acknowledge(session, job_id=job_id, context=context, allowed=False)
    with store.factory() as session:
        event = session.scalar(
            select(EvolutionEvent).where(EvolutionEvent.event_type == COMPARISON_DECIDED)
        )
        assert event.payload["allowed"] is True
    assert _event_count(store, COMPARISON_DECIDED) == 1
