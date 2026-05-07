from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SqlAlchemySession

import loreley.scheduler.baselines as baselines
from loreley.core.campaign_program import parse_campaign_program
from loreley.core.worker.evaluator import (
    EvaluationArtifact,
    EvaluationMetric,
    EvaluationOutcome,
    EvaluationResult,
)
from loreley.db.models import (
    CampaignBaseline,
    CommitCard,
    EvaluationArtifactRecord,
    EvolutionJob,
    JobArtifacts,
    MapElitesArchiveCell,
    Metric,
)
from loreley.scheduler.baselines import (
    BASELINE_STATUS_DEGRADED,
    BASELINE_STATUS_FAILED,
    BASELINE_STATUS_VALID,
    BaselineBootstrapService,
    BaselineMetricSpec,
    baseline_evaluator_version,
    baseline_effective_settings_fingerprint,
    build_baseline_key,
    load_latest_matching_baseline,
    resolve_status_campaign_program_hash,
    validate_baseline_primary_metric,
)
from tests.support import TestSettings


class _Result:
    def __init__(self, value: object = None) -> None:
        self.value = value

    def scalar_one_or_none(self) -> object:
        return self.value


class _BaselineStore:
    def __init__(self) -> None:
        self.baselines: list[CampaignBaseline] = []
        self.commit_cards: list[CommitCard] = []
        self.metrics: list[Metric] = []
        self.archive_cells: list[MapElitesArchiveCell] = []
        self.evolution_jobs: list[EvolutionJob] = []
        self.job_artifacts: list[JobArtifacts] = []
        self.evaluation_artifacts: list[EvaluationArtifactRecord] = []

    def session(self) -> "_Session":
        return _Session(self)


class _Session:
    def __init__(self, store: _BaselineStore) -> None:
        self.store = store

    def execute(self, stmt: Any) -> _Result:
        entity = _selected_entity(stmt)
        params = {str(value) for value in stmt.compile().params.values()}
        if entity is CampaignBaseline:
            return _Result(_first_with_param(self.store.baselines, "baseline_key_hash", params))
        if entity is CommitCard:
            return _Result(_first_with_param(self.store.commit_cards, "commit_hash", params))
        if entity is Metric:
            return _Result(_first_metric(self.store.metrics, params))
        return _Result(None)

    def add(self, obj: object) -> None:
        if isinstance(obj, CampaignBaseline):
            self.store.baselines.append(obj)
        elif isinstance(obj, CommitCard):
            self.store.commit_cards.append(obj)
        elif isinstance(obj, Metric):
            self.store.metrics.append(obj)
        elif isinstance(obj, MapElitesArchiveCell):
            self.store.archive_cells.append(obj)
        elif isinstance(obj, EvolutionJob):
            self.store.evolution_jobs.append(obj)
        elif isinstance(obj, JobArtifacts):
            self.store.job_artifacts.append(obj)
        elif isinstance(obj, EvaluationArtifactRecord):
            self.store.evaluation_artifacts.append(obj)

    def flush(self) -> None:
        for obj in [*self.store.baselines, *self.store.commit_cards, *self.store.metrics]:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
        for metric in self.store.metrics:
            commit = getattr(metric, "commit", None)
            if commit is not None and getattr(metric, "commit_card_id", None) is None:
                metric.commit_card_id = commit.id


def _selected_entity(stmt: Any) -> object:
    descriptions = list(getattr(stmt, "column_descriptions", []) or [])
    return descriptions[0].get("entity") if descriptions else None


def _first_with_param(rows: list[Any], attr: str, params: set[str]) -> object | None:
    return next((row for row in rows if str(getattr(row, attr, "")) in params), None)


def _first_metric(rows: list[Metric], params: set[str]) -> Metric | None:
    return next((row for row in rows if str(row.name) in params), None)


def _install_session(monkeypatch: pytest.MonkeyPatch, store: _BaselineStore) -> None:
    @contextmanager
    def _scope():
        yield store.session()

    monkeypatch.setattr(baselines, "session_scope", _scope)


def _install_baseline_eval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    outcome: EvaluationOutcome,
    contexts: list[object] | None = None,
) -> None:
    contexts = contexts if contexts is not None else []

    class _WorkerRepository:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @contextmanager
        def checkout_lease_for_job(self, **kwargs: object):
            assert kwargs["job_id"] is None
            assert kwargs["create_branch"] is False
            yield SimpleNamespace(worktree=tmp_path)

    class _Evaluator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def evaluate_outcome(self, context: object) -> EvaluationOutcome:
            contexts.append(context)
            return outcome

    monkeypatch.setattr(baselines, "WorkerRepository", _WorkerRepository)
    monkeypatch.setattr(baselines, "Evaluator", _Evaluator)


def _settings(*, policy: str = "required") -> TestSettings:
    return TestSettings(
        MAPELITES_EXPERIMENT_ROOT_COMMIT="root123",
        MAPELITES_FITNESS_METRIC="score",
        MAPELITES_FITNESS_HIGHER_IS_BETTER=True,
        WORKER_EVALUATOR_PLUGIN="tests.support:plugin",
        BASELINE_BOOTSTRAP_POLICY=policy,
    )


def _passed_outcome(*, metrics: tuple[EvaluationMetric, ...]) -> EvaluationOutcome:
    return EvaluationOutcome(
        evaluator_name="unit-evaluator",
        evaluator_version="1.0",
        candidate_commit_hash="root123",
        outcome_kind="passed",
        result=EvaluationResult(summary="baseline evaluated", metrics=metrics),
    )


def test_required_policy_blocks_when_primary_metric_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store = _BaselineStore()
    _install_session(monkeypatch, store)
    _install_baseline_eval(
        monkeypatch,
        tmp_path,
        _passed_outcome(metrics=(EvaluationMetric(name="other", value=1.0),)),
    )

    service = BaselineBootstrapService(
        settings=_settings(policy="required"),
        repo_root=tmp_path,
        console=baselines.Console(record=True),
    )
    result = service.ensure_or_load_baseline(root_commit_hash="root123", campaign_program=None)

    assert result.can_dispatch_or_schedule is False
    assert result.status == BASELINE_STATUS_FAILED
    assert store.baselines[0].status == BASELINE_STATUS_FAILED
    assert store.baselines[0].failure_kind == "primary_metric_missing"


def test_warn_policy_persists_degraded_baseline_and_allows_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store = _BaselineStore()
    _install_session(monkeypatch, store)
    _install_baseline_eval(
        monkeypatch,
        tmp_path,
        _passed_outcome(metrics=(EvaluationMetric(name="other", value=1.0),)),
    )

    service = BaselineBootstrapService(
        settings=_settings(policy="warn"),
        repo_root=tmp_path,
        console=baselines.Console(record=True),
    )
    result = service.ensure_or_load_baseline(root_commit_hash="root123", campaign_program=None)

    assert result.can_dispatch_or_schedule is True
    assert result.status == BASELINE_STATUS_DEGRADED
    assert store.baselines[0].status == BASELINE_STATUS_DEGRADED
    assert store.baselines[0].failure_kind == "primary_metric_missing"


def test_valid_baseline_persists_source_of_truth_and_compat_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store = _BaselineStore()
    contexts: list[object] = []
    _install_session(monkeypatch, store)
    _install_baseline_eval(
        monkeypatch,
        tmp_path,
        _passed_outcome(metrics=(EvaluationMetric(name="score", value=2.5, unit="pts"),)),
        contexts,
    )

    service = BaselineBootstrapService(
        settings=_settings(policy="required"),
        repo_root=tmp_path,
        console=baselines.Console(record=True),
    )
    result = service.ensure_or_load_baseline(root_commit_hash="root123", campaign_program=None)

    assert result.can_dispatch_or_schedule is True
    assert store.baselines[0].status == BASELINE_STATUS_VALID
    assert store.baselines[0].metric_value == pytest.approx(2.5)
    assert [row.commit_hash for row in store.commit_cards] == ["root123"]
    assert [(row.name, row.value) for row in store.metrics] == [("score", pytest.approx(2.5))]
    assert contexts[0].metadata["kind"] == "baseline"
    assert contexts[0].metadata["baseline_key_hash"] == store.baselines[0].baseline_key_hash


def test_valid_matching_baseline_is_loaded_without_rerunning_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = _settings(policy="required")
    key = build_baseline_key(settings=settings, root_commit_hash="root123", campaign_program=None)
    row = CampaignBaseline(
        baseline_key_hash=key.hash,
        root_commit_hash="root123",
        campaign_program_hash=None,
        evaluator_name="tests.support:plugin",
        evaluator_version=None,
        primary_metric_name="score",
        primary_metric_higher_is_better=True,
        runtime_profile=settings.profile,
        effective_settings_fingerprint=key.effective_settings_fingerprint,
        status=BASELINE_STATUS_VALID,
        metric_value=1.0,
    )
    row.id = uuid.uuid4()
    store = _BaselineStore()
    store.baselines.append(row)
    _install_session(monkeypatch, store)

    class _ForbiddenEvaluator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("valid baseline rows should be loaded without evaluation")

    monkeypatch.setattr(baselines, "Evaluator", _ForbiddenEvaluator)

    service = BaselineBootstrapService(
        settings=settings,
        repo_root=tmp_path,
        console=baselines.Console(record=True),
    )
    result = service.ensure_or_load_baseline(root_commit_hash="root123", campaign_program=None)

    assert result.can_dispatch_or_schedule is True
    assert result.status == BASELINE_STATUS_VALID


def test_failed_matching_baseline_is_retried_before_blocking_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = _settings(policy="required")
    key = build_baseline_key(settings=settings, root_commit_hash="root123", campaign_program=None)
    row = CampaignBaseline(
        baseline_key_hash=key.hash,
        root_commit_hash="root123",
        campaign_program_hash=None,
        evaluator_name="tests.support:plugin",
        evaluator_version=key.evaluator_version,
        primary_metric_name="score",
        primary_metric_higher_is_better=True,
        runtime_profile=settings.profile,
        effective_settings_fingerprint=key.effective_settings_fingerprint,
        status=BASELINE_STATUS_FAILED,
        failure_kind="primary_metric_missing",
        failure_summary="transient setup problem",
    )
    row.id = uuid.uuid4()
    store = _BaselineStore()
    store.baselines.append(row)
    _install_session(monkeypatch, store)
    _install_baseline_eval(
        monkeypatch,
        tmp_path,
        _passed_outcome(metrics=(EvaluationMetric(name="score", value=4.0),)),
    )

    service = BaselineBootstrapService(
        settings=settings,
        repo_root=tmp_path,
        console=baselines.Console(record=True),
    )
    result = service.ensure_or_load_baseline(root_commit_hash="root123", campaign_program=None)

    assert result.can_dispatch_or_schedule is True
    assert result.status == BASELINE_STATUS_VALID
    assert store.baselines == [row]
    assert row.failure_kind is None
    assert row.metric_value == pytest.approx(4.0)


def test_baseline_key_changes_with_evaluator_version() -> None:
    settings_v1 = _settings()
    settings_v1.worker_evaluator_version = "1.0.0"
    settings_v2 = _settings()
    settings_v2.worker_evaluator_version = "2.0.0"

    key_v1 = build_baseline_key(
        settings=settings_v1,
        root_commit_hash="root123",
        campaign_program=None,
    )
    key_v2 = build_baseline_key(
        settings=settings_v2,
        root_commit_hash="root123",
        campaign_program=None,
    )

    assert baseline_evaluator_version(settings_v1) == "1.0.0"
    assert key_v1.evaluator_version == "1.0.0"
    assert key_v2.evaluator_version == "2.0.0"
    assert key_v1.effective_settings_fingerprint != key_v2.effective_settings_fingerprint
    assert key_v1.hash != key_v2.hash


def test_baseline_key_uses_evaluator_source_fingerprint_when_version_unset(tmp_path) -> None:
    plugin_path = tmp_path / "baseline_plugin.py"
    plugin_path.write_text(
        "def plugin(context):\n"
        "    return {'summary': 'ok', 'metrics': []}\n",
        encoding="utf-8",
    )
    settings = _settings()
    settings.worker_evaluator_plugin = "baseline_plugin:plugin"
    settings.worker_evaluator_python_paths = [str(tmp_path)]

    key = build_baseline_key(
        settings=settings,
        root_commit_hash="root123",
        campaign_program=None,
    )
    plugin_path.write_text(
        "def plugin(context):\n"
        "    return {'summary': 'changed', 'metrics': []}\n",
        encoding="utf-8",
    )
    changed_key = build_baseline_key(
        settings=settings,
        root_commit_hash="root123",
        campaign_program=None,
    )

    assert key.evaluator_version is not None
    assert key.evaluator_version.startswith("source-sha256:")
    assert changed_key.evaluator_version is not None
    assert changed_key.evaluator_version.startswith("source-sha256:")
    assert key.hash != changed_key.hash


def test_program_hash_changes_require_distinct_baseline_keys() -> None:
    settings = _settings()
    program_a = parse_campaign_program(
        b"# Campaign\n\n## Goal\nImprove A\n",
        source_path="loreley.program.md",
    )
    program_b = parse_campaign_program(
        b"# Campaign\n\n## Goal\nImprove B\n",
        source_path="loreley.program.md",
    )

    key_a = build_baseline_key(
        settings=settings,
        root_commit_hash="root123",
        campaign_program=program_a,
    )
    key_b = build_baseline_key(
        settings=settings,
        root_commit_hash="root123",
        campaign_program=program_b,
    )

    assert program_a.raw_sha256 != program_b.raw_sha256
    assert key_a.hash != key_b.hash


def test_load_latest_matching_baseline_scopes_explicit_null_campaign_program() -> None:
    """Regression: no-program baseline reads must not fall through to another campaign."""

    settings = _settings()
    fingerprint = baseline_effective_settings_fingerprint(settings)
    no_program = CampaignBaseline(
        baseline_key_hash="a" * 64,
        root_commit_hash="root123",
        campaign_program_hash=None,
        evaluator_name="tests.support:plugin",
        evaluator_version=None,
        primary_metric_name="score",
        primary_metric_higher_is_better=True,
        runtime_profile=settings.profile,
        effective_settings_fingerprint=fingerprint,
        status=BASELINE_STATUS_VALID,
        metric_value=1.0,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    other_program = CampaignBaseline(
        baseline_key_hash="b" * 64,
        root_commit_hash="root123",
        campaign_program_hash="program-b",
        evaluator_name="tests.support:plugin",
        evaluator_version=None,
        primary_metric_name="score",
        primary_metric_higher_is_better=True,
        runtime_profile=settings.profile,
        effective_settings_fingerprint=fingerprint,
        status=BASELINE_STATUS_VALID,
        metric_value=9.0,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    engine = create_engine("sqlite:///:memory:")
    CampaignBaseline.__table__.create(engine)

    with SqlAlchemySession(engine) as session:
        session.add_all([no_program, other_program])
        session.commit()

        selected = load_latest_matching_baseline(
            session=session,
            settings=settings,
            campaign_program_hash=None,
        )

    assert selected is not None
    assert selected.campaign_program_hash is None
    assert selected.metric_value == pytest.approx(1.0)


def test_status_campaign_program_hash_uses_newest_persisted_scheduler_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status should follow persisted scheduler state, not only the visible program file."""

    settings = _settings()
    job_program_hash = "a" * 64
    baseline_program_hash = "b" * 64
    persisted_rows = iter(
        [
            (
                job_program_hash,
                datetime(2026, 3, 25, 7, 59, tzinfo=timezone.utc),
                datetime(2026, 3, 25, 7, 58, tzinfo=timezone.utc),
                datetime(2026, 3, 25, 7, 58, tzinfo=timezone.utc),
            ),
            (
                baseline_program_hash,
                datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc),
            ),
        ]
    )
    fallback_calls: list[TestSettings] = []

    class DummyResult:
        def __init__(self, row: object) -> None:
            self._row = row

        def first(self) -> object:
            return self._row

    class DummySession:
        def execute(self, _stmt: object) -> DummyResult:
            return DummyResult(next(persisted_rows))

    def fake_resolve_current_campaign_program_hash(active_settings: TestSettings) -> object:
        fallback_calls.append(active_settings)
        return object()

    monkeypatch.setattr(
        baselines,
        "resolve_current_campaign_program_hash",
        fake_resolve_current_campaign_program_hash,
    )

    resolution = resolve_status_campaign_program_hash(
        session=DummySession(),  # type: ignore[arg-type]
        settings=settings,
    )

    assert resolution.known is True
    assert resolution.campaign_program_hash == baseline_program_hash
    assert resolution.source_path == "database:campaign_baselines"
    assert fallback_calls == []


@pytest.mark.parametrize(
    ("result", "expected_kind"),
    [
        (EvaluationResult(summary="missing", metrics=()), "primary_metric_missing"),
        (
            EvaluationResult(
                summary="nan",
                metrics=(EvaluationMetric(name="score", value=float("nan")),),
            ),
            "primary_metric_non_finite",
        ),
        (
            EvaluationResult(
                summary="inf",
                metrics=(EvaluationMetric(name="score", value=float("inf")),),
            ),
            "primary_metric_non_finite",
        ),
        (
            EvaluationResult(
                summary="direction",
                metrics=(EvaluationMetric(name="score", value=1.0, higher_is_better=False),),
            ),
            "primary_metric_direction_conflict",
        ),
    ],
)
def test_primary_metric_validation_rejects_missing_nonfinite_and_wrong_direction(
    result: EvaluationResult,
    expected_kind: str,
) -> None:
    validation = validate_baseline_primary_metric(
        result=result,
        spec=BaselineMetricSpec(name="score", higher_is_better=True),
    )

    assert validation.ok is False
    assert validation.failure_kind == expected_kind


def test_existing_root_commit_metric_does_not_satisfy_required_without_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store = _BaselineStore()
    card = CommitCard(
        commit_hash="root123",
        parent_commit_hash=None,
        island_id="main",
        author=None,
        subject="Root",
        change_summary="Root baseline commit.",
        evaluation_summary=None,
        tags=[],
        key_files=[],
        highlights=[],
        job_id=None,
    )
    card.id = uuid.uuid4()
    metric = Metric(
        commit=card,
        name="score",
        value=10.0,
        unit=None,
        higher_is_better=True,
        details={},
    )
    metric.id = uuid.uuid4()
    metric.commit_card_id = card.id
    store.commit_cards.append(card)
    store.metrics.append(metric)
    _install_session(monkeypatch, store)
    _install_baseline_eval(
        monkeypatch,
        tmp_path,
        _passed_outcome(metrics=(EvaluationMetric(name="other", value=1.0),)),
    )

    service = BaselineBootstrapService(
        settings=_settings(policy="required"),
        repo_root=tmp_path,
        console=baselines.Console(record=True),
    )
    result = service.ensure_or_load_baseline(root_commit_hash="root123", campaign_program=None)

    assert result.can_dispatch_or_schedule is False
    assert len(store.baselines) == 1
    assert store.baselines[0].status == BASELINE_STATUS_FAILED


def test_baseline_evaluation_does_not_create_fake_jobs_or_job_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store = _BaselineStore()
    _install_session(monkeypatch, store)
    outcome = EvaluationOutcome(
        evaluator_name="unit-evaluator",
        evaluator_version="1.0",
        candidate_commit_hash="root123",
        outcome_kind="passed",
        result=EvaluationResult(
            summary="baseline evaluated",
            metrics=(EvaluationMetric(name="score", value=3.0),),
            artifacts=(
                EvaluationArtifact(
                    key="baseline-report",
                    kind="benchmark_json",
                    mime_type="application/json",
                    inline_payload={"score": 3.0},
                    summary="not job scoped",
                    visibility="agent_visible",
                ),
            ),
        ),
    )
    _install_baseline_eval(monkeypatch, tmp_path, outcome)

    service = BaselineBootstrapService(
        settings=_settings(policy="required"),
        repo_root=tmp_path,
        console=baselines.Console(record=True),
    )
    result = service.ensure_or_load_baseline(root_commit_hash="root123", campaign_program=None)

    assert result.can_dispatch_or_schedule is True
    assert store.baselines[0].status == BASELINE_STATUS_VALID
    assert store.evolution_jobs == []
    assert store.job_artifacts == []
    assert store.evaluation_artifacts == []
    assert store.archive_cells == []
