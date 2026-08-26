from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from rich.console import Console

import loreley.scheduler.job_scheduler as job_scheduler_module
import loreley.scheduler.seed_portfolios as seed_portfolio_module
from loreley.core.map_elites.objectives import ObjectiveContract, ObjectiveSpec
from loreley.core.seed_portfolio import (
    SeedPortfolioDraft,
    SeedPortfolioPlanningRequest,
    SeedPortfolioPlanningResponse,
    SeedRootEvidence,
    materialize_seed_portfolio,
)
from loreley.db.models import JobStatus, SeedDirection, SeedPortfolio
from loreley.scheduler.job_scheduler import JobScheduler
from loreley.scheduler.main import EvolutionScheduler
from loreley.scheduler.seed_portfolios import (
    SeedPortfolioCoordinator,
    SeedPortfolioEvidence,
    SeedPortfolioStateError,
)


def _request() -> SeedPortfolioPlanningRequest:
    contract = ObjectiveContract((ObjectiveSpec(name="quality", direction="max"),))
    return SeedPortfolioPlanningRequest(
        configured_direction_count=1,
        direction_count=1,
        root_commit_hash="root",
        campaign_program_hash=None,
        campaign_title=None,
        goal="Improve quality.",
        constraints=(),
        acceptance_criteria=(),
        notes=(),
        objective_contract=tuple(contract.as_payload()),
        objective_contract_fingerprint=contract.fingerprint,
        root_evidence=SeedRootEvidence(
            metrics=({"name": "quality", "value": 1.0, "higher_is_better": True},)
        ),
        input_evidence_fingerprints={"baseline_key": "b" * 64},
        model_route={
            "backend": "kilo",
            "provider": "openai",
            "model": "openai/gpt-5.6-sol",
        },
        reasoning_effort="high",
        max_pairwise_overlap=0.65,
    )


def _draft() -> SeedPortfolioDraft:
    return SeedPortfolioDraft.model_validate(
        {
            "directions": [
                {
                    "direction_id": "cache-layout",
                    "title": "Cache layout",
                    "bottleneck": "Repeated decoding work dominates the measured path.",
                    "causal_mechanism": (
                        "Reuse decoded state through a compact cache representation."
                    ),
                    "likely_files": ["src/cache.py"],
                    "first_implementation": (
                        "Add the smallest bounded cache representation to one hot path."
                    ),
                    "expected_immediate_signals": [
                        "The quality objective may improve."
                    ],
                    "acceptable_neutral_results": [
                        "Neutral quality still validates the reusable representation."
                    ],
                    "roadmap": [
                        "Measure state reuse.",
                        "Specialize only stable cache entries.",
                    ],
                    "risks": ["Incorrect reuse could alter decoded output."],
                    "local_checks": ["Run the existing focused cache tests."],
                    "admission_intent": "exploratory_stepping_stone",
                    "selection_reason": (
                        "This is the smallest distinct data-reuse mechanism."
                    ),
                }
            ],
            "pairwise_overlaps": [],
            "rejected_directions": [
                {
                    "title": "Alternative cache container",
                    "causal_mechanism": (
                        "Use another container for the same decoded-state reuse mechanism."
                    ),
                    "duplicate_of_direction_id": "cache-layout",
                    "rejection_reason": (
                        "It is a superficial variant of the selected cache mechanism."
                    ),
                }
            ],
            "curation_summary": (
                "The single-slot portfolio retains one implementation-ready mechanism."
            ),
        }
    )


def _two_direction_artifact():
    request = replace(
        _request(),
        configured_direction_count=2,
        direction_count=2,
    )
    payload = _draft().model_dump(mode="json")
    second = dict(payload["directions"][0])
    second.update(
        {
            "direction_id": "branch-shaping",
            "title": "Branch shaping",
            "bottleneck": "Unpredictable branching dominates the measured path.",
            "causal_mechanism": (
                "Separate the common control-flow case so later specialization is possible."
            ),
            "likely_files": ["src/branch.py"],
            "admission_intent": "immediate_evidence",
            "selection_reason": (
                "This direction covers a control-flow mechanism distinct from data reuse."
            ),
        }
    )
    payload["directions"].append(second)
    payload["pairwise_overlaps"] = [
        {
            "direction_a": "cache-layout",
            "direction_b": "branch-shaping",
            "overlap_score": 0.2,
            "shared_surface": "Both may touch the decoder.",
            "mechanism_distinction": (
                "One changes state reuse and the other changes control flow."
            ),
        }
    ]
    return materialize_seed_portfolio(
        request,
        SeedPortfolioDraft.model_validate(payload),
    )


class _DirectionRows:
    def __init__(self, rows) -> None:
        self.rows = rows

    def all(self):
        return list(self.rows)


class _DirectionSession:
    def __init__(self, rows=()) -> None:
        self.rows = rows

    def execute(self, _statement):
        return _DirectionRows(self.rows)

    @staticmethod
    def flush() -> None:
        return None


class _FakePlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, request, *, working_dir: Path) -> SeedPortfolioPlanningResponse:
        self.calls += 1
        return SeedPortfolioPlanningResponse(
            draft=_draft(),
            prompt="prompt",
            raw_output="output",
            prompt_sha256="c" * 64,
            output_sha256="d" * 64,
            attempts=1,
            duration_seconds=0.1,
        )


class _FailingPlanner:
    def __init__(self, usage_event: object) -> None:
        self.usage_event = usage_event

    def plan(self, request, *, working_dir: Path) -> SeedPortfolioPlanningResponse:
        del request, working_dir
        error = RuntimeError("paid portfolio call failed")
        error.usage_events = (self.usage_event,)  # type: ignore[attr-defined]
        raise error


class _MemoryCoordinator(SeedPortfolioCoordinator):
    def __init__(self, *, settings, repo_root: Path, planner, store) -> None:
        super().__init__(settings=settings, repo_root=repo_root, planner=planner)
        self.store = store

    def _build_request(self, **_kwargs):
        return _request()

    @contextmanager
    def _planning_worktree(self, _root_commit_hash: str):
        yield self.repo_root

    def _load_by_request_fingerprint(self, request_fingerprint: str):
        row = self.store.get(request_fingerprint)
        return row

    def _stage_request(self, *, request, baseline_key_hash: str):
        del baseline_key_hash
        self.store[request.request_fingerprint] = SimpleNamespace(
            id="portfolio-row",
            status="planning",
            payload={},
            error_summary=None,
        )
        return "portfolio-row"

    def _persist_ready_portfolio(self, *, portfolio_id, artifact, response) -> None:
        del portfolio_id, response
        self.store[artifact.request_fingerprint] = SimpleNamespace(
            id="portfolio-row",
            status="ready",
            payload=artifact.model_dump(mode="json"),
            error_summary=None,
        )

    def _persist_failure(self, *, portfolio_id, exc: Exception) -> None:
        raise AssertionError(f"unexpected portfolio failure {portfolio_id}: {exc}")

    def _persist_usage_events(self, events) -> None:
        assert tuple(events) == ()


class _FailureMemoryCoordinator(_MemoryCoordinator):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.failure: Exception | None = None
        self.persisted_usage: tuple[object, ...] = ()

    def _persist_failure(self, *, portfolio_id, exc: Exception) -> None:
        assert portfolio_id == "portfolio-row"
        self.failure = exc

    def _persist_usage_events(self, events) -> None:
        self.persisted_usage = tuple(events)


def test_scheduler_restart_reuses_persisted_portfolio_without_second_call(
    tmp_path: Path,
    settings,
) -> None:
    store = {}
    first_planner = _FakePlanner()
    first = _MemoryCoordinator(
        settings=settings,
        repo_root=tmp_path,
        planner=first_planner,
        store=store,
    )

    created = first.ensure_ready(
        root_commit_hash="root",
        baseline_key_hash="b" * 64,
        direction_count=1,
        campaign_program=None,
    )

    second_planner = _FakePlanner()
    restarted = _MemoryCoordinator(
        settings=settings,
        repo_root=tmp_path,
        planner=second_planner,
        store=store,
    )
    reused = restarted.ensure_ready(
        root_commit_hash="root",
        baseline_key_hash="b" * 64,
        direction_count=1,
        campaign_program=None,
    )

    assert first_planner.calls == 1
    assert second_planner.calls == 0
    assert reused.portfolio_hash == created.portfolio_hash
    assert reused.request_fingerprint == created.request_fingerprint


def test_failed_paid_portfolio_call_persists_usage_before_safe_stop(
    tmp_path: Path,
    settings,
) -> None:
    usage_event = object()
    coordinator = _FailureMemoryCoordinator(
        settings=settings,
        repo_root=tmp_path,
        planner=_FailingPlanner(usage_event),
        store={},
    )

    with pytest.raises(SeedPortfolioStateError, match="untracked duplicate"):
        coordinator.ensure_ready(
            root_commit_hash="root",
            baseline_key_hash="b" * 64,
            direction_count=1,
            campaign_program=None,
        )

    assert str(coordinator.failure) == "paid portfolio call failed"
    assert coordinator.persisted_usage == (usage_event,)


def test_seed_job_assignment_uses_each_portfolio_direction_before_reuse() -> None:
    assigned = JobScheduler._seed_direction_assignments(
        session=_DirectionSession(),
        seed_portfolio=_two_direction_artifact(),
        count=2,
    )

    assert [direction.direction_id for direction in assigned] == [
        "cache-layout",
        "branch-shaping",
    ]


def test_seed_direction_with_unfinished_attempt_gets_no_concurrent_duplicate() -> None:
    assigned = JobScheduler._seed_direction_assignments(
        session=_DirectionSession((("cache-layout", JobStatus.RUNNING),)),
        seed_portfolio=_two_direction_artifact(),
        count=2,
    )

    assert [direction.direction_id for direction in assigned] == ["branch-shaping"]


def test_failed_direction_retry_waits_for_first_coverage_and_cannot_starve_slate() -> (
    None
):
    assigned = JobScheduler._seed_direction_assignments(
        session=_DirectionSession((("cache-layout", JobStatus.FAILED),)),
        seed_portfolio=_two_direction_artifact(),
        count=2,
    )

    assert [direction.direction_id for direction in assigned] == [
        "branch-shaping",
        "cache-layout",
    ]


def test_direction_is_exhausted_after_two_unsuccessful_terminal_attempts() -> None:
    rows = (
        ("cache-layout", JobStatus.FAILED),
        ("cache-layout", JobStatus.CANCELLED),
        ("branch-shaping", JobStatus.FAILED),
    )

    plan = JobScheduler._seed_direction_assignment_plan(
        session=_DirectionSession(rows),
        seed_portfolio=_two_direction_artifact(),
        count=2,
        max_unsuccessful_attempts=2,
    )

    assert [direction.direction_id for direction in plan.assignments] == [
        "branch-shaping"
    ]
    assert plan.safe_stop_reason is None


def test_all_failed_directions_produce_restart_stable_safe_stop() -> None:
    rows = (
        ("cache-layout", JobStatus.FAILED),
        ("cache-layout", JobStatus.CANCELLED),
        ("branch-shaping", JobStatus.FAILED),
        ("branch-shaping", JobStatus.CANCELLED),
    )

    first = JobScheduler._seed_direction_assignment_plan(
        session=_DirectionSession(rows),
        seed_portfolio=_two_direction_artifact(),
        count=2,
        max_unsuccessful_attempts=2,
    )
    restarted = JobScheduler._seed_direction_assignment_plan(
        session=_DirectionSession(rows),
        seed_portfolio=_two_direction_artifact(),
        count=2,
        max_unsuccessful_attempts=2,
    )

    assert first.assignments == restarted.assignments == ()
    assert first.safe_stop_reason == restarted.safe_stop_reason
    assert "all 2 portfolio directions exhausted" in str(first.safe_stop_reason)


def test_all_failed_directions_create_no_more_seed_jobs_and_surface_safe_stop(
    monkeypatch,
    settings,
) -> None:
    rows = (
        ("cache-layout", JobStatus.FAILED),
        ("cache-layout", JobStatus.CANCELLED),
        ("branch-shaping", JobStatus.FAILED),
        ("branch-shaping", JobStatus.CANCELLED),
    )

    @contextmanager
    def _scope():
        yield _DirectionSession(rows)

    monkeypatch.setattr(job_scheduler_module, "session_scope", _scope)
    scheduler = JobScheduler.__new__(JobScheduler)
    scheduler.settings = settings
    scheduler.console = Console(record=True)
    scheduler._campaign_program_snapshot = None
    scheduler._seed_portfolio_safe_stop_reason = None
    monkeypatch.setattr(
        JobScheduler,
        "_enqueue_jobs",
        lambda _self, _job_ids: pytest.fail(
            "exhausted directions must not enqueue another paid seed job"
        ),
    )

    created = scheduler.create_seed_jobs(
        base_commit_hash="root",
        count=2,
        refresh_campaign_program=False,
        seed_portfolio=_two_direction_artifact(),
    )

    assert created == 0
    assert "all 2 portfolio directions exhausted" in str(
        scheduler.seed_portfolio_safe_stop_reason
    )
    assert "Seed portfolio safe stop" in scheduler.console.export_text()


def test_successful_directions_are_reused_in_deterministic_balanced_order() -> None:
    rows = (
        ("cache-layout", JobStatus.SUCCEEDED),
        ("cache-layout", JobStatus.SUCCEEDED),
        ("branch-shaping", JobStatus.SUCCEEDED),
    )

    assigned = JobScheduler._seed_direction_assignments(
        session=_DirectionSession(rows),
        seed_portfolio=_two_direction_artifact(),
        count=1,
    )

    assert [direction.direction_id for direction in assigned] == ["branch-shaping"]


def test_smaller_direction_slate_is_distributed_evenly_across_islands() -> None:
    scheduler = EvolutionScheduler.__new__(EvolutionScheduler)
    scheduler.console = Console(record=True)
    calls: list[str] = []

    class _JobScheduler:
        @staticmethod
        def create_seed_jobs(**kwargs) -> int:
            calls.append(str(kwargs["island_id"]))
            return 1

    scheduler.job_scheduler = _JobScheduler()
    budget = SimpleNamespace(
        root_commit_hash="root",
        target_samples=128,
        unfinished_jobs=0,
    )
    allocations = (
        SimpleNamespace(
            demand=SimpleNamespace(
                island_id="island-a",
                warmup_samples=0,
            ),
            count=3,
        ),
        SimpleNamespace(
            demand=SimpleNamespace(
                island_id="island-b",
                warmup_samples=0,
            ),
            count=3,
        ),
    )

    created = scheduler._create_portfolio_seed_allocations(
        budget,
        allocations,
        seed_portfolio=_two_direction_artifact(),
    )

    assert created == 6
    assert calls == [
        "island-a",
        "island-b",
        "island-a",
        "island-b",
        "island-a",
        "island-b",
    ]


def test_large_warmup_uses_bounded_campaign_direction_count(settings) -> None:
    bounded = settings.model_copy(
        update={
            "mapelites_seed_population_size": 32,
            "mapelites_feature_normalization_warmup_samples": 128,
            "mapelites_seed_portfolio_direction_count": 8,
            "mapelites_islands": tuple(f"island-{index}" for index in range(8)),
            "scheduler_max_total_jobs": 30_000,
        }
    )
    scheduler = EvolutionScheduler.__new__(EvolutionScheduler)
    scheduler.settings = bounded

    direction_count = scheduler._seed_portfolio_direction_count()

    assert direction_count == 8
    assert direction_count * (direction_count - 1) // 2 == 28
    assert direction_count != 128


@pytest.mark.parametrize(
    ("configured", "seed_target", "warmup_target", "total_cap", "expected"),
    (
        (12, 4, 4, 30_000, 4),
        (12, 32, 128, 5, 5),
        (16, 32, 128, 30_000, 16),
    ),
)
def test_effective_direction_count_is_minimum_of_stable_configured_caps(
    settings,
    configured: int,
    seed_target: int,
    warmup_target: int,
    total_cap: int,
    expected: int,
) -> None:
    bounded = settings.model_copy(
        update={
            "mapelites_seed_portfolio_direction_count": configured,
            "mapelites_seed_population_size": seed_target,
            "mapelites_feature_normalization_warmup_samples": warmup_target,
            "scheduler_max_total_jobs": total_cap,
        }
    )
    scheduler = EvolutionScheduler.__new__(EvolutionScheduler)
    scheduler.settings = bounded

    assert scheduler._seed_portfolio_direction_count() == expected


def test_portfolio_is_opt_in_and_disabled_path_never_calls_planner(settings) -> None:
    scheduler = EvolutionScheduler.__new__(EvolutionScheduler)
    scheduler.settings = settings
    scheduler._active_seed_portfolio = object()
    calls: list[object] = []
    scheduler.seed_portfolio_coordinator = SimpleNamespace(
        ensure_ready=lambda **kwargs: calls.append(kwargs)
    )

    scheduler._ensure_seed_portfolio_ready(
        baseline=SimpleNamespace(baseline_key_hash="b" * 64)
    )

    assert settings.mapelites_seed_portfolio_enabled is False
    assert scheduler._active_seed_portfolio is None
    assert calls == []


def test_coordinator_loads_ordered_root_evidence_and_complete_fingerprints(
    monkeypatch,
    tmp_path: Path,
    settings,
) -> None:
    card_id = uuid4()
    baseline = SimpleNamespace(
        status="valid",
        evaluation_summary="baseline summary",
    )
    card = SimpleNamespace(
        id=card_id,
        evaluation_summary="root summary",
    )
    metrics = [
        SimpleNamespace(
            name="memory_mib",
            value=20.0,
            unit="MiB",
            higher_is_better=False,
            details={},
        ),
        SimpleNamespace(
            name="composite_score",
            value=1.0,
            unit=None,
            higher_is_better=True,
            details={"summary": "primary"},
        ),
    ]
    artifacts = [
        SimpleNamespace(
            key="profile",
            kind="profile",
            sha256="e" * 64,
            summary="root hotspot",
            diagnostics=[{"message": "decoder dominates"}],
        )
    ]
    manifest = SimpleNamespace(fingerprint="f" * 64)

    class _One:
        def __init__(self, value) -> None:
            self.value = value

        def scalar_one_or_none(self):
            return self.value

    class _Many:
        def __init__(self, values) -> None:
            self.values = values

        def all(self):
            return list(self.values)

    class _Session:
        def __init__(self) -> None:
            self.execute_values = iter((baseline, card, manifest))
            self.scalar_values = iter((metrics, artifacts))

        def execute(self, _statement):
            return _One(next(self.execute_values))

        def scalars(self, _statement):
            return _Many(next(self.scalar_values))

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(seed_portfolio_module, "session_scope", _scope)
    coordinator = SeedPortfolioCoordinator(
        settings=settings,
        repo_root=tmp_path,
        planner=_FakePlanner(),
    )

    evidence = coordinator._load_root_evidence(
        root_commit_hash="root",
        baseline_key_hash="b" * 64,
    )

    assert [item["name"] for item in evidence.root_evidence.metrics] == [
        "composite_score",
        "memory_mib",
    ]
    assert evidence.root_evidence.diagnostics == (
        "root hotspot",
        "decoder dominates",
    )
    assert evidence.fingerprints["repo_state_embedding_manifest"] == "f" * 64
    assert all(evidence.fingerprints.values())


def test_coordinator_build_request_pins_route_and_campaign_inputs(
    monkeypatch,
    tmp_path: Path,
    settings,
) -> None:
    settings.worker_evolution_global_goal = "Improve safely."
    evidence = SeedPortfolioEvidence(
        root_evidence=SeedRootEvidence(
            metrics=(
                {
                    "name": "composite_score",
                    "value": 1.0,
                    "higher_is_better": True,
                },
            )
        ),
        fingerprints={"root_metrics": "e" * 64},
    )
    monkeypatch.setattr(
        SeedPortfolioCoordinator,
        "_load_root_evidence",
        lambda *_args, **_kwargs: evidence,
    )
    coordinator = SeedPortfolioCoordinator(
        settings=settings,
        repo_root=tmp_path,
        planner=_FakePlanner(),
    )

    request = coordinator._build_request(
        root_commit_hash="root",
        baseline_key_hash="b" * 64,
        direction_count=3,
        campaign_program=None,
    )

    assert request.configured_direction_count == 8
    assert request.direction_count == 3
    assert request.model_route["model"] == "openai/gpt-5.6-sol"
    assert request.reasoning_effort == "high"
    assert request.input_evidence_fingerprints["baseline_key"] == "b" * 64
    assert request.goal == "Improve safely."


def test_coordinator_rejects_non_sol_model_before_any_planner_call(
    monkeypatch,
    tmp_path: Path,
    settings,
) -> None:
    settings.worker_seed_portfolio_model = "openai/gpt-5.6-luna"
    coordinator = SeedPortfolioCoordinator(
        settings=settings,
        repo_root=tmp_path,
        planner=_FakePlanner(),
    )

    with pytest.raises(SeedPortfolioStateError, match="pinned"):
        coordinator._build_request(
            root_commit_hash="root",
            baseline_key_hash="b" * 64,
            direction_count=1,
            campaign_program=None,
        )


def test_coordinator_stages_then_persists_portfolio_and_direction_rows(
    monkeypatch,
    tmp_path: Path,
    settings,
) -> None:
    stored: dict[object, SeedPortfolio] = {}
    directions: list[SeedDirection] = []

    class _Session:
        def add(self, row) -> None:
            if isinstance(row, SeedPortfolio):
                if row.id is None:
                    row.id = uuid4()
                stored[row.id] = row
            elif isinstance(row, SeedDirection):
                if row.id is None:
                    row.id = uuid4()
                directions.append(row)

        @staticmethod
        def flush() -> None:
            return None

        def get(self, model, row_id):
            assert model is SeedPortfolio
            return stored.get(row_id)

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(seed_portfolio_module, "session_scope", _scope)
    request = _request()
    row_id = SeedPortfolioCoordinator._stage_request(
        request=request,
        baseline_key_hash="b" * 64,
    )
    artifact = materialize_seed_portfolio(request, _draft())
    response = SeedPortfolioPlanningResponse(
        draft=_draft(),
        prompt="prompt",
        raw_output="output",
        prompt_sha256="c" * 64,
        output_sha256="d" * 64,
        attempts=1,
        duration_seconds=0.5,
    )

    SeedPortfolioCoordinator._persist_ready_portfolio(
        portfolio_id=row_id,
        artifact=artifact,
        response=response,
    )

    row = stored[row_id]
    assert row.status == "ready"
    assert row.portfolio_hash == artifact.portfolio_hash
    assert row.payload["portfolio_hash"] == artifact.portfolio_hash
    assert row.planner_prompt_sha256 == "c" * 64
    assert [item.direction_id for item in directions] == ["cache-layout"]
