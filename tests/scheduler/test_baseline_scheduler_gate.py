from __future__ import annotations

from typing import Any, cast

from rich.console import Console

from loreley.scheduler.baselines import BaselineBootstrapResult
from loreley.scheduler.main import EvolutionScheduler
from tests.support import TestSettings


class _Ingestion:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def ingest_completed_jobs(self) -> int:
        self.events.append("ingest")
        return 0


class _BlockedJobScheduler:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reclaim_stale_running_jobs(self) -> object:
        self.events.append("reclaim")
        return object()

    def refresh_campaign_program_for_policy(self) -> None:
        self.events.append("refresh_program")

    @property
    def campaign_program_snapshot(self) -> None:
        return None

    def dispatch_pending_jobs(self) -> int:
        raise AssertionError("pending jobs must not dispatch while baseline is blocked")

    def schedule_jobs(self, *_args: object, **_kwargs: object) -> int:
        raise AssertionError("new jobs must not schedule while baseline is blocked")

    def count_unfinished_jobs(self) -> int:
        self.events.append("count_unfinished")
        return 3


class _RunnableJobScheduler(_BlockedJobScheduler):
    def dispatch_pending_jobs(self) -> int:
        self.events.append("dispatch")
        return 1

    def schedule_jobs(
        self,
        unfinished_jobs: int,
        *,
        total_jobs: int,
        refresh_campaign_program: bool = True,
    ) -> int:
        self.events.append(f"schedule:{unfinished_jobs}:{total_jobs}:{refresh_campaign_program}")
        return 2

    def count_total_jobs(self) -> int:
        raise AssertionError("tick should use cached total jobs")


class _BaselineService:
    def __init__(self, events: list[str], *, can_run: bool) -> None:
        self.events = events
        self.can_run = can_run

    def ensure_or_load_baseline(self, **_kwargs: object) -> BaselineBootstrapResult:
        self.events.append("baseline")
        return BaselineBootstrapResult(
            can_dispatch_or_schedule=self.can_run,
            status="valid" if self.can_run else "failed",
            policy="required",
            baseline_key_hash="a" * 64,
            failure_kind=None if self.can_run else "primary_metric_missing",
        )


def _scheduler(events: list[str], *, can_run: bool, job_scheduler: object) -> EvolutionScheduler:
    scheduler = cast(Any, EvolutionScheduler.__new__(EvolutionScheduler))
    scheduler.settings = TestSettings(
        MAPELITES_EXPERIMENT_ROOT_COMMIT="root123",
        MAPELITES_FITNESS_METRIC="score",
        SCHEDULER_MAX_TOTAL_JOBS=10,
    )
    scheduler.console = Console(record=True)
    scheduler._root_commit_hash = "root123"
    scheduler._stop_requested = False
    scheduler._max_total_jobs = 10
    scheduler._total_jobs_count = 0
    scheduler.ingestion = _Ingestion(events)
    scheduler.job_scheduler = job_scheduler
    scheduler.baseline_bootstrap = _BaselineService(events, can_run=can_run)
    scheduler._maybe_schedule_seed_jobs = lambda unfinished_jobs: events.append(
        f"seed:{unfinished_jobs}"
    ) or 1
    scheduler._create_best_fitness_branch_if_possible = lambda: None
    scheduler.stop = lambda: None
    return cast(EvolutionScheduler, scheduler)


def test_tick_blocks_dispatch_and_scheduling_when_required_baseline_is_invalid() -> None:
    events: list[str] = []
    scheduler = _scheduler(
        events,
        can_run=False,
        job_scheduler=_BlockedJobScheduler(events),
    )

    stats = scheduler.tick()

    assert stats["baseline_blocked"] == 1
    assert stats["dispatched"] == 0
    assert stats["scheduled"] == 0
    assert events == ["ingest", "reclaim", "refresh_program", "baseline", "count_unfinished"]


def test_tick_establishes_baseline_before_dispatch_seed_and_sampler_scheduling() -> None:
    events: list[str] = []
    scheduler = _scheduler(
        events,
        can_run=True,
        job_scheduler=_RunnableJobScheduler(events),
    )

    stats = scheduler.tick()

    assert stats["baseline_blocked"] == 0
    assert stats["dispatched"] == 1
    assert stats["seed_scheduled"] == 1
    assert stats["scheduled"] == 2
    assert events == [
        "ingest",
        "reclaim",
        "refresh_program",
        "baseline",
        "dispatch",
        "count_unfinished",
        "seed:3",
        "schedule:4:1:False",
    ]
