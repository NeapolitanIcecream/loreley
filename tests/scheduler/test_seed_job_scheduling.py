from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, cast

import pytest
from rich.console import Console

import loreley.scheduler.main as scheduler_main
from loreley.config import Settings
from loreley.db.models import EvolutionJob
from loreley.scheduler.main import EvolutionScheduler


class _FakeResult:
    def __init__(self, row: object) -> None:
        self._row = row

    def one(self) -> object:
        return self._row

    def scalar_one(self) -> object:
        row = self._row
        if isinstance(row, tuple):
            if len(row) != 1:
                raise AssertionError("scalar_one() expected a single-column row")
            return row[0]
        return row


@dataclass
class _FakeSession:
    total_jobs: int
    seed_count: int
    executed: list[object]

    def execute(self, stmt: Any) -> _FakeResult:
        self.executed.append(stmt)

        # Guardrail: the scheduler must not load ORM rows for this check.
        descriptions = list(getattr(stmt, "column_descriptions", []))
        if any(d.get("expr") is EvolutionJob or d.get("type") is EvolutionJob for d in descriptions):
            raise AssertionError("seed scheduling must not execute select(EvolutionJob)")

        descriptions = list(getattr(stmt, "column_descriptions", []))
        if len(descriptions) == 1:
            return _FakeResult((self.seed_count,))
        return _FakeResult((self.total_jobs, self.seed_count))


@dataclass
class _DummyManager:
    records: list[object]

    def get_records(self, _island_id: str) -> list[object]:
        return self.records


@dataclass
class _DummyJobScheduler:
    created_calls: list[dict[str, object]]
    created_return: int = 0

    def create_seed_jobs(self, *, base_commit_hash: str, count: int, island_id: str | None = None) -> int:
        self.created_calls.append(
            {
                "base_commit_hash": base_commit_hash,
                "count": count,
                "island_id": island_id,
            }
        )
        return self.created_return


def _make_scheduler(*, settings: Settings, records: list[object]) -> EvolutionScheduler:
    scheduler = cast(Any, EvolutionScheduler.__new__(EvolutionScheduler))
    scheduler.settings = settings
    scheduler.console = Console(record=True)
    scheduler._root_commit_hash = (settings.mapelites_experiment_root_commit or "").strip() or None
    scheduler._max_total_jobs = 100
    scheduler.manager = _DummyManager(records=records)
    scheduler.job_scheduler = _DummyJobScheduler(created_calls=[], created_return=0)
    scheduler._total_jobs_count = 0
    return cast(EvolutionScheduler, scheduler)


def test_seed_scheduling_skips_when_archive_has_records(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    # Arrange
    scheduler = _make_scheduler(settings=settings, records=[object()])

    @contextmanager
    def _no_session_scope() -> Iterator[object]:
        raise AssertionError("DB should not be queried when archive has records")
        yield  # pragma: no cover

    monkeypatch.setattr(scheduler_main, "session_scope", _no_session_scope)

    # Act
    created = scheduler._maybe_schedule_seed_jobs(unfinished_jobs=0)

    # Assert
    assert created == 0


def test_seed_scheduling_is_noop_when_non_seed_jobs_exist(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    # Arrange
    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": "deadbeef",
            "mapelites_seed_population_size": 16,
            "scheduler_max_unfinished_jobs": 4,
        }
    )
    scheduler = _make_scheduler(settings=settings, records=[])
    scheduler._total_jobs_count = 3
    executed: list[object] = []
    fake_session = _FakeSession(total_jobs=3, seed_count=2, executed=executed)

    @contextmanager
    def _session_scope() -> Iterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(scheduler_main, "session_scope", _session_scope)

    # Act
    created = scheduler._maybe_schedule_seed_jobs(unfinished_jobs=0)

    # Assert
    assert created == 0
    assert cast(Any, scheduler.job_scheduler).created_calls == []
    assert len(executed) >= 1


def test_seed_scheduling_creates_limited_jobs_without_loading_rows(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    # Arrange
    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": "deadbeef",
            "mapelites_seed_population_size": 10,
            "scheduler_max_unfinished_jobs": 4,
        }
    )
    scheduler = _make_scheduler(settings=settings, records=[])
    scheduler._max_total_jobs = 20
    scheduler._total_jobs_count = 2
    cast(Any, scheduler.job_scheduler).created_return = 3

    executed: list[object] = []
    fake_session = _FakeSession(total_jobs=2, seed_count=2, executed=executed)

    @contextmanager
    def _session_scope() -> Iterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(scheduler_main, "session_scope", _session_scope)

    # Act
    created = scheduler._maybe_schedule_seed_jobs(unfinished_jobs=1)

    # Assert
    # remaining_seed = 10 - 2 = 8
    # capacity = 4 - 1 = 3
    # remaining_total = 20 - 2 = 18
    # to_create = min(8, 3, 18) = 3
    assert created == 3
    assert cast(Any, scheduler.job_scheduler).created_calls == [
        {
            "base_commit_hash": "deadbeef",
            "count": 3,
            "island_id": "main",
        }
    ]
    assert len(executed) == 1


def test_tick_reuses_cached_total_job_count(settings: Settings) -> None:
    scheduler = cast(Any, EvolutionScheduler.__new__(EvolutionScheduler))
    scheduler.settings = settings
    scheduler.console = Console(record=True)
    scheduler._max_total_jobs = 10
    scheduler._stop_requested = False
    scheduler._total_jobs_count = 4

    class _DummyIngestion:
        def ingest_completed_jobs(self) -> int:
            return 0

    class _DummyJobScheduler:
        def dispatch_pending_jobs(self) -> int:
            return 0

        def count_unfinished_jobs(self) -> int:
            return 1

        def count_total_jobs(self) -> int:
            raise AssertionError("tick should reuse the cached total job count")

        def schedule_jobs(self, unfinished_jobs: int, *, total_jobs: int) -> int:
            assert unfinished_jobs == 1
            assert total_jobs == 4
            return 2

    scheduler.ingestion = _DummyIngestion()
    scheduler.job_scheduler = _DummyJobScheduler()
    scheduler._maybe_schedule_seed_jobs = lambda unfinished_jobs: 0
    scheduler._create_best_fitness_branch_if_possible = lambda: None
    scheduler.stop = lambda: None

    stats = EvolutionScheduler.tick(cast(EvolutionScheduler, scheduler))

    assert stats["scheduled"] == 2
    assert scheduler._total_jobs_count == 6


def test_tick_accounts_for_seed_jobs_before_sampler_scheduling(settings: Settings) -> None:
    scheduler = cast(Any, EvolutionScheduler.__new__(EvolutionScheduler))
    scheduler.settings = settings
    scheduler.console = Console(record=True)
    scheduler._max_total_jobs = 10
    scheduler._stop_requested = False
    scheduler._total_jobs_count = 4

    class _DummyIngestion:
        def ingest_completed_jobs(self) -> int:
            return 0

    class _DummyJobScheduler:
        def dispatch_pending_jobs(self) -> int:
            return 0

        def count_unfinished_jobs(self) -> int:
            return 1

        def count_total_jobs(self) -> int:
            raise AssertionError("tick should not refresh total job count from the database")

        def schedule_jobs(self, unfinished_jobs: int, *, total_jobs: int) -> int:
            assert unfinished_jobs == 3
            assert total_jobs == 6
            return 1

    scheduler.ingestion = _DummyIngestion()
    scheduler.job_scheduler = _DummyJobScheduler()
    scheduler._maybe_schedule_seed_jobs = lambda unfinished_jobs: 2
    scheduler._create_best_fitness_branch_if_possible = lambda: None
    scheduler.stop = lambda: None

    stats = EvolutionScheduler.tick(cast(EvolutionScheduler, scheduler))

    assert stats["seed_scheduled"] == 2
    assert stats["scheduled"] == 1
    assert scheduler._total_jobs_count == 7
