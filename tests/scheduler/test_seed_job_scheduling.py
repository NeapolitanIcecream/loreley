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
    def __init__(self, row: tuple[object, object]) -> None:
        self._row = row

    def one(self) -> tuple[object, object]:
        return self._row


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

