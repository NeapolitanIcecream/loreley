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
    unfinished_seed_count: int = 0
    pending_ingestion_seed_count: int = 0

    def execute(self, stmt: Any) -> _FakeResult:
        self.executed.append(stmt)

        # Guardrail: the scheduler must not load ORM rows for this check.
        descriptions = list(getattr(stmt, "column_descriptions", []))
        if any(d.get("expr") is EvolutionJob or d.get("type") is EvolutionJob for d in descriptions):
            raise AssertionError("seed scheduling must not execute select(EvolutionJob)")

        descriptions = list(getattr(stmt, "column_descriptions", []))
        if len(descriptions) == 3:
            return _FakeResult(
                (
                    self.seed_count,
                    self.unfinished_seed_count,
                    self.pending_ingestion_seed_count,
                )
            )
        if len(descriptions) == 2:
            return _FakeResult((self.seed_count, self.unfinished_seed_count))
        if len(descriptions) == 1:
            return _FakeResult((self.seed_count,))
        return _FakeResult((self.total_jobs, self.seed_count))


@dataclass
class _DummyManager:
    records: list[object]
    history_count: int = 0

    def get_records(self, _island_id: str) -> list[object]:
        return self.records

    def count_pca_history_samples(self, _island_id: str | None = None) -> int:
        return self.history_count


@dataclass
class _DummyJobScheduler:
    created_calls: list[dict[str, object]]
    created_return: int = 0

    def create_seed_jobs(
        self,
        *,
        base_commit_hash: str,
        count: int,
        island_id: str | None = None,
        refresh_campaign_program: bool = True,
    ) -> int:
        self.created_calls.append(
            {
                "base_commit_hash": base_commit_hash,
                "count": count,
                "island_id": island_id,
                "refresh_campaign_program": refresh_campaign_program,
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


def test_seed_scheduling_is_not_blocked_by_unrelated_non_seed_jobs(
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
    cast(Any, scheduler.job_scheduler).created_return = 4
    executed: list[object] = []
    fake_session = _FakeSession(total_jobs=3, seed_count=2, executed=executed)

    @contextmanager
    def _session_scope() -> Iterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(scheduler_main, "session_scope", _session_scope)

    # Act
    created = scheduler._maybe_schedule_seed_jobs(unfinished_jobs=0)

    # Assert
    assert created == 4
    assert cast(Any, scheduler.job_scheduler).created_calls == [
        {
            "base_commit_hash": "deadbeef",
            "count": 4,
            "island_id": "main",
            "refresh_campaign_program": False,
        }
    ]
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
    # remaining_seed = 10 - 0 = 10
    # capacity = 4 - 1 = 3
    # remaining_total = 20 - 2 = 18
    # to_create = min(10, 3, 18) = 3
    assert created == 3
    assert cast(Any, scheduler.job_scheduler).created_calls == [
        {
            "base_commit_hash": "deadbeef",
            "count": 3,
            "island_id": "main",
            "refresh_campaign_program": False,
        }
    ]
    assert len(executed) == 1


def test_seed_scheduling_replenishes_failed_seed_jobs_until_warmup_samples_exist(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    # Arrange
    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": "deadbeef",
            "mapelites_seed_population_size": 10,
            "mapelites_feature_normalization_warmup_samples": 10,
            "scheduler_max_unfinished_jobs": 4,
        }
    )
    scheduler = _make_scheduler(settings=settings, records=[])
    scheduler._max_total_jobs = 100
    scheduler._total_jobs_count = 10
    cast(Any, scheduler.manager).history_count = 6
    cast(Any, scheduler.job_scheduler).created_return = 4

    executed: list[object] = []
    fake_session = _FakeSession(
        total_jobs=10,
        seed_count=10,
        unfinished_seed_count=0,
        executed=executed,
    )

    @contextmanager
    def _session_scope() -> Iterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(scheduler_main, "session_scope", _session_scope)

    # Act
    created = scheduler._maybe_schedule_seed_jobs(unfinished_jobs=0)

    # Assert
    assert created == 4
    assert cast(Any, scheduler.job_scheduler).created_calls == [
        {
            "base_commit_hash": "deadbeef",
            "count": 4,
            "island_id": "main",
            "refresh_campaign_program": False,
        }
    ]
    assert len(executed) == 1


def test_seed_scheduling_counts_in_flight_seed_jobs_as_warmup_candidates(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    # Arrange
    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": "deadbeef",
            "mapelites_seed_population_size": 10,
            "mapelites_feature_normalization_warmup_samples": 10,
            "scheduler_max_unfinished_jobs": 4,
        }
    )
    scheduler = _make_scheduler(settings=settings, records=[])
    scheduler._max_total_jobs = 100
    scheduler._total_jobs_count = 10
    cast(Any, scheduler.manager).history_count = 8
    cast(Any, scheduler.job_scheduler).created_return = 2

    executed: list[object] = []
    fake_session = _FakeSession(
        total_jobs=10,
        seed_count=10,
        unfinished_seed_count=2,
        executed=executed,
    )

    @contextmanager
    def _session_scope() -> Iterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(scheduler_main, "session_scope", _session_scope)

    # Act
    created = scheduler._maybe_schedule_seed_jobs(unfinished_jobs=2)

    # Assert
    assert created == 0
    assert cast(Any, scheduler.job_scheduler).created_calls == []
    assert len(executed) == 1


def test_seed_scheduling_counts_uningested_succeeded_seed_jobs_as_warmup_candidates(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    # Arrange
    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": "deadbeef",
            "mapelites_seed_population_size": 10,
            "mapelites_feature_normalization_warmup_samples": 10,
            "scheduler_max_unfinished_jobs": 4,
        }
    )
    scheduler = _make_scheduler(settings=settings, records=[])
    scheduler._max_total_jobs = 100
    scheduler._total_jobs_count = 10
    cast(Any, scheduler.manager).history_count = 8
    cast(Any, scheduler.job_scheduler).created_return = 2

    executed: list[object] = []
    fake_session = _FakeSession(
        total_jobs=10,
        seed_count=10,
        unfinished_seed_count=0,
        pending_ingestion_seed_count=2,
        executed=executed,
    )

    @contextmanager
    def _session_scope() -> Iterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(scheduler_main, "session_scope", _session_scope)

    # Act
    created = scheduler._maybe_schedule_seed_jobs(unfinished_jobs=0)

    # Assert
    assert created == 0
    assert cast(Any, scheduler.job_scheduler).created_calls == []
    assert len(executed) == 1


def test_seed_scheduling_keeps_one_readiness_probe_after_warmup(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": "deadbeef",
            "mapelites_seed_population_size": 10,
            "mapelites_feature_normalization_warmup_samples": 10,
            "scheduler_max_unfinished_jobs": 4,
        }
    )
    scheduler = _make_scheduler(settings=settings, records=[])
    scheduler._total_jobs_count = 10
    cast(Any, scheduler.manager).history_count = 10
    cast(Any, scheduler.job_scheduler).created_return = 1
    fake_session = _FakeSession(total_jobs=10, seed_count=10, executed=[])

    @contextmanager
    def _session_scope() -> Iterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(scheduler_main, "session_scope", _session_scope)

    created = scheduler._maybe_schedule_seed_jobs(unfinished_jobs=0)

    assert created == 1
    assert cast(Any, scheduler.job_scheduler).created_calls == [
        {
            "base_commit_hash": "deadbeef",
            "count": 1,
            "island_id": "main",
            "refresh_campaign_program": False,
        }
    ]
    assert "phase=readiness" in scheduler.console.export_text()


@pytest.mark.parametrize(
    ("unfinished_seed_jobs", "pending_ingestion_seed_jobs"),
    [(1, 0), (0, 1)],
)
def test_seed_scheduling_does_not_duplicate_a_post_warmup_readiness_probe(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    unfinished_seed_jobs: int,
    pending_ingestion_seed_jobs: int,
) -> None:
    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": "deadbeef",
            "mapelites_seed_population_size": 10,
            "mapelites_feature_normalization_warmup_samples": 10,
            "scheduler_max_unfinished_jobs": 4,
        }
    )
    scheduler = _make_scheduler(settings=settings, records=[])
    scheduler._total_jobs_count = 10
    cast(Any, scheduler.manager).history_count = 10
    fake_session = _FakeSession(
        total_jobs=10,
        seed_count=10,
        unfinished_seed_count=unfinished_seed_jobs,
        pending_ingestion_seed_count=pending_ingestion_seed_jobs,
        executed=[],
    )

    @contextmanager
    def _session_scope() -> Iterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(scheduler_main, "session_scope", _session_scope)

    created = scheduler._maybe_schedule_seed_jobs(
        unfinished_jobs=unfinished_seed_jobs
    )

    assert created == 0
    assert cast(Any, scheduler.job_scheduler).created_calls == []


def test_seed_scheduling_distributes_capacity_across_empty_islands(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": "deadbeef",
            "mapelites_islands": ("alpha", "beta", "gamma"),
            "mapelites_seed_population_size": 10,
            "scheduler_max_unfinished_jobs": 5,
        }
    )
    scheduler = _make_scheduler(settings=settings, records=[])
    scheduler._max_total_jobs = 20
    scheduler._total_jobs_count = 0

    class _CreatingJobScheduler(_DummyJobScheduler):
        def create_seed_jobs(
            self,
            *,
            base_commit_hash: str,
            count: int,
            island_id: str | None = None,
            refresh_campaign_program: bool = True,
        ) -> int:
            super().create_seed_jobs(
                base_commit_hash=base_commit_hash,
                count=count,
                island_id=island_id,
                refresh_campaign_program=refresh_campaign_program,
            )
            return count

    scheduler.job_scheduler = _CreatingJobScheduler(created_calls=[])
    monkeypatch.setattr(
        scheduler,
        "_count_seed_warmup_job_counts",
        lambda *, island_id: scheduler_main._SeedWarmupJobCounts(0, 0, 0),
    )

    created = scheduler._maybe_schedule_seed_jobs(unfinished_jobs=0)

    assert created == 5
    assert cast(Any, scheduler.job_scheduler).created_calls == [
        {
            "base_commit_hash": "deadbeef",
            "count": 2,
            "island_id": "alpha",
            "refresh_campaign_program": False,
        },
        {
            "base_commit_hash": "deadbeef",
            "count": 2,
            "island_id": "beta",
            "refresh_campaign_program": False,
        },
        {
            "base_commit_hash": "deadbeef",
            "count": 1,
            "island_id": "gamma",
            "refresh_campaign_program": False,
        },
    ]


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

        def schedule_jobs(
            self,
            unfinished_jobs: int,
            *,
            total_jobs: int,
            refresh_campaign_program: bool = True,
        ) -> int:
            assert unfinished_jobs == 1
            assert total_jobs == 4
            assert refresh_campaign_program is False
            return 2

    scheduler.ingestion = _DummyIngestion()
    scheduler.job_scheduler = _DummyJobScheduler()
    scheduler._maybe_schedule_seed_jobs = lambda unfinished_jobs: 0
    scheduler._create_primary_objective_branch_if_possible = lambda: None
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

        def schedule_jobs(
            self,
            unfinished_jobs: int,
            *,
            total_jobs: int,
            refresh_campaign_program: bool = True,
        ) -> int:
            assert unfinished_jobs == 3
            assert total_jobs == 6
            assert refresh_campaign_program is False
            return 1

    scheduler.ingestion = _DummyIngestion()
    scheduler.job_scheduler = _DummyJobScheduler()
    scheduler._maybe_schedule_seed_jobs = lambda unfinished_jobs: 2
    scheduler._create_primary_objective_branch_if_possible = lambda: None
    scheduler.stop = lambda: None

    stats = EvolutionScheduler.tick(cast(EvolutionScheduler, scheduler))

    assert stats["seed_scheduled"] == 2
    assert stats["scheduled"] == 1
    assert scheduler._total_jobs_count == 7


def test_tick_at_job_limit_waits_for_pending_ingestion(settings: Settings) -> None:
    scheduler = cast(Any, EvolutionScheduler.__new__(EvolutionScheduler))
    scheduler.settings = settings
    scheduler.console = Console(record=True)
    scheduler._max_total_jobs = 10
    scheduler._stop_requested = False
    scheduler._total_jobs_count = 10

    class _DummyIngestion:
        def ingest_completed_jobs(self) -> int:
            return 2

        def count_pending_ingestion_jobs(self) -> int:
            return 3

    class _DummyJobScheduler:
        def reclaim_stale_running_jobs(self) -> object:
            return object()

        def dispatch_pending_jobs(self) -> int:
            return 0

        def count_unfinished_jobs(self) -> int:
            return 0

        def schedule_jobs(
            self,
            unfinished_jobs: int,
            *,
            total_jobs: int,
            refresh_campaign_program: bool = True,
        ) -> int:
            assert unfinished_jobs == 0
            assert total_jobs == 10
            assert refresh_campaign_program is False
            return 0

    scheduler.ingestion = _DummyIngestion()
    scheduler.job_scheduler = _DummyJobScheduler()
    scheduler._ensure_campaign_baseline_ready = lambda: None
    scheduler._maybe_schedule_seed_jobs = lambda unfinished_jobs: 0
    branch_updates: list[object] = []
    scheduler._create_primary_objective_branch_if_possible = lambda: branch_updates.append(
        object()
    )

    stats = EvolutionScheduler.tick(cast(EvolutionScheduler, scheduler))

    assert stats["pending_ingestion"] == 3
    assert scheduler._stop_requested is False
    assert branch_updates == []
    assert "pending_ingestion=3" in scheduler.console.export_text()


def test_missing_primary_candidate_is_a_clean_terminal_state(
    settings: Settings,
) -> None:
    scheduler = cast(Any, EvolutionScheduler.__new__(EvolutionScheduler))
    scheduler.settings = settings
    scheduler.console = Console(record=True)
    scheduler._resolve_best_primary_commit = lambda: (None, {})

    created = EvolutionScheduler._create_primary_objective_branch_if_possible(
        cast(EvolutionScheduler, scheduler)
    )

    assert created is False
    output = scheduler.console.export_text()
    assert "Primary-objective branch not created" in output
    assert "candidate with the configured primary objective" in output
