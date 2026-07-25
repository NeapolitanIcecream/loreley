from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import uuid
from typing import Any, cast
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from rich.console import Console

import loreley.scheduler.job_scheduler as job_scheduler
from loreley.config import Settings
from loreley.core.map_elites.sampler import (
    MapElitesSampler,
    SamplingSnapshot,
    ScheduleJobRequest,
)
from loreley.core.worker.repair import REPAIR_MODE_REBASE_FROM_NEAREST_VIABLE
from loreley.db.models import JobStatus
from loreley.scheduler.baselines import BASELINE_STATUS_DEGRADED, BASELINE_STATUS_VALID
from loreley.scheduler.job_scheduler import (
    FailedCandidateRepairSampler,
    JobLeaseReclaimResult,
    JobScheduler,
    ScheduledRepairJob,
)
from tests.support import TestSettings


class DummySenderActor:
    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.sent: list[str] = []

    def send(self, job_id: str) -> None:
        if job_id in self.fail_on:
            raise RuntimeError("send failed")
        self.sent.append(job_id)


class DummyLog:
    def __init__(self) -> None:
        self.debug_calls: list[tuple[str, tuple[object, ...]]] = []

    def debug(self, message: str, *args: object) -> None:
        self.debug_calls.append((message, args))

    def exception(self, _message: str, *_args: object) -> None:  # pragma: no cover - not used here
        return


def test_enqueue_jobs_marks_only_sent_jobs(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_ids = [uuid.uuid4(), uuid.uuid4()]
    sender = DummySenderActor(fail_on={str(job_ids[1])})
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    marked: list[uuid.UUID] = []
    monkeypatch.setattr(
        JobScheduler,
        "_mark_jobs_queued",
        lambda _self, ids: marked.extend(list(ids)) or list(ids),
    )

    dispatched = scheduler._enqueue_jobs(job_ids)

    assert dispatched == 1
    assert sender.sent == [str(job_ids[0])]
    assert marked == [job_ids[0]]


def test_enqueue_jobs_logs_marked_vs_sent_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_ids = [uuid.uuid4(), uuid.uuid4()]
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    dummy_log = DummyLog()
    monkeypatch.setattr(job_scheduler, "log", dummy_log)
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    monkeypatch.setattr(JobScheduler, "_mark_jobs_queued", lambda _self, _ids: [])

    dispatched = scheduler._enqueue_jobs(job_ids)

    assert dispatched == len(job_ids)
    assert len(dummy_log.debug_calls) == 1
    message, args = dummy_log.debug_calls[0]
    assert "marked {} job(s) as QUEUED" in message
    assert args[0] == len(job_ids)
    assert args[1] == 0


def test_fetch_pending_job_ids_includes_stale_queued_jobs_for_redispatch(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: broker loss or restart could leave QUEUED jobs undispatchable."""

    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.worker_job_lease_ttl_seconds = 600
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    db_now = datetime(2026, 3, 25, 8, 30, tzinfo=timezone.utc)
    captured_stmt: list[Any] = []

    class DummyExecuteResult:
        def scalars(self) -> list[uuid.UUID]:
            return []

    class DummySession:
        def execute(self, stmt: Any) -> DummyExecuteResult:
            captured_stmt.append(stmt)
            return DummyExecuteResult()

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_scheduler, "session_scope", fake_scope)
    monkeypatch.setattr(job_scheduler, "_db_utc_now", lambda _session: db_now)

    fetched = scheduler._fetch_pending_job_ids(limit=10)

    assert fetched == []
    stmt = captured_stmt[0]
    params = stmt.compile().params
    assert JobStatus.PENDING in params.values()
    assert JobStatus.QUEUED in params.values()
    assert db_now - timedelta(seconds=600) in params.values()


def test_mark_jobs_queued_refreshes_stale_queued_redispatch_timestamp_pr24(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression PR #24: stale QUEUED redispatches kept an old queue timestamp."""

    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    db_now = datetime(2026, 3, 25, 8, 30, tzinfo=timezone.utc)
    stale_scheduled_at = db_now - timedelta(hours=2)
    job_row = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.QUEUED,
        scheduled_at=stale_scheduled_at,
    )

    class DummyExecuteResult:
        def scalars(self) -> list[object]:
            return [job_row]

    class DummySession:
        def execute(self, _stmt: Any) -> DummyExecuteResult:
            return DummyExecuteResult()

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_scheduler, "session_scope", fake_scope)
    monkeypatch.setattr(job_scheduler, "_db_utc_now", lambda _session: db_now)

    marked = scheduler._mark_jobs_queued([job_row.id])

    assert marked == [job_row.id]
    assert job_row.status is JobStatus.QUEUED
    assert job_row.scheduled_at == db_now


def test_schedule_jobs_reuses_single_sampling_snapshot_and_avoids_duplicate_bases(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )

    settings.scheduler_max_unfinished_jobs = 3
    settings.scheduler_schedule_batch_size = 2

    class DummySampler:
        def __init__(self) -> None:
            self.snapshot_calls = 0
            self.schedule_calls: list[tuple[str | None, tuple[str, ...]]] = []

        def get_sampling_snapshot(self, island_id: str | None = None):
            self.snapshot_calls += 1
            return SamplingSnapshot(
                island_id=island_id or "main",
                cell_fronts={0: ("base-a",), 1: ("base-b",)},
                items=((0, "base-a"), (1, "base-b")),
                neighbor_cell_indices=None,
                neighbor_commits=(),
                neighbor_coords=None,
            )

        def schedule_job(self, request: ScheduleJobRequest):
            excluded = tuple(
                sorted(str(commit) for commit in (request.excluded_base_commits or ()))
            )
            self.schedule_calls.append((request.island_id, excluded))
            if request.sampling_snapshot is None:
                raise AssertionError("sampling_snapshot should be reused across the batch")
            for _cell_index, commit_hash in request.sampling_snapshot.items:
                if commit_hash in set(excluded):
                    continue
                return SimpleNamespace(
                    job_id=uuid.uuid4(),
                    island_id=request.island_id or "main",
                    base_commit_hash=commit_hash,
                    inspiration_commit_hashes=(),
                )
            return None

    sampler = DummySampler()
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, sampler),
    )
    monkeypatch.setattr(JobScheduler, "_enqueue_jobs", lambda _self, ids: len(list(ids)))

    scheduled = scheduler.schedule_jobs(unfinished_jobs=0, total_jobs=0)

    assert scheduled == 2
    assert sampler.snapshot_calls == 1
    assert sampler.schedule_calls == [
        ("main", ()),
        ("main", ("base-a",)),
    ]


def test_schedule_jobs_round_robins_across_configured_islands(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.mapelites_islands = ("alpha", "beta", "gamma")
    settings.mapelites_migration_interval_jobs = 0
    settings.scheduler_max_unfinished_jobs = 6
    settings.scheduler_schedule_batch_size = 6

    class DummySampler:
        def __init__(self) -> None:
            self.snapshot_calls: list[str] = []
            self.schedule_calls: list[str] = []

        def get_sampling_snapshot(self, island_id: str | None = None) -> SamplingSnapshot:
            assert island_id is not None
            self.snapshot_calls.append(island_id)
            items = (
                (0, f"{island_id}-base-1"),
                (1, f"{island_id}-base-2"),
            )
            return SamplingSnapshot(
                island_id=island_id,
                cell_fronts={
                    0: (f"{island_id}-base-1",),
                    1: (f"{island_id}-base-2",),
                },
                items=items,
                neighbor_cell_indices=None,
                neighbor_commits=(),
                neighbor_coords=None,
            )

        def schedule_job(self, request: ScheduleJobRequest) -> object | None:
            assert request.island_id is not None
            assert request.sampling_snapshot is not None
            excluded = set(request.excluded_base_commits or ())
            for _cell_index, commit_hash in request.sampling_snapshot.items:
                if commit_hash in excluded:
                    continue
                self.schedule_calls.append(request.island_id)
                return SimpleNamespace(
                    job_id=uuid.uuid4(),
                    island_id=request.island_id,
                    base_commit_hash=commit_hash,
                    inspiration_commit_hashes=(),
                )
            return None

    sampler = DummySampler()
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, sampler),
    )
    monkeypatch.setattr(JobScheduler, "_enqueue_jobs", lambda _self, ids: len(list(ids)))

    scheduled = scheduler.schedule_jobs(unfinished_jobs=0, total_jobs=0)

    assert scheduled == 6
    assert sampler.snapshot_calls == ["alpha", "beta", "gamma"]
    assert sampler.schedule_calls == [
        "alpha",
        "beta",
        "gamma",
        "alpha",
        "beta",
        "gamma",
    ]


def test_schedule_jobs_records_periodic_cross_island_migration(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.mapelites_islands = ("alpha", "beta", "gamma")
    settings.mapelites_migration_interval_jobs = 2
    settings.scheduler_max_unfinished_jobs = 2
    settings.scheduler_schedule_batch_size = 2

    class DummySampler:
        def __init__(self) -> None:
            self.schedule_calls: list[dict[str, object]] = []

        def get_sampling_snapshot(self, island_id: str | None = None) -> SamplingSnapshot:
            assert island_id is not None
            commit_hash = f"{island_id}-elite"
            return SamplingSnapshot(
                island_id=island_id,
                cell_fronts={0: (commit_hash,)},
                items=((0, commit_hash),),
                neighbor_cell_indices=None,
                neighbor_commits=(),
                neighbor_coords=None,
            )

        def schedule_job(self, request: ScheduleJobRequest) -> object:
            assert request.island_id is not None
            assert request.sampling_snapshot is not None
            base_commit_hash = request.sampling_snapshot.items[0][1]
            self.schedule_calls.append(
                {
                    "island_id": request.island_id,
                    "base_commit_hash": base_commit_hash,
                    "migration_source_island_id": request.migration_source_island_id,
                    "migration_commit_hash": request.migration_commit_hash,
                }
            )
            return SimpleNamespace(
                job_id=uuid.uuid4(),
                island_id=request.island_id,
                base_commit_hash=base_commit_hash,
                inspiration_commit_hashes=(),
            )

    sampler = DummySampler()
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, sampler),
    )
    monkeypatch.setattr(JobScheduler, "_enqueue_jobs", lambda _self, ids: len(list(ids)))

    assert scheduler.schedule_jobs(unfinished_jobs=0, total_jobs=0) == 2
    assert sampler.schedule_calls == [
        {
            "island_id": "alpha",
            "base_commit_hash": "alpha-elite",
            "migration_source_island_id": None,
            "migration_commit_hash": None,
        },
        {
            "island_id": "beta",
            "base_commit_hash": "beta-elite",
            "migration_source_island_id": "gamma",
            "migration_commit_hash": "gamma-elite",
        },
    ]


def test_schedule_jobs_can_reuse_baseline_checked_campaign_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression ADR 0050: scheduler tick must not refresh program after baseline gating."""

    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    monkeypatch.setattr(
        JobScheduler,
        "_refresh_campaign_program_for_policy",
        lambda _self: (_ for _ in ()).throw(AssertionError("unexpected campaign refresh")),
    )
    settings.scheduler_max_unfinished_jobs = 1
    settings.scheduler_schedule_batch_size = 1

    class DummySampler:
        def get_sampling_snapshot(self, island_id: str | None = None) -> None:
            return None

    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, DummySampler()),
    )

    assert scheduler.schedule_jobs(
        unfinished_jobs=0,
        total_jobs=0,
        refresh_campaign_program=False,
    ) == 0


def test_seed_jobs_can_reuse_baseline_checked_campaign_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression ADR 0050: seed jobs must use the same program hash that passed the gate."""

    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    monkeypatch.setattr(
        JobScheduler,
        "_refresh_campaign_program_for_policy",
        lambda _self: (_ for _ in ()).throw(AssertionError("unexpected campaign refresh")),
    )
    settings.worker_evolution_global_goal = ""
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )

    assert scheduler.create_seed_jobs(
        base_commit_hash="root",
        count=1,
        refresh_campaign_program=False,
    ) == 0


def test_repair_scheduler_noops_when_repair_pool_deprecated(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    monkeypatch.setattr(
        JobScheduler,
        "_count_completed_normal_jobs_best_effort",
        lambda _self: 0,
    )
    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_normal_jobs_per_token = 9
    settings.failed_candidate_repair_max_tokens = 3
    settings.failed_candidate_repair_max_active_jobs = 1
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    repair_job_id = uuid.uuid4()
    repair_source_id = uuid.uuid4()

    class DummyRepairSampler:
        def __init__(self) -> None:
            self.scheduled = 0

        def count_active_repair_jobs(self, *, session=None) -> int:
            return 0

        def schedule_one(self, *, session=None) -> ScheduledRepairJob:
            self.scheduled += 1
            return ScheduledRepairJob(
                job_id=repair_job_id,
                repair_source_candidate_id=repair_source_id,
                base_commit_hash="base",
            )

    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    scheduler.repair_sampler = DummyRepairSampler()
    scheduler._repair_completed_normal_jobs_seen = 0
    monkeypatch.setattr(
        JobScheduler,
        "_count_completed_normal_jobs_best_effort",
        lambda _self: 9,
    )

    scheduled = scheduler._schedule_repair_jobs(capacity=1)

    assert scheduled == []
    assert scheduler._repair_tokens == 0
    assert scheduler.repair_sampler.scheduled == 0


def test_repair_scheduler_ignores_persistent_budget_after_deprecation(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: scheduler restarts must not lose already earned repair tokens."""

    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    monkeypatch.setattr(
        JobScheduler,
        "_count_completed_normal_jobs_best_effort",
        lambda _self: 9,
    )
    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_normal_jobs_per_token = 9
    settings.failed_candidate_repair_max_tokens = 3
    settings.failed_candidate_repair_max_active_jobs = 1
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    repair_job_id = uuid.uuid4()
    repair_source_id = uuid.uuid4()

    class DummyRepairSampler:
        def __init__(self) -> None:
            self.scheduled = 0

        def count_active_repair_jobs(self, *, session=None) -> int:
            return 0

        def schedule_one(self, *, session=None) -> ScheduledRepairJob:
            self.scheduled += 1
            return ScheduledRepairJob(
                job_id=repair_job_id,
                repair_source_candidate_id=repair_source_id,
                base_commit_hash="base",
            )

    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    scheduler.repair_sampler = DummyRepairSampler()
    scheduler._repair_tokens = 0

    scheduled = scheduler._schedule_repair_jobs(capacity=1)

    assert scheduled == []
    assert scheduler._repair_tokens == 0
    assert scheduler.repair_sampler.scheduled == 0


def test_repair_scheduler_does_not_enter_repair_lock_after_deprecation(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: scheduler repair dispatch must share the manual repair lock."""

    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_active_jobs = 1
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    repair_job_id = uuid.uuid4()
    repair_source_id = uuid.uuid4()
    events: list[str] = []
    locked_session = object()

    def _lock(**kwargs):
        events.append("lock.enter")
        try:
            return kwargs["callback"](locked_session)
        finally:
            events.append("lock.exit")

    def _persistent_tokens(*, session=None, **_kwargs) -> int:
        assert session is locked_session
        assert events == ["lock.enter"]
        events.append("tokens.available")
        return 1

    class DummyRepairSampler:
        def count_active_repair_jobs(self, *, session=None) -> int:
            assert session is locked_session
            assert events == ["lock.enter", "tokens.available"]
            events.append("active.count")
            return 0

        def schedule_one(self, *, session=None) -> ScheduledRepairJob:
            assert session is locked_session
            assert events == ["lock.enter", "tokens.available", "active.count"]
            events.append("schedule.one")
            return ScheduledRepairJob(
                job_id=repair_job_id,
                repair_source_candidate_id=repair_source_id,
                base_commit_hash="base",
            )

    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    scheduler.repair_sampler = DummyRepairSampler()
    scheduler._repair_tokens = 1

    scheduled = scheduler._schedule_repair_jobs(capacity=1)

    assert scheduled == []
    assert scheduler._repair_tokens == 1
    assert events == []


def test_repair_scheduler_recomputes_persistent_token_budget_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: manual API repair scheduling must consume scheduler-visible budget."""

    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_active_jobs = 1
    settings.failed_candidate_repair_max_jobs_per_tick = 1

    class DummyRepairSampler:
        def count_active_repair_jobs(self, *, session=None) -> int:
            raise AssertionError("persistent token budget should block before active-job count")

        def schedule_one(self, *, session=None) -> ScheduledRepairJob:
            raise AssertionError("persistent token budget should block repair scheduling")

    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    scheduler.repair_sampler = DummyRepairSampler()
    scheduler._repair_tokens = 1

    scheduled = scheduler._schedule_repair_jobs(capacity=1)

    assert scheduled == []
    assert scheduler._repair_tokens == 1


def test_repair_sampler_schedule_one_noops_after_deprecation(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: FAILED_CANDIDATE_REPAIR_MODE must control scheduled repair jobs."""

    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_mode = REPAIR_MODE_REBASE_FROM_NEAREST_VIABLE
    job_id = uuid.uuid4()
    source_id = uuid.uuid4()
    candidate = SimpleNamespace(
        id=source_id,
        nearest_viable_ancestor_hash="base",
        island_id="main",
        failure_summary="tests failed",
        failure_kind="test_failed",
        repair_state="eligible",
        repair_attempts=0,
        last_repair_job_id=None,
        campaign_program_hash=None,
    )

    class DummySession:
        def __init__(self) -> None:
            self.added_jobs: list[Any] = []

        def add(self, job: Any) -> None:
            self.added_jobs.append(job)

        def flush(self) -> None:
            self.added_jobs[-1].id = job_id

    session = DummySession()

    @contextmanager
    def fake_scope() -> Any:
        yield session

    monkeypatch.setattr(job_scheduler, "session_scope", fake_scope)
    monkeypatch.setattr(
        job_scheduler,
        "_db_utc_now",
        lambda _session: datetime(2026, 5, 8, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        job_scheduler,
        "load_campaign_program_snapshot_by_hash",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        FailedCandidateRepairSampler,
        "_select_candidate",
        lambda _self, *, session: candidate,
    )
    monkeypatch.setattr(
        FailedCandidateRepairSampler,
        "_has_active_repair_job",
        staticmethod(lambda **_kwargs: False),
    )
    sampler = FailedCandidateRepairSampler(settings=settings)

    scheduled = sampler.schedule_one()

    assert scheduled is None
    assert session.added_jobs == []
    assert candidate.repair_state == "eligible"
    assert candidate.repair_attempts == 0
    assert candidate.last_repair_job_id is None


def test_settings_rejects_unsupported_failed_candidate_repair_mode() -> None:
    """Unsupported repair modes should fail during settings construction."""

    with pytest.raises(ValidationError) as exc_info:
        TestSettings(
            FAILED_CANDIDATE_REPAIR_MODE="apply_from_failed_candidate",
            MAPELITES_CODE_EMBEDDING_DIMENSIONS=8,
            EXPERIMENT_ID="test",
        )

    assert "rebase_from_nearest_viable" in str(exc_info.value)


def test_repair_sampler_rejects_mutated_unsupported_repair_mode(settings: Settings) -> None:
    """Directly mutated settings must not silently schedule an unsupported mode."""

    settings.failed_candidate_repair_mode = "apply_from_failed_candidate"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="FAILED_CANDIDATE_REPAIR_MODE"):
        FailedCandidateRepairSampler(settings=settings)


def test_schedule_jobs_does_not_reserve_repair_slot_after_deprecation(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: normal MAP-Elites scheduling could consume every repair-capable tick."""

    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.scheduler_max_unfinished_jobs = 4
    settings.scheduler_schedule_batch_size = 2
    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_normal_jobs_per_token = 1
    settings.failed_candidate_repair_max_tokens = 3
    settings.failed_candidate_repair_max_active_jobs = 1
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    normal_job_ids = [uuid.uuid4(), uuid.uuid4()]
    repair_job_id = uuid.uuid4()
    enqueued: list[uuid.UUID] = []

    class DummySampler:
        def __init__(self) -> None:
            self.normal_scheduled = 0

        def get_sampling_snapshot(self, island_id: str | None = None):
            return SamplingSnapshot(
                island_id=island_id or "main",
                cell_fronts={0: ("base-a",), 1: ("base-b",)},
                items=((0, "base-a"), (1, "base-b")),
                neighbor_cell_indices=None,
                neighbor_commits=(),
                neighbor_coords=None,
            )

        def schedule_job(self, request: ScheduleJobRequest):
            assert request.sampling_snapshot is not None
            excluded = set(
                str(commit) for commit in (request.excluded_base_commits or ())
            )
            for index, (_cell_index, commit_hash) in enumerate(
                request.sampling_snapshot.items
            ):
                if commit_hash in excluded:
                    continue
                self.normal_scheduled += 1
                return SimpleNamespace(
                    job_id=normal_job_ids[index],
                    island_id=request.island_id or "main",
                    base_commit_hash=commit_hash,
                    inspiration_commit_hashes=(),
                )
            return None

    class DummyRepairSampler:
        def __init__(self) -> None:
            self.scheduled = 0

        def count_active_repair_jobs(self, *, session=None) -> int:
            return 0

        def schedule_one(self, *, session=None) -> ScheduledRepairJob:
            self.scheduled += 1
            return ScheduledRepairJob(
                job_id=repair_job_id,
                repair_source_candidate_id=uuid.uuid4(),
                base_commit_hash="base",
            )

    monkeypatch.setattr(
        JobScheduler,
        "_count_completed_normal_jobs_best_effort",
        lambda _self: 1,
    )
    sampler = DummySampler()
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, sampler),
    )
    scheduler.repair_sampler = DummyRepairSampler()
    scheduler._repair_completed_normal_jobs_seen = 0
    monkeypatch.setattr(
        JobScheduler,
        "_enqueue_jobs",
        lambda _self, ids: enqueued.extend(list(ids)) or len(list(ids)),
    )

    scheduled = scheduler.schedule_jobs(unfinished_jobs=0, total_jobs=0)

    assert scheduled == 2
    assert sampler.normal_scheduled == 2
    assert scheduler.repair_sampler.scheduled == 0
    assert repair_job_id not in enqueued
    assert len(enqueued) == 2


@pytest.mark.parametrize("source_baseline_status", [None, BASELINE_STATUS_DEGRADED])
def test_repair_candidate_requires_own_valid_campaign_baseline_under_required(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    captured_logs: list[dict[str, Any]],
    source_baseline_status: str | None,
) -> None:
    """Regression ADR 0050: an active program baseline cannot authorize another program's repair."""

    settings.baseline_bootstrap_policy = "required"
    source_program_hash = "a" * 64
    active_program_hash = "b" * 64
    attempt_id = uuid.uuid4()
    capsule_id = uuid.uuid4()
    candidate = SimpleNamespace(
        id=uuid.uuid4(),
        latest_evaluation_attempt_id=attempt_id,
        failure_evidence_id=capsule_id,
        nearest_viable_ancestor_hash="base",
        campaign_program_hash=source_program_hash,
    )
    baselines = {
        active_program_hash: SimpleNamespace(
            status=BASELINE_STATUS_VALID,
            campaign_program_hash=active_program_hash,
        ),
    }
    if source_baseline_status is not None:
        baselines[source_program_hash] = SimpleNamespace(
            status=source_baseline_status,
            campaign_program_hash=source_program_hash,
        )
    requested_hashes: list[tuple[str | None, bool]] = []

    class DummySession:
        def get(self, model: object, row_id: object) -> object | None:
            if model is job_scheduler.EvaluationAttempt and row_id == attempt_id:
                return SimpleNamespace(repairability="repairable")
            if model is job_scheduler.DiagnosticCapsule and row_id == capsule_id:
                return SimpleNamespace(policy_passed=True)
            return None

    def fake_load_latest_matching_baseline(
        *,
        session: object,
        settings: Settings,
        campaign_program_hash: str | None,
        valid_only: bool = False,
    ) -> object | None:
        requested_hashes.append((campaign_program_hash, valid_only))
        row = baselines.get(campaign_program_hash)
        if row is None:
            return None
        if valid_only and row.status != BASELINE_STATUS_VALID:
            return None
        return row

    monkeypatch.setattr(
        job_scheduler,
        "load_latest_matching_baseline",
        fake_load_latest_matching_baseline,
        raising=False,
    )
    monkeypatch.setattr(
        FailedCandidateRepairSampler,
        "_has_active_repair_job",
        staticmethod(lambda **_kwargs: False),
    )
    monkeypatch.setattr(
        FailedCandidateRepairSampler,
        "_ancestor_aggregate_ready",
        staticmethod(lambda **_kwargs: True),
    )

    sampler = FailedCandidateRepairSampler(settings=settings)

    assert sampler._strictly_eligible(session=DummySession(), candidate=candidate) is False
    assert requested_hashes == [(source_program_hash, True)]
    assert any(
        record["module"] == "scheduler.job_scheduler"
        and record["message"] == "Repair candidate blocked by campaign baseline"
        and record["extra"].get("campaign_program_hash") == source_program_hash
        and record["extra"].get("baseline_policy") == "required"
        for record in captured_logs
    )


def test_repair_candidate_with_degraded_source_baseline_remains_eligible_under_warn(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings.baseline_bootstrap_policy = "warn"
    source_program_hash = "a" * 64
    attempt_id = uuid.uuid4()
    capsule_id = uuid.uuid4()
    candidate = SimpleNamespace(
        id=uuid.uuid4(),
        latest_evaluation_attempt_id=attempt_id,
        failure_evidence_id=capsule_id,
        nearest_viable_ancestor_hash="base",
        campaign_program_hash=source_program_hash,
    )

    class DummySession:
        def get(self, model: object, row_id: object) -> object | None:
            if model is job_scheduler.EvaluationAttempt and row_id == attempt_id:
                return SimpleNamespace(repairability="repairable")
            if model is job_scheduler.DiagnosticCapsule and row_id == capsule_id:
                return SimpleNamespace(policy_passed=True)
            return None

    def fake_load_latest_matching_baseline(
        *,
        session: object,
        settings: Settings,
        campaign_program_hash: str | None,
        valid_only: bool = False,
    ) -> object | None:
        assert campaign_program_hash == source_program_hash
        assert valid_only is False
        return SimpleNamespace(
            status=BASELINE_STATUS_DEGRADED,
            campaign_program_hash=source_program_hash,
        )

    monkeypatch.setattr(
        job_scheduler,
        "load_latest_matching_baseline",
        fake_load_latest_matching_baseline,
        raising=False,
    )
    monkeypatch.setattr(
        FailedCandidateRepairSampler,
        "_has_active_repair_job",
        staticmethod(lambda **_kwargs: False),
    )
    monkeypatch.setattr(
        FailedCandidateRepairSampler,
        "_ancestor_aggregate_ready",
        staticmethod(lambda **_kwargs: True),
    )

    sampler = FailedCandidateRepairSampler(settings=settings)

    assert sampler._strictly_eligible(session=DummySession(), candidate=candidate) is True


def test_repair_candidate_without_source_baseline_is_not_eligible_under_warn(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    captured_logs: list[dict[str, Any]],
) -> None:
    settings.baseline_bootstrap_policy = "warn"
    source_program_hash = "a" * 64
    attempt_id = uuid.uuid4()
    capsule_id = uuid.uuid4()
    candidate = SimpleNamespace(
        id=uuid.uuid4(),
        latest_evaluation_attempt_id=attempt_id,
        failure_evidence_id=capsule_id,
        nearest_viable_ancestor_hash="base",
        campaign_program_hash=source_program_hash,
    )

    class DummySession:
        def get(self, model: object, row_id: object) -> object | None:
            if model is job_scheduler.EvaluationAttempt and row_id == attempt_id:
                return SimpleNamespace(repairability="repairable")
            if model is job_scheduler.DiagnosticCapsule and row_id == capsule_id:
                return SimpleNamespace(policy_passed=True)
            return None

    def fake_load_latest_matching_baseline(
        *,
        session: object,
        settings: Settings,
        campaign_program_hash: str | None,
        valid_only: bool = False,
    ) -> object | None:
        assert campaign_program_hash == source_program_hash
        assert valid_only is False
        return None

    monkeypatch.setattr(
        job_scheduler,
        "load_latest_matching_baseline",
        fake_load_latest_matching_baseline,
        raising=False,
    )
    monkeypatch.setattr(
        FailedCandidateRepairSampler,
        "_has_active_repair_job",
        staticmethod(lambda **_kwargs: False),
    )
    monkeypatch.setattr(
        FailedCandidateRepairSampler,
        "_ancestor_aggregate_ready",
        staticmethod(lambda **_kwargs: True),
    )

    sampler = FailedCandidateRepairSampler(settings=settings)

    assert sampler._strictly_eligible(session=DummySession(), candidate=candidate) is False
    assert any(
        record["module"] == "scheduler.job_scheduler"
        and record["message"] == "Repair candidate blocked by missing campaign baseline under warn policy"
        and record["extra"].get("campaign_program_hash") == source_program_hash
        and record["extra"].get("baseline_policy") == "warn"
        for record in captured_logs
    )


def test_repair_sampler_queries_only_original_depth_zero_candidates(settings: Settings) -> None:
    """MVP repair scheduling is one-generation: repair-produced failures stay audit-only."""

    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_depth = 99
    captured: list[Any] = []

    class DummyResult:
        def scalars(self) -> list[object]:
            return []

    class DummySession:
        def execute(self, stmt: Any) -> DummyResult:
            captured.append(stmt)
            return DummyResult()

    sampler = FailedCandidateRepairSampler(settings=settings)

    assert sampler._select_candidate(session=DummySession()) is None
    compiled = captured[0].compile()
    sql = str(compiled)
    params = list(compiled.params.values())
    assert "candidate_commits.repair_source_candidate_id IS NULL" in sql
    assert "candidate_commits.failed_depth = " in sql
    assert 0 in params
    assert 99 not in params


def test_reclaim_stale_running_jobs_requeues_expired_attempts(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.scheduler_stale_running_reclaim_batch_size = 10
    settings.scheduler_stale_running_max_recovery_attempts = 3
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    now = datetime.now(timezone.utc)
    job_row = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.RUNNING,
        recovery_count=0,
        run_token=uuid.uuid4(),
        worker_id="worker-01",
        lease_expires_at=now - timedelta(minutes=5),
        heartbeat_at=now - timedelta(minutes=10),
        started_at=now - timedelta(hours=1),
        completed_at=None,
        scheduled_at=None,
        last_error=None,
    )

    class DummyExecuteResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return list(self._rows)

    class DummySession:
        def execute(self, _stmt: Any) -> DummyExecuteResult:
            return DummyExecuteResult([job_row])

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_scheduler, "session_scope", fake_scope)

    reclaimed = scheduler.reclaim_stale_running_jobs(now=now)

    assert reclaimed == JobLeaseReclaimResult(requeued=1, failed=0)
    assert job_row.status is JobStatus.PENDING
    assert job_row.recovery_count == 1
    assert job_row.run_token is None
    assert job_row.worker_id is None
    assert job_row.heartbeat_at is None
    assert job_row.lease_expires_at is None
    assert job_row.started_at is None
    assert "lease expired" in str(job_row.last_error).lower()


def test_reclaim_stale_running_jobs_requeues_running_jobs_missing_lease_metadata(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.scheduler_stale_running_reclaim_batch_size = 10
    settings.scheduler_stale_running_max_recovery_attempts = 3
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    now = datetime.now(timezone.utc)
    job_row = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.RUNNING,
        recovery_count=0,
        run_token=None,
        worker_id=None,
        lease_expires_at=None,
        heartbeat_at=None,
        started_at=now - timedelta(hours=1),
        completed_at=None,
        scheduled_at=None,
        last_error=None,
    )

    class DummyExecuteResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return list(self._rows)

    class DummySession:
        def execute(self, _stmt: Any) -> DummyExecuteResult:
            return DummyExecuteResult([job_row])

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_scheduler, "session_scope", fake_scope)

    reclaimed = scheduler.reclaim_stale_running_jobs(now=now)

    assert reclaimed == JobLeaseReclaimResult(requeued=1, failed=0)
    assert job_row.status is JobStatus.PENDING
    assert job_row.run_token is None
    assert job_row.worker_id is None
    assert "lease metadata missing" in str(job_row.last_error).lower()


def test_reclaim_stale_running_jobs_uses_database_time_when_now_omitted(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.scheduler_stale_running_reclaim_batch_size = 10
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    db_now = datetime(2026, 3, 25, 8, 30, tzinfo=timezone.utc)
    captured_stmt: list[Any] = []

    class DummyExecuteResult:
        def __init__(self) -> None:
            self._rows: list[object] = []

        def scalars(self) -> list[object]:
            return list(self._rows)

    class DummySession:
        def execute(self, stmt: Any) -> DummyExecuteResult:
            captured_stmt.append(stmt)
            return DummyExecuteResult()

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_scheduler, "session_scope", fake_scope)
    monkeypatch.setattr(job_scheduler, "_db_utc_now", lambda _session: db_now)

    scheduler.reclaim_stale_running_jobs()

    stmt = captured_stmt[0]
    params = stmt.compile().params
    assert db_now in params.values()


def test_reclaim_stale_running_jobs_fails_after_recovery_budget_and_logs_signal(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    captured_logs,
) -> None:
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.scheduler_stale_running_reclaim_batch_size = 10
    settings.scheduler_stale_running_max_recovery_attempts = 2
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    now = datetime.now(timezone.utc)
    job_row = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.RUNNING,
        recovery_count=2,
        run_token=uuid.uuid4(),
        worker_id="worker-01",
        lease_expires_at=now - timedelta(minutes=5),
        heartbeat_at=now - timedelta(minutes=10),
        started_at=now - timedelta(hours=1),
        completed_at=None,
        scheduled_at=None,
        last_error=None,
    )

    class DummyExecuteResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return list(self._rows)

    class DummySession:
        def execute(self, _stmt: Any) -> DummyExecuteResult:
            return DummyExecuteResult([job_row])

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_scheduler, "session_scope", fake_scope)

    reclaimed = scheduler.reclaim_stale_running_jobs(now=now)

    assert reclaimed == JobLeaseReclaimResult(requeued=0, failed=1)
    assert job_row.status is JobStatus.FAILED
    assert job_row.completed_at is not None
    assert any(
        record["module"] == "scheduler.job_scheduler"
        and "failed after stale lease reclaim" in record["message"].lower()
        for record in captured_logs
    )


@pytest.mark.parametrize(
    ("repair_attempts", "max_attempts", "expected_state"),
    [
        (1, 1, "exhausted"),
        (1, 2, "eligible"),
    ],
)
def test_reclaim_stale_running_repair_job_restores_source_after_recovery_budget(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    repair_attempts: int,
    max_attempts: int,
    expected_state: str,
) -> None:
    """Regression: stale repair jobs could strand their source in repairing."""

    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    settings.scheduler_stale_running_reclaim_batch_size = 10
    settings.scheduler_stale_running_max_recovery_attempts = 1
    settings.failed_candidate_repair_max_attempts = max_attempts
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    now = datetime.now(timezone.utc)
    source_id = uuid.uuid4()
    source = SimpleNamespace(repair_state="repairing", repair_attempts=repair_attempts)
    job_row = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.RUNNING,
        recovery_count=1,
        run_token=uuid.uuid4(),
        worker_id="worker-01",
        lease_expires_at=now - timedelta(minutes=5),
        heartbeat_at=now - timedelta(minutes=10),
        started_at=now - timedelta(hours=1),
        completed_at=None,
        scheduled_at=None,
        last_error=None,
        job_kind="repair",
        repair_source_candidate_id=source_id,
    )

    class DummyExecuteResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return list(self._rows)

    class DummySession:
        def execute(self, _stmt: Any) -> DummyExecuteResult:
            return DummyExecuteResult([job_row])

        def get(self, model: Any, row_id: uuid.UUID) -> object | None:
            if model is job_scheduler.CandidateCommit and row_id == source_id:
                return source
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr(job_scheduler, "session_scope", fake_scope)

    reclaimed = scheduler.reclaim_stale_running_jobs(now=now)

    assert reclaimed == JobLeaseReclaimResult(requeued=0, failed=1)
    assert job_row.status is JobStatus.FAILED
    assert source.repair_state == expected_state
