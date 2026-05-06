from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import uuid
from typing import Any, cast
from types import SimpleNamespace

import pytest
from rich.console import Console

import loreley.scheduler.job_scheduler as job_scheduler
from loreley.config import Settings
from loreley.core.map_elites.sampler import MapElitesSampler, SamplingSnapshot
from loreley.db.models import JobStatus
from loreley.scheduler.job_scheduler import JobLeaseReclaimResult, JobScheduler, ScheduledRepairJob


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
                cell_commits={0: "base-a", 1: "base-b"},
                cell_objectives={0: 1.0, 1: 2.0},
                items=((0, "base-a"), (1, "base-b")),
                neighbor_cell_indices=None,
                neighbor_commits=(),
                neighbor_coords=None,
            )

        def schedule_job(
            self,
            *,
            island_id: str | None = None,
            sampling_snapshot: SamplingSnapshot | None = None,
            excluded_base_commits=None,
            **_kwargs: object,
        ):
            excluded = tuple(sorted(str(commit) for commit in (excluded_base_commits or ())))
            self.schedule_calls.append((island_id, excluded))
            if sampling_snapshot is None:
                raise AssertionError("sampling_snapshot should be reused across the batch")
            for _cell_index, commit_hash in sampling_snapshot.items:
                if commit_hash in set(excluded):
                    continue
                return SimpleNamespace(
                    job_id=uuid.uuid4(),
                    island_id=island_id or "main",
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


def test_repair_scheduler_consumes_token_for_scheduled_repair_job(
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

        def count_active_repair_jobs(self) -> int:
            return 0

        def schedule_one(self) -> ScheduledRepairJob:
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

    assert scheduled == [repair_job_id]
    assert scheduler._repair_tokens == 0
    assert scheduler.repair_sampler.scheduled == 1


def test_schedule_jobs_reserves_repair_slot_when_normal_sampling_can_fill_batch(
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
                cell_commits={0: "base-a", 1: "base-b"},
                cell_objectives={0: 1.0, 1: 2.0},
                items=((0, "base-a"), (1, "base-b")),
                neighbor_cell_indices=None,
                neighbor_commits=(),
                neighbor_coords=None,
            )

        def schedule_job(
            self,
            *,
            island_id: str | None = None,
            sampling_snapshot: SamplingSnapshot | None = None,
            excluded_base_commits=None,
            **_kwargs: object,
        ):
            excluded = set(str(commit) for commit in (excluded_base_commits or ()))
            for index, (_cell_index, commit_hash) in enumerate(sampling_snapshot.items):
                if commit_hash in excluded:
                    continue
                self.normal_scheduled += 1
                return SimpleNamespace(
                    job_id=normal_job_ids[index],
                    island_id=island_id or "main",
                    base_commit_hash=commit_hash,
                    inspiration_commit_hashes=(),
                )
            return None

    class DummyRepairSampler:
        def __init__(self) -> None:
            self.scheduled = 0

        def count_active_repair_jobs(self) -> int:
            return 0

        def schedule_one(self) -> ScheduledRepairJob:
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
    assert sampler.normal_scheduled == 1
    assert scheduler.repair_sampler.scheduled == 1
    assert repair_job_id in enqueued
    assert len(enqueued) == 2


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
