from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from uuid import uuid4

from rich.console import Console

import loreley.scheduler.ingestion as ingestion_mod
from loreley.config import Settings
from loreley.scheduler.ingestion import IngestionError, JobSnapshot, MapElitesIngestion


def _make_ingestion(tmp_path) -> MapElitesIngestion:
    settings = Settings.model_validate({"mapelites_code_embedding_dimensions": 8})
    return MapElitesIngestion(
        settings=settings,
        console=Console(record=True),
        repo_root=tmp_path,
        repo=cast(Any, object()),
        manager=cast(Any, object()),  # patched per test
    )


def test_ingest_completed_jobs_continues_after_snapshot_error(monkeypatch, tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)

    snapshots = [
        JobSnapshot(
            job_id=uuid4(),
            base_commit_hash=None,
            island_id=None,
            result_commit_hash="bad",
            completed_at=None,
        ),
        JobSnapshot(
            job_id=uuid4(),
            base_commit_hash=None,
            island_id=None,
            result_commit_hash="good",
            completed_at=None,
        ),
    ]
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_jobs_requiring_ingestion",
        lambda _self, *, limit: snapshots,
    )

    recorded: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_ingestion_state",
        lambda _self, _snapshot, *, status, reason=None, **_kwargs: recorded.append((status, reason)),
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_load_metrics_payload_batch",
        lambda _self, _commit_hashes, *, session: ({}, {}),
    )

    @contextmanager
    def fake_scope():
        yield object()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    def fake_ingest_snapshot(
        _self,
        snapshot: JobSnapshot,
        *,
        snapshot_session: Any | None = None,
    ) -> bool:
        if snapshot.result_commit_hash == "bad":
            raise RuntimeError("boom")
        return True

    monkeypatch.setattr(ingestion_mod.MapElitesIngestion, "_ingest_snapshot", fake_ingest_snapshot)

    assert ingestion.ingest_completed_jobs() == 1
    assert recorded == [("failed", "boom")]


def test_ingest_snapshot_records_failed_when_commit_unavailable(monkeypatch, tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)

    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_commit_available",
        lambda _self, _commit_hash: (_ for _ in ()).throw(IngestionError("missing commit")),
    )

    recorded: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_ingestion_state",
        lambda _self, _snapshot, *, status, reason=None, **_kwargs: recorded.append((status, reason)),
    )

    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash=None,
        island_id=None,
        result_commit_hash="deadbeef",
        completed_at=None,
    )

    assert ingestion._ingest_snapshot(snapshot) is False
    assert recorded == [("failed", "missing commit")]


def test_ingest_completed_jobs_uses_prefetch_session_only_for_metrics_prefetch(
    monkeypatch, tmp_path
) -> None:
    ingestion = _make_ingestion(tmp_path)
    snapshots = [
        JobSnapshot(
            job_id=uuid4(),
            base_commit_hash=None,
            island_id=None,
            result_commit_hash="c1",
            completed_at=None,
        ),
        JobSnapshot(
            job_id=uuid4(),
            base_commit_hash=None,
            island_id=None,
            result_commit_hash="c2",
            completed_at=None,
        ),
    ]
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_jobs_requiring_ingestion",
        lambda _self, *, limit: snapshots,
    )

    created_sessions: list[object] = []

    @contextmanager
    def fake_scope():
        marker = object()
        created_sessions.append(marker)
        yield marker

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    seen_sessions: list[object | None] = []

    def fake_ingest_snapshot(
        _self,
        snapshot: JobSnapshot,
        *,
        snapshot_session: Any | None = None,
    ) -> bool:
        seen_sessions.append(snapshot_session)
        return True

    monkeypatch.setattr(ingestion_mod.MapElitesIngestion, "_ingest_snapshot", fake_ingest_snapshot)
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_load_metrics_payload_batch",
        lambda _self, _commit_hashes, *, session: ({}, {}),
    )

    assert ingestion.ingest_completed_jobs() == 2
    assert len(created_sessions) == 1
    assert seen_sessions == [None, None]


def test_ingest_completed_jobs_prefetches_metrics_with_canonical_hashes(
    monkeypatch,
    tmp_path,
) -> None:
    ingestion = _make_ingestion(tmp_path)
    snapshots = [
        JobSnapshot(
            job_id=uuid4(),
            base_commit_hash=None,
            island_id=None,
            result_commit_hash="abc1234",
            completed_at=None,
        ),
    ]
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_jobs_requiring_ingestion",
        lambda _self, *, limit: snapshots,
    )

    canonical_hash = "a" * 40

    class DummyCommit:
        def __init__(self, hexsha: str) -> None:
            self.hexsha = hexsha

    class DummyRepo:
        def commit(self, ref: str) -> DummyCommit:
            if ref == "abc1234":
                return DummyCommit(canonical_hash)
            raise ValueError("unknown commit")

    ingestion.repo = cast(Any, DummyRepo())

    seen_prefetch_hashes: list[list[str]] = []

    def fake_load_metrics_payload_batch(
        _self,
        commit_hashes: list[str],
        *,
        session: Any,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
        seen_prefetch_hashes.append(list(commit_hashes))
        return {}, {}

    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_load_metrics_payload_batch",
        fake_load_metrics_payload_batch,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ingest_snapshot",
        lambda *_args, **_kwargs: False,
    )

    @contextmanager
    def fake_scope():
        yield object()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    assert ingestion.ingest_completed_jobs() == 0
    assert seen_prefetch_hashes == [[canonical_hash]]


def test_ingest_completed_jobs_records_failed_when_prefetched_metrics_payload_invalid(
    monkeypatch,
    tmp_path,
) -> None:
    ingestion = _make_ingestion(tmp_path)
    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash=None,
        island_id=None,
        result_commit_hash="deadbeef",
        completed_at=None,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_jobs_requiring_ingestion",
        lambda _self, *, limit: [snapshot],
    )

    canonical_hash = "d" * 40
    failure_reason = "Failed to build metrics payload (commit=deadbeef metric='score' reason=boom)."
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_commit_available",
        lambda _self, _commit_hash: canonical_hash,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_load_metrics_payload_batch",
        lambda _self, _commit_hashes, *, session: (
            {canonical_hash: []},
            {canonical_hash: failure_reason},
        ),
    )

    class DummyManager:
        def ingest(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("manager.ingest should not be called when metrics prefetch failed")

    ingestion.manager = DummyManager()  # type: ignore[assignment]

    @contextmanager
    def fake_scope():
        yield object()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    recorded: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_ingestion_state",
        lambda _self, _snapshot, *, status, reason=None, **_kwargs: recorded.append((status, reason)),
    )

    assert ingestion.ingest_completed_jobs() == 0
    assert len(recorded) == 1
    assert recorded[0][0] == "failed"
    assert recorded[0][1] is not None
    assert "Failed to build metrics payload" in cast(str, recorded[0][1])


def test_ingest_completed_jobs_raises_when_metrics_prefetch_session_commit_fails(
    monkeypatch, tmp_path
) -> None:
    ingestion = _make_ingestion(tmp_path)
    snapshots = [
        JobSnapshot(
            job_id=uuid4(),
            base_commit_hash=None,
            island_id=None,
            result_commit_hash="c1",
            completed_at=None,
        ),
        JobSnapshot(
            job_id=uuid4(),
            base_commit_hash=None,
            island_id=None,
            result_commit_hash="c2",
            completed_at=None,
        ),
    ]
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_jobs_requiring_ingestion",
        lambda _self, *, limit: snapshots,
    )

    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_ingestion_state",
        lambda *_args, **_kwargs: None,
    )

    @contextmanager
    def fake_scope():
        yield object()
        raise RuntimeError("commit failed")

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    seen_snapshots: list[JobSnapshot] = []

    def fake_ingest_snapshot(
        _self,
        snapshot: JobSnapshot,
        *,
        snapshot_session: Any | None = None,
    ) -> bool:
        seen_snapshots.append(snapshot)
        return True

    monkeypatch.setattr(ingestion_mod.MapElitesIngestion, "_ingest_snapshot", fake_ingest_snapshot)
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_load_metrics_payload_batch",
        lambda _self, _commit_hashes, *, session: ({}, {}),
    )

    try:
        ingestion.ingest_completed_jobs()
    except RuntimeError as exc:
        assert str(exc) == "commit failed"
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ingestion prefetch failure to propagate")
    assert seen_snapshots == []


def test_ingest_snapshot_records_failed_when_manager_raises(monkeypatch, tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)

    monkeypatch.setattr(ingestion_mod.MapElitesIngestion, "_ensure_commit_available", lambda _self, commit_hash: commit_hash)

    class DummyManager:
        def ingest(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("manager failed")

    ingestion.manager = DummyManager()  # type: ignore[assignment]

    class DummyScalarResult:
        def scalar_one_or_none(self) -> None:
            return None

    class DummyScalars:
        def all(self) -> list[object]:
            return []

    class DummySession:
        def execute(self, _stmt: Any) -> DummyScalarResult:
            return DummyScalarResult()

        def scalars(self, _stmt: Any) -> DummyScalars:
            return DummyScalars()

    @contextmanager
    def fake_scope():
        yield DummySession()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    recorded: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_ingestion_state",
        lambda _self, _snapshot, *, status, reason=None, **_kwargs: recorded.append((status, reason)),
    )

    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash=None,
        island_id=None,
        result_commit_hash="cafebabe",
        completed_at=None,
    )
    assert ingestion._ingest_snapshot(snapshot) is False
    assert recorded == [("failed", "manager failed")]


def test_backoff_computation_skips_recent_failures(tmp_path) -> None:
    settings = Settings.model_validate(
        {
            "mapelites_code_embedding_dimensions": 8,
            "scheduler_poll_interval_seconds": 30.0,
        }
    )
    ingestion = MapElitesIngestion(
        settings=settings,
        console=Console(record=True),
        repo_root=tmp_path,
        repo=cast(Any, object()),
        manager=cast(Any, object()),
    )
    now = datetime.now(timezone.utc)

    class DummyJob:
        ingestion_last_attempt_at: datetime
        ingestion_attempts: int

        def __init__(self, last_attempt: datetime, attempts: int) -> None:
            self.ingestion_last_attempt_at = last_attempt
            self.ingestion_attempts = attempts

    assert ingestion._should_backoff_failed_job(cast(Any, DummyJob(now - timedelta(seconds=10), 1)), now=now) is True
    assert ingestion._should_backoff_failed_job(cast(Any, DummyJob(now - timedelta(seconds=31), 1)), now=now) is False


def test_jobs_requiring_ingestion_skips_backoff_failed_jobs(monkeypatch, tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)

    now = datetime.now(timezone.utc)

    class DummyJob:
        def __init__(
            self,
            *,
            commit: str,
            ingestion_status: str | None,
            attempts: int,
            last_attempt: datetime | None,
        ) -> None:
            self.id = uuid4()
            self.base_commit_hash = None
            self.island_id = None
            self.result_commit_hash = commit
            self.completed_at = now
            self.ingestion_status = ingestion_status
            self.ingestion_attempts = attempts
            self.ingestion_last_attempt_at = last_attempt

    failed_recent = DummyJob(
        commit="failed",
        ingestion_status="failed",
        attempts=1,
        last_attempt=now,
    )
    fresh = DummyJob(
        commit="fresh",
        ingestion_status=None,
        attempts=0,
        last_attempt=None,
    )

    class DummyResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return self._rows

    class DummySession:
        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult([failed_recent, fresh])

    @contextmanager
    def fake_scope():
        yield DummySession()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    snapshots = ingestion._jobs_requiring_ingestion(limit=5)
    assert [snap.result_commit_hash for snap in snapshots] == ["fresh"]


def test_jobs_requiring_ingestion_pages_through_failed_backlog(monkeypatch, tmp_path) -> None:
    """Regression: backoff-filtered failed rows must not hide later retryable jobs."""

    ingestion = _make_ingestion(tmp_path)
    now = datetime.now(timezone.utc)

    class DummyJob:
        def __init__(
            self,
            *,
            commit: str,
            attempts: int,
            last_attempt: datetime | None,
        ) -> None:
            self.id = uuid4()
            self.base_commit_hash = None
            self.island_id = None
            self.result_commit_hash = commit
            self.completed_at = now
            self.ingestion_status = "failed"
            self.ingestion_attempts = attempts
            self.ingestion_last_attempt_at = last_attempt

    blocked = [
        DummyJob(
            commit=f"blocked-{idx}",
            attempts=1,
            last_attempt=now,
        )
        for idx in range(40)
    ]
    retryable = DummyJob(
        commit="retryable",
        attempts=1,
        last_attempt=now - timedelta(seconds=31),
    )

    class DummyResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return self._rows

    class DummySession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt: Any) -> DummyResult:
            self.calls += 1
            if self.calls == 1:
                return DummyResult([])
            if self.calls == 2:
                return DummyResult(blocked[:32])
            if self.calls == 3:
                return DummyResult([*blocked[32:], retryable])
            return DummyResult([])

    @contextmanager
    def fake_scope():
        yield DummySession()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    snapshots = ingestion._jobs_requiring_ingestion(limit=1)

    assert [snap.result_commit_hash for snap in snapshots] == ["retryable"]


def test_jobs_requiring_ingestion_uses_cursor_pagination_not_offset(monkeypatch, tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)
    now = datetime.now(timezone.utc)

    class DummyJob:
        def __init__(self, commit: str) -> None:
            self.id = uuid4()
            self.base_commit_hash = None
            self.island_id = None
            self.result_commit_hash = commit
            self.completed_at = now
            self.ingestion_status = None
            self.ingestion_attempts = 0
            self.ingestion_last_attempt_at = None

    page_one = [DummyJob("c1")]
    page_two = [DummyJob("c2")]

    class DummyResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return self._rows

    class DummySession:
        def __init__(self) -> None:
            self.calls = 0
            self.statements: list[Any] = []

        def execute(self, stmt: Any) -> DummyResult:
            self.statements.append(stmt)
            self.calls += 1
            if self.calls == 1:
                return DummyResult(page_one)
            if self.calls == 2:
                return DummyResult(page_two)
            return DummyResult([])

    session = DummySession()

    @contextmanager
    def fake_scope():
        yield session

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    snapshots = ingestion._jobs_requiring_ingestion(limit=2)

    assert [snap.result_commit_hash for snap in snapshots] == ["c1", "c2"]
    assert all(getattr(stmt, "_offset_clause", None) is None for stmt in session.statements)


def test_record_ingestion_state_clamps_long_reason(monkeypatch, tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)

    job_id = uuid4()
    snapshot = JobSnapshot(
        job_id=job_id,
        base_commit_hash=None,
        island_id=None,
        result_commit_hash="abc123",
        completed_at=None,
    )
    long_reason = "x" * 10000

    class DummyJob:
        def __init__(self) -> None:
            self.ingestion_attempts = 0
            self.ingestion_status = None
            self.ingestion_last_attempt_at = None
            self.ingestion_reason = None
            self.ingestion_delta = None
            self.ingestion_status_code = None
            self.ingestion_message = None
            self.ingestion_cell_index = None

    dummy_job = DummyJob()

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return dummy_job
            return None

    @contextmanager
    def fake_scope():
        yield DummySession()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    ingestion._record_ingestion_state(snapshot, status="failed", reason=long_reason)
    assert dummy_job.ingestion_attempts == 1
    assert dummy_job.ingestion_reason is not None
    assert dummy_job.ingestion_reason.endswith("…")
    assert len(dummy_job.ingestion_reason) == 4096
