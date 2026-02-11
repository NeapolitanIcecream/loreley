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


def test_ingest_completed_jobs_reuses_single_snapshot_transaction(monkeypatch, tmp_path) -> None:
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

    assert ingestion.ingest_completed_jobs() == 2
    assert len(created_sessions) == 1
    assert seen_sessions == [created_sessions[0], created_sessions[0]]


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

