from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
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
    assert len(created_sessions) == 2
    assert seen_sessions == [created_sessions[1], created_sessions[1]]


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


def test_ingest_snapshot_reloads_island_after_manager_mutation_fails(
    monkeypatch,
    tmp_path,
) -> None:
    ingestion = _make_ingestion(tmp_path)
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_commit_available",
        lambda _self, commit_hash: commit_hash,
    )
    reloads: list[tuple[str, Any | None]] = []

    class DummyManager:
        def ingest(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("snapshot persist failed")

        def reload_island(
            self,
            island_id: str,
            *,
            snapshot_session: Any | None = None,
        ) -> None:
            reloads.append((island_id, snapshot_session))

    ingestion.manager = DummyManager()  # type: ignore[assignment]
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_ingestion_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_load_metrics_payload_for_commit",
        lambda *_args, **_kwargs: [],
    )
    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash=None,
        island_id="beta",
        result_commit_hash="cafebabe",
        completed_at=None,
    )

    assert ingestion._ingest_snapshot(snapshot, snapshot_session=None) is False
    assert reloads == [("beta", None)]


def test_ingest_snapshot_falls_back_when_canonical_prefetch_payload_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    ingestion = _make_ingestion(tmp_path)
    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash=None,
        island_id="island-a",
        result_commit_hash="cafebabe",
        completed_at=None,
    )
    canonical_hash = "c" * 40
    fallback_payload = [
        {
            "name": "fitness",
            "value": 1.5,
            "unit": "score",
            "higher_is_better": True,
        }
    ]

    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_commit_available",
        lambda _self, _commit_hash: canonical_hash,
    )
    ingestion._prefetched_metrics_payload_by_commit = {"cafebabe": []}
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_load_metrics_payload_for_commit",
        lambda _self, commit_hash: fallback_payload if commit_hash == canonical_hash else [],
    )

    seen_calls: list[dict[str, Any]] = []

    class DummyManager:
        def ingest(self, *args: Any, **kwargs: Any) -> Any:
            seen_calls.append(dict(kwargs))
            return SimpleNamespace(
                record=None,
                delta=0.0,
                status=0,
                message="unchanged",
                inserted=False,
            )

    ingestion.manager = DummyManager()  # type: ignore[assignment]
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_ingestion_state",
        lambda *_args, **_kwargs: None,
    )

    assert ingestion._ingest_snapshot(snapshot) is False
    assert seen_calls == [
        {
            "commit_hash": canonical_hash,
            "metrics": fallback_payload,
            "island_id": "island-a",
            "repo_root": tmp_path,
            "snapshot_session": None,
        }
    ]


def test_ingest_snapshot_records_skipped_state_when_archive_unchanged(
    monkeypatch,
    tmp_path,
) -> None:
    ingestion = _make_ingestion(tmp_path)
    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash=None,
        island_id=None,
        result_commit_hash="cafebabe",
        completed_at=None,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_commit_available",
        lambda _self, commit_hash: commit_hash,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_load_metrics_payload_for_commit",
        lambda _self, _commit_hash: [],
    )

    class DummyManager:
        def ingest(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(
                record=None,
                delta=-0.25,
                status=0,
                message="Commit not inserted; objective below cell threshold.",
                inserted=False,
            )

    ingestion.manager = DummyManager()  # type: ignore[assignment]

    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_ingestion_state",
        lambda _self, _snapshot, **kwargs: recorded.append(dict(kwargs)),
    )

    assert ingestion._ingest_snapshot(snapshot) is False
    assert recorded == [
        {
            "status": "skipped",
            "delta": -0.25,
            "status_code": 0,
            "message": "Commit not inserted; objective below cell threshold.",
            "record": None,
        }
    ]


def test_ingest_snapshot_skips_evaluator_equivalent_ingested_candidate(
    monkeypatch,
    tmp_path,
) -> None:
    ingestion = _make_ingestion(tmp_path)
    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash=None,
        island_id="island-a",
        result_commit_hash="new-commit",
        completed_at=None,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_commit_available",
        lambda _self, commit_hash: commit_hash,
    )

    class _ScalarResult:
        def __init__(self, value: Any) -> None:
            self.value = value

        def scalar_one_or_none(self) -> Any:
            return self.value

    class _Session:
        def __init__(self) -> None:
            self.values = iter(
                [
                    SimpleNamespace(evaluation_identity_key="identity-key"),
                    "existing-commit",
                ]
            )

        def execute(self, _stmt: Any) -> _ScalarResult:
            return _ScalarResult(next(self.values))

    class _Manager:
        def ingest(self, **_kwargs: Any) -> Any:
            raise AssertionError("equivalent candidates must not reach MAP-Elites")

    ingestion.manager = _Manager()  # type: ignore[assignment]
    recorded: list[Any] = []
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_successful_ingestion",
        lambda _self, _snapshot, *, insertion, session: recorded.append(
            (insertion, session)
        ),
    )
    session = _Session()

    assert ingestion._ingest_snapshot(snapshot, snapshot_session=session) is False
    insertion, observed_session = recorded[0]
    assert observed_session is session
    assert insertion.status == 0
    assert insertion.record is None
    assert "existing-commit" in insertion.message


def test_equivalent_candidate_check_ignores_unprocessed_peer(tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)
    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash=None,
        island_id="island-a",
        result_commit_hash="new-commit",
        completed_at=None,
    )

    class _ScalarResult:
        def __init__(self, value: Any) -> None:
            self.value = value

        def scalar_one_or_none(self) -> Any:
            return self.value

    class _Session:
        def __init__(self) -> None:
            self.values = iter(
                [
                    SimpleNamespace(evaluation_identity_key="identity-key"),
                    None,
                ]
            )

        def execute(self, _stmt: Any) -> _ScalarResult:
            return _ScalarResult(next(self.values))

    assert (
        ingestion._equivalent_ingested_candidate(
            snapshot,
            commit_hash="new-commit",
            session=_Session(),  # type: ignore[arg-type]
        )
        is None
    )


def test_ingest_snapshot_wraps_batch_session_work_in_savepoint(monkeypatch, tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)
    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash=None,
        island_id=None,
        result_commit_hash="cafebabe",
        completed_at=None,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_ensure_commit_available",
        lambda _self, commit_hash: commit_hash,
    )
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_load_metrics_payload_for_commit",
        lambda _self, _commit_hash: [],
    )

    class DummyNested:
        def __init__(self, owner: "DummySession") -> None:
            self.owner = owner

        def __enter__(self) -> "DummyNested":
            self.owner.begin_nested_calls += 1
            self.owner._in_nested = True
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            self.owner._in_nested = False
            return False

    class DummySession:
        def __init__(self) -> None:
            self.begin_nested_calls = 0
            self._in_nested = False

        def begin_nested(self) -> DummyNested:
            return DummyNested(self)

        def in_nested_transaction(self) -> bool:
            return self._in_nested

    batch_session = DummySession()

    class DummyManager:
        def ingest(self, *args: Any, **kwargs: Any) -> Any:
            assert kwargs.get("snapshot_session") is batch_session
            return cast(
                Any,
                type(
                    "DummyInsertion",
                    (),
                    {
                        "record": None,
                        "delta": 0.0,
                        "status": 0,
                        "message": "unchanged",
                        "inserted": False,
                    },
                )(),
            )

    ingestion.manager = DummyManager()  # type: ignore[assignment]

    recorded_sessions: list[Any] = []
    monkeypatch.setattr(
        ingestion_mod.MapElitesIngestion,
        "_record_ingestion_state",
        lambda _self, _snapshot, *, session=None, **_kwargs: recorded_sessions.append(session),
    )

    assert ingestion._ingest_snapshot(snapshot, snapshot_session=batch_session) is False
    assert batch_session.begin_nested_calls == 1
    assert recorded_sessions == [batch_session]


def test_record_ingestion_state_reuses_provided_session(monkeypatch, tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)

    job_id = uuid4()
    snapshot = JobSnapshot(
        job_id=job_id,
        base_commit_hash=None,
        island_id=None,
        result_commit_hash="abc123",
        completed_at=None,
    )

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

    class DummyNested:
        def __init__(self, owner: "DummySession") -> None:
            self.owner = owner

        def __enter__(self) -> "DummyNested":
            self.owner.begin_nested_calls += 1
            self.owner._in_nested = True
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            self.owner._in_nested = False
            return False

    class DummySession:
        def __init__(self) -> None:
            self.begin_nested_calls = 0
            self._in_nested = False

        def begin_nested(self) -> DummyNested:
            return DummyNested(self)

        def in_nested_transaction(self) -> bool:
            return self._in_nested

        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return dummy_job
            return None

    session = DummySession()

    @contextmanager
    def fake_scope():
        raise AssertionError("session_scope() should not be used when a session is provided")
        yield session

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    ingestion._record_ingestion_state(
        snapshot,
        status="failed",
        reason="boom",
        session=session,
    )

    assert session.begin_nested_calls == 1
    assert dummy_job.ingestion_attempts == 1
    assert dummy_job.ingestion_status == "failed"
    assert dummy_job.ingestion_reason == "boom"


def test_record_ingestion_state_records_result_payload(monkeypatch, tmp_path) -> None:
    ingestion = _make_ingestion(tmp_path)

    job_id = uuid4()
    snapshot = JobSnapshot(
        job_id=job_id,
        base_commit_hash=None,
        island_id=None,
        result_commit_hash="abc123",
        completed_at=None,
    )

    class DummyJob:
        def __init__(self) -> None:
            self.ingestion_attempts = 2
            self.ingestion_status = None
            self.ingestion_last_attempt_at = None
            self.ingestion_reason = "old reason"
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

    ingestion._record_ingestion_state(
        snapshot,
        status="succeeded",
        reason="updated",
        delta=1.25,
        status_code=3,
        message="Inserted\ninto archive",
        record=SimpleNamespace(cell_index="7"),
    )

    assert dummy_job.ingestion_attempts == 3
    assert dummy_job.ingestion_status == "succeeded"
    assert dummy_job.ingestion_last_attempt_at is not None
    assert dummy_job.ingestion_reason == "updated"
    assert dummy_job.ingestion_delta == 1.25
    assert dummy_job.ingestion_status_code == 3
    assert dummy_job.ingestion_message == "Inserted into archive"
    assert dummy_job.ingestion_cell_index == 7


def test_failed_ingestion_becomes_terminal_after_retry_budget(
    monkeypatch,
    tmp_path,
) -> None:
    ingestion = _make_ingestion(tmp_path)
    job_id = uuid4()
    snapshot = JobSnapshot(
        job_id=job_id,
        base_commit_hash=None,
        island_id="alpha",
        result_commit_hash="abc123",
        completed_at=None,
    )
    dummy_job = SimpleNamespace(
        ingestion_attempts=10_000,
        ingestion_status="failed",
        ingestion_last_attempt_at=None,
        ingestion_reason="old reason",
        ingestion_delta=None,
        ingestion_status_code=None,
        ingestion_message=None,
        ingestion_cell_index=None,
        result_commit_hash="abc123",
    )

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            return dummy_job if key == job_id else None

    @contextmanager
    def fake_scope():
        yield DummySession()

    monkeypatch.setattr(ingestion_mod, "session_scope", fake_scope)

    ingestion._record_ingestion_state(
        snapshot,
        status="failed",
        reason="commit remains unavailable",
    )

    assert dummy_job.ingestion_status == "skipped"
    assert "retry limit reached" in dummy_job.ingestion_reason.lower()
    assert "Ingestion retry limit reached" in ingestion.console.export_text()


class _ScalarOneOrNone:
    def __init__(self, value: object | None) -> None:
        self.value = value

    def scalar_one_or_none(self) -> object | None:
        return self.value


def test_skipped_ingestion_does_not_downgrade_existing_archive_member() -> None:
    """Regression: skipped duplicate ingests must not mark archive members rejected."""
    archived_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candidate = SimpleNamespace(archive_status="member", archived_at=archived_at)

    class DummySession:
        def __init__(self) -> None:
            self.execute_calls = 0

        def execute(self, _stmt: Any) -> _ScalarOneOrNone:
            self.execute_calls += 1
            return _ScalarOneOrNone(candidate)

    payload = ingestion_mod._IngestionStatePayload(
        status="skipped",
        reason=None,
        delta=None,
        status_code=0,
        message=None,
        record=None,
    )
    session = DummySession()

    ingestion_mod.MapElitesIngestion._apply_candidate_archive_state(
        session=cast(Any, session),
        commit_hash="abc123",
        payload=payload,
    )

    assert candidate.archive_status == "member"
    assert candidate.archived_at is archived_at
    assert session.execute_calls == 1


def test_skipped_ingestion_reconciles_archive_cell_membership() -> None:
    candidate = SimpleNamespace(archive_status="not_considered", archived_at=None)

    class DummySession:
        def __init__(self) -> None:
            self.execute_calls = 0

        def execute(self, _stmt: Any) -> _ScalarOneOrNone:
            self.execute_calls += 1
            if self.execute_calls == 1:
                return _ScalarOneOrNone(candidate)
            return _ScalarOneOrNone(1)

    payload = ingestion_mod._IngestionStatePayload(
        status="skipped",
        reason=None,
        delta=None,
        status_code=0,
        message=None,
        record=None,
    )
    session = DummySession()

    ingestion_mod.MapElitesIngestion._apply_candidate_archive_state(
        session=cast(Any, session),
        commit_hash="abc123",
        payload=payload,
    )

    assert candidate.archive_status == "member"
    assert candidate.archived_at is not None
    assert session.execute_calls == 2


def test_skipped_ingestion_marks_non_member_candidate_rejected() -> None:
    candidate = SimpleNamespace(archive_status="not_considered", archived_at=None)

    class DummySession:
        def __init__(self) -> None:
            self.execute_calls = 0

        def execute(self, _stmt: Any) -> _ScalarOneOrNone:
            self.execute_calls += 1
            if self.execute_calls == 1:
                return _ScalarOneOrNone(candidate)
            return _ScalarOneOrNone(None)

    payload = ingestion_mod._IngestionStatePayload(
        status="skipped",
        reason=None,
        delta=None,
        status_code=0,
        message=None,
        record=None,
    )
    session = DummySession()

    ingestion_mod.MapElitesIngestion._apply_candidate_archive_state(
        session=cast(Any, session),
        commit_hash="abc123",
        payload=payload,
    )

    assert candidate.archive_status == "rejected"
    assert candidate.archived_at is None
    assert session.execute_calls == 2


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
