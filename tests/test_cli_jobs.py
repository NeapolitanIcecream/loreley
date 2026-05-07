from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

import loreley.api.services.candidate_fates as candidate_fate_service
from loreley.cli import _job_detail_payload, main
from loreley.core.candidate_fate import CandidateFate
from loreley.db.models import JobStatus
from tests.support import TestSettings


def _make_settings() -> TestSettings:
    return TestSettings()


def _patch_cli_db_now(
    monkeypatch: pytest.MonkeyPatch,
    value: datetime | None = None,
) -> datetime:
    db_now = value or datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("loreley.cli._db_utc_now", lambda _session: db_now)
    return db_now


def test_jobs_retry_requeues_failed_job_and_resets_lease_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    now = _patch_cli_db_now(monkeypatch)

    job_id = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.FAILED,
        scheduled_at=None,
        started_at=object(),
        completed_at=object(),
        heartbeat_at=object(),
        lease_expires_at=object(),
        run_token=uuid.uuid4(),
        worker_id="worker-01",
        recovery_count=4,
        result_commit_hash="deadbeef",
        last_error="Lease expired after missing heartbeat",
    )

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return job
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "retry", str(job_id), "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["job_id"] == str(job_id)
    assert payload["previous_status"] == "failed"
    assert payload["new_status"] == "pending"
    assert payload["recovery_count_reset_from"] == 4
    assert job.status is JobStatus.PENDING
    assert job.scheduled_at == now
    assert job.started_at is None
    assert job.completed_at is None
    assert job.heartbeat_at is None
    assert job.lease_expires_at is None
    assert job.run_token is None
    assert job.worker_id is None
    assert job.recovery_count == 0
    assert job.result_commit_hash is None


def test_jobs_retry_clears_candidate_metadata_from_previous_attempt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: retried jobs exposed stale candidate refs before the new attempt."""

    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    published_at = datetime(2026, 3, 25, 7, 30, tzinfo=timezone.utc)
    _patch_cli_db_now(monkeypatch)

    job_id = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.FAILED,
        scheduled_at=None,
        started_at=object(),
        completed_at=object(),
        heartbeat_at=object(),
        lease_expires_at=object(),
        run_token=uuid.uuid4(),
        worker_id="worker-01",
        recovery_count=1,
        candidate_commit_hash="oldcandidate",
        candidate_branch_name="exp/job-old",
        candidate_published_at=published_at,
        result_commit_hash=None,
        last_error="published candidate failed evaluation",
    )

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return job
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "retry", str(job_id), "--json"])
    capsys.readouterr()

    assert code == 0
    assert job.status is JobStatus.PENDING
    assert job.candidate_commit_hash is None
    assert job.candidate_branch_name is None
    assert job.candidate_published_at is None


def test_jobs_retry_rejects_non_failed_jobs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    now = _patch_cli_db_now(monkeypatch)

    job_id = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.RUNNING,
        heartbeat_at=now,
        lease_expires_at=now + timedelta(minutes=10),
        run_token=uuid.uuid4(),
        worker_id="worker-01",
        recovery_count=1,
        last_error=None,
    )

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return job
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "retry", str(job_id)])
    captured = capsys.readouterr()

    assert code == 1
    assert "only failed or stuck running jobs can be retried" in captured.out.lower()
    assert job.status is JobStatus.RUNNING


def test_jobs_retry_requeues_running_job_with_missing_lease_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    now = _patch_cli_db_now(monkeypatch)

    job_id = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.RUNNING,
        scheduled_at=None,
        started_at=object(),
        completed_at=None,
        heartbeat_at=None,
        lease_expires_at=None,
        run_token=None,
        worker_id=None,
        recovery_count=1,
        result_commit_hash="deadbeef",
        last_error="missing lease metadata",
    )

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return job
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "retry", str(job_id), "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["job_id"] == str(job_id)
    assert payload["previous_status"] == "running"
    assert payload["new_status"] == "pending"
    assert job.status is JobStatus.PENDING
    assert job.scheduled_at == now
    assert job.run_token is None
    assert job.worker_id is None
    assert job.result_commit_hash is None


def test_jobs_retry_uses_database_time_to_keep_active_running_job_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)

    db_now = _patch_cli_db_now(monkeypatch, datetime.now(timezone.utc) - timedelta(minutes=5))
    job_id = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.RUNNING,
        heartbeat_at=db_now - timedelta(seconds=30),
        lease_expires_at=db_now + timedelta(minutes=1),
        run_token=uuid.uuid4(),
        worker_id="worker-01",
        recovery_count=1,
        last_error=None,
    )

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return job
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "retry", str(job_id)])
    captured = capsys.readouterr()

    assert code == 1
    assert "only failed or stuck running jobs can be retried" in captured.out.lower()
    assert "lease_state=active" in captured.out.lower()
    assert job.status is JobStatus.RUNNING


def test_jobs_inspect_json_reports_stale_lease_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    now = _patch_cli_db_now(monkeypatch)

    job_id = uuid.uuid4()
    run_token = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.RUNNING,
        base_commit_hash="abc123",
        island_id="main",
        created_at=now - timedelta(hours=2),
        scheduled_at=now - timedelta(hours=1, minutes=55),
        started_at=now - timedelta(hours=1, minutes=50),
        heartbeat_at=now - timedelta(minutes=40),
        lease_expires_at=now - timedelta(minutes=10),
        run_token=run_token,
        worker_id="worker-02",
        recovery_count=2,
        result_commit_hash=None,
        last_error="still running",
    )

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return job
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "inspect", str(job_id), "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["job_id"] == str(job_id)
    assert payload["status"] == "running"
    assert payload["lease"]["state"] == "stale"
    assert payload["lease"]["worker_id"] == "worker-02"
    assert payload["lease"]["run_token"] == str(run_token)
    assert payload["recovery_count"] == 2


def test_jobs_inspect_json_uses_contextual_fate_for_repair_eligible_candidate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: CLI inspect ignored CandidateCommit repair state and reported unknown."""

    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    now = _patch_cli_db_now(monkeypatch)

    job_id = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.FAILED,
        base_commit_hash="base-a",
        island_id="main",
        created_at=now - timedelta(hours=2),
        scheduled_at=now - timedelta(hours=1, minutes=55),
        started_at=now - timedelta(hours=1, minutes=50),
        completed_at=now,
        heartbeat_at=None,
        lease_expires_at=None,
        run_token=None,
        worker_id=None,
        recovery_count=1,
        candidate_commit_hash="candidate-a",
        result_commit_hash=None,
        last_error="candidate failed evaluation",
    )
    candidate = SimpleNamespace(
        commit_hash="candidate-a",
        produced_by_job_id=job_id,
        evaluation_status="candidate_failed",
        repair_state="eligible",
        failure_stage="evaluation",
        failure_kind="regression",
    )

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return job
            return None

    class CandidateFateExecuteResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return list(self._rows)

        def all(self) -> list[object]:
            return list(self._rows)

    class CandidateFateSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt: Any) -> CandidateFateExecuteResult:
            self.calls += 1
            if self.calls == 1:
                return CandidateFateExecuteResult([candidate])
            if self.calls == 2:
                return CandidateFateExecuteResult([job])
            if self.calls == 3:
                return CandidateFateExecuteResult([])
            raise AssertionError("unexpected candidate fate query")

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    @contextmanager
    def fake_candidate_fate_scope() -> Any:
        yield CandidateFateSession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)
    monkeypatch.setattr(candidate_fate_service, "session_scope", fake_candidate_fate_scope)

    code = main(["jobs", "inspect", str(job_id), "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["candidate_fate_label"] == "repair_pending"
    assert (
        payload["candidate_fate_reason"]
        == "Candidate repair_state=eligible. Failure stage=evaluation kind=regression."
    )


def test_job_detail_payload_includes_summary_and_nested_lease_state() -> None:
    now = datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc)
    run_token = uuid.uuid4()
    job = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.RUNNING,
        base_commit_hash="abc123",
        island_id="main",
        created_at=now - timedelta(hours=2),
        scheduled_at=now - timedelta(hours=1, minutes=55),
        started_at=now - timedelta(hours=1, minutes=50),
        completed_at=None,
        heartbeat_at=now - timedelta(minutes=40),
        lease_expires_at=now - timedelta(minutes=10),
        run_token=run_token,
        worker_id="worker-02",
        recovery_count=2,
        result_commit_hash=None,
        last_error="still running",
    )

    payload = _job_detail_payload(job=job, now=now)

    assert payload["status"] == "running"
    assert payload["lease_state"] == "stale"
    assert payload["lease"]["state"] == "stale"
    assert payload["lease"]["run_token"] == str(run_token)
    assert payload["scheduled_at"] == (now - timedelta(hours=1, minutes=55)).isoformat()


def test_job_detail_payload_includes_candidate_fate() -> None:
    now = datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc)
    job = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.SUCCEEDED,
        base_commit_hash="abc123",
        island_id="main",
        created_at=now - timedelta(hours=2),
        scheduled_at=now - timedelta(hours=1, minutes=55),
        started_at=now - timedelta(hours=1, minutes=50),
        completed_at=now,
        heartbeat_at=None,
        lease_expires_at=None,
        run_token=None,
        worker_id=None,
        recovery_count=0,
        result_commit_hash="def456",
        last_error=None,
    )

    payload = _job_detail_payload(
        job=job,
        now=now,
        candidate_fate=CandidateFate(
            label="elite_replaced",
            reason="Candidate improved an occupied archive niche.",
        ),
    )

    assert payload["candidate_fate_label"] == "elite_replaced"
    assert payload["candidate_fate_reason"] == "Candidate improved an occupied archive niche."


def test_jobs_inspect_table_preserves_zero_recovery_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    now = _patch_cli_db_now(monkeypatch)

    job_id = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.RUNNING,
        base_commit_hash="abc123",
        island_id="main",
        created_at=now - timedelta(hours=2),
        scheduled_at=now - timedelta(hours=1, minutes=55),
        started_at=now - timedelta(hours=1, minutes=50),
        heartbeat_at=now - timedelta(minutes=1),
        lease_expires_at=now + timedelta(minutes=10),
        run_token=uuid.uuid4(),
        worker_id="worker-02",
        recovery_count=0,
        result_commit_hash=None,
        last_error=None,
    )

    class DummySession:
        def get(self, _model: Any, key: Any) -> Any:
            if key == job_id:
                return job
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "inspect", str(job_id)])
    captured = capsys.readouterr()

    assert code == 0
    assert any("recovery_count" in line and "0" in line for line in captured.out.splitlines())


def test_jobs_inspect_rejects_missing_job(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)

    class DummySession:
        def get(self, _model: Any, _key: Any) -> Any:
            return None

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "inspect", str(uuid.uuid4())])
    captured = capsys.readouterr()

    assert code == 1
    assert "job not found" in captured.out.lower()


def test_jobs_ls_failed_stale_json_filters_to_recovery_exhausted_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    settings.scheduler_stale_running_max_recovery_attempts = 3
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)

    now = datetime.now(timezone.utc)
    matching = SimpleNamespace(
        id=uuid.uuid4(),
        status=JobStatus.FAILED,
        base_commit_hash="base-a",
        island_id="main",
        recovery_count=4,
        last_error="Lease expired after missing heartbeat; recovered by scheduler (attempt=4).",
        completed_at=now,
        created_at=now - timedelta(hours=1),
    )

    class DummyExecuteResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return list(self._rows)

    class DummySession:
        def execute(self, _stmt: Any) -> DummyExecuteResult:
            return DummyExecuteResult([matching])

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "ls", "--failed-stale", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["filters"] == {"failed_stale": True}
    assert len(payload["jobs"]) == 1
    assert payload["jobs"][0]["job_id"] == str(matching.id)
    assert payload["jobs"][0]["status"] == "failed"
    assert payload["jobs"][0]["recovery_count"] == 4


def test_jobs_ls_failed_stale_filter_includes_missing_lease_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    settings.scheduler_stale_running_max_recovery_attempts = 3
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)

    seen_sql: list[str] = []

    class DummyExecuteResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return list(self._rows)

    class DummySession:
        def execute(self, stmt: Any) -> DummyExecuteResult:
            compiled = stmt.compile(compile_kwargs={"literal_binds": True})
            seen_sql.append(str(compiled).lower())
            return DummyExecuteResult([])

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "ls", "--failed-stale", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    assert json.loads(captured.out)["jobs"] == []
    assert any("lease expired after missing heartbeat;%" in sql for sql in seen_sql)
    assert any("lease metadata missing for running job;%" in sql for sql in seen_sql)


def test_jobs_retry_failed_stale_limit_requeues_multiple_jobs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    settings.scheduler_stale_running_max_recovery_attempts = 3
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    now = _patch_cli_db_now(monkeypatch)

    jobs = [
        SimpleNamespace(
            id=uuid.uuid4(),
            status=JobStatus.FAILED,
            scheduled_at=None,
            started_at=object(),
            completed_at=object(),
            heartbeat_at=object(),
            lease_expires_at=object(),
            run_token=uuid.uuid4(),
            worker_id="worker-a",
            recovery_count=4,
            result_commit_hash="c1",
            last_error="Lease expired after missing heartbeat; recovered by scheduler (attempt=4).",
        ),
        SimpleNamespace(
            id=uuid.uuid4(),
            status=JobStatus.FAILED,
            scheduled_at=None,
            started_at=object(),
            completed_at=object(),
            heartbeat_at=object(),
            lease_expires_at=object(),
            run_token=uuid.uuid4(),
            worker_id="worker-b",
            recovery_count=5,
            result_commit_hash="c2",
            last_error="Lease expired after missing heartbeat; recovered by scheduler (attempt=5).",
        ),
    ]

    class DummyExecuteResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return list(self._rows)

    class DummySession:
        def execute(self, _stmt: Any) -> DummyExecuteResult:
            return DummyExecuteResult(jobs)

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "retry", "--failed-stale", "--limit", "2", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["filters"] == {"all": False, "failed_stale": True, "limit": 2}
    assert payload["count"] == 2
    assert [item["job_id"] for item in payload["retried_jobs"]] == [str(job.id) for job in jobs]
    for job in jobs:
        assert job.status is JobStatus.PENDING
        assert job.scheduled_at == now
        assert job.recovery_count == 0
        assert job.run_token is None
        assert job.worker_id is None
        assert job.result_commit_hash is None


def test_jobs_retry_failed_stale_filter_includes_missing_lease_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    settings.scheduler_stale_running_max_recovery_attempts = 3
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)

    seen_sql: list[str] = []

    class DummyExecuteResult:
        def __init__(self, rows: list[object]) -> None:
            self._rows = rows

        def scalars(self) -> list[object]:
            return list(self._rows)

    class DummySession:
        def execute(self, stmt: Any) -> DummyExecuteResult:
            compiled = stmt.compile(compile_kwargs={"literal_binds": True})
            seen_sql.append(str(compiled).lower())
            return DummyExecuteResult([])

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["jobs", "retry", "--failed-stale", "--limit", "1", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    assert json.loads(captured.out)["count"] == 0
    assert any("lease expired after missing heartbeat;%" in sql for sql in seen_sql)
    assert any("lease metadata missing for running job;%" in sql for sql in seen_sql)


def test_jobs_retry_failed_stale_requires_all_or_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)

    code = main(["jobs", "retry", "--failed-stale"])
    captured = capsys.readouterr()

    assert code == 1
    assert "use --all or --limit with --failed-stale" in captured.out.lower()
