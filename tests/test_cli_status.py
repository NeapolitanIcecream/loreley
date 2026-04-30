from __future__ import annotations

from datetime import datetime, timezone
import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from loreley.cli import _status_response_payload, main
from tests.support import TestSettings


def _make_settings() -> TestSettings:
    return TestSettings(
        MAPELITES_FITNESS_METRIC="",
        WORKER_JOB_LEASE_TTL_SECONDS=1800,
        WORKER_JOB_HEARTBEAT_INTERVAL_SECONDS=60,
        SCHEDULER_STALE_RUNNING_MAX_RECOVERY_ATTEMPTS=3,
    )


def _patch_cli_db_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "loreley.cli._db_utc_now",
        lambda _session: datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc),
    )


def test_status_json_includes_job_lease_health(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)
    monkeypatch.setattr(
        "loreley.cli._load_archive_stats_or_exit",
        lambda **_kwargs: {"island_id": "main", "occupied": 1, "cells": 4, "coverage": 0.25},
    )

    instance = SimpleNamespace(
        experiment_id_raw="exp-demo",
        experiment_uuid="11111111-1111-1111-1111-111111111111",
        root_commit_hash="abcdef1234567890",
        repository_slug="demo/repo",
        repository_canonical_origin="https://example.com/demo/repo.git",
    )
    counts = iter([7, 2, 3, 1, 0, 2])

    class DummyScalarResult:
        def __init__(self, value: Any) -> None:
            self._value = value

        def scalar_one(self) -> Any:
            return self._value

    class DummySession:
        def get(self, _model: Any, _key: Any) -> Any:
            return instance

        def execute(self, _stmt: Any) -> DummyScalarResult:
            return DummyScalarResult(next(counts))

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["status", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["jobs"] == {"unfinished": 7, "pending_ingestion": 2}
    assert payload["job_leases"] == {
        "heartbeat_interval_seconds": 60,
        "lease_ttl_seconds": 1800,
        "max_recovery_attempts": 3,
        "recovery_exhausted_failed": 2,
        "running": 3,
        "running_without_lease": 0,
        "stale_running": 1,
    }


def test_status_response_payload_preserves_nested_sections() -> None:
    payload = _status_response_payload(
        instance_payload={"experiment_id_raw": "exp-demo"},
        jobs_payload={"unfinished": 2, "pending_ingestion": 1},
        lease_payload={"running": 1, "stale_running": 0},
        archive_stats={"island_id": "main"},
        best_commit={"commit_hash": "abc123"},
    )

    assert payload == {
        "instance": {"experiment_id_raw": "exp-demo"},
        "jobs": {"unfinished": 2, "pending_ingestion": 1},
        "job_leases": {"running": 1, "stale_running": 0},
        "archive": {"island_id": "main"},
        "best_commit": {"commit_hash": "abc123"},
    }


def test_status_table_prints_job_lease_health_section(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)
    monkeypatch.setattr(
        "loreley.cli._load_archive_stats_or_exit",
        lambda **_kwargs: {"island_id": "main", "occupied": 1, "cells": 4, "coverage": 0.25},
    )

    instance = SimpleNamespace(
        experiment_id_raw="exp-demo",
        experiment_uuid="11111111-1111-1111-1111-111111111111",
        root_commit_hash="abcdef1234567890",
        repository_slug=None,
        repository_canonical_origin=None,
    )
    counts = iter([4, 0, 2, 1, 1, 1])

    class DummyScalarResult:
        def __init__(self, value: Any) -> None:
            self._value = value

        def scalar_one(self) -> Any:
            return self._value

    class DummySession:
        def get(self, _model: Any, _key: Any) -> Any:
            return instance

        def execute(self, _stmt: Any) -> DummyScalarResult:
            return DummyScalarResult(next(counts))

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)

    code = main(["status"])
    captured = capsys.readouterr()

    assert code == 0
    assert "running_jobs" in captured.out
    assert "stale_running" in captured.out
    assert "running_without_lease" in captured.out
    assert "recovery_exhausted_failed" in captured.out
