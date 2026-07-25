from __future__ import annotations

from datetime import datetime, timezone
import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from loreley.cli import _load_best_commit_status_payload, _status_response_payload, main
from loreley.core.campaign_program import parse_campaign_program
from loreley.core.map_elites.objectives import ObjectiveSpec
from loreley.db.models import CommitCard, MapElitesArchiveCell, Metric
from tests.support import TestSettings


def _make_settings() -> TestSettings:
    return TestSettings(
        WORKER_JOB_LEASE_TTL_SECONDS=1800,
        WORKER_JOB_HEARTBEAT_INTERVAL_SECONDS=60,
        SCHEDULER_STALE_RUNNING_MAX_RECOVERY_ATTEMPTS=3,
    )


def _patch_cli_db_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "loreley.cli._db_utc_now",
        lambda _session: datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc),
    )


def _patch_no_baseline_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "loreley.scheduler.baselines.resolve_status_campaign_program_hash",
        lambda **_kwargs: SimpleNamespace(known=False),
    )


def test_status_json_includes_job_lease_health(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)
    _patch_no_baseline_resolution(monkeypatch)
    monkeypatch.setattr("loreley.cli._load_best_commit_status_payload", lambda **_kwargs: None)
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
        "baseline": None,
    }


def test_best_commit_status_query_ignores_retired_islands() -> None:
    settings = _make_settings()
    settings.mapelites_islands = ("alpha", "beta")
    settings.mapelites_objectives = (ObjectiveSpec(name="score", direction="max"),)
    statements: list[Any] = []

    class DummyResult:
        def first(self) -> None:
            return None

    class DummySession:
        def execute(self, stmt: Any) -> DummyResult:
            statements.append(stmt)
            return DummyResult()

    payload = _load_best_commit_status_payload(
        session=DummySession(),
        settings=settings,
        instance=SimpleNamespace(root_commit_hash="root"),
        CommitCard=CommitCard,
        MapElitesArchiveCell=MapElitesArchiveCell,
        Metric=Metric,
    )

    assert payload is None
    assert len(statements) == 1
    statement = statements[0]
    parameter_values = statement.compile().params.values()
    assert ["alpha", "beta"] in parameter_values
    selected_island = list(statement.selected_columns)[2]
    assert selected_island.table is MapElitesArchiveCell.__table__


def test_status_table_prints_job_lease_health_section(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)
    _patch_no_baseline_resolution(monkeypatch)
    monkeypatch.setattr("loreley.cli._load_best_commit_status_payload", lambda **_kwargs: None)
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


def test_status_json_scopes_baseline_to_current_campaign_program(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """Regression: status must not report a baseline from a different campaign program."""

    raw_program = b"## Goal\nImprove program A.\n"
    (tmp_path / "loreley.program.md").write_bytes(raw_program)
    expected_program_hash = parse_campaign_program(
        raw_program,
        source_path="loreley.program.md",
    ).raw_sha256
    settings = _make_settings()
    settings.mapelites_experiment_root_commit = "root123"
    settings.mapelites_objectives = (ObjectiveSpec(name="score", direction="max"),)
    settings.scheduler_repo_root = str(tmp_path)
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
        root_commit_hash="root123",
        repository_slug="demo/repo",
        repository_canonical_origin="https://example.com/demo/repo.git",
    )
    counts = iter([7, 2, 3, 1, 0, 2])
    baseline_calls: list[str | None] = []

    class DummyResult:
        def __init__(self, *, value: Any = None, row: Any = None) -> None:
            self._value = value
            self._row = row

        def scalar_one(self) -> Any:
            return self._value

        def first(self) -> Any:
            return self._row

    class DummySession:
        def get(self, _model: Any, _key: Any) -> Any:
            return instance

        def execute(self, _stmt: Any) -> DummyResult:
            try:
                return DummyResult(value=next(counts))
            except StopIteration:
                return DummyResult(row=None)

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    def fake_load_latest_matching_baseline(**kwargs: Any) -> Any:
        campaign_program_hash = kwargs.get("campaign_program_hash")
        baseline_calls.append(campaign_program_hash)
        return SimpleNamespace(
            id="aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa",
            baseline_key_hash="a" * 64,
            root_commit_hash="root123",
            primary_metric_name="score",
            metric_value=1.0,
            primary_metric_higher_is_better=True,
            status="valid",
            campaign_program_hash=campaign_program_hash,
            failure_kind=None,
            failure_summary=None,
        )

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)
    monkeypatch.setattr(
        "loreley.scheduler.baselines.load_latest_matching_baseline",
        fake_load_latest_matching_baseline,
    )

    code = main(["status", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert baseline_calls == [expected_program_hash]
    assert payload["baseline"]["baseline_campaign_program_hash"] == expected_program_hash


def test_status_json_uses_persisted_scheduler_campaign_program(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression: locked schedulers can run against a hash that differs from disk."""

    active_program_hash = "a" * 64
    disk_program_hash = "b" * 64
    settings = _make_settings()
    settings.mapelites_experiment_root_commit = "root123"
    settings.mapelites_objectives = (ObjectiveSpec(name="score", direction="max"),)
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    monkeypatch.setattr(
        "loreley.cli._load_archive_stats_or_exit",
        lambda **_kwargs: {"island_id": "main", "occupied": 1, "cells": 4, "coverage": 0.25},
    )
    monkeypatch.setattr(
        "loreley.cli._load_status_job_payloads",
        lambda **_kwargs: (
            {"unfinished": 1, "pending_ingestion": 0},
            {
                "heartbeat_interval_seconds": 60,
                "lease_ttl_seconds": 1800,
                "max_recovery_attempts": 3,
                "recovery_exhausted_failed": 0,
                "running": 0,
                "running_without_lease": 0,
                "stale_running": 0,
            },
        ),
    )
    monkeypatch.setattr("loreley.cli._load_best_commit_status_payload", lambda **_kwargs: None)
    _patch_cli_db_now(monkeypatch)

    instance = SimpleNamespace(
        experiment_id_raw="exp-demo",
        experiment_uuid="11111111-1111-1111-1111-111111111111",
        root_commit_hash="root123",
        repository_slug="demo/repo",
        repository_canonical_origin="https://example.com/demo/repo.git",
    )
    persisted_rows = iter(
        [
            (
                active_program_hash,
                datetime(2026, 3, 25, 7, 59, tzinfo=timezone.utc),
                datetime(2026, 3, 25, 7, 58, tzinfo=timezone.utc),
                datetime(2026, 3, 25, 7, 58, tzinfo=timezone.utc),
            ),
            None,
        ]
    )
    baseline_calls: list[str | None] = []
    fallback_calls: list[object] = []

    class DummyResult:
        def __init__(self, *, row: Any = None) -> None:
            self._row = row

        def first(self) -> Any:
            return self._row

    class DummySession:
        def get(self, _model: Any, _key: Any) -> Any:
            return instance

        def execute(self, _stmt: Any) -> DummyResult:
            return DummyResult(row=next(persisted_rows))

    @contextmanager
    def fake_scope() -> Any:
        yield DummySession()

    def fake_resolve_current_campaign_program_hash(_settings: Any) -> Any:
        fallback_calls.append(_settings)
        return SimpleNamespace(known=True, campaign_program_hash=disk_program_hash)

    def fake_load_latest_matching_baseline(**kwargs: Any) -> Any:
        campaign_program_hash = kwargs.get("campaign_program_hash")
        baseline_calls.append(campaign_program_hash)
        return SimpleNamespace(
            id="aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa",
            baseline_key_hash="a" * 64,
            root_commit_hash="root123",
            primary_metric_name="score",
            metric_value=1.0,
            primary_metric_higher_is_better=True,
            status="valid",
            campaign_program_hash=campaign_program_hash,
            failure_kind=None,
            failure_summary=None,
        )

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)
    monkeypatch.setattr(
        "loreley.scheduler.baselines.resolve_current_campaign_program_hash",
        fake_resolve_current_campaign_program_hash,
    )
    monkeypatch.setattr(
        "loreley.scheduler.baselines.load_latest_matching_baseline",
        fake_load_latest_matching_baseline,
    )

    code = main(["status", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert fallback_calls == []
    assert baseline_calls == [active_program_hash]
    assert payload["baseline"]["baseline_campaign_program_hash"] == active_program_hash
