from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from loreley.cli import main
from loreley.core.evolution_timeline import (
    EvolutionTimelineExport,
    TimelineCompletenessError,
    TimelineIssue,
)
from tests.support import TestSettings


def _patch_timeline_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "loreley.cli.get_settings",
        lambda: TestSettings(
            EXPERIMENT_ID="timeline-cli",
            MAPELITES_EXPERIMENT_ROOT_COMMIT="a" * 40,
        ),
    )
    monkeypatch.setattr(
        "loreley.db.base.ensure_database_schema",
        lambda **_kwargs: None,
    )

    @contextmanager
    def fake_scope():  # type: ignore[no-untyped-def]
        yield object()

    monkeypatch.setattr("loreley.db.base.session_scope", fake_scope)


def test_timeline_export_stdout_is_jsonl_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_timeline_cli(monkeypatch)
    occurred_at = datetime(2026, 8, 25, 3, 0, tzinfo=UTC)
    exported = EvolutionTimelineExport(
        metadata={
            "record_type": "metadata",
            "timeline_schema_version": 1,
            "strict_valid": True,
        },
        events=(
            {
                "record_type": "event",
                "timeline_schema_version": 1,
                "event_id": "event-1",
                "event_type": "job.run.started",
                "occurred_at": occurred_at.isoformat(),
                "payload": {},
            },
        ),
        issues=(),
    )
    monkeypatch.setattr(
        "loreley.core.evolution_timeline.export_evolution_timeline",
        lambda _session, *, strict: exported,
    )

    code = main(["timeline", "export", "--strict"])
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]

    assert code == 0
    assert [record["record_type"] for record in records] == [
        "metadata",
        "event",
    ]


def test_timeline_strict_failure_returns_machine_readable_issues(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_timeline_cli(monkeypatch)
    issue = TimelineIssue(
        code="terminal_without_run_start",
        message="terminal job lacks run start",
        job_id="job-1",
    )

    def fail(_session, *, strict: bool):  # type: ignore[no-untyped-def]
        assert strict is True
        raise TimelineCompletenessError((issue,))

    monkeypatch.setattr(
        "loreley.core.evolution_timeline.export_evolution_timeline",
        fail,
    )

    code = main(["timeline", "export", "--strict"])
    captured = capsys.readouterr()
    error = json.loads(captured.err.splitlines()[-1])

    assert code == 1
    assert captured.out == ""
    assert error["error"] == "timeline_incomplete"
    assert error["issues"][0]["code"] == "terminal_without_run_start"
