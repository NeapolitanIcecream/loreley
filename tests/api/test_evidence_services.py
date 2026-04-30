from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

import loreley.api.services.evidence as evidence_service


class _ScalarRows:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)


class _ExecRows:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return _ScalarRows(self._rows)


def _artifact_row(
    *,
    commit_hash: str,
    key: str,
    visibility: str,
    summary: str | None = None,
    diagnostics: list[dict[str, object]] | None = None,
    agent_projection: str = "summary",
):
    now = datetime(2026, 4, 29, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=uuid4(),
        job_id=uuid4(),
        commit_hash=commit_hash,
        key=key,
        kind="benchmark_json",
        mime_type="application/json",
        label=None,
        summary=summary,
        visibility=visibility,
        agent_projection=agent_projection,
        storage_path="/worker/artifacts/report.json",
        size_bytes=12,
        sha256="a" * 64,
        diagnostics=diagnostics or [],
        created_at=now,
    )


def test_load_evidence_indicators_ignores_hidden_and_uses_agent_visible_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _artifact_row(
            commit_hash="c1",
            key="hidden",
            visibility="hidden",
            summary="hidden summary",
        ),
        _artifact_row(
            commit_hash="c1",
            key="bench",
            visibility="agent_visible",
            summary="fallback summary",
            diagnostics=[{"kind": "regression", "message": "p95 latency regressed"}],
        ),
        _artifact_row(
            commit_hash="c1",
            key="stderr",
            visibility="human_only",
            summary="human summary",
        ),
    ]

    class _Session:
        def execute(self, _stmt):
            return _ExecRows(rows)

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(evidence_service, "session_scope", _fake_scope)

    indicators = evidence_service.load_evidence_indicators_by_commit_hash(["c1", "c2"])

    assert indicators["c1"].has_evaluation_evidence is True
    assert indicators["c1"].agent_visible_evidence_count == 1
    assert indicators["c1"].top_evaluation_diagnosis == "p95 latency regressed"
    assert indicators["c2"].has_evaluation_evidence is False


def test_build_evaluation_artifact_payload_exposes_download_url_not_storage_path() -> None:
    row = _artifact_row(
        commit_hash="c1",
        key="bench",
        visibility="agent_visible",
        summary="summary",
    )

    payload = evidence_service.build_evaluation_artifact_payload(row)

    assert payload["download_url"] == f"/api/v1/jobs/{row.job_id}/evaluation-artifacts/bench"
    assert "storage_path" not in payload


def test_build_agent_feedback_payload_respects_artifact_manifest_projection() -> None:
    settings = evidence_service.Settings()
    settings.worker_evaluation_agent_feedback_mode = "path"
    row = _artifact_row(
        commit_hash="c1",
        key="bench",
        visibility="agent_visible",
        summary="summary prose must not reach agents",
        diagnostics=[{"kind": "regression", "message": "diagnostic prose must not reach agents"}],
        agent_projection="manifest",
    )

    payload = evidence_service.build_agent_feedback_payload([row], settings=settings)

    assert payload is not None
    text = str(payload["text"])
    assert "bench" in text
    assert "mime=application/json" in text
    assert "summary prose must not reach agents" not in text
    assert "diagnostic prose must not reach agents" not in text
    assert payload["included_artifact_keys"] == ["bench"]
