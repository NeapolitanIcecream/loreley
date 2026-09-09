from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

import loreley.core.fresh_comparison as comparison_service
import loreley.scheduler.ingestion as ingestion_module
from loreley.scheduler.ingestion import IngestionError, JobSnapshot, MapElitesIngestion
from tests.scheduler.test_ingestion_resilience import _make_ingestion


class _Session:
    @contextmanager
    def begin_nested(self):
        yield


def _snapshot(**kwargs):
    return JobSnapshot(
        job_id=uuid4(), base_commit_hash="base", island_id="main",
        result_commit_hash="candidate", completed_at=None, **kwargs,
    )


@pytest.fixture
def ingestion_harness(tmp_path, monkeypatch):
    ingestion = _make_ingestion(tmp_path)
    calls = []
    context = {"context_id": "issued", "candidate_commit_hash": "candidate"}
    decision = {"allowed": True}
    record = SimpleNamespace(cell_index=0)
    result = SimpleNamespace(
        inserted=True, record=record, status=1, delta=1.0, message=None,
    )

    def ingest_comparison(**kwargs):
        calls.append(("fresh", kwargs))
        return result

    def ingest(**kwargs):
        calls.append(("pareto", kwargs))
        return result

    ingestion.manager = SimpleNamespace(
        ingest_comparison=ingest_comparison, ingest=ingest,
        reload_island=lambda *args, **kwargs: calls.append(("reload", kwargs)),
    )
    monkeypatch.setattr(
        MapElitesIngestion, "_ensure_commit_available", lambda self, commit: commit,
    )
    monkeypatch.setattr(
        MapElitesIngestion, "_metrics_payload_for_ingestion",
        lambda *args, **kwargs: [{"name": "score", "value": 1.0, "higher_is_better": True}],
    )
    monkeypatch.setattr(
        MapElitesIngestion, "_record_successful_ingestion",
        lambda self, snapshot, **kwargs: calls.append(("success", snapshot)),
    )
    monkeypatch.setattr(
        MapElitesIngestion, "_record_ingestion_state",
        lambda self, snapshot, **kwargs: calls.append(("state", (snapshot, kwargs))),
    )
    monkeypatch.setattr(
        comparison_service, "load_handoff", lambda **kwargs: (context, decision),
    )
    monkeypatch.setattr(
        comparison_service, "acknowledge",
        lambda *args, **kwargs: calls.append(("ack", kwargs)),
    )
    monkeypatch.setattr(comparison_service, "requires_fresh_admission", lambda *args: False)
    return SimpleNamespace(
        ingestion=ingestion, calls=calls, context=context, decision=decision,
        result=result, snapshot=_snapshot(), session=_Session(),
    )


def _duplicate_lookup_must_not_run(*args, **kwargs):
    raise AssertionError("old equivalent identity must not veto a fresh comparison")


def test_fresh_admission_bypasses_historical_equivalent_identity(ingestion_harness, monkeypatch):
    h = ingestion_harness
    h.ingestion.settings.mapelites_admission_policy = "fresh_comparison"
    monkeypatch.setattr(
        MapElitesIngestion, "_equivalent_ingested_candidate", _duplicate_lookup_must_not_run,
    )
    assert h.ingestion._ingest_snapshot(h.snapshot, snapshot_session=h.session) is True
    names = [name for name, value in h.calls]
    assert names == ["fresh", "ack", "success"]
    kwargs = h.calls[0][1]
    assert kwargs["replacement_allowed"] is True
    assert kwargs["comparison_context"] is h.context
    assert kwargs["snapshot_session"] is h.session
    assert h.calls[-1][1].comparison_required is True


def test_default_scheduler_rejects_ledger_intent_before_duplicate_or_pareto_path(
    ingestion_harness, monkeypatch,
):
    h = ingestion_harness
    monkeypatch.setattr(comparison_service, "requires_fresh_admission", lambda *args: True)
    monkeypatch.setattr(
        MapElitesIngestion, "_equivalent_ingested_candidate", _duplicate_lookup_must_not_run,
    )
    assert h.ingestion._ingest_snapshot(h.snapshot, snapshot_session=h.session) is False
    assert not any(name in {"fresh", "pareto", "ack", "success"} for name, _ in h.calls)
    failed_snapshot, failed_payload = next(value for name, value in h.calls if name == "state")
    assert failed_snapshot.comparison_required is True
    assert failed_payload["status"] == "failed"
    assert "requires fresh_comparison" in failed_payload["reason"]


def test_direct_default_ingestion_rechecks_ledger_intent(ingestion_harness, monkeypatch):
    h = ingestion_harness
    monkeypatch.setattr(comparison_service, "requires_fresh_admission", lambda *args: True)
    with pytest.raises(IngestionError, match="requires fresh_comparison"):
        h.ingestion._invoke_manager_ingest(
            h.snapshot, commit_hash="candidate", metrics_payload=[], snapshot_session=h.session,
        )
    assert h.calls == []


def test_default_without_fresh_intent_preserves_duplicate_shortcut(ingestion_harness, monkeypatch):
    h = ingestion_harness
    monkeypatch.setattr(
        MapElitesIngestion, "_equivalent_ingested_candidate", lambda *args, **kwargs: "old-binary",
    )
    assert h.ingestion._ingest_snapshot(h.snapshot, snapshot_session=h.session) is False
    assert [name for name, _ in h.calls] == ["success"]
    assert h.calls[0][1].comparison_required is False


def test_fresh_scheduler_cannot_admit_worker_result_without_handoff(ingestion_harness, monkeypatch):
    h = ingestion_harness
    h.ingestion.settings.mapelites_admission_policy = "fresh_comparison"

    def missing_handoff(**kwargs):
        raise ValueError("No issued context for this attempt")

    monkeypatch.setattr(comparison_service, "load_handoff", missing_handoff)
    assert h.ingestion._ingest_snapshot(h.snapshot, snapshot_session=h.session) is False
    assert not any(name in {"fresh", "pareto", "ack", "success"} for name, _ in h.calls)


@pytest.mark.parametrize("comparison_required", [True, False])
def test_explicit_fresh_intent_controls_retry_exhaustion(
    tmp_path, comparison_required,
):
    ingestion = _make_ingestion(tmp_path)
    snapshot = _snapshot(comparison_required=comparison_required)
    payload = ingestion_module._IngestionStatePayload(
        status="failed", reason="stale comparison", delta=None,
        status_code=None, message=None, record=None,
    )
    result = ingestion._terminalize_failed_ingestion(snapshot, payload=payload, attempts=100)
    assert result.status == ("failed" if comparison_required else "skipped")


def test_failure_record_rechecks_durable_intent_even_without_snapshot_flag(tmp_path, monkeypatch):
    ingestion = _make_ingestion(tmp_path)
    snapshot = _snapshot()
    job = SimpleNamespace(
        ingestion_attempts=100, result_commit_hash="candidate", ingestion_status="failed",
    )
    session = SimpleNamespace(get=lambda model, key: job)
    monkeypatch.setattr(comparison_service, "requires_fresh_admission", lambda *args: True)
    payload = ingestion_module._IngestionStatePayload(
        status="failed", reason="commit unavailable", delta=None,
        status_code=None, message=None, record=None,
    )
    ingestion._apply_ingestion_state(snapshot, payload=payload, session=session)
    assert job.ingestion_status == "failed"
    assert job.ingestion_attempts == 101
    assert "retry limit" not in job.ingestion_reason.lower()


def test_outer_commit_failure_invalidates_uncommitted_archive_cache(tmp_path, monkeypatch):
    ingestion = _make_ingestion(tmp_path)
    snapshots = [_snapshot(), replace(_snapshot(), island_id="second")]
    cache = {"main": "old", "second": "old"}
    invalidated = []
    transaction_closed = []

    def invalidate(island):
        assert transaction_closed == [True]
        invalidated.append(island)
        cache.pop(island, None)

    ingestion.manager = SimpleNamespace(invalidate_island=invalidate)
    monkeypatch.setattr(MapElitesIngestion, "_jobs_requiring_ingestion", lambda *args, **kwargs: snapshots)
    monkeypatch.setattr(MapElitesIngestion, "_load_metrics_payload_batch", lambda *args, **kwargs: ({}, {}))

    def ingest_snapshot(self, snapshot, **kwargs):
        cache[snapshot.island_id] = "uncommitted candidate"
        return True

    monkeypatch.setattr(MapElitesIngestion, "_ingest_snapshot", ingest_snapshot)
    scope_count = 0

    @contextmanager
    def session_scope():
        nonlocal scope_count
        scope_count += 1
        yield object()
        if scope_count == 2:
            transaction_closed.append(True)
            raise RuntimeError("outer commit failed")

    monkeypatch.setattr(ingestion_module, "session_scope", session_scope)
    with pytest.raises(RuntimeError, match="outer commit failed"):
        ingestion.ingest_completed_jobs()
    assert sorted(invalidated) == ["main", "second"]
    assert cache == {}
    assert ingestion._prefetched_metrics_payload_by_commit is None
    assert ingestion._record_events_for_batch is False
