from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

import loreley.api.services.candidate_fates as candidate_fate_service
from loreley.db.models import JobStatus


class _ExecResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return list(self._rows)

    def all(self):
        return list(self._rows)


def test_load_candidate_fates_for_jobs_uses_candidate_repair_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.FAILED,
        island_id="main",
        candidate_commit_hash="candidate-a",
        result_commit_hash=None,
        ingestion_status=None,
    )
    candidate = SimpleNamespace(
        commit_hash="candidate-a",
        island_id="main",
        produced_by_job_id=job_id,
        evaluation_status="candidate_failed",
        repair_state="eligible",
        failure_stage="evaluation",
        failure_kind="regression",
    )

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult([candidate])
            if self.calls == 2:
                return _ExecResult([job])
            if self.calls == 3:
                return _ExecResult([])
            raise AssertionError("unexpected query")

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(candidate_fate_service, "session_scope", _fake_scope)

    fates = candidate_fate_service.load_candidate_fates_for_jobs([job])

    assert fates[str(job_id)].label == "repair_pending"
    assert fates[str(job_id)].reason == "Candidate repair_state=eligible. Failure stage=evaluation kind=regression."


def test_load_candidate_fates_for_commits_uses_current_archive_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = uuid4()
    commit = SimpleNamespace(commit_hash="candidate-a", island_id="main", job_id=job_id)
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.SUCCEEDED,
        island_id="main",
        result_commit_hash="candidate-a",
        ingestion_status=None,
    )
    candidate = SimpleNamespace(
        commit_hash="candidate-a",
        island_id="main",
        produced_by_job_id=job_id,
        evaluation_status="passed",
        repair_state="audit_only",
        archive_status="member",
    )

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult([candidate])
            if self.calls == 2:
                return _ExecResult([job])
            if self.calls == 3:
                return _ExecResult([("candidate-a", "main", 9)])
            raise AssertionError("unexpected query")

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(candidate_fate_service, "session_scope", _fake_scope)

    fates = candidate_fate_service.load_candidate_fates_for_commits([commit])

    assert fates["candidate-a"].label == "elite_retained"
    assert fates["candidate-a"].reason == "Candidate is a current archive elite. cell=9."


def test_load_candidate_fates_for_commits_scopes_archive_membership_to_island(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = uuid4()
    commit = SimpleNamespace(commit_hash="candidate-a", island_id="beta", job_id=job_id)
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.SUCCEEDED,
        island_id="beta",
        result_commit_hash="candidate-a",
        ingestion_status=None,
    )
    candidate = SimpleNamespace(
        commit_hash="candidate-a",
        island_id="beta",
        produced_by_job_id=job_id,
        evaluation_status="passed",
        repair_state="audit_only",
        archive_status="member",
    )

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult([candidate])
            if self.calls == 2:
                return _ExecResult([job])
            if self.calls == 3:
                return _ExecResult([("candidate-a", "alpha", 9)])
            raise AssertionError("unexpected query")

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(candidate_fate_service, "session_scope", _fake_scope)

    fates = candidate_fate_service.load_candidate_fates_for_commits([commit])

    assert fates["candidate-a"].label == "valid_not_elite"
    assert (
        fates["candidate-a"].reason
        == "Candidate passed evaluation and has archive_status=member, but is not a current archive elite."
    )


def test_load_candidate_fates_for_jobs_scopes_archive_membership_to_island(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = uuid4()
    job = SimpleNamespace(
        id=job_id,
        status=JobStatus.SUCCEEDED,
        island_id="beta",
        candidate_commit_hash="candidate-a",
        result_commit_hash="candidate-a",
        ingestion_status=None,
    )
    candidate = SimpleNamespace(
        commit_hash="candidate-a",
        island_id="beta",
        produced_by_job_id=job_id,
        evaluation_status="passed",
        repair_state="audit_only",
        archive_status="member",
    )

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt):
            self.calls += 1
            if self.calls == 1:
                return _ExecResult([candidate])
            if self.calls == 2:
                return _ExecResult([job])
            if self.calls == 3:
                return _ExecResult([("candidate-a", "alpha", 9)])
            raise AssertionError("unexpected query")

    @contextmanager
    def _fake_scope():
        yield _Session()

    monkeypatch.setattr(candidate_fate_service, "session_scope", _fake_scope)

    fates = candidate_fate_service.load_candidate_fates_for_jobs([job])

    assert fates[str(job_id)].label == "valid_not_elite"
    assert (
        fates[str(job_id)].reason
        == "Candidate passed evaluation and has archive_status=member, but is not a current archive elite."
    )
