from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from loreley.db.migrations.versions import v0012_agent_actions_and_cleanup as migration


class _FakeMappingsResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> list[dict[str, Any]]:
        return self._rows


class _FakeInsertResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.inserted: list[dict[str, Any]] = []
        self._inserted_commit_hashes: set[str] = set()

    def execute(self, _statement, params: dict[str, Any] | None = None):
        if params is None:
            return _FakeMappingsResult(self._rows)
        if params["commit_hash"] in self._inserted_commit_hashes:
            return _FakeInsertResult(0)
        self._inserted_commit_hashes.add(params["commit_hash"])
        self.inserted.append(params)
        return _FakeInsertResult(1)


def _historical_job_row(
    *,
    job_id: UUID,
    commit_hash: str,
    base_commit_hash: str,
    completed_at: datetime,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "commit_hash": commit_hash,
        "base_commit_hash": base_commit_hash,
        "island_id": "main",
        "run_token": UUID("00000000-0000-0000-0000-000000000099"),
        "job_kind": "evolution",
        "candidate_branch_name": None,
        "candidate_published_at": None,
        "completed_at": completed_at,
        "commit_card_id": None,
        "archive_member": False,
    }


def test_v0012_candidate_backfill_chooses_latest_job_for_duplicate_commit_hash() -> None:
    """Regression: duplicate historical commits must select a canonical source job."""

    older_job = UUID("00000000-0000-0000-0000-000000000001")
    latest_job = UUID("00000000-0000-0000-0000-000000000002")
    conn = _FakeConnection(
        [
            _historical_job_row(
                job_id=older_job,
                commit_hash="commit-a",
                base_commit_hash="base-old",
                completed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            _historical_job_row(
                job_id=latest_job,
                commit_hash="commit-a",
                base_commit_hash="base-latest",
                completed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
        ]
    )

    migration._backfill_candidate_commits(conn)

    assert len(conn.inserted) == 1
    assert conn.inserted[0]["commit_hash"] == "commit-a"
    assert conn.inserted[0]["produced_by_job_id"] == latest_job
    assert conn.inserted[0]["git_parent_commit_hash"] == "base-latest"
