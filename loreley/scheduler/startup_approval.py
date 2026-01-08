from __future__ import annotations

"""Scheduler startup guards that are safe to unit-test without a live database."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from git import Repo

from loreley.config import Settings
from loreley.core.map_elites.repository_files import list_repository_files


@dataclass(frozen=True, slots=True)
class RepoStateRootScan:
    root_commit: str
    eligible_files: int


def scan_repo_state_root(
    *,
    settings: Settings,
    repo_root: Path,
    repo: Repo,
    root_commit: str,
) -> RepoStateRootScan:
    """Scan eligible repo-state files at the experiment root commit (count only)."""

    files = list_repository_files(
        repo_root=Path(repo_root).resolve(),
        commit_hash=str(root_commit).strip(),
        settings=settings,
        repo=repo,
    )
    return RepoStateRootScan(root_commit=str(root_commit).strip(), eligible_files=len(files))


def validate_repo_state_eligible_files_approval(
    *,
    observed_eligible_files: int,
    approved_count: Any,
    root_commit: str,
) -> None:
    """Validate the operator-provided approval count against the observed count."""

    if approved_count is None:
        raise ValueError(
            "Scheduler startup requires explicit approval of the root eligible file count. "
            f"Observed eligible_files={observed_eligible_files} at root_commit={root_commit}. "
            "Set SCHEDULER_REPO_STATE_ELIGIBLE_FILES_APPROVED_COUNT to proceed."
        )
    try:
        approved_value = int(approved_count)
    except (TypeError, ValueError):
        raise ValueError(
            "Invalid SCHEDULER_REPO_STATE_ELIGIBLE_FILES_APPROVED_COUNT; must be an integer."
        ) from None
    if approved_value != int(observed_eligible_files):
        raise ValueError(
            "Repo-state eligible file approval mismatch. "
            f"Observed eligible_files={observed_eligible_files} at root_commit={root_commit}, "
            f"but approved_count={approved_value}."
        )


