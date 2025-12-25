"""CommitCard builders for Loreley's evolution worker.

This module intentionally uses lightweight, deterministic heuristics (git diff)
to produce bounded fields suitable for inspiration prompts and UI displays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from loguru import logger

log = logger.bind(module="worker.commit_card")


@dataclass(frozen=True, slots=True)
class CommitCardBuildResult:
    """Derived CommitCard fields computed from git history."""

    key_files: tuple[str, ...]
    highlights: tuple[str, ...]


def build_commit_card_from_git(
    *,
    worktree: Path,
    base_commit: str,
    candidate_commit: str,
    max_key_files: int = 20,
    max_highlights: int = 8,
) -> CommitCardBuildResult:
    """Build (key_files, highlights) using git diff heuristics."""

    worktree = Path(worktree).expanduser().resolve()
    try:
        repo = Repo(worktree)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Invalid git worktree for commit card builder: {worktree}") from exc

    key_files: list[str] = []
    highlights: list[str] = []

    diff_range = f"{base_commit}..{candidate_commit}"

    try:
        names = repo.git.diff("--name-only", diff_range).splitlines()
        key_files = [line.strip() for line in names if line.strip()]
    except GitCommandError as exc:
        log.warning("Failed to compute key_files from git diff {}: {}", diff_range, exc)
        key_files = []

    if max_key_files > 0 and len(key_files) > max_key_files:
        key_files = key_files[:max_key_files]

    try:
        numstat_lines = repo.git.diff("--numstat", diff_range).splitlines()
    except GitCommandError as exc:
        log.warning("Failed to compute highlights from git numstat {}: {}", diff_range, exc)
        numstat_lines = []

    # Parse: "<added>\t<deleted>\t<path>"
    parsed: list[tuple[int, int, str]] = []
    for line in numstat_lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added_raw, deleted_raw, path = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not path:
            continue
        try:
            added = int(added_raw) if added_raw.isdigit() else 0
            deleted = int(deleted_raw) if deleted_raw.isdigit() else 0
        except Exception:
            added, deleted = 0, 0
        parsed.append((added, deleted, path))

    # Sort by total churn, then added.
    parsed.sort(key=lambda item: (item[0] + item[1], item[0]), reverse=True)

    for added, deleted, path in parsed[: max(0, max_highlights)]:
        highlights.append(f"{path}: +{added}/-{deleted}")

    if not highlights:
        # Ensure non-empty highlights for downstream prompt contracts.
        if key_files:
            highlights = [f"Touched files: {', '.join(key_files[: min(5, len(key_files))])}"]
        else:
            highlights = ["No file-level highlights available."]

    return CommitCardBuildResult(
        key_files=tuple(key_files),
        highlights=tuple(highlights[: max(1, max_highlights)]),
    )


