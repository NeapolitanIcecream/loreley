"""Repository file enumeration utilities for repo-state embeddings.

This module provides a lightweight way to enumerate *eligible* files for a given
git commit hash while applying basic filtering:

- Respect pinned repository-root ignore rules provided by the experiment snapshot
  (`Settings.mapelites_repo_state_ignore_text`) using gitignore-compatible matching
  via `pathspec.gitignore.GitIgnoreSpec`.
- Respect MAP-Elites preprocessing filters (allowed extensions/filenames, excluded globs).
- Exclude obviously unsuitable files (oversized blobs).

The primary use-case is repo-state embeddings where we need (path, blob_sha)
pairs to drive a file-level embedding cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from git import Repo
from git.exc import BadName, InvalidGitRepositoryError
from loguru import logger
from pathspec.gitignore import GitIgnoreSpec

from loreley.config import Settings, get_settings
from .preprocess import CodePreprocessor

log = logger.bind(module="map_elites.repository_files")

__all__ = [
    "RepositoryFile",
    "ROOT_IGNORE_FILES",
    "RepositoryFileCatalog",
    "list_repository_files",
]

ROOT_IGNORE_FILES: tuple[str, ...] = (".gitignore", ".loreleyignore")


@dataclass(frozen=True, slots=True)
class RepositoryFile:
    """File entry resolved from a git commit hash."""

    path: Path
    blob_sha: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

def _to_posix_repo_path(path: Path | str) -> str:
    """Return a repo-relative, POSIX-style path suitable for gitignore matching."""
    if isinstance(path, Path):
        value = path.as_posix()
    else:
        value = str(path).replace("\\", "/")
    return value.lstrip("/")


def build_pinned_ignore_spec(ignore_text: str | None) -> GitIgnoreSpec | None:
    """Compile pinned root ignore rules into a GitIgnoreSpec.

    The input is the concatenation of repository-root `.gitignore` and `.loreleyignore`
    text pinned in the experiment snapshot. Matching is root-only (no nested ignore
    files and no global excludes).
    """
    raw = str(ignore_text or "")
    if not raw.strip():
        return None
    try:
        return cast(GitIgnoreSpec, GitIgnoreSpec.from_lines(raw.splitlines()))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Failed to compile pinned ignore rules: {}", exc)
        return None


def is_ignored_path(ignore_spec: GitIgnoreSpec, path: Path | str) -> bool:
    """Return True when the given path should be ignored by the pinned rules."""
    candidate = _to_posix_repo_path(path)
    if not candidate:
        return False
    try:
        return bool(ignore_spec.match_file(candidate))
    except Exception:  # pragma: no cover - defensive
        # If path matching fails, do not fail ingestion.
        return False


class RepositoryFileCatalog:
    """Enumerate eligible repository files at a given commit hash."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        settings: Settings | None = None,
        commit_hash: str | None = None,
        repo: Repo | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.settings = settings or get_settings()
        self.commit_hash = commit_hash
        self._repo = repo or self._init_repo()
        self._git_root, self._git_prefix = self._resolve_git_root_and_prefix()

        # Reuse existing preprocess filters for file-type gating and excluded globs.
        self._preprocess_filter = CodePreprocessor(
            repo_root=self.repo_root,
            settings=self.settings,
            commit_hash=None,  # only using filtering helpers; content loads happen elsewhere
        )

        self._max_file_size_bytes = (
            max(self.settings.mapelites_preprocess_max_file_size_kb, 1) * 1024
        )
        self._ignore_spec = self._load_root_ignore_spec()

    def list_files(self) -> list[RepositoryFile]:
        """Return eligible files for this catalog.

        Returned paths are relative to `repo_root`.
        """

        if not self.commit_hash:
            raise ValueError("RepositoryFileCatalog requires commit_hash for git-tree enumeration.")
        if not self._repo:
            return []

        try:
            tree = self._repo.tree(self.commit_hash)
        except BadName as exc:
            raise ValueError(f"Unknown commit {self.commit_hash!r}") from exc

        prefix = self._git_prefix
        prefix_str = prefix.as_posix().rstrip("/") if prefix else ""

        results: list[RepositoryFile] = []
        for blob in tree.traverse():
            if getattr(blob, "type", None) != "blob":
                continue

            git_rel = Path(getattr(blob, "path", ""))
            if not git_rel.as_posix():
                continue

            if prefix_str:
                # Only include files under repo_root when repo_root is a subdir.
                try:
                    git_rel.relative_to(prefix_str)
                except ValueError:
                    continue

            # Apply root ignore filtering relative to git root.
            if self._ignore_spec and is_ignored_path(self._ignore_spec, git_rel):
                continue

            repo_rel = self._to_repo_relative(git_rel)
            if repo_rel is None:
                continue

            # Apply preprocessing file filters (extension allowlist + excluded globs).
            if self._preprocess_filter.is_excluded(repo_rel):
                continue
            if not self._preprocess_filter.is_code_file(repo_rel):
                continue

            size = int(getattr(blob, "size", 0) or 0)
            if size <= 0 or size > self._max_file_size_bytes:
                continue

            sha = str(getattr(blob, "hexsha", "") or "")
            if not sha:
                continue

            results.append(
                RepositoryFile(
                    path=repo_rel,
                    blob_sha=sha,
                    size_bytes=size,
                )
            )

        results.sort(key=lambda entry: entry.path.as_posix())
        log.info(
            "Enumerated {} eligible repository files at commit {} (repo_root={})",
            len(results),
            self.commit_hash,
            self.repo_root,
        )
        return results

    # Internals -------------------------------------------------------------

    def _init_repo(self) -> Repo | None:
        try:
            return Repo(self.repo_root, search_parent_directories=True)
        except InvalidGitRepositoryError:
            log.warning("Unable to locate git repository for repo_root={}", self.repo_root)
            return None

    def _resolve_git_root_and_prefix(self) -> tuple[Path | None, Path | None]:
        if not self._repo or not self._repo.working_tree_dir:
            return None, None
        git_root = Path(self._repo.working_tree_dir).resolve()
        try:
            prefix = self.repo_root.relative_to(git_root)
        except ValueError:
            log.warning(
                "Cannot align repo_root={} with git root={} (commit_hash={})",
                self.repo_root,
                git_root,
                self.commit_hash,
            )
            prefix = None
        if prefix and str(prefix) == ".":
            prefix = None
        return git_root, prefix

    def _to_repo_relative(self, git_rel_path: Path) -> Path | None:
        """Convert a git-root-relative path into a repo_root-relative path."""
        if self._git_prefix:
            try:
                return git_rel_path.relative_to(self._git_prefix.as_posix())
            except ValueError:
                return None
        return git_rel_path

    def _load_root_ignore_spec(self) -> GitIgnoreSpec | None:
        """Load pinned ignore rules from Settings (experiment snapshot)."""
        if not self._repo:
            return None

        pinned = str(getattr(self.settings, "mapelites_repo_state_ignore_text", "") or "").strip()
        if not pinned:
            return None
        return build_pinned_ignore_spec(pinned)


def list_repository_files(
    *,
    repo_root: Path | None = None,
    commit_hash: str | None = None,
    settings: Settings | None = None,
    repo: Repo | None = None,
) -> list[RepositoryFile]:
    """Convenience wrapper for `RepositoryFileCatalog`."""
    catalog = RepositoryFileCatalog(
        repo_root=repo_root,
        commit_hash=commit_hash,
        settings=settings,
        repo=repo,
    )
    return catalog.list_files()


