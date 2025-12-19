"""Repository file enumeration utilities for repo-state embeddings.

This module provides a lightweight way to enumerate *eligible* files for a given
git treeish (commit hash, tag, etc.) while applying basic filtering:

- Respect the repository root `.gitignore` (best-effort, glob-based matching).
- Respect MAP-Elites preprocessing filters (allowed extensions/filenames, excluded globs).
- Exclude obviously unsuitable files (oversized blobs).

The primary use-case is repo-state embeddings where we need (path, blob_sha)
pairs to drive a file-level embedding cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError
from loguru import logger

from loreley.config import Settings, get_settings
from .preprocess import CodePreprocessor

log = logger.bind(module="map_elites.repository_files")

__all__ = [
    "RepositoryFile",
    "GitignoreMatcher",
    "RepositoryFileCatalog",
    "list_repository_files",
]


@dataclass(frozen=True, slots=True)
class RepositoryFile:
    """File entry resolved from a git treeish."""

    path: Path
    blob_sha: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class _IgnoreRule:
    raw: str
    pattern: str
    negated: bool


class GitignoreMatcher:
    """Best-effort `.gitignore` matcher for repository-relative paths.

    Notes:
    - This intentionally implements a pragmatic subset of gitignore semantics,
      sufficient for filtering typical build/test artifacts in LLM embedding
      pipelines.
    - Patterns are applied in order; last match wins.
    - Negation patterns (`!foo`) re-include previously ignored paths.
    """

    def __init__(self, patterns: Sequence[str]) -> None:
        self._rules: tuple[_IgnoreRule, ...] = tuple(self._parse(patterns))

    @classmethod
    def from_gitignore_text(cls, text: str) -> "GitignoreMatcher":
        patterns: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip("\n").strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            patterns.append(line)
        return cls(patterns)

    def is_ignored(self, path: Path | str) -> bool:
        candidate = self._to_posix(path)
        ignored = False
        for rule in self._rules:
            if not rule.pattern:
                continue
            if self._match(candidate, rule.pattern):
                ignored = not rule.negated
        return ignored

    @staticmethod
    def _to_posix(path: Path | str) -> str:
        if isinstance(path, Path):
            return path.as_posix().lstrip("/")
        return str(path).replace("\\", "/").lstrip("/")

    def _parse(self, patterns: Sequence[str]) -> Iterable[_IgnoreRule]:
        for raw in patterns:
            if not raw:
                continue
            cleaned = raw.strip()
            if not cleaned or cleaned.startswith("#"):
                continue

            negated = cleaned.startswith("!")
            if negated:
                cleaned = cleaned[1:].lstrip()
            if not cleaned:
                continue

            # Directory rules like "dist/" should ignore everything beneath.
            directory_rule = cleaned.endswith("/")
            cleaned = cleaned.rstrip("/")

            anchored = cleaned.startswith("/")
            cleaned = cleaned.lstrip("/")

            # Turn gitignore-ish patterns into a small set of glob patterns.
            # We use PurePosixPath.match which treats path separators sanely.
            globs: list[str] = []
            if directory_rule:
                base = cleaned
                if anchored:
                    globs.append(f"{base}/**")
                else:
                    globs.append(f"{base}/**")
                    globs.append(f"**/{base}/**")
            else:
                base = cleaned
                if anchored:
                    globs.append(base)
                else:
                    # Patterns without slashes behave like matching any basename.
                    if "/" not in base:
                        globs.append(base)
                        globs.append(f"**/{base}")
                    else:
                        globs.append(base)
                        globs.append(f"**/{base}")

            for glob in globs:
                yield _IgnoreRule(raw=raw, pattern=glob, negated=negated)

    @staticmethod
    def _match(path_posix: str, pattern: str) -> bool:
        try:
            return PurePosixPath(path_posix).match(pattern)
        except Exception:  # pragma: no cover - defensive
            # If a pattern is malformed, ignore it rather than failing ingestion.
            return False


class RepositoryFileCatalog:
    """Enumerate eligible repository files at a given treeish."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        settings: Settings | None = None,
        treeish: str | None = None,
        repo: Repo | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.settings = settings or get_settings()
        self.treeish = treeish
        self._repo = repo or self._init_repo()
        self._git_root, self._git_prefix = self._resolve_git_root_and_prefix()

        # Reuse existing preprocess filters for file-type gating and excluded globs.
        self._preprocess_filter = CodePreprocessor(
            repo_root=self.repo_root,
            settings=self.settings,
            treeish=None,  # only using filtering helpers; content loads happen elsewhere
        )

        self._max_file_size_bytes = (
            max(self.settings.mapelites_preprocess_max_file_size_kb, 1) * 1024
        )
        self._gitignore = self._load_root_gitignore()

    def list_files(self) -> list[RepositoryFile]:
        """Return eligible files for this catalog.

        Returned paths are relative to `repo_root`.
        """

        if not self.treeish:
            raise ValueError("RepositoryFileCatalog requires treeish for git-tree enumeration.")
        if not self._repo:
            return []

        try:
            tree = self._repo.tree(self.treeish)
        except BadName as exc:
            raise ValueError(f"Unknown treeish {self.treeish!r}") from exc

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

            # Apply .gitignore filtering relative to git root.
            if self._gitignore and self._gitignore.is_ignored(git_rel):
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
        limit = int(getattr(self.settings, "mapelites_repo_state_max_files", 0) or 0)
        if limit > 0 and len(results) > limit:
            original = len(results)
            # Deterministic sub-sampling to reduce path-prefix bias while still
            # keeping results stable across runs.
            sampled = sorted(
                results,
                key=lambda entry: hashlib.sha1(entry.path.as_posix().encode("utf-8")).hexdigest(),
            )
            results = sampled[:limit]
            results.sort(key=lambda entry: entry.path.as_posix())
            log.warning(
                "Truncated eligible repository files from {} to {} due to MAPELITES_REPO_STATE_MAX_FILES (stable-hash sampling).",
                original,
                limit,
            )
        log.info(
            "Enumerated {} eligible repository files at treeish={} (repo_root={})",
            len(results),
            self.treeish,
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
                "Cannot align repo_root={} with git root={} (treeish={})",
                self.repo_root,
                git_root,
                self.treeish,
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

    def _load_root_gitignore(self) -> GitignoreMatcher | None:
        """Load `.gitignore` from git root at the requested treeish."""
        if not self._repo:
            return None

        content: str | None = None
        if self.treeish:
            try:
                content = self._repo.git.show(f"{self.treeish}:.gitignore")
            except (GitCommandError, BadName):
                content = None
        if content is None:
            # Fall back to working tree .gitignore when available.
            git_root = self._git_root or self.repo_root
            path = (git_root / ".gitignore").resolve()
            try:
                if path.exists():
                    content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover - filesystem edge cases
                content = None

        if not content:
            return None
        return GitignoreMatcher.from_gitignore_text(content)


def list_repository_files(
    *,
    repo_root: Path | None = None,
    treeish: str | None = None,
    settings: Settings | None = None,
    repo: Repo | None = None,
) -> list[RepositoryFile]:
    """Convenience wrapper for `RepositoryFileCatalog`."""
    catalog = RepositoryFileCatalog(
        repo_root=repo_root,
        treeish=treeish,
        settings=settings,
        repo=repo,
    )
    return catalog.list_files()


