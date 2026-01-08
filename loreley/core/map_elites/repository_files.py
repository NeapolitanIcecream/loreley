"""Repository file enumeration utilities for repo-state embeddings.

This module provides a lightweight way to enumerate *eligible* files for a given
git commit hash while applying basic filtering:

- Respect the repository root `.gitignore` and `.loreleyignore` (best-effort, glob-based matching).
- Respect MAP-Elites preprocessing filters (allowed extensions/filenames, excluded globs).
- Exclude obviously unsuitable files (oversized blobs).

The primary use-case is repo-state embeddings where we need (path, blob_sha)
pairs to drive a file-level embedding cache.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    "ROOT_IGNORE_FILES",
    "load_root_ignore_matcher_from_commit",
    "RepositoryFileCatalog",
    "list_repository_files",
]

ROOT_IGNORE_FILES: tuple[str, ...] = (".gitignore", ".loreleyignore")


def load_root_ignore_matcher_from_commit(repo: Repo, commit_hash: str) -> GitignoreMatcher | None:
    """Load repository-root ignore rules from a specific commit.

    Files:
    - `.gitignore`
    - `.loreleyignore`

    Notes:
    - Files are read strictly from the specified commit; missing files are treated as absent.
    - Rules are applied in order: `.gitignore` first, then `.loreleyignore`.
    """
    commit = str(commit_hash or "").strip()
    if not commit:
        return None

    chunks: list[str] = []
    for filename in ROOT_IGNORE_FILES:
        try:
            chunks.append(repo.git.show(f"{commit}:{filename}"))
        except (GitCommandError, BadName):
            chunks.append("")

    combined = "\n".join(chunks).strip()
    if not combined:
        return None
    return GitignoreMatcher.from_gitignore_text(combined)


@dataclass(frozen=True, slots=True)
class RepositoryFile:
    """File entry resolved from a git commit hash."""

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
    """Best-effort `.gitignore`-style matcher for repository-relative paths.

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
        self._ignore_matcher = self._load_root_ignore_matcher()

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
            if self._ignore_matcher and self._ignore_matcher.is_ignored(git_rel):
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

    def _load_root_ignore_matcher(self) -> GitignoreMatcher | None:
        """Load ignore rules from git root at the requested commit hash.

        Files:
        - `.gitignore`
        - `.loreleyignore`

        Rules are applied in order: `.gitignore` first, then `.loreleyignore`.
        """
        if not self._repo:
            return None

        if self.commit_hash:
            # When a commit hash is requested, do not fall back to the working tree.
            # If ignore files do not exist at that commit, treat them as absent.
            return load_root_ignore_matcher_from_commit(self._repo, self.commit_hash)
        else:
            # Fall back to working tree ignore files when commit hash is not specified.
            git_root = self._git_root or self.repo_root
            gitignore_path = (git_root / ".gitignore").resolve()
            loreleyignore_path = (git_root / ".loreleyignore").resolve()
            gitignore: str | None = None
            loreleyignore: str | None = None
            try:
                if gitignore_path.exists():
                    gitignore = gitignore_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover - filesystem edge cases
                gitignore = None
            try:
                if loreleyignore_path.exists():
                    loreleyignore = loreleyignore_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:  # pragma: no cover - filesystem edge cases
                loreleyignore = None

        combined = "\n".join([gitignore or "", loreleyignore or ""]).strip()
        if not combined:
            return None
        return GitignoreMatcher.from_gitignore_text(combined)


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


