"""Preprocess code files before feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError
from loguru import logger

from loreley.config import Settings, get_settings

log = logger.bind(module="map_elites.preprocess")

__all__ = [
    "PreprocessedFile",
    "CodePreprocessor",
]


@dataclass(slots=True, frozen=True)
class PreprocessedFile:
    """Result of lightweight preprocessing."""

    path: Path
    change_count: int
    content: str


class CodePreprocessor:
    """Filter and cleanup repository files prior to embedding."""

    _block_comment_pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
    _single_line_comment_prefixes = ("#", "//", "--")

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        settings: Settings | None = None,
        commit_hash: str | None = None,
        repo: Repo | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.settings = settings or get_settings()
        self.commit_hash = commit_hash
        self._allowed_extensions = {
            ext if ext.startswith(".") else f".{ext}"
            for ext in self.settings.mapelites_preprocess_allowed_extensions
        }
        self._allowed_filenames = {
            name for name in self.settings.mapelites_preprocess_allowed_filenames
        }
        self._excluded_globs = self._prepare_excluded_globs(
            self.settings.mapelites_preprocess_excluded_globs
        )
        self._max_file_size_bytes = (
            max(self.settings.mapelites_preprocess_max_file_size_kb, 1) * 1024
        )
        self._tab_replacement = (
            " " * self.settings.mapelites_preprocess_tab_width
            if self.settings.mapelites_preprocess_tab_width > 0
            else "\t"
        )
        self._repo: Repo | None = None
        self._git_prefix: Path | None = None
        if self.commit_hash:
            self._repo = repo or self._init_repo()
            if self._repo and self._repo.working_tree_dir:
                git_root = Path(self._repo.working_tree_dir).resolve()
                try:
                    self._git_prefix = self.repo_root.relative_to(git_root)
                except ValueError:
                    log.warning(
                        "Cannot align repo_root={} with git root={} for commit_hash={}",
                        self.repo_root,
                        git_root,
                        self.commit_hash,
                    )
                    self._git_prefix = None

    # Public helpers -------------------------------------------------------

    def is_code_file(self, relative_path: Path) -> bool:
        """Return True if the file looks like a code file under current settings."""
        return self._is_code_file(relative_path)

    def is_excluded(self, relative_path: Path) -> bool:
        """Return True if the path should be excluded under current settings."""
        return self._is_excluded(relative_path)

    def cleanup_text(self, content: str) -> str:
        """Apply preprocessing cleanup (comment stripping, whitespace normalisation)."""
        return self._cleanup_text(content)

    def load_text(self, relative_path: Path) -> str | None:
        """Load file content either from `commit_hash` or from disk."""
        return self._load_text(relative_path)

    def _resolve_on_disk(self, relative_path: Path) -> Path | None:
        absolute = (self.repo_root / relative_path).resolve()
        try:
            absolute.relative_to(self.repo_root)
        except ValueError:
            return None
        return absolute

    def _is_code_file(self, relative_path: Path) -> bool:
        suffix = relative_path.suffix.lower()
        if suffix in self._allowed_extensions:
            return True
        if relative_path.name in self._allowed_filenames:
            return True
        return False

    def _is_excluded(self, relative_path: Path) -> bool:
        if not self._excluded_globs:
            return False
        return any(relative_path.match(pattern) for pattern in self._excluded_globs)

    def _prepare_excluded_globs(self, patterns: Sequence[str]) -> tuple[str, ...]:
        expanded: list[str] = []
        for raw in patterns:
            if not raw:
                continue
            cleaned = raw.strip().replace("\\", "/")
            if cleaned.startswith("./"):
                cleaned = cleaned[2:]
            cleaned = cleaned.lstrip("/")
            if not cleaned:
                continue
            variants = {cleaned}
            if "/" in cleaned and not cleaned.startswith("**/"):
                variants.add(f"**/{cleaned}")
            for variant in sorted(variants):
                if variant not in expanded:
                    expanded.append(variant)
        return tuple(expanded)

    def _cleanup_text(self, content: str) -> str:
        normalised = content.replace("\r\n", "\n").replace("\r", "\n")
        if self.settings.mapelites_preprocess_strip_block_comments:
            normalised = self._block_comment_pattern.sub("\n", normalised)

        lines = []
        blank_streak = 0
        for raw_line in normalised.split("\n"):
            line = raw_line.rstrip()
            if self.settings.mapelites_preprocess_strip_comments:
                stripped = line.lstrip()
                if stripped and stripped.startswith(self._single_line_comment_prefixes):
                    continue

            if self._tab_replacement != "\t":
                line = line.replace("\t", self._tab_replacement)

            if not line.strip():
                blank_streak += 1
                if blank_streak > self.settings.mapelites_preprocess_max_blank_lines:
                    continue
                lines.append("")
                continue

            blank_streak = 0
            lines.append(line)

        cleaned = "\n".join(lines).strip()
        return cleaned

    def _init_repo(self) -> Repo | None:
        try:
            return Repo(self.repo_root, search_parent_directories=True)
        except InvalidGitRepositoryError:
            log.warning(
                "Unable to locate git repository for repo_root={} when commit_hash={} requested",
                self.repo_root,
                self.commit_hash,
            )
            return None

    def _load_text(self, relative_path: Path) -> str | None:
        if self.commit_hash:
            git_content = self._read_from_git(relative_path)
            if git_content is not None:
                return git_content
        return self._read_from_disk(relative_path)

    def _read_from_git(self, relative_path: Path) -> str | None:
        if not self.commit_hash or not self._repo or not self._git_prefix:
            return None
        git_path = (self._git_prefix / relative_path).as_posix()
        spec = f"{self.commit_hash}:{git_path}"
        try:
            size_str = self._repo.git.cat_file("-s", spec)
            blob_size = int(size_str.strip())
        except (GitCommandError, BadName, ValueError) as exc:
            log.error("Unable to stat {} at {}: {}", git_path, self.commit_hash, exc)
            return None

        if self._exceeds_size_limit(blob_size):
            log.info(
                "Skipping {}@{} because it exceeds {} KB",
                git_path,
                self.commit_hash,
                self.settings.mapelites_preprocess_max_file_size_kb,
            )
            return None

        try:
            return self._repo.git.show(spec)
        except (GitCommandError, BadName) as exc:
            log.error("Unable to read {} at {}: {}", git_path, self.commit_hash, exc)
            return None

    def _read_from_disk(self, relative_path: Path) -> str | None:
        file_path = self._resolve_on_disk(relative_path)
        if file_path is None:
            return None
        if not file_path.exists():
            log.warning("Changed file no longer exists on disk: {}", relative_path)
            return None
        try:
            file_size = file_path.stat().st_size
        except OSError as exc:
            log.error("Unable to stat {}: {}", relative_path, exc)
            return None
        if self._exceeds_size_limit(file_size):
            log.info(
                "Skipping {} because it exceeds {} KB",
                relative_path,
                self.settings.mapelites_preprocess_max_file_size_kb,
            )
            return None
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            log.error("Unable to read {}: {}", relative_path, exc)
            return None

    def _exceeds_size_limit(self, num_bytes: int) -> bool:
        return num_bytes > self._max_file_size_bytes

    @staticmethod
    def _coerce_int(value: object | None) -> int:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return 0
        return 0

