"""Preprocess code files before feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from loreley.config import Settings, get_settings

log = logger.bind(module="map_elites.preprocess")

__all__ = [
    "ChangedFile",
    "PreprocessedFile",
    "CodePreprocessor",
    "preprocess_changed_files",
]


@dataclass(slots=True, frozen=True)
class ChangedFile:
    """Minimal information about a file touched by a commit."""

    path: Path
    change_count: int = 0
    content: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(slots=True, frozen=True)
class PreprocessedFile:
    """Result of lightweight preprocessing."""

    path: Path
    change_count: int
    content: str


class CodePreprocessor:
    """Filter and cleanup changed files prior to embedding."""

    _block_comment_pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
    _single_line_comment_prefixes = ("#", "//", "--")

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        settings: Settings | None = None,
        treeish: str | None = None,
        repo: Repo | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.settings = settings or get_settings()
        self.treeish = treeish
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
        if self.treeish:
            self._repo = repo or self._init_repo()
            if self._repo and self._repo.working_tree_dir:
                git_root = Path(self._repo.working_tree_dir).resolve()
                try:
                    self._git_prefix = self.repo_root.relative_to(git_root)
                except ValueError:
                    log.warning(
                        "Cannot align repo_root={} with git root={} for treeish={}",
                        self.repo_root,
                        git_root,
                        self.treeish,
                    )
                    self._git_prefix = None

    def run(self, changed_files: Sequence[ChangedFile | Mapping[str, object]]) -> list[PreprocessedFile]:
        """Return cleaned textual content for top-N changed files."""
        candidates = self._select_candidates(changed_files)
        if not candidates:
            log.info("No eligible files for preprocessing.")
            return []

        artifacts: list[PreprocessedFile] = []
        progress = self._build_progress()

        with progress:
            task_id = progress.add_task(
                "[cyan]Preprocessing changed files",
                total=len(candidates),
            )
            for candidate in candidates:
                relative_path = self._relative_path(candidate.path)
                if relative_path is None:
                    log.warning("Skipping file outside repository root: {}", candidate.path)
                    progress.update(task_id, advance=1)
                    continue

                inline_content = candidate.content
                if inline_content is not None:
                    byte_size = len(inline_content.encode("utf-8"))
                    if self._exceeds_size_limit(byte_size):
                        log.info(
                            "Skipping {} because provided content exceeds {} KB",
                            relative_path,
                            self.settings.mapelites_preprocess_max_file_size_kb,
                        )
                        progress.update(task_id, advance=1)
                        continue
                    raw_text = inline_content
                else:
                    raw_text = self._load_text(relative_path)

                if raw_text is None:
                    log.warning(
                        "Unable to load content for {}; skipping",
                        relative_path,
                    )
                    progress.update(task_id, advance=1)
                    continue

                processed = self._cleanup_text(raw_text)
                if not processed:
                    log.debug("File {} became empty after cleanup; skipping", relative_path)
                    progress.update(task_id, advance=1)
                    continue

                artifacts.append(
                    PreprocessedFile(
                        path=relative_path,
                        change_count=candidate.change_count,
                        content=processed,
                    )
                )
                progress.update(task_id, advance=1)

        log.info("Preprocessed {} files.", len(artifacts))
        return artifacts

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
        """Load file content either from `treeish` or from disk."""
        return self._load_text(relative_path)

    def _select_candidates(
        self,
        changed_files: Sequence[ChangedFile | Mapping[str, object]],
    ) -> list[ChangedFile]:
        normalised = [
            cf
            for cf in (self._coerce_changed_file(entry) for entry in changed_files)
            if cf is not None
        ]

        filtered: list[ChangedFile] = []
        for file in normalised:
            rel_path = self._relative_path(file.path)
            if rel_path is None:
                continue
            if self._is_excluded(rel_path):
                continue
            if not self._is_code_file(rel_path):
                continue
            filtered.append(
                ChangedFile(
                    path=rel_path,
                    change_count=file.change_count,
                    content=file.content,
                )
            )

        filtered.sort(key=lambda item: item.change_count, reverse=True)
        limit = max(self.settings.mapelites_preprocess_max_files, 0)
        if limit:
            filtered = filtered[:limit]

        return filtered

    def _coerce_changed_file(
        self,
        entry: ChangedFile | Mapping[str, object],
    ) -> ChangedFile | None:
        if isinstance(entry, ChangedFile):
            return entry

        if isinstance(entry, (str, Path)):
            return ChangedFile(path=Path(entry), change_count=0)

        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            raw_path, raw_delta = entry
            if not isinstance(raw_path, (str, Path)):
                return None
            delta = self._coerce_int(raw_delta)
            return ChangedFile(path=Path(raw_path), change_count=delta)

        if isinstance(entry, Mapping):
            path_value = entry.get("path") or entry.get("file") or entry.get("filename")
            if not path_value:
                return None
            if not isinstance(path_value, (str, Path)):
                return None
            change_count_value = entry.get("change_count") or entry.get("lines_changed") or entry.get("delta")
            content_value = entry.get("content")
            change_count = self._coerce_int(change_count_value)
            return ChangedFile(
                path=Path(path_value),
                change_count=change_count,
                content=content_value if isinstance(content_value, str) else None,
            )

        return None

    def _relative_path(self, candidate: Path) -> Path | None:
        candidate_path = Path(candidate)
        combined = (
            candidate_path if candidate_path.is_absolute() else self.repo_root / candidate_path
        )
        try:
            absolute = combined.resolve()
        except OSError:
            return None
        try:
            return absolute.relative_to(self.repo_root)
        except ValueError:
            return None

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

    def _build_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(style="green"),
            TextColumn("{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
        )

    def _init_repo(self) -> Repo | None:
        try:
            return Repo(self.repo_root, search_parent_directories=True)
        except InvalidGitRepositoryError:
            log.warning(
                "Unable to locate git repository for repo_root={} when treeish={} requested",
                self.repo_root,
                self.treeish,
            )
            return None

    def _load_text(self, relative_path: Path) -> str | None:
        if self.treeish:
            git_content = self._read_from_git(relative_path)
            if git_content is not None:
                return git_content
        return self._read_from_disk(relative_path)

    def _read_from_git(self, relative_path: Path) -> str | None:
        if not self.treeish or not self._repo or not self._git_prefix:
            return None
        git_path = (self._git_prefix / relative_path).as_posix()
        spec = f"{self.treeish}:{git_path}"
        try:
            size_str = self._repo.git.cat_file("-s", spec)
            blob_size = int(size_str.strip())
        except (GitCommandError, BadName, ValueError) as exc:
            log.error("Unable to stat {} at {}: {}", git_path, self.treeish, exc)
            return None

        if self._exceeds_size_limit(blob_size):
            log.info(
                "Skipping {}@{} because it exceeds {} KB",
                git_path,
                self.treeish,
                self.settings.mapelites_preprocess_max_file_size_kb,
            )
            return None

        try:
            return self._repo.git.show(spec)
        except (GitCommandError, BadName) as exc:
            log.error("Unable to read {} at {}: {}", git_path, self.treeish, exc)
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


def preprocess_changed_files(
    changed_files: Sequence[ChangedFile | Mapping[str, object]],
    *,
    repo_root: Path | None = None,
    settings: Settings | None = None,
    treeish: str | None = None,
    repo: Repo | None = None,
) -> list[PreprocessedFile]:
    """Functional wrapper for the preprocessor."""
    preprocessor = CodePreprocessor(
        repo_root=repo_root,
        settings=settings,
        treeish=treeish,
        repo=repo,
    )
    return preprocessor.run(changed_files)

