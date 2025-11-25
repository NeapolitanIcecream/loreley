"""Preprocess commit diffs before feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import fnmatch
import re
from typing import Mapping, Sequence

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from app.config import Settings, get_settings

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
    ) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.settings = settings or get_settings()
        self._allowed_extensions = {
            ext if ext.startswith(".") else f".{ext}"
            for ext in self.settings.mapelites_preprocess_allowed_extensions
        }
        self._allowed_filenames = {
            name for name in self.settings.mapelites_preprocess_allowed_filenames
        }
        self._excluded_globs = tuple(self.settings.mapelites_preprocess_excluded_globs)
        self._max_file_size_bytes = (
            max(self.settings.mapelites_preprocess_max_file_size_kb, 1) * 1024
        )
        self._tab_replacement = (
            " " * self.settings.mapelites_preprocess_tab_width
            if self.settings.mapelites_preprocess_tab_width > 0
            else "\t"
        )

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

                file_path = self._resolve_on_disk(relative_path)
                if file_path is None:
                    log.warning(
                        "Skipping {} because it cannot be safely resolved under repo root",
                        relative_path,
                    )
                    progress.update(task_id, advance=1)
                    continue
                if not file_path.exists():
                    log.warning("Changed file no longer exists on disk: {}", relative_path)
                    progress.update(task_id, advance=1)
                    continue

                try:
                    if file_path.stat().st_size > self._max_file_size_bytes:
                        log.info(
                            "Skipping {} because it exceeds {} KB",
                            relative_path,
                            self.settings.mapelites_preprocess_max_file_size_kb,
                        )
                        progress.update(task_id, advance=1)
                        continue
                except OSError as exc:
                    log.error("Unable to stat {}: {}", relative_path, exc)
                    progress.update(task_id, advance=1)
                    continue

                try:
                    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError as exc:
                    log.error("Unable to read {}: {}", relative_path, exc)
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
            filtered.append(ChangedFile(path=rel_path, change_count=file.change_count))

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
            try:
                delta = int(raw_delta)
            except (TypeError, ValueError):
                delta = 0
            return ChangedFile(path=Path(raw_path), change_count=delta)

        if isinstance(entry, Mapping):
            path_value = entry.get("path") or entry.get("file") or entry.get("filename")
            if not path_value:
                return None
            change_count_value = entry.get("change_count") or entry.get("lines_changed") or entry.get("delta")
            try:
                change_count = int(change_count_value) if change_count_value is not None else 0
            except (TypeError, ValueError):
                change_count = 0
            return ChangedFile(path=Path(path_value), change_count=change_count)

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
        unix_path = relative_path.as_posix()
        return any(fnmatch.fnmatch(unix_path, pattern) for pattern in self._excluded_globs)

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


def preprocess_changed_files(
    changed_files: Sequence[ChangedFile | Mapping[str, object]],
    *,
    repo_root: Path | None = None,
    settings: Settings | None = None,
) -> list[PreprocessedFile]:
    """Functional wrapper for the preprocessor."""
    preprocessor = CodePreprocessor(repo_root=repo_root, settings=settings)
    return preprocessor.run(changed_files)

