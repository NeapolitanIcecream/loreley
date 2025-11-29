"""Split preprocessed files into semantically aware chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, TYPE_CHECKING

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from loreley.config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover
    from .preprocess import PreprocessedFile as _PreprocessedFile

log = logger.bind(module="map_elites.chunk")

__all__ = [
    "FileChunk",
    "ChunkedFile",
    "CodeChunker",
    "chunk_preprocessed_files",
]


@dataclass(slots=True, frozen=True)
class FileChunk:
    """Chunked segment of a single file."""

    file_path: Path
    chunk_id: str
    index: int
    start_line: int
    end_line: int
    content: str
    line_count: int
    change_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_path", Path(self.file_path))


@dataclass(slots=True, frozen=True)
class ChunkedFile:
    """Collection of chunks generated from one file."""

    path: Path
    change_count: int
    total_lines: int
    chunks: tuple[FileChunk, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


class PreprocessedArtifact(Protocol):
    """Protocol describing the inputs consumed by the chunker."""

    path: Path
    change_count: int
    content: str


class CodeChunker:
    """Split files into manageable chunks for embedding."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._min_lines = max(1, self.settings.mapelites_chunk_min_lines)
        self._target_lines = max(self._min_lines, self.settings.mapelites_chunk_target_lines)
        self._overlap = max(0, self.settings.mapelites_chunk_overlap_lines)
        self._max_chunks_per_file = max(1, self.settings.mapelites_chunk_max_chunks_per_file)
        self._boundary_keywords = tuple(
            keyword.lower()
            for keyword in self.settings.mapelites_chunk_boundary_keywords
            if keyword
        )

    def run(self, files: Sequence[PreprocessedArtifact]) -> list[ChunkedFile]:
        """Chunk every preprocessed file."""
        if not files:
            log.info("No files provided to chunker; skipping.")
            return []

        chunked_files: list[ChunkedFile] = []
        progress = self._build_progress()

        with progress:
            task_id = progress.add_task(
                "[cyan]Chunking files",
                total=len(files),
            )
            for file in files:
                chunks = self._chunk_file(file)
                chunked_files.append(
                    ChunkedFile(
                        path=file.path,
                        change_count=file.change_count,
                        total_lines=self._count_lines(file.content),
                        chunks=tuple(chunks),
                    )
                )
                progress.update(task_id, advance=1)

        chunk_count = sum(len(entry.chunks) for entry in chunked_files)
        log.info("Chunked {} files into {} chunks.", len(chunked_files), chunk_count)
        return chunked_files

    def _chunk_file(self, file: PreprocessedArtifact) -> list[FileChunk]:
        """Chunk a single file."""
        lines = file.content.splitlines()
        total_lines = len(lines)
        if not total_lines:
            return []

        chunks: list[FileChunk] = []
        cursor = 0
        windows_processed = 0

        while cursor < total_lines and windows_processed < self._max_chunks_per_file:
            start = cursor
            rough_end = min(start + self._target_lines, total_lines)
            end = self._find_break_point(lines, start, rough_end, total_lines)
            if end <= start:
                end = min(start + self._target_lines, total_lines)

            chunk_lines = lines[start:end]
            joined = "\n".join(chunk_lines)
            if joined.strip():
                content = self._trim_surrounding_newlines(joined)
                if content:
                    chunk_index = len(chunks)
                    chunks.append(
                        FileChunk(
                            file_path=file.path,
                            chunk_id=self._build_chunk_id(file.path, chunk_index),
                            index=chunk_index,
                            start_line=start + 1,
                            end_line=end,
                            content=content,
                            line_count=end - start,
                            change_count=file.change_count,
                        )
                    )
                else:
                    log.debug(
                        "Discarded empty chunk for {} [{}:{}]",
                        file.path,
                        start,
                        end,
                    )
            else:
                log.debug(
                    "Discarded whitespace-only chunk for {} [{}:{}]",
                    file.path,
                    start,
                    end,
                )

            cursor = self._advance_cursor(start, end, total_lines)
            windows_processed += 1

        if cursor < total_lines:
            log.debug(
                "Reached chunk limit ({} windows) for {}; truncated tail.",
                self._max_chunks_per_file,
                file.path,
            )

        return chunks

    def _find_break_point(
        self,
        lines: list[str],
        start: int,
        proposed_end: int,
        total_lines: int,
    ) -> int:
        min_break = min(start + self._min_lines, total_lines)
        if proposed_end <= min_break:
            return proposed_end

        lower_bound = min_break
        upper_bound = min(proposed_end, total_lines)

        for idx in range(upper_bound - 1, lower_bound - 1, -1):
            line = lines[idx]
            stripped = line.strip()
            if not stripped:
                return idx + 1
            if self._looks_like_boundary(stripped, line):
                return idx + 1

        return upper_bound

    def _looks_like_boundary(self, stripped: str, raw_line: str) -> bool:
        lowered = stripped.lower()
        if any(lowered.startswith(keyword) for keyword in self._boundary_keywords):
            return True
        if stripped.endswith((":", "{", "}")):
            return True

        leading_whitespace = len(raw_line) - len(raw_line.lstrip())
        return leading_whitespace == 0

    def _advance_cursor(self, start: int, end: int, total_lines: int) -> int:
        if end >= total_lines:
            return total_lines
        if self._overlap <= 0:
            return end
        next_cursor = end - self._overlap
        if next_cursor <= start:
            next_cursor = end
        return max(0, next_cursor)

    @staticmethod
    def _count_lines(content: str) -> int:
        return len(content.splitlines())

    @staticmethod
    def _build_chunk_id(path: Path, index: int) -> str:
        safe_path = path.as_posix()
        return f"{safe_path}::chunk-{index:04d}"

    @staticmethod
    def _build_progress() -> Progress:
        return Progress(
            SpinnerColumn(style="green"),
            TextColumn("{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
        )

    @staticmethod
    def _trim_surrounding_newlines(text: str) -> str:
        """Remove only leading/trailing empty lines while preserving indentation."""
        start = 0
        end = len(text)
        while start < end and text[start] == "\n":
            start += 1
        while end > start and text[end - 1] == "\n":
            end -= 1
        return text[start:end]


def chunk_preprocessed_files(
    files: Sequence[PreprocessedArtifact],
    *,
    settings: Settings | None = None,
) -> list[ChunkedFile]:
    """Convenience function mirroring :func:`preprocess_changed_files`."""
    chunker = CodeChunker(settings=settings)
    return chunker.run(files)


