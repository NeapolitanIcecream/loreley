"""Generate structured summaries and embeddings for preprocessed code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Sequence

from loguru import logger
from openai import OpenAI, OpenAIError
from rich.progress import Progress, SpinnerColumn, TextColumn

from app.config import Settings, get_settings
from .preprocess import PreprocessedFile

log = logger.bind(module="map_elites.summarization_embedding")

__all__ = [
    "FileSummary",
    "SummaryEmbedding",
    "CommitSummaryEmbedding",
    "SummaryEmbedder",
    "summarize_preprocessed_files",
]

Vector = tuple[float, ...]

_SUMMARY_INSTRUCTIONS = (
    "You are a senior code reviewer helping an evolutionary search agent. "
    "Summarize each file into concise markdown with the following sections in order:\n"
    "1. Overview – single sentence describing responsibility.\n"
    "2. Key Elements – bullet list of functions, classes, or flows.\n"
    "3. Recent Changes – bullet list referencing impact of provided change count; "
    "write 'None' if unknown.\n"
    "4. Risks & Opportunities – bullet list of potential follow-up ideas.\n"
    "Keep the response under 180 words and avoid verbatim quoting of the code."
)


@dataclass(slots=True, frozen=True)
class FileSummary:
    """Structured summary for a single file."""

    path: Path
    change_count: int
    summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(slots=True, frozen=True)
class SummaryEmbedding:
    """Embedding derived from a file summary."""

    file_summary: FileSummary
    vector: Vector
    weight: float


@dataclass(slots=True, frozen=True)
class CommitSummaryEmbedding:
    """Commit-level embedding composed from file summaries."""

    summaries: tuple[SummaryEmbedding, ...]
    vector: Vector
    summary_model: str
    embedding_model: str
    dimensions: int

    @property
    def file_count(self) -> int:
        return len(self.summaries)


class SummaryEmbedder:
    """Generate structured summaries and embed them into a single vector."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or OpenAI()
        self._summary_model = self.settings.mapelites_summary_model
        self._summary_temperature = self.settings.mapelites_summary_temperature
        self._summary_max_tokens = max(
            32,
            self.settings.mapelites_summary_max_output_tokens,
        )
        self._source_char_limit = max(
            512,
            self.settings.mapelites_summary_source_char_limit,
        )
        self._summary_max_retries = max(1, self.settings.mapelites_summary_max_retries)
        self._summary_retry_backoff = max(
            0.0, self.settings.mapelites_summary_retry_backoff_seconds
        )
        self._embedding_model = self.settings.mapelites_summary_embedding_model
        self._embedding_dimensions = (
            self.settings.mapelites_summary_embedding_dimensions
        )
        self._embedding_batch_size = max(
            1,
            self.settings.mapelites_summary_embedding_batch_size,
        )

    def run(self, files: Sequence[PreprocessedFile]) -> CommitSummaryEmbedding | None:
        """Return commit-level summary embedding for supplied files."""
        if not files:
            log.info("Summary embedder received no files; skipping.")
            return None

        summaries = self._summarize_files(files)
        if not summaries:
            log.warning("No summaries were generated.")
            return None

        embeddings = self._embed_summaries(summaries)
        if not embeddings:
            log.warning("Summary embeddings could not be computed.")
            return None

        embeddings.sort(key=lambda entry: entry.file_summary.path.as_posix())
        commit_vector = self._weighted_average(
            [entry.vector for entry in embeddings],
            [entry.weight for entry in embeddings],
        )
        if not commit_vector:
            log.warning("Commit summary aggregation produced an empty vector.")
            return None

        return CommitSummaryEmbedding(
            summaries=tuple(embeddings),
            vector=commit_vector,
            summary_model=self._summary_model,
            embedding_model=self._embedding_model,
            dimensions=len(commit_vector),
        )

    def _summarize_files(self, files: Sequence[PreprocessedFile]) -> list[FileSummary]:
        progress = self._build_progress()
        summaries: list[FileSummary] = []

        with progress:
            task_id = progress.add_task(
                "[cyan]Summarizing files",
                total=len(files),
            )
            for file in files:
                if not file.content.strip():
                    log.debug("File {} empty after preprocessing; skipping summary.", file.path)
                    progress.update(task_id, advance=1)
                    continue

                summary_text = self._call_summary_model(file.path, file.change_count, file.content)
                if summary_text:
                    summaries.append(
                        FileSummary(
                            path=file.path,
                            change_count=file.change_count,
                            summary=summary_text,
                        )
                    )
                progress.update(task_id, advance=1)

        return summaries

    def _call_summary_model(self, path: Path, change_count: int, excerpt: str) -> str | None:
        payload = self._build_summary_prompt(path, change_count, excerpt)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._client.responses.create(
                    model=self._summary_model,
                    instructions=_SUMMARY_INSTRUCTIONS,
                    input=payload,
                    temperature=self._summary_temperature,
                    max_output_tokens=self._summary_max_tokens,
                )
                text = (response.output_text or "").strip()
                if not text:
                    log.warning("Summary model returned empty output for {}", path)
                    return None
                return text
            except OpenAIError as exc:
                if attempt >= self._summary_max_retries:
                    log.error(
                        "Summarization failed for {} after {} attempts: {}",
                        path,
                        attempt,
                        exc,
                    )
                    return None
                delay = self._summary_retry_backoff * attempt
                log.warning(
                    "Summarization attempt {} for {} failed: {}. Retrying in {:.1f}s",
                    attempt,
                    path,
                    exc,
                    delay,
                )
                time.sleep(delay)

    def _embed_summaries(self, summaries: Sequence[FileSummary]) -> list[SummaryEmbedding]:
        progress = self._build_progress()
        embeddings: list[SummaryEmbedding] = []

        with progress:
            task_id = progress.add_task(
                "[cyan]Embedding summaries",
                total=len(summaries),
            )
            for batch in self._batched(summaries, self._embedding_batch_size):
                vectors = self._embed_batch([entry.summary for entry in batch])
                if len(vectors) != len(batch):
                    log.error(
                        "Embedding API returned %s vectors for %s summaries",
                        len(vectors),
                        len(batch),
                    )
                    raise RuntimeError("Summary embedding response/input mismatch")
                for summary, vector in zip(batch, vectors):
                    embeddings.append(
                        SummaryEmbedding(
                            file_summary=summary,
                            vector=vector,
                            weight=self._summary_weight(summary),
                        )
                    )
                progress.update(task_id, advance=len(batch))

        return embeddings

    def _embed_batch(self, inputs: Sequence[str]) -> list[Vector]:
        payload = list(inputs)
        attempt = 0
        while True:
            attempt += 1
            try:
                if self._embedding_dimensions:
                    response = self._client.embeddings.create(
                        model=self._embedding_model,
                        input=payload,
                        dimensions=self._embedding_dimensions,
                    )
                else:
                    response = self._client.embeddings.create(
                        model=self._embedding_model,
                        input=payload,
                    )
                return [tuple(item.embedding) for item in response.data]
            except OpenAIError as exc:
                if attempt >= self._summary_max_retries:
                    log.error(
                        "Summary embedding batch failed after {} attempts: {}",
                        attempt,
                        exc,
                    )
                    raise
                delay = self._summary_retry_backoff * attempt
                log.warning(
                    "Summary embedding batch failed on attempt {}: {}. Retrying in {:.1f}s",
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)

    def _summary_weight(self, summary: FileSummary) -> float:
        if summary.change_count > 0:
            return float(summary.change_count)
        return float(max(len(summary.summary.split()), 1))

    def _build_summary_prompt(self, path: Path, change_count: int, excerpt: str) -> str:
        relative = path.as_posix()
        limited = excerpt
        if len(limited) > self._source_char_limit:
            head_len = int(self._source_char_limit * 0.7)
            tail_len = self._source_char_limit - head_len
            limited = f"{excerpt[:head_len].rstrip()}\n...\n{excerpt[-tail_len:].lstrip()}"
        return (
            f"File: {relative}\n"
            f"Estimated changed lines: {change_count}\n"
            "Code excerpt:\n"
            "```code\n"
            f"{limited}\n"
            "```\n"
            "Produce the structured summary now."
        )

    @staticmethod
    def _weighted_average(
        vectors: Sequence[Vector],
        weights: Sequence[float],
    ) -> Vector:
        if not vectors:
            return ()
        dims = len(vectors[0])
        totals = [0.0] * dims
        weight_sum = 0.0
        for vector, weight in zip(vectors, weights):
            if weight <= 0:
                continue
            if len(vector) != dims:
                raise ValueError("Summary embedding dimension mismatch during aggregation")
            for idx in range(dims):
                totals[idx] += vector[idx] * weight
            weight_sum += weight
        if weight_sum == 0.0:
            weight_sum = float(len(vectors))
            totals = [
                sum(vector[idx] for vector in vectors)
                for idx in range(dims)
            ]
        return tuple(value / weight_sum for value in totals)

    @staticmethod
    def _batched(sequence: Sequence[FileSummary], batch_size: int) -> Iterable[Sequence[FileSummary]]:
        for start in range(0, len(sequence), batch_size):
            yield sequence[start : start + batch_size]

    @staticmethod
    def _build_progress() -> Progress:
        return Progress(
            SpinnerColumn(style="green"),
            TextColumn("{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
        )


def summarize_preprocessed_files(
    files: Sequence[PreprocessedFile],
    *,
    settings: Settings | None = None,
    client: OpenAI | None = None,
) -> CommitSummaryEmbedding | None:
    """Convenience wrapper mirroring :func:`chunk_preprocessed_files`."""
    embedder = SummaryEmbedder(settings=settings, client=client)
    return embedder.run(files)

