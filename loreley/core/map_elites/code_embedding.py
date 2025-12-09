"""Compute commit-level code embeddings from chunked code artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Iterable, Sequence

from loguru import logger
from openai import OpenAI, OpenAIError
from rich.progress import Progress, SpinnerColumn, TextColumn

from loreley.config import Settings, get_settings
from .chunk import ChunkedFile, FileChunk


log = logger.bind(module="map_elites.code_embedding")

__all__ = [
    "ChunkEmbedding",
    "FileEmbedding",
    "CommitCodeEmbedding",
    "CodeEmbedder",
    "embed_chunked_files",
]

Vector = tuple[float, ...]


@dataclass(slots=True, frozen=True)
class ChunkEmbedding:
    """Embedding vector derived from a single chunk of code."""

    chunk: FileChunk
    vector: Vector
    weight: float


@dataclass(slots=True, frozen=True)
class FileEmbedding:
    """Aggregated embedding for one file."""

    file: ChunkedFile
    chunk_embeddings: tuple[ChunkEmbedding, ...]
    vector: Vector
    weight: float


@dataclass(slots=True, frozen=True)
class CommitCodeEmbedding:
    """Commit-level representation built from all files."""

    files: tuple[FileEmbedding, ...]
    vector: Vector
    model: str
    dimensions: int

    @property
    def chunk_count(self) -> int:
        return sum(len(file.chunk_embeddings) for file in self.files)


class CodeEmbedder:
    """Produce chunk, file and commit embeddings via the OpenAI API."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client: OpenAI | None = client
        self._client_factory: Callable[[], OpenAI] | None = None
        if client is None:
            client_kwargs: dict[str, object] = {}
            if self.settings.openai_api_key:
                client_kwargs["api_key"] = self.settings.openai_api_key
            if self.settings.openai_base_url:
                client_kwargs["base_url"] = self.settings.openai_base_url
            self._client_factory = lambda: (
                OpenAI(**client_kwargs)  # type: ignore[call-arg]
                if client_kwargs
                else OpenAI()
            )
        self._model = self.settings.mapelites_code_embedding_model
        self._dimensions = self.settings.mapelites_code_embedding_dimensions
        self._batch_size = max(1, self.settings.mapelites_code_embedding_batch_size)
        self._max_chunks = max(
            0,
            self.settings.mapelites_code_embedding_max_chunks_per_commit,
        )
        self._max_retries = max(1, self.settings.mapelites_code_embedding_max_retries)
        self._retry_backoff = max(
            0.0,
            self.settings.mapelites_code_embedding_retry_backoff_seconds,
        )

    def run(self, chunked_files: Sequence[ChunkedFile]) -> CommitCodeEmbedding | None:
        """Return commit-level embedding for the provided chunked files."""
        if not chunked_files:
            log.info("No chunked files supplied to embedder; skipping.")
            return None

        files_with_chunks = [entry for entry in chunked_files if entry.chunks]
        if not files_with_chunks:
            log.info("All chunked files are empty; skipping embedding.")
            return None

        chunk_payloads, chunk_owner_lookup = self._prepare_chunk_payloads(files_with_chunks)
        if not chunk_payloads:
            log.info("No chunk payloads collected for embedding.")
            return None

        chunk_embeddings = self._embed_chunks(chunk_payloads)
        if not chunk_embeddings:
            log.warning("Chunk embedding step produced no results.")
            return None

        file_embeddings = self._aggregate_file_embeddings(
            chunk_embeddings,
            chunk_owner_lookup,
        )
        if not file_embeddings:
            log.warning("Unable to aggregate file embeddings.")
            return None

        commit_vector = self._weighted_average(
            [file.vector for file in file_embeddings],
            [file.weight for file in file_embeddings],
        )
        if not commit_vector:
            log.warning("Commit-level aggregation resulted in an empty vector.")
            return None

        file_embeddings.sort(key=lambda entry: entry.file.path.as_posix())

        commit_embedding = CommitCodeEmbedding(
            files=tuple(file_embeddings),
            vector=commit_vector,
            model=self._model,
            dimensions=len(commit_vector),
        )
        log.info(
            "Computed commit code embedding: files={} chunks={} dims={}",
            len(file_embeddings),
            commit_embedding.chunk_count,
            commit_embedding.dimensions,
        )
        return commit_embedding

    def _prepare_chunk_payloads(
        self,
        files: Sequence[ChunkedFile],
    ) -> tuple[list[FileChunk], dict[Path, ChunkedFile]]:
        payloads: list[FileChunk] = []
        owner_lookup: dict[Path, ChunkedFile] = {file.path: file for file in files}
        for file in files:
            for chunk in file.chunks:
                payloads.append(chunk)
                if self._max_chunks and len(payloads) >= self._max_chunks:
                    log.debug(
                        "Hit chunk cap {} while preparing {}; remaining chunks dropped.",
                        self._max_chunks,
                        file.path,
                    )
                    return payloads, owner_lookup
        return payloads, owner_lookup

    def _embed_chunks(self, chunks: Sequence[FileChunk]) -> list[ChunkEmbedding]:
        progress = self._build_progress()
        embeddings: list[ChunkEmbedding] = []
        total = len(chunks)

        with progress:
            task_id = progress.add_task(
                "[cyan]Embedding code chunks",
                total=total,
            )
            for batch in self._batched(chunks, self._batch_size):
                vectors = self._embed_batch(tuple(chunk.content for chunk in batch))
                if len(vectors) != len(batch):
                    log.error(
                        "Embedding API returned %s vectors for %s inputs",
                        len(vectors),
                        len(batch),
                    )
                    raise RuntimeError("Embedding response/input size mismatch")
                for chunk, vector in zip(batch, vectors):
                    embeddings.append(
                        ChunkEmbedding(
                            chunk=chunk,
                            vector=vector,
                            weight=self._chunk_weight(chunk),
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
                client = self._get_client()
                if self._dimensions:
                    response = client.embeddings.create(
                        model=self._model,
                        input=payload,
                        dimensions=self._dimensions,
                    )
                else:
                    response = client.embeddings.create(
                        model=self._model,
                        input=payload,
                    )
                return [tuple(item.embedding) for item in response.data]
            except OpenAIError as exc:
                if attempt >= self._max_retries:
                    log.error("Embedding batch failed after {} attempts: {}", attempt, exc)
                    raise
                delay = self._retry_backoff * attempt
                log.warning(
                    "Embedding batch failed on attempt {}: {}. Retrying in {:.1f}s",
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)

    def _aggregate_file_embeddings(
        self,
        chunk_embeddings: Sequence[ChunkEmbedding],
        owner_lookup: dict[Path, ChunkedFile],
    ) -> list[FileEmbedding]:
        grouped: dict[Path, list[ChunkEmbedding]] = {}
        for embedding in chunk_embeddings:
            grouped.setdefault(embedding.chunk.file_path, []).append(embedding)

        file_embeddings: list[FileEmbedding] = []
        for path, embeddings in grouped.items():
            file = owner_lookup.get(path)
            if not file:
                log.warning("Missing chunk owner for %s; skipping aggregation", path)
                continue

            vector = self._weighted_average(
                [entry.vector for entry in embeddings],
                [entry.weight for entry in embeddings],
            )
            if not vector:
                continue

            chunk_sorted = tuple(sorted(embeddings, key=lambda entry: entry.chunk.index))
            file_weight = self._file_weight(file, chunk_sorted)
            file_embeddings.append(
                FileEmbedding(
                    file=file,
                    chunk_embeddings=chunk_sorted,
                    vector=vector,
                    weight=file_weight,
                )
            )

        return file_embeddings

    def _file_weight(self, file: ChunkedFile, chunks: Sequence[ChunkEmbedding]) -> float:
        if file.change_count > 0:
            return float(file.change_count)
        total = sum(chunk.weight for chunk in chunks)
        return float(total if total > 0 else file.total_lines or 1)

    @staticmethod
    def _chunk_weight(chunk: FileChunk) -> float:
        return float(max(chunk.line_count, 1))

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
                raise ValueError("Embedding dimension mismatch during aggregation")
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
    def _batched(sequence: Sequence[FileChunk], batch_size: int) -> Iterable[Sequence[FileChunk]]:
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

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if self._client_factory is None:
                raise RuntimeError("OpenAI client factory is not configured.")
            self._client = self._client_factory()
        return self._client


def embed_chunked_files(
    chunked_files: Sequence[ChunkedFile],
    *,
    settings: Settings | None = None,
    client: OpenAI | None = None,
) -> CommitCodeEmbedding | None:
    """Convenience wrapper for :class:`CodeEmbedder`."""
    embedder = CodeEmbedder(settings=settings, client=client)
    return embedder.run(chunked_files)

