"""Compute commit-level code embeddings from chunked code artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Iterable, Sequence

from loguru import logger
import numpy as np
from openai import OpenAI, OpenAIError
from rich.progress import Progress, SpinnerColumn, TextColumn
from tenacity import RetryError

from loreley.config import Settings, get_settings
from loreley.core.openai_auth import DynamicOpenAIKeyUnavailableError, get_internal_openai_api_key
from loreley.core.openai_retry import openai_retrying, retry_error_details
from loreley.core.usage import normalize_openai_usage_event, record_usage_event
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
        self._model = self.settings.mapelites_code_embedding_model
        self._use_local_hash = str(self._model or "").strip().lower().startswith("local-hash")
        self._local_hash_warned = False
        self._dimensions = int(self.settings.mapelites_code_embedding_dimensions or 0)
        if self._dimensions <= 0:
            raise ValueError("MAPELITES_CODE_EMBEDDING_DIMENSIONS must be a positive integer.")
        self._batch_size = max(1, self.settings.mapelites_code_embedding_batch_size)
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
        if not inputs:
            return []
        if self._use_local_hash:
            if not self._local_hash_warned:
                log.warning(
                    "Using deterministic local-hash embeddings model={} dims={}",
                    self._model,
                    self._dimensions,
                )
                self._local_hash_warned = True
            return [self._hash_vector(text) for text in inputs]

        payload = list(inputs)
        retryer = openai_retrying(
            max_attempts=self._max_retries,
            backoff_seconds=self._retry_backoff,
            retry_on=(OpenAIError, DynamicOpenAIKeyUnavailableError),
            log=log,
            operation="Embedding batch",
        )
        try:
            for attempt in retryer:
                with attempt:
                    client = self._get_client()
                    response = client.embeddings.create(
                        model=self._model,
                        input=payload,
                        dimensions=self._dimensions,
                    )
                    vectors: list[Vector | None] = [None] * len(payload)
                    for item in response.data:
                        index = getattr(item, "index", None)
                        if index is None:
                            raise RuntimeError("Embedding response item is missing 'index'.")
                        if not isinstance(index, int):
                            raise RuntimeError(
                                f"Embedding response 'index' must be int, got {type(index)!r}."
                            )
                        if index < 0 or index >= len(payload):
                            raise RuntimeError(
                                "Embedding response 'index' out of range: "
                                f"{index} for payload size {len(payload)}."
                            )
                        if vectors[index] is not None:
                            raise RuntimeError(f"Duplicate embedding index returned: {index}.")
                        vectors[index] = tuple(item.embedding)

                    missing = [idx for idx, vector in enumerate(vectors) if vector is None]
                    if missing:
                        raise RuntimeError(f"Embedding response missing indices: {missing}.")

                    ordered_vectors = [vector for vector in vectors if vector is not None]
                    self._record_usage(response)
                    return ordered_vectors
        except RetryError as exc:
            attempts, last_exc = retry_error_details(exc, default_attempts=self._max_retries)
            log.error("Embedding batch failed after {} attempts: {}", attempts, last_exc)
            if last_exc is not None:
                raise last_exc
            raise
        raise RuntimeError("Embedding batch retry loop terminated without a result.")

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

    def _hash_vector(self, text: str) -> Vector:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        width = max(1, self._dimensions)
        return tuple(((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(width))

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
        if len(weights) != len(vectors):
            raise ValueError("Embedding aggregation requires one weight per vector.")

        dims = len(vectors[0])
        try:
            matrix = np.asarray(vectors, dtype=np.float64)
        except ValueError as exc:
            raise ValueError("Embedding dimension mismatch during aggregation") from exc

        if matrix.ndim != 2 or matrix.shape[1] != dims:
            raise ValueError("Embedding dimension mismatch during aggregation")
        if dims == 0:
            return ()

        weight_array = np.asarray(weights, dtype=np.float64)
        positive_mask = weight_array > 0.0
        if not np.any(positive_mask):
            mean_vector = matrix.mean(axis=0)
            return tuple(float(value) for value in mean_vector.tolist())

        positive_weights = weight_array[positive_mask]
        positive_matrix = matrix[positive_mask]
        weight_sum = float(positive_weights.sum())
        if weight_sum <= 0.0:
            mean_vector = matrix.mean(axis=0)
            return tuple(float(value) for value in mean_vector.tolist())

        weighted_totals = positive_matrix.T @ positive_weights
        return tuple(float(value) for value in (weighted_totals / weight_sum).tolist())

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
            disable=not sys.stdout.isatty(),
        )

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client
        client_kwargs: dict[str, object] = {}
        api_key = get_internal_openai_api_key(self.settings)
        if api_key:
            client_kwargs["api_key"] = api_key
        if self.settings.openai_base_url:
            client_kwargs["base_url"] = self.settings.openai_base_url
        return OpenAI(**client_kwargs) if client_kwargs else OpenAI()

    def _record_usage(self, response: object) -> None:
        event = normalize_openai_usage_event(
            response,
            phase="embedding",
            model=self._model,
            api_surface="embeddings",
            settings=self.settings,
        )
        record_usage_event(event, settings=self.settings)


def embed_chunked_files(
    chunked_files: Sequence[ChunkedFile],
    *,
    settings: Settings | None = None,
    client: OpenAI | None = None,
) -> CommitCodeEmbedding | None:
    """Convenience wrapper for :class:`CodeEmbedder`."""
    embedder = CodeEmbedder(settings=settings, client=client)
    return embedder.run(chunked_files)
