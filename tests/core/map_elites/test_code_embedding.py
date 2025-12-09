from __future__ import annotations

from pathlib import Path

import pytest

from loreley.config import Settings
from loreley.core.map_elites.chunk import ChunkedFile, FileChunk
from loreley.core.map_elites.code_embedding import CodeEmbedder


class _DummyProgress:
    def __enter__(self) -> "_DummyProgress":
        return self

    def __exit__(self, *args, **kwargs) -> None:  # pragma: no cover - no cleanup needed
        return None

    def add_task(self, *args, **kwargs) -> int:
        return 1

    def update(self, *args, **kwargs) -> None:
        return None


def _make_chunked_file(chunk_count: int = 2, change_count: int = 3) -> ChunkedFile:
    chunks = []
    for idx in range(chunk_count):
        start = idx * 2 + 1
        end = start + 1
        chunks.append(
            FileChunk(
                file_path=Path("file.py"),
                chunk_id=f"file.py::chunk-{idx:04d}",
                index=idx,
                start_line=start,
                end_line=end,
                content=f"chunk {idx}",
                line_count=end - start + 1,
                change_count=change_count,
            )
        )
    return ChunkedFile(
        path=Path("file.py"),
        change_count=change_count,
        total_lines=len(chunks) * 2,
        chunks=tuple(chunks),
    )


def test_run_builds_commit_embedding(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    chunked = _make_chunked_file(chunk_count=2, change_count=3)
    embedder = CodeEmbedder(settings=settings, client=None)

    monkeypatch.setattr(embedder, "_build_progress", lambda: _DummyProgress())
    monkeypatch.setattr(embedder, "_embed_batch", lambda inputs: [(1.0, 0.0), (3.0, 2.0)])

    result = embedder.run([chunked])

    assert result is not None
    assert result.model == settings.mapelites_code_embedding_model
    assert len(result.files) == 1
    file_embedding = result.files[0]
    assert file_embedding.weight == float(chunked.change_count)
    # chunk weights default to line_count (=2), so this is the weighted average of (1,0) and (3,2)
    assert file_embedding.vector == pytest.approx((2.0, 1.0))
    assert result.vector == pytest.approx(file_embedding.vector)


def test_prepare_chunk_payloads_respects_max_chunks(settings: Settings) -> None:
    settings.mapelites_code_embedding_max_chunks_per_commit = 2
    embedder = CodeEmbedder(settings=settings, client=None)

    chunked = _make_chunked_file(chunk_count=4)
    payloads, lookup = embedder._prepare_chunk_payloads([chunked])

    assert len(payloads) == 2
    assert payloads[0].chunk_id.endswith("0000")
    assert payloads[1].chunk_id.endswith("0001")
    assert lookup[chunked.path] is chunked


def test_weighted_average_falls_back_to_mean_when_weights_zero() -> None:
    vectors = [(1.0, 2.0), (3.0, 4.0)]
    weights = [0.0, 0.0]

    result = CodeEmbedder._weighted_average(vectors, weights)

    assert result == (2.0, 3.0)
