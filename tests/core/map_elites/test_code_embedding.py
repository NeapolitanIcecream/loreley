from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import loreley.core.map_elites.code_embedding as code_embedding_module
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


def test_weighted_average_raises_when_weight_count_mismatch() -> None:
    vectors = [(1.0, 2.0), (3.0, 4.0)]

    with pytest.raises(ValueError, match="one weight per vector"):
        CodeEmbedder._weighted_average(vectors, [1.0])


def test_weighted_average_raises_on_dimension_mismatch() -> None:
    vectors = [(1.0, 2.0), (3.0,)]
    weights = [1.0, 1.0]

    with pytest.raises(ValueError, match="dimension mismatch"):
        CodeEmbedder._weighted_average(vectors, weights)


def test_build_progress_disables_render_when_stdout_is_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStdout:
        @staticmethod
        def isatty() -> bool:
            return False

    monkeypatch.setattr(code_embedding_module.sys, "stdout", _FakeStdout())
    progress = CodeEmbedder._build_progress()

    assert progress.disable is True


def test_embed_batch_aligns_vectors_by_response_index(settings: Settings) -> None:
    dims = int(settings.mapelites_code_embedding_dimensions or 0)

    response = SimpleNamespace(
        data=[
            SimpleNamespace(index=2, embedding=[2.0] * dims),
            SimpleNamespace(index=0, embedding=[0.0] * dims),
            SimpleNamespace(index=1, embedding=[1.0] * dims),
        ],
    )

    class _FakeEmbeddings:
        def __init__(self, response_obj):  # type: ignore[no-untyped-def]
            self._response = response_obj

        def create(self, *, model, input, dimensions):  # type: ignore[no-untyped-def]
            assert model == settings.mapelites_code_embedding_model
            assert input == ["a", "b", "c"]
            assert dimensions == dims
            return self._response

    class _FakeClient:
        def __init__(self, response_obj):  # type: ignore[no-untyped-def]
            self.embeddings = _FakeEmbeddings(response_obj)

    embedder = CodeEmbedder(settings=settings, client=_FakeClient(response))  # type: ignore[arg-type]

    vectors = embedder._embed_batch(["a", "b", "c"])

    assert vectors == [
        (0.0,) * dims,
        (1.0,) * dims,
        (2.0,) * dims,
    ]


# ---------------------------------------------------------------------------
# Observability: failure signal tests
# ---------------------------------------------------------------------------


def test_embedder_returns_none_and_logs_info_when_no_files_supplied(
    settings: Settings,
    captured_logs: list[dict[str, Any]],
) -> None:
    """Observability: embedder logs INFO when invoked with an empty file list."""
    embedder = CodeEmbedder(settings=settings, client=None)

    result = embedder.run([])

    assert result is None
    info_logs = [
        r
        for r in captured_logs
        if r["module"] == "map_elites.code_embedding" and r["level"] == "INFO"
    ]
    assert any("No chunked files" in str(r["message"]) for r in info_logs)


def test_embedder_logs_warning_when_chunk_embedding_produces_no_results(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    captured_logs: list[dict[str, Any]],
) -> None:
    """Observability: WARNING is emitted when the embedding step yields no vectors."""
    chunked = _make_chunked_file(chunk_count=2, change_count=3)
    embedder = CodeEmbedder(settings=settings, client=None)

    # Patch _embed_chunks (not _embed_batch) to return empty, simulating a
    # scenario where the embedding API returns nothing actionable.
    monkeypatch.setattr(embedder, "_embed_chunks", lambda _chunks: [])

    result = embedder.run([chunked])

    assert result is None
    warn_logs = [
        r
        for r in captured_logs
        if r["level"] == "WARNING" and r["module"] == "map_elites.code_embedding"
    ]
    assert any("no results" in str(r["message"]).lower() for r in warn_logs)
