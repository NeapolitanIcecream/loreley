from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from loreley.config import Settings
from loreley.core.map_elites.chunk import CodeChunker, chunk_preprocessed_files


def make_chunker(settings: Settings | None = None) -> CodeChunker:
    test_settings = settings or Settings()
    return CodeChunker(settings=test_settings)


def test_looks_like_boundary_heuristics(settings: Settings) -> None:
    chunker = make_chunker(settings)

    assert chunker._looks_like_boundary("def foo():", "def foo():")  # type: ignore[attr-defined]
    assert chunker._looks_like_boundary("class Bar:", "class Bar:")  # type: ignore[attr-defined]
    assert chunker._looks_like_boundary("something:", "something:")  # type: ignore[attr-defined]

    # A normally indented line that is not flush-left should not be treated as a boundary
    assert not chunker._looks_like_boundary("x = 1", "    x = 1")  # type: ignore[attr-defined]


def test_chunk_file_single_chunk_no_overlap(settings: Settings) -> None:
    settings.mapelites_chunk_min_lines = 1
    settings.mapelites_chunk_target_lines = 100
    settings.mapelites_chunk_overlap_lines = 0
    settings.mapelites_chunk_max_chunks_per_file = 16

    chunker = make_chunker(settings)

    content = "\n".join(f"line {i}" for i in range(1, 11))
    artifact = SimpleNamespace(path=Path("file.py"), change_count=3, content=content)

    chunks = chunker._chunk_file(artifact)  # type: ignore[attr-defined]
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start_line == 1
    assert chunk.end_line == 10
    assert chunk.line_count == 10
    assert chunk.chunk_id.endswith("file.py::chunk-0000")


def test_chunk_file_overlap_and_max_chunks(settings: Settings) -> None:
    settings.mapelites_chunk_min_lines = 1
    settings.mapelites_chunk_target_lines = 4
    settings.mapelites_chunk_overlap_lines = 2
    settings.mapelites_chunk_max_chunks_per_file = 3

    chunker = make_chunker(settings)

    lines = [f"line {i}" for i in range(1, 13)]
    content = "\n".join(lines)
    artifact = SimpleNamespace(path=Path("file.py"), change_count=1, content=content)

    chunks = chunker._chunk_file(artifact)  # type: ignore[attr-defined]

    assert 1 <= len(chunks) <= settings.mapelites_chunk_max_chunks_per_file

    for first, second in zip(chunks, chunks[1:]):
        # Validate overlapping window logic: each subsequent chunk must start on or before the previous chunk's end line
        assert second.start_line <= second.end_line
        assert second.start_line <= first.end_line


def test_run_and_wrapper_chunk_preprocessed_files(settings: Settings) -> None:
    settings.mapelites_chunk_min_lines = 1
    settings.mapelites_chunk_target_lines = 4
    settings.mapelites_chunk_overlap_lines = 0
    settings.mapelites_chunk_max_chunks_per_file = 8

    content1 = "line1\nline2\nline3\nline4\nline5"
    content2 = "a\nb\nc"
    artifacts = [
        SimpleNamespace(path=Path("a.py"), change_count=2, content=content1),
        SimpleNamespace(path=Path("b.py"), change_count=1, content=content2),
    ]

    chunked = chunk_preprocessed_files(artifacts, settings=settings)
    assert len(chunked) == 2

    total_lines = sum(entry.total_lines for entry in chunked)
    assert total_lines == len(content1.splitlines()) + len(content2.splitlines())


