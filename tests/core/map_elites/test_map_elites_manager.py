from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

import loreley.core.map_elites.map_elites as map_elites_module
from loreley.config import Settings
from loreley.core.map_elites.chunk import ChunkedFile, FileChunk
from loreley.core.map_elites.code_embedding import CommitCodeEmbedding
from loreley.core.map_elites.dimension_reduction import FinalEmbedding, PenultimateEmbedding
from loreley.core.map_elites.map_elites import MapElitesManager, MapElitesRecord
from loreley.core.map_elites.preprocess import PreprocessedFile
from loreley.core.map_elites.summarization_embedding import CommitSummaryEmbedding


def test_ingest_short_circuits_when_no_preprocessed(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(map_elites_module, "preprocess_changed_files", lambda *args, **kwargs: [])

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    result = manager.ingest(
        commit_hash="abc",
        changed_files=[{"path": "a.py", "change_count": 1}],
    )

    assert result.status == 0
    assert result.record is None
    assert result.artifacts.preprocessed_files == ()
    assert "No eligible files" in (result.message or "")


def test_ingest_builds_record_with_stubbed_dependencies(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_feature_clip = True
    settings.mapelites_feature_truncation_k = 1.0
    settings.mapelites_fitness_metric = "score"

    preprocessed = PreprocessedFile(
        path=Path("a.py"),
        change_count=2,
        content="print('a')",
    )
    chunk = FileChunk(
        file_path=Path("a.py"),
        chunk_id="a.py::chunk-0000",
        index=0,
        start_line=1,
        end_line=1,
        content="print('a')",
        line_count=1,
        change_count=preprocessed.change_count,
    )
    chunked = ChunkedFile(
        path=Path("a.py"),
        change_count=preprocessed.change_count,
        total_lines=1,
        chunks=(chunk,),
    )

    code_embedding = CommitCodeEmbedding(
        files=(),
        vector=(0.5, -0.5),
        model="code",
        dimensions=2,
    )
    summary_embedding = CommitSummaryEmbedding(
        summaries=(),
        vector=(0.2, 0.3),
        summary_model="summary",
        embedding_model="embed",
        dimensions=2,
    )
    penultimate = PenultimateEmbedding(
        commit_hash="abc",
        vector=(0.1, 0.2, 0.3, 0.4),
        code_dimensions=2,
        summary_dimensions=2,
        code_model="code",
        summary_model="summary",
        summary_embedding_model="embed",
    )
    final_embedding = FinalEmbedding(
        commit_hash="abc",
        vector=(0.2, 0.8),
        dimensions=2,
        penultimate=penultimate,
        projection=None,
    )

    monkeypatch.setattr(
        map_elites_module, "preprocess_changed_files", lambda *args, **kwargs: [preprocessed]
    )
    monkeypatch.setattr(
        map_elites_module, "chunk_preprocessed_files", lambda *args, **kwargs: [chunked]
    )
    monkeypatch.setattr(
        map_elites_module, "embed_chunked_files", lambda *args, **kwargs: code_embedding
    )
    monkeypatch.setattr(
        map_elites_module, "summarize_preprocessed_files", lambda *args, **kwargs: summary_embedding
    )
    monkeypatch.setattr(
        map_elites_module,
        "reduce_commit_embeddings",
        lambda **kwargs: (final_embedding, (penultimate,), None),
    )

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    monkeypatch.setattr(manager, "_persist_island_state", lambda *args, **kwargs: None)

    captured: dict[str, object] = {}

    def _fake_add_to_archive(
        *,
        state: object,
        island_id: str,
        commit_hash: str,
        fitness: float,
        measures: np.ndarray,
        metadata: Mapping[str, object],
    ) -> tuple[int, float, MapElitesRecord]:
        captured["measures"] = measures
        captured["fitness"] = fitness
        record = MapElitesRecord(
            commit_hash=commit_hash,
            island_id=island_id,
            cell_index=0,
            fitness=fitness,
            measures=tuple(measures.tolist()),
            solution=tuple(measures.tolist()),
            metadata=metadata,
            timestamp=123.0,
        )
        return 1, 0.1, record

    monkeypatch.setattr(manager, "_add_to_archive", _fake_add_to_archive)

    result = manager.ingest(
        commit_hash="abc",
        changed_files=[{"path": "a.py", "change_count": 2}],
        metrics={"score": 1.2},
    )

    assert result.inserted
    assert captured["fitness"] == 1.2
    assert captured["measures"] is not None
    assert tuple(captured["measures"].tolist()) == pytest.approx((0.6, 0.9))  # type: ignore[index]
    assert result.record is not None
    assert result.record.commit_hash == "abc"
    assert result.artifacts.code_embedding is code_embedding
    assert result.artifacts.summary_embedding is summary_embedding
    assert result.artifacts.final_embedding is final_embedding
    assert result.record.metadata["metrics"]["score"] == 1.2
