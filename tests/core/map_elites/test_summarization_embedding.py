from __future__ import annotations

from pathlib import Path

import pytest

from loreley.config import Settings
from loreley.core.map_elites.preprocess import PreprocessedFile
from loreley.core.map_elites.summarization_embedding import SummaryEmbedder


class _DummyProgress:
    def __enter__(self) -> "_DummyProgress":
        return self

    def __exit__(self, *args, **kwargs) -> None:  # pragma: no cover - no cleanup needed
        return None

    def add_task(self, *args, **kwargs) -> int:
        return 1

    def update(self, *args, **kwargs) -> None:
        return None


def test_build_summary_prompt_truncates_long_excerpt(settings: Settings) -> None:
    settings.mapelites_summary_source_char_limit = 600
    embedder = SummaryEmbedder(settings=settings, client=None)

    long_text = "a" * 700 + "TAIL"
    prompt = embedder._build_summary_prompt(Path("foo.py"), 3, long_text)

    assert "File: foo.py" in prompt
    assert "...\n" in prompt
    assert long_text[:10] in prompt
    assert "TAIL" in prompt


def test_run_produces_commit_embedding(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    embedder = SummaryEmbedder(settings=settings, client=None)
    files = [
        PreprocessedFile(path=Path("a.py"), change_count=2, content="print('a')"),
        PreprocessedFile(path=Path("b.py"), change_count=0, content="print('b')"),
    ]

    monkeypatch.setattr(embedder, "_build_progress", lambda: _DummyProgress())
    monkeypatch.setattr(
        embedder,
        "_call_summary_model",
        lambda path, change_count, excerpt: f"{path.name} summary {change_count}",
    )

    def _fake_embed_batch(inputs: list[str]) -> list[tuple[float, float]]:
        return [(float(idx + 1), float(idx + 2)) for idx, _ in enumerate(inputs)]

    monkeypatch.setattr(embedder, "_embed_batch", _fake_embed_batch)

    result = embedder.run(files)

    assert result is not None
    assert result.summary_model == settings.mapelites_summary_model
    assert result.embedding_model == settings.mapelites_summary_embedding_model
    assert len(result.summaries) == 2

    # weight = change_count when present; otherwise falls back to word count
    expected_vector = (
        (1.0 * 2 + 2.0 * 3) / 5,
        (2.0 * 2 + 3.0 * 3) / 5,
    )
    assert result.vector == pytest.approx(expected_vector)
