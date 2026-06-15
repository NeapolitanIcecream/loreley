from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import loreley.core.map_elites.code_embedding as code_embedding_module
from loreley.config import Settings
from loreley.core.map_elites.chunk import ChunkedFile, FileChunk
from loreley.core.map_elites.code_embedding import CodeEmbedder
from loreley.core.openai_auth import DynamicOpenAIKeyUnavailableError


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


def test_prepare_chunk_payloads_includes_all_chunks(settings: Settings) -> None:
    embedder = CodeEmbedder(settings=settings, client=None)

    chunked = _make_chunked_file(chunk_count=4)
    payloads, lookup = embedder._prepare_chunk_payloads([chunked])

    assert len(payloads) == 4
    assert payloads[0].chunk_id.endswith("0000")
    assert payloads[1].chunk_id.endswith("0001")
    assert payloads[2].chunk_id.endswith("0002")
    assert payloads[3].chunk_id.endswith("0003")
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


def test_embed_batch_records_usage_after_validation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dims = int(settings.mapelites_code_embedding_dimensions or 0)
    response = SimpleNamespace(
        model=settings.mapelites_code_embedding_model,
        usage=SimpleNamespace(prompt_tokens=11, total_tokens=11),
        data=[SimpleNamespace(index=0, embedding=[1.0] * dims)],
    )
    recorded: list[object] = []

    class _FakeEmbeddings:
        def create(self, *, model, input, dimensions):  # type: ignore[no-untyped-def]
            return response

    class _FakeClient:
        embeddings = _FakeEmbeddings()

    def _record(event: object, **_kwargs: object) -> int:
        recorded.append(event)
        return 1

    monkeypatch.setattr(code_embedding_module, "record_usage_event", _record)

    embedder = CodeEmbedder(settings=settings, client=_FakeClient())  # type: ignore[arg-type]

    vectors = embedder._embed_batch(["a"])

    assert vectors == [(1.0,) * dims]
    assert len(recorded) == 1


def test_embed_batch_does_not_record_usage_for_invalid_response(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dims = int(settings.mapelites_code_embedding_dimensions or 0)
    response = SimpleNamespace(
        model=settings.mapelites_code_embedding_model,
        usage=SimpleNamespace(prompt_tokens=11, total_tokens=11),
        data=[SimpleNamespace(index=0, embedding=[1.0] * dims)],
    )
    recorded: list[object] = []

    class _FakeEmbeddings:
        def create(self, *, model, input, dimensions):  # type: ignore[no-untyped-def]
            return response

    class _FakeClient:
        embeddings = _FakeEmbeddings()

    def _record(event: object, **_kwargs: object) -> int:
        recorded.append(event)
        return 1

    monkeypatch.setattr(code_embedding_module, "record_usage_event", _record)

    embedder = CodeEmbedder(settings=settings, client=_FakeClient())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="missing indices"):
        embedder._embed_batch(["a", "b"])

    assert recorded == []


def test_embed_batch_supports_local_hash_model(
    settings: Settings,
    captured_logs: list[dict[str, Any]],
) -> None:
    settings.mapelites_code_embedding_model = "local-hash-v1"
    embedder = CodeEmbedder(settings=settings, client=None)

    vectors = embedder._embed_batch(["alpha", "beta"])

    assert len(vectors) == 2
    assert len(vectors[0]) == int(settings.mapelites_code_embedding_dimensions or 0)
    assert vectors[0] != vectors[1]
    warn_logs = [
        record
        for record in captured_logs
        if record["module"] == "map_elites.code_embedding" and record["level"] == "WARNING"
    ]
    assert any("local-hash embeddings" in str(record["message"]) for record in warn_logs)


def test_embed_batch_rebuilds_client_with_current_runtime_api_key(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_api_keys: list[str] = []
    responses = [
        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0] * 8)]),
        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[2.0] * 8)]),
    ]

    class _FakeEmbeddings:
        def create(self, *, model, input, dimensions):  # type: ignore[no-untyped-def]
            assert model == settings.mapelites_code_embedding_model
            assert dimensions == 8
            return responses.pop(0)

    api_keys = iter(["dyn-1", "dyn-2"])
    monkeypatch.setattr(
        code_embedding_module,
        "get_internal_openai_api_key",
        lambda _settings: next(api_keys),
    )
    monkeypatch.setattr(
        code_embedding_module,
        "OpenAI",
        lambda **kwargs: seen_api_keys.append(str(kwargs["api_key"])) or SimpleNamespace(
            embeddings=_FakeEmbeddings()
        ),
    )

    embedder = CodeEmbedder(settings=settings, client=None)

    embedder._embed_batch(["chunk-a"])
    embedder._embed_batch(["chunk-b"])

    assert seen_api_keys == ["dyn-1", "dyn-2"]


def test_embed_batch_retries_when_runtime_api_key_lookup_transiently_fails(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.mapelites_code_embedding_max_retries = 2
    settings.mapelites_code_embedding_retry_backoff_seconds = 0

    lookup_calls = {"count": 0}
    seen_api_keys: list[str] = []

    class _FakeEmbeddings:
        def create(self, *, model, input, dimensions):  # type: ignore[no-untyped-def]
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0] * 8)])

    def flaky_lookup(_settings):  # noqa: ANN001
        lookup_calls["count"] += 1
        if lookup_calls["count"] == 1:
            raise DynamicOpenAIKeyUnavailableError("provider unavailable")
        return "dyn-2"

    monkeypatch.setattr(code_embedding_module, "get_internal_openai_api_key", flaky_lookup)
    monkeypatch.setattr(
        code_embedding_module,
        "OpenAI",
        lambda **kwargs: seen_api_keys.append(str(kwargs["api_key"])) or SimpleNamespace(
            embeddings=_FakeEmbeddings()
        ),
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    embedder = CodeEmbedder(settings=settings, client=None)

    vectors = embedder._embed_batch(["chunk-a"])

    assert len(vectors) == 1
    assert lookup_calls["count"] == 2
    assert seen_api_keys == ["dyn-2"]


def test_embed_batch_retries_sdk_no_embedding_data_error_then_returns_embedding(
    settings: Settings,
) -> None:
    settings.mapelites_code_embedding_max_retries = 2
    settings.mapelites_code_embedding_retry_backoff_seconds = 0
    dims = int(settings.mapelites_code_embedding_dimensions or 0)
    calls = {"count": 0}

    class _FakeEmbeddings:
        def create(self, *, model, input, dimensions):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError("No embedding data received")
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0] * dims)])

    class _FakeClient:
        embeddings = _FakeEmbeddings()

    embedder = CodeEmbedder(settings=settings, client=_FakeClient())  # type: ignore[arg-type]

    vectors = embedder._embed_batch(["chunk-a"])

    assert vectors == [(1.0,) * dims]
    assert calls["count"] == 2


def test_embed_batch_retries_empty_embedding_data_until_exhausted_without_usage(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.mapelites_code_embedding_max_retries = 3
    settings.mapelites_code_embedding_retry_backoff_seconds = 0
    calls = {"count": 0}
    recorded: list[object] = []

    class _FakeEmbeddings:
        def create(self, *, model, input, dimensions):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            return SimpleNamespace(
                model=settings.mapelites_code_embedding_model,
                usage=SimpleNamespace(prompt_tokens=11, total_tokens=11),
                data=[],
            )

    class _FakeClient:
        embeddings = _FakeEmbeddings()

    def _record(event: object, **_kwargs: object) -> int:
        recorded.append(event)
        return 1

    monkeypatch.setattr(code_embedding_module, "record_usage_event", _record)

    embedder = CodeEmbedder(settings=settings, client=_FakeClient())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="no embedding data"):
        embedder._embed_batch(["chunk-a"])

    assert calls["count"] == 3
    assert recorded == []


def test_embed_batch_does_not_retry_unrelated_value_error(settings: Settings) -> None:
    settings.mapelites_code_embedding_max_retries = 3
    settings.mapelites_code_embedding_retry_backoff_seconds = 0
    calls = {"count": 0}

    class _FakeEmbeddings:
        def create(self, *, model, input, dimensions):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            raise ValueError("client-side input invariant failed")

    class _FakeClient:
        embeddings = _FakeEmbeddings()

    embedder = CodeEmbedder(settings=settings, client=_FakeClient())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="client-side input invariant failed"):
        embedder._embed_batch(["chunk-a"])

    assert calls["count"] == 1


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
