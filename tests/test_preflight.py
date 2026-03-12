from __future__ import annotations

from tests.support import TestSettings
from loreley.preflight import (
    _check_openai_api_key_for_scheduler,
    _check_openai_api_key_for_worker,
    check_embedding_dimensions,
)


def test_check_embedding_dimensions_missing(monkeypatch) -> None:
    monkeypatch.delenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", raising=False)
    settings = TestSettings()
    result = check_embedding_dimensions(settings)
    assert result.status == "fail"


def test_check_embedding_dimensions_positive(monkeypatch) -> None:
    monkeypatch.setenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", "8")
    settings = TestSettings()
    result = check_embedding_dimensions(settings)
    assert result.status == "ok"


def test_openai_key_not_required_for_local_hash_scheduler(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LORELEY_LLM_API_KEY", raising=False)
    monkeypatch.setenv("MAPELITES_CODE_EMBEDDING_MODEL", "local-hash-v1")
    settings = TestSettings()

    result = _check_openai_api_key_for_scheduler(settings)

    assert result.status == "ok"
    assert "local-hash embeddings" in result.details


def test_openai_key_not_required_for_local_hash_worker_when_trajectory_disabled(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LORELEY_LLM_API_KEY", raising=False)
    monkeypatch.setenv("MAPELITES_CODE_EMBEDDING_MODEL", "local-hash-v1")
    monkeypatch.setenv("WORKER_PLANNING_TRAJECTORY_MAX_CHUNKS", "0")
    settings = TestSettings()

    result = _check_openai_api_key_for_worker(settings)

    assert result.status == "ok"
    assert "trajectory summarization is disabled" in result.details

