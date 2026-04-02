from __future__ import annotations

from tests.support import TestSettings
from loreley.preflight import (
    _check_openai_api_key_for_scheduler,
    _check_openai_api_key_for_worker,
    check_embedding_dimensions,
    preflight_worker,
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


def test_dynamic_provider_satisfies_scheduler_openai_auth(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LORELEY_LLM_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENAI_DYNAMIC_API_KEY_PROVIDER",
        "tests.support_dynamic_provider:token_provider",
    )
    monkeypatch.setenv("OPENAI_DYNAMIC_API_KEY_TTL_SECONDS", "600")
    settings = TestSettings()

    result = _check_openai_api_key_for_scheduler(settings)

    assert result.status == "ok"
    assert "dynamic provider" in result.details.lower()


def test_dynamic_provider_missing_ttl_fails_openai_auth(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LORELEY_LLM_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENAI_DYNAMIC_API_KEY_PROVIDER",
        "tests.support_dynamic_provider:token_provider",
    )
    monkeypatch.delenv("OPENAI_DYNAMIC_API_KEY_TTL_SECONDS", raising=False)
    settings = TestSettings()

    result = _check_openai_api_key_for_worker(settings)

    assert result.status == "fail"
    assert "OPENAI_DYNAMIC_API_KEY_TTL_SECONDS" in result.details


def test_dynamic_ttl_without_provider_fails_openai_auth(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LORELEY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_DYNAMIC_API_KEY_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_DYNAMIC_API_KEY_TTL_SECONDS", "600")
    settings = TestSettings()

    result = _check_openai_api_key_for_worker(settings)

    assert result.status == "fail"
    assert "OPENAI_DYNAMIC_API_KEY_PROVIDER" in result.details


def test_worker_preflight_warns_when_dynamic_ttl_is_shorter_than_kilocode_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OPENAI_DYNAMIC_API_KEY_PROVIDER",
        "tests.support_dynamic_provider:token_provider",
    )
    monkeypatch.setenv("OPENAI_DYNAMIC_API_KEY_TTL_SECONDS", "600")
    monkeypatch.setenv("WORKER_REPO_REMOTE_URL", "https://example.com/repo.git")
    monkeypatch.setenv("WORKER_EVALUATOR_PLUGIN", "tests.support_dynamic_provider:token_provider")
    settings = TestSettings()

    results = preflight_worker(settings)

    ttl_warnings = [item for item in results if item.name == "openai_dynamic_api_key_ttl"]
    assert ttl_warnings
    assert any(item.status == "warn" for item in ttl_warnings)
