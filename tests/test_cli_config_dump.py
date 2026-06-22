from __future__ import annotations

import json

import pytest

from loreley.cli import main
from tests.support import TestSettings


def _make_settings() -> TestSettings:
    return TestSettings(
        OPENAI_API_KEY="sk-test-secret",
        OPENAI_BASE_URL="https://gateway.example.com/v1",
        OPENAI_DYNAMIC_API_KEY_PROVIDER="tests.support_dynamic_provider:token_provider",
        OPENAI_DYNAMIC_API_KEY_TTL_SECONDS=600,
        OPENAI_DYNAMIC_API_KEY_REFRESH_SKEW_SECONDS=30,
        DB_PASSWORD="db-secret-password",
        DB_HOST="db.internal",
        DB_PORT=5433,
        DB_NAME="loreley_dev",
        TASKS_REDIS_PASSWORD="redis-secret",
        TASKS_REDIS_URL="redis://:redis-secret@redis.internal:6380/2",
        WORKER_KILOCODE_OPENAI_API_KEY="kilo-secret",
        WORKER_EVOLUTION_COMMIT_API_KEY="commit-secret",
        WORKER_EVOLUTION_COMMIT_BASE_URL="https://commit.example.com/v1",
        WORKER_REPO_REMOTE_URL="https://token@example.com/repo.git",
        LORELEY_AGENT_API_TOKEN="agent-secret",
        LORELEY_API_WRITE_TOKEN="write-secret",
    )


def test_config_dump_json_masks_secrets_by_default(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)

    code = main(["config", "dump", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["openai_api_key"] == "***"
    assert payload["openai_dynamic_api_key_provider"] == "tests.support_dynamic_provider:token_provider"
    assert payload["openai_dynamic_api_key_ttl_seconds"] == 600
    assert payload["openai_dynamic_api_key_refresh_skew_seconds"] == 30
    assert payload["tasks_redis_password"] == "***"
    assert payload["loreley_agent_api_token"] == "***"
    assert payload["loreley_api_write_token"] == "***"
    assert payload["database_dsn"] == "postgresql+psycopg://db.internal:5433/loreley_dev"
    assert payload["worker_repo_remote_url"] == "https://example.com/repo.git"


def test_config_dump_json_sanitizes_ipv6_urls(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    settings = TestSettings(
        OPENAI_API_KEY="sk-test-secret",
        OPENAI_BASE_URL="https://gateway.example.com/v1",
        DB_PASSWORD="db-secret-password",
        DB_HOST="db.internal",
        DB_PORT=5433,
        DB_NAME="loreley_dev",
        TASKS_REDIS_PASSWORD="redis-secret",
        TASKS_REDIS_URL="redis://:redis-secret@[::1]:6380/2",
        WORKER_KILOCODE_OPENAI_API_KEY="kilo-secret",
        WORKER_REPO_REMOTE_URL="https://token@[::1]:8443/repo.git",
    )
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)

    code = main(["config", "dump", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["tasks_redis_url"] == "redis://[::1]:6380/2"
    assert payload["worker_repo_remote_url"] == "https://[::1]:8443/repo.git"


def test_config_dump_json_can_disable_secret_masking(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)

    code = main(["config", "dump", "--json", "--no-mask-secrets"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["openai_api_key"] == "sk-test-secret"
    assert payload["tasks_redis_password"] == "redis-secret"
    assert payload["worker_evolution_commit_api_key"] == "commit-secret"
    assert "db-secret-password" in payload["database_dsn"]


def test_large_repo_profile_effective_defaults_snapshot() -> None:
    settings = TestSettings(LORELEY_PROFILE="large_repo_1m_30k")

    snapshot = {
        "scheduler_poll_interval_seconds": settings.scheduler_poll_interval_seconds,
        "tasks_worker_time_limit_seconds": settings.tasks_worker_time_limit_seconds,
        "worker_planning_timeout_seconds": settings.worker_planning_timeout_seconds,
        "worker_coding_timeout_seconds": settings.worker_coding_timeout_seconds,
        "worker_evaluator_timeout_seconds": settings.worker_evaluator_timeout_seconds,
        "mapelites_preprocess_max_file_size_kb": settings.mapelites_preprocess_max_file_size_kb,
        "mapelites_chunk_target_lines": settings.mapelites_chunk_target_lines,
        "mapelites_chunk_min_lines": settings.mapelites_chunk_min_lines,
        "mapelites_chunk_overlap_lines": settings.mapelites_chunk_overlap_lines,
        "mapelites_code_embedding_batch_size": settings.mapelites_code_embedding_batch_size,
        "mapelites_dimensionality_min_fit_samples": settings.mapelites_dimensionality_min_fit_samples,
        "mapelites_dimensionality_history_size": settings.mapelites_dimensionality_history_size,
        "mapelites_dimensionality_refit_interval": settings.mapelites_dimensionality_refit_interval,
        "mapelites_feature_normalization_warmup_samples": (
            settings.mapelites_feature_normalization_warmup_samples
        ),
        "mapelites_seed_population_size": settings.mapelites_seed_population_size,
        "mapelites_sampler_inspiration_count": settings.mapelites_sampler_inspiration_count,
        "mapelites_sampler_neighbor_max_radius": settings.mapelites_sampler_neighbor_max_radius,
        "mapelites_sampler_fallback_sample_size": settings.mapelites_sampler_fallback_sample_size,
    }

    assert snapshot == {
        "scheduler_poll_interval_seconds": 15.0,
        "tasks_worker_time_limit_seconds": 0,
        "worker_planning_timeout_seconds": 1200,
        "worker_coding_timeout_seconds": 3600,
        "worker_evaluator_timeout_seconds": 3600,
        "mapelites_preprocess_max_file_size_kb": 256,
        "mapelites_chunk_target_lines": 120,
        "mapelites_chunk_min_lines": 40,
        "mapelites_chunk_overlap_lines": 12,
        "mapelites_code_embedding_batch_size": 64,
        "mapelites_dimensionality_min_fit_samples": 32,
        "mapelites_dimensionality_history_size": 8192,
        "mapelites_dimensionality_refit_interval": 250,
        "mapelites_feature_normalization_warmup_samples": 128,
        "mapelites_seed_population_size": 32,
        "mapelites_sampler_inspiration_count": 4,
        "mapelites_sampler_neighbor_max_radius": 4,
        "mapelites_sampler_fallback_sample_size": 32,
    }


def test_export_safe_masks_sensitive_fields_snapshot() -> None:
    settings = _make_settings()

    payload = settings.export_safe(mask_secrets=True)
    snapshot = {
        "openai_api_key": payload["openai_api_key"],
        "openai_base_url": payload["openai_base_url"],
        "database_dsn": payload["database_dsn"],
        "db_password": payload["db_password"],
        "tasks_redis_url": payload["tasks_redis_url"],
        "tasks_redis_password": payload["tasks_redis_password"],
        "worker_kilocode_openai_api_key": payload["worker_kilocode_openai_api_key"],
        "worker_evolution_commit_api_key": payload["worker_evolution_commit_api_key"],
        "worker_evolution_commit_base_url": payload["worker_evolution_commit_base_url"],
        "worker_repo_remote_url": payload["worker_repo_remote_url"],
        "loreley_agent_api_token": payload["loreley_agent_api_token"],
        "loreley_api_write_token": payload["loreley_api_write_token"],
    }

    assert snapshot == {
        "openai_api_key": "***",
        "openai_base_url": "https://gateway.example.com/v1",
        "database_dsn": "postgresql+psycopg://db.internal:5433/loreley_dev",
        "db_password": "***",
        "tasks_redis_url": "redis://redis.internal:6380/2",
        "tasks_redis_password": "***",
        "worker_kilocode_openai_api_key": "***",
        "worker_evolution_commit_api_key": "***",
        "worker_evolution_commit_base_url": "https://commit.example.com/v1",
        "worker_repo_remote_url": "https://example.com/repo.git",
        "loreley_agent_api_token": "***",
        "loreley_api_write_token": "***",
    }


def test_config_dump_yaml_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = pytest.importorskip("yaml")

    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)

    code = main(["config", "dump", "--yaml"])
    captured = capsys.readouterr()

    assert code == 0
    payload = yaml.safe_load(captured.out)
    assert payload["openai_base_url"] == "https://gateway.example.com/v1"
    assert payload["tasks_redis_url"] == "redis://redis.internal:6380/2"
    assert "scheduler_poll_interval_seconds" in payload


def test_config_dump_rejects_multiple_output_formats(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)

    code = main(["config", "dump", "--json", "--yaml"])
    captured = capsys.readouterr()

    assert code == 1
    assert "choose exactly one output format" in captured.out.lower()


def test_settings_accept_loreley_llm_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.support import TestSettings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LORELEY_LLM_API_KEY", "sk-alias")
    monkeypatch.setenv("LORELEY_LLM_BASE_URL", "https://alias.example.com/v1")

    settings = TestSettings()

    assert settings.openai_api_key == "sk-alias"
    assert settings.openai_base_url == "https://alias.example.com/v1"
