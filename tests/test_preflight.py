from __future__ import annotations

from tests.support import TestSettings
from loreley.preflight import (
    _check_agent_backend,
    _check_openai_api_key_for_scheduler,
    _check_openai_api_key_for_worker,
    _check_dynamic_openai_agent_ttl,
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


def test_dynamic_provider_import_failure_fails_openai_auth_check(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LORELEY_LLM_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENAI_DYNAMIC_API_KEY_PROVIDER",
        "tests.support_broken_dynamic_provider:token_provider",
    )
    monkeypatch.setenv("OPENAI_DYNAMIC_API_KEY_TTL_SECONDS", "600")
    settings = TestSettings()

    result = _check_openai_api_key_for_worker(settings)

    assert result.status == "fail"
    assert "broken provider import" in result.details


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


def test_worker_preflight_warns_when_backend_refs_are_blank_strings(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OPENAI_DYNAMIC_API_KEY_PROVIDER",
        "tests.support_dynamic_provider:token_provider",
    )
    monkeypatch.setenv("OPENAI_DYNAMIC_API_KEY_TTL_SECONDS", "600")
    monkeypatch.setenv("WORKER_REPO_REMOTE_URL", "https://example.com/repo.git")
    monkeypatch.setenv("WORKER_EVALUATOR_PLUGIN", "tests.support_dynamic_provider:token_provider")
    monkeypatch.setenv("WORKER_PLANNING_BACKEND", "")
    monkeypatch.setenv("WORKER_CODING_BACKEND", "")
    settings = TestSettings()

    results = preflight_worker(settings)

    ttl_warnings = [item for item in results if item.name == "openai_dynamic_api_key_ttl"]
    assert ttl_warnings
    assert any(item.status == "warn" for item in ttl_warnings)


def test_kilocode_preflight_fails_when_required_run_flags_are_missing(monkeypatch) -> None:
    from loreley.core.worker.agent.backends import kilocode_cli

    monkeypatch.setattr("loreley.preflight.shutil.which", lambda _name: "/usr/bin/kilo")
    monkeypatch.setenv("WORKER_KILOCODE_PROVIDER_CONFIG_MODE", "none")
    monkeypatch.setenv("LLM_USAGE_TRACKING_ENABLED", "false")

    def fake_discover(*_args, **_kwargs):  # noqa: ANN002
        return kilocode_cli.KiloCliCapabilities(
            version="7.3.16",
            run_flags=frozenset({"--agent", "--model"}),
            supports_db_path=True,
        )

    monkeypatch.setattr(kilocode_cli, "discover_kilo_cli_capabilities", fake_discover)
    settings = TestSettings(WORKER_PLANNING_BACKEND="")

    results = _check_agent_backend(kind="planning", settings=settings)

    cli_result = next(item for item in results if item.name == "planning_kilocode_cli")
    assert cli_result.status == "fail"
    assert "--auto" in cli_result.details
    assert "--dir" in cli_result.details


def test_kilocode_preflight_uses_default_backend_settings_when_ref_is_blank(monkeypatch) -> None:
    from loreley.core.worker.agent.backends import kilocode_cli

    monkeypatch.setattr("loreley.preflight.shutil.which", lambda _name: "/usr/bin/kilo")
    monkeypatch.setenv("WORKER_KILOCODE_PROVIDER_CONFIG_MODE", "none")
    monkeypatch.setenv("LLM_USAGE_TRACKING_ENABLED", "false")

    def fake_discover(*_args, **_kwargs):  # noqa: ANN002
        return kilocode_cli.KiloCliCapabilities(
            version="7.3.16",
            run_flags=frozenset({"--auto"}),
            supports_db_path=True,
        )

    monkeypatch.setattr(kilocode_cli, "discover_kilo_cli_capabilities", fake_discover)
    settings = TestSettings(
        WORKER_PLANNING_BACKEND="",
        WORKER_KILOCODE_JSON_OUTPUT=True,
        WORKER_KILOCODE_AGENT="architect",
        WORKER_KILOCODE_MODEL="openai/gpt-5.4",
        WORKER_KILOCODE_VARIANT="high",
        WORKER_KILOCODE_PURE=True,
    )

    results = _check_agent_backend(kind="planning", settings=settings)

    cli_result = next(item for item in results if item.name == "planning_kilocode_cli")
    assert cli_result.status == "fail"
    assert "--format" in cli_result.details
    assert "--agent" in cli_result.details
    assert "--model" in cli_result.details
    assert "--variant" in cli_result.details
    assert "--pure" in cli_result.details
    assert "--dir" in cli_result.details


def test_kilocode_preflight_fails_when_usage_tracking_requires_unsupported_title(
    monkeypatch,
) -> None:
    from loreley.core.worker.agent.backends import kilocode_cli

    monkeypatch.setattr("loreley.preflight.shutil.which", lambda _name: "/usr/bin/kilo")
    monkeypatch.setenv("WORKER_KILOCODE_PROVIDER_CONFIG_MODE", "none")
    monkeypatch.setenv("LLM_USAGE_TRACKING_ENABLED", "true")

    def fake_discover(*_args, **_kwargs):  # noqa: ANN002
        return kilocode_cli.KiloCliCapabilities(
            version="7.3.16",
            run_flags=frozenset({"--auto"}),
            supports_db_path=True,
        )

    monkeypatch.setattr(kilocode_cli, "discover_kilo_cli_capabilities", fake_discover)
    settings = TestSettings(WORKER_PLANNING_BACKEND="")

    results = _check_agent_backend(kind="planning", settings=settings)

    cli_result = next(item for item in results if item.name == "planning_kilocode_cli")
    usage_result = next(item for item in results if item.name == "planning_kilocode_usage_db")
    assert cli_result.status == "fail"
    assert "--title" in cli_result.details
    assert usage_result.status == "fail"
    assert "--title" in usage_result.details


def test_kilocode_preflight_fails_config_mode_when_provider_probe_fails(monkeypatch) -> None:
    from loreley.core.worker.agent.backends import kilocode_cli

    monkeypatch.setattr("loreley.preflight.shutil.which", lambda _name: "/usr/bin/kilo")
    monkeypatch.setenv("LLM_USAGE_TRACKING_ENABLED", "false")

    def fake_discover(*_args, **_kwargs):  # noqa: ANN002
        return kilocode_cli.KiloCliCapabilities(
            version="7.3.16",
            run_flags=frozenset(
                {"--auto", "--agent", "--model", "--format", "--title", "--variant", "--dir"}
            ),
            supports_db_path=True,
        )

    monkeypatch.setattr(kilocode_cli, "discover_kilo_cli_capabilities", fake_discover)
    monkeypatch.setattr(
        kilocode_cli,
        "probe_kilo_config_content_support",
        lambda *_args, **_kwargs: (False, "debug config ignored KILO_CONFIG_CONTENT"),
    )
    settings = TestSettings(
        WORKER_PLANNING_BACKEND="",
        WORKER_KILOCODE_PROVIDER_CONFIG_MODE="config",
        WORKER_KILOCODE_OPENAI_BASE_URL="https://worker.example.com/v1",
        WORKER_KILOCODE_OPENAI_API_KEY="sk-worker",
    )

    results = _check_agent_backend(kind="planning", settings=settings)

    provider_result = next(item for item in results if item.name == "planning_kilocode_provider")
    assert provider_result.status == "fail"
    assert "KILO_CONFIG_CONTENT" in provider_result.details


def test_dynamic_openai_agent_ttl_check_ignores_broken_provider_import(monkeypatch) -> None:
    monkeypatch.setenv(
        "OPENAI_DYNAMIC_API_KEY_PROVIDER",
        "tests.support_broken_dynamic_provider:token_provider",
    )
    monkeypatch.setenv("OPENAI_DYNAMIC_API_KEY_TTL_SECONDS", "600")
    settings = TestSettings()

    assert _check_dynamic_openai_agent_ttl(settings) == []
