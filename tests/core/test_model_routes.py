from __future__ import annotations

from loreley.core.model_routes import resolve_effective_routes
from tests.support import TestSettings


def test_effective_routes_use_phase_specific_kilo_models_without_secrets() -> None:
    settings = TestSettings(
        WORKER_KILOCODE_PLANNING_MODEL="openai/gpt-plan",
        WORKER_KILOCODE_CODING_MODEL="openai/gpt-code",
        WORKER_KILOCODE_VARIANT="max",
        WORKER_KILOCODE_OPENAI_BASE_URL="https://user:secret@example.invalid/v1",
        WORKER_KILOCODE_OPENAI_API_KEY="do-not-export",
        WORKER_KILOCODE_PROVIDER_CONFIG_MODE="native",
        WORKER_PLANNING_TRAJECTORY_SUMMARY_MODEL="gpt-summary",
        MAPELITES_CODE_EMBEDDING_MODEL="text-embedding-3-small",
        MAPELITES_CODE_EMBEDDING_DIMENSIONS=1536,
    )

    routes = resolve_effective_routes(settings)

    assert routes["planning"]["model"] == "openai/gpt-plan"
    assert routes["coding"]["model"] == "openai/gpt-code"
    assert routes["planning"]["reasoning"] == "max"
    assert routes["planning"]["base_url_host"] == "example.invalid"
    assert "secret" not in repr(routes)
    assert routes["commit_summary"]["model_call"] is False


def test_native_kilo_provider_is_not_misattributed_to_embedding_gateway() -> None:
    settings = TestSettings(
        OPENAI_API_KEY="embedding-only",
        OPENAI_BASE_URL="https://openrouter.ai/api/v1",
        WORKER_KILOCODE_PLANNING_MODEL="deepseek/deepseek-v4-flash",
        WORKER_KILOCODE_CODING_MODEL="deepseek/deepseek-v4-flash",
    )

    routes = resolve_effective_routes(settings)

    assert routes["planning"]["provider"] == "deepseek"
    assert routes["planning"]["provider_mode"] == "native"
    assert routes["embedding"]["provider"] == "openrouter"


def test_auto_mode_reports_config_route_selected_by_openai_phase_model() -> None:
    settings = TestSettings(
        WORKER_KILOCODE_PLANNING_MODEL="openai/gpt-plan",
        WORKER_KILOCODE_PROVIDER_CONFIG_MODE="auto",
        WORKER_KILOCODE_OPENAI_API_SPEC="chat_completions",
    )

    route = resolve_effective_routes(settings)["planning"]

    assert route["provider_mode"] == "config"
    assert route["provider"] == "loreley-openai-compatible"
    assert route["api_surface"] == "chat_completions"


def test_none_mode_reports_native_route_and_ignores_gateway_host() -> None:
    settings = TestSettings(
        WORKER_KILOCODE_PLANNING_MODEL="openai/gpt-plan",
        WORKER_KILOCODE_PROVIDER_CONFIG_MODE="none",
        WORKER_KILOCODE_OPENAI_BASE_URL="https://gateway.invalid/v1",
    )

    route = resolve_effective_routes(settings)["planning"]

    assert route["provider_mode"] == "none"
    assert route["provider"] == "openai"
    assert route["api_surface"] == "kilo_native"
    assert route["base_url_host"] is None
