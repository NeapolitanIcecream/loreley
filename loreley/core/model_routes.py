"""Secret-free resolution of the effective model and embedding routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from loreley.config import Settings


@dataclass(frozen=True, slots=True)
class _KiloRouteResolution:
    mode: str
    provider: str
    api_spec: object
    base_url: str | None
    gateway: bool


def resolve_effective_routes(settings: Settings) -> dict[str, dict[str, Any]]:
    """Return the routes runtime and operator diagnostics should agree on."""

    return {
        "seed_portfolio": _seed_portfolio_route(settings),
        "planning": _agent_route(settings, "planning"),
        "coding": _agent_route(settings, "coding"),
        "trajectory_summary": _trajectory_route(settings),
        "embedding": _embedding_route(settings),
        "commit_summary": {
            "enabled": True,
            "model_call": False,
            "source": "coding.report.summary",
            "fallback_source": "plan.summary",
        },
    }


def _seed_portfolio_route(settings: Settings) -> dict[str, Any]:
    backend_ref = str(
        getattr(settings, "worker_seed_portfolio_backend", "") or ""
    ).strip()
    model = _non_empty(getattr(settings, "worker_seed_portfolio_model", None))
    reasoning = _non_empty(
        getattr(settings, "worker_seed_portfolio_reasoning_effort", None)
    )
    if not _is_kilocode_backend(backend_ref):
        return {
            "backend": backend_ref or "backend_defined",
            "provider_mode": "backend_defined",
            "provider": "unknown",
            "model": model,
            "variant": reasoning,
            "reasoning": reasoning or "backend_defined",
            "api_surface": "backend_defined",
            "base_url_host": None,
        }
    route = _resolve_kilo_route(settings, model)
    return {
        "backend": "kilo",
        "provider_mode": route.mode,
        "provider": route.provider,
        "model": model,
        "variant": reasoning,
        "reasoning": reasoning or "provider_default",
        "api_surface": route.api_spec if route.gateway else "kilo_native",
        "base_url_host": _url_host(route.base_url) if route.gateway else None,
    }


def resolve_kilocode_phase_model(
    settings: Settings,
    phase: Literal["planning", "coding"],
) -> str | None:
    selected = (
        getattr(settings, f"worker_kilocode_{phase}_model", None)
        or settings.worker_kilocode_model
        or ""
    )
    return str(selected).strip() or None


def resolve_kilocode_provider_input(
    settings: Settings,
    *,
    api_key: str | None = None,
    selected_model: str | None = None,
) -> dict[str, object]:
    """Resolve the provider inputs shared by Kilo runtime and diagnostics."""

    worker_base_url = _non_empty(settings.worker_kilocode_openai_base_url)
    worker_model = _kilocode_gateway_model(settings, selected_model=selected_model)
    worker_api_spec = settings.worker_kilocode_openai_api_spec
    base_url = worker_base_url or _non_empty(settings.openai_base_url)
    api_spec = worker_api_spec or settings.openai_api_spec
    has_api_key_source = any(
        _non_empty(value)
        for value in (
            api_key,
            settings.worker_kilocode_openai_api_key,
            settings.openai_api_key,
            settings.openai_dynamic_api_key_provider,
        )
    )
    has_provider_config = any(
        (has_api_key_source, base_url, worker_model, worker_api_spec)
    )
    provider_id: str | None = None
    if api_spec == "responses":
        provider_id = "loreley-openai-responses"
    elif api_spec == "chat_completions" or has_provider_config:
        provider_id = "loreley-openai-compatible"
    return {
        "api_spec": api_spec,
        "base_url": base_url,
        "model": worker_model,
        "provider_id": provider_id,
        "has_api_key_source": has_api_key_source,
        "has_provider_config": has_provider_config,
    }


def _agent_route(
    settings: Settings,
    phase: Literal["planning", "coding"],
) -> dict[str, Any]:
    backend_ref = str(getattr(settings, f"worker_{phase}_backend", "") or "").strip()
    if not _is_kilocode_backend(backend_ref):
        return _backend_defined_route(settings, phase, backend_ref)
    model = resolve_kilocode_phase_model(settings, phase)
    route = _resolve_kilo_route(settings, model)
    variant = _non_empty(settings.worker_kilocode_variant)
    return {
        "backend": "kilo",
        "provider_mode": route.mode,
        "provider": route.provider,
        "model": model,
        "variant": variant,
        "reasoning": variant or "provider_default",
        "api_surface": route.api_spec if route.gateway else "kilo_native",
        "base_url_host": _url_host(route.base_url) if route.gateway else None,
    }


def _backend_defined_route(
    settings: Settings,
    phase: Literal["planning", "coding"],
    backend_ref: str,
) -> dict[str, Any]:
    return {
        "backend": backend_ref or "backend_defined",
        "provider_mode": "backend_defined",
        "provider": "unknown",
        "model": _non_empty(getattr(settings, f"worker_{phase}_codex_model", None)),
        "variant": None,
        "reasoning": "backend_defined",
        "api_surface": "backend_defined",
        "base_url_host": None,
    }


def _resolve_kilo_route(
    settings: Settings,
    model: str | None,
) -> _KiloRouteResolution:
    configured_mode = (
        str(settings.worker_kilocode_provider_config_mode or "auto").strip().lower()
    )
    provider_input = resolve_kilocode_provider_input(
        settings,
        selected_model=model,
    )
    base_url = _non_empty(provider_input["base_url"])
    api_spec = provider_input["api_spec"]
    effective_gateway = _kilo_gateway_enabled(
        settings=settings,
        model=model,
        configured_mode=configured_mode,
        has_provider_config=bool(provider_input["has_provider_config"]),
    )
    effective_mode = _effective_kilo_mode(configured_mode, effective_gateway)
    provider = _effective_kilo_provider(
        model=model,
        api_spec=api_spec,
        mode=effective_mode,
        gateway=effective_gateway,
    )
    return _KiloRouteResolution(
        mode=effective_mode,
        provider=provider,
        api_spec=api_spec,
        base_url=base_url,
        gateway=effective_gateway,
    )


def _kilo_gateway_enabled(
    *,
    settings: Settings,
    model: str | None,
    configured_mode: str,
    has_provider_config: bool,
) -> bool:
    if configured_mode == "none" or not has_provider_config:
        return False
    if _gateway_compatible_model(model):
        return True
    return bool(
        configured_mode == "native"
        and _model_provider(model) == "deepseek"
        and _non_empty(settings.worker_kilocode_openai_base_url)
    )


def _effective_kilo_mode(configured_mode: str, gateway: bool) -> str:
    if configured_mode != "auto":
        return configured_mode
    return "config" if gateway else "native"


def _effective_kilo_provider(
    *,
    model: str | None,
    api_spec: object,
    mode: str,
    gateway: bool,
) -> str:
    model_provider = _model_provider(model)
    if not gateway:
        return model_provider
    if mode in {"config", "legacy_env"}:
        if api_spec == "responses":
            return "loreley-openai-responses"
        return "loreley-openai-compatible"
    if mode == "native":
        return model_provider if model_provider in {"deepseek", "openai"} else "openai"
    return model_provider


def _trajectory_route(settings: Settings) -> dict[str, Any]:
    enabled = int(settings.worker_planning_trajectory_max_chunks or 0) > 0
    mode = settings.worker_planning_trajectory_summary_provider_mode
    base_url = (
        settings.worker_planning_trajectory_summary_base_url
        if mode == "custom"
        else settings.openai_base_url
    )
    api_spec = (
        settings.worker_planning_trajectory_summary_api_spec
        if mode == "custom"
        else settings.openai_api_spec
    )
    return {
        "enabled": enabled,
        "provider_mode": mode,
        "provider": _provider_from_host(_url_host(base_url)),
        "model": _non_empty(settings.worker_planning_trajectory_summary_model),
        "api_surface": api_spec,
        "thinking": settings.worker_planning_trajectory_summary_thinking,
        "reasoning_effort": (
            settings.worker_planning_trajectory_summary_reasoning_effort
        ),
        "base_url_host": _url_host(base_url),
    }


def _embedding_route(settings: Settings) -> dict[str, Any]:
    model = str(settings.mapelites_code_embedding_model or "").strip()
    local = model.lower().startswith("local-hash")
    host = None if local else _url_host(settings.openai_base_url)
    return {
        "provider_mode": "local" if local else "global_openai",
        "provider": "local-hash" if local else _provider_from_host(host),
        "model": model or None,
        "dimensions": settings.mapelites_code_embedding_dimensions,
        "api_surface": "local" if local else "embeddings",
        "base_url_host": host,
        "local_hash_acknowledged": (
            bool(settings.mapelites_local_hash_embedding_acknowledged)
            if local
            else None
        ),
    }


def _is_kilocode_backend(backend_ref: str) -> bool:
    return not backend_ref or "kilocode" in backend_ref.lower()


def _model_provider(model: str | None) -> str:
    raw = str(model or "").strip()
    if "/" in raw:
        return raw.partition("/")[0]
    return "kilo_default" if raw else "unknown"


def _gateway_compatible_model(model: str | None) -> bool:
    raw = str(model or "").strip()
    if "/" not in raw:
        return True
    return raw.partition("/")[0] in {
        "openai",
        "openai-responses",
        "loreley-openai-compatible",
        "loreley-openai-responses",
    }


def _kilocode_gateway_model(settings: Settings, *, selected_model: str | None) -> str:
    """Mirror the model selection used by Kilo provider-config construction."""

    explicit = _non_empty(settings.worker_kilocode_openai_model)
    if explicit and not selected_model:
        return explicit
    selected = (
        str(selected_model or "").strip()
        or str(settings.worker_kilocode_model or "").strip()
    )
    provider_id, separator, model_id = selected.partition("/")
    if not separator:
        return selected
    if provider_id in {
        "openai",
        "openai-responses",
        "loreley-openai-compatible",
        "loreley-openai-responses",
    }:
        return model_id
    return ""


def _url_host(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    return (parsed.hostname or "").lower() or None


def _provider_from_host(host: str | None) -> str:
    if not host:
        return "openai_compatible"
    if host == "api.openai.com" or host.endswith(".openai.com"):
        return "openai"
    if "openrouter" in host:
        return "openrouter"
    return "openai_compatible"


def _non_empty(value: object) -> str | None:
    return str(value or "").strip() or None


__all__ = [
    "resolve_effective_routes",
    "resolve_kilocode_phase_model",
    "resolve_kilocode_provider_input",
]
