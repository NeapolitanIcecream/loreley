from __future__ import annotations

from decimal import Decimal
import json
from typing import Any, Iterable, Mapping
from uuid import UUID

from loreley.core.usage.events import (
    COST_SOURCE_PROVIDER_REPORTED,
    COST_SOURCE_UNAVAILABLE,
    COST_SOURCE_UNPRICED,
    LLMUsageEventPayload,
    current_usage_context,
    sanitized_usage_payload,
)
from loreley.core.usage.pricing import price_usage_event


def normalize_openai_usage_event(
    response: object,
    *,
    source: str = "openai_sdk",
    phase: str,
    provider: str = "openai",
    model: str,
    api_surface: str,
    job_id: UUID | None = None,
    run_token: UUID | None = None,
    settings: object | None = None,
    external_usage_id: str | None = None,
) -> LLMUsageEventPayload | None:
    usage = _attr(response, "usage")
    if usage is None:
        return None
    context = current_usage_context()
    phase_value = phase or context.phase or ""
    event = LLMUsageEventPayload(
        source=source,
        phase=phase_value,
        provider=provider,
        model=model or str(_attr(response, "model") or ""),
        api_surface=api_surface,
        job_id=job_id or context.job_id,
        run_token=run_token or context.run_token,
        input_tokens=_input_tokens(usage, api_surface=api_surface),
        cached_input_tokens=_cached_input_tokens(usage),
        cache_write_tokens=_cache_write_tokens(usage),
        output_tokens=_output_tokens(usage, api_surface=api_surface),
        reasoning_output_tokens=_reasoning_output_tokens(usage),
        total_tokens=_int_attr(usage, "total_tokens"),
        cost_source=COST_SOURCE_UNPRICED,
        raw_usage={"usage": sanitized_usage_payload(usage)},
        external_usage_id=external_usage_id or str(_attr(response, "id") or ""),
    )
    if event.total_tokens <= 0 and event.input_tokens <= 0 and event.output_tokens <= 0:
        return None
    return price_usage_event(event, settings=settings)  # type: ignore[arg-type]


def codex_usage_event_from_jsonl(
    jsonl_text: str,
    *,
    phase: str,
    job_id: UUID | None = None,
    run_token: UUID | None = None,
    model: str | None = None,
    settings: object | None = None,
    external_usage_id: str = "",
) -> LLMUsageEventPayload | None:
    token_event = _last_codex_token_count(jsonl_text)
    if token_event is None:
        return None
    info = _mapping(_attr(token_event, "info"))
    usage = _mapping(info.get("total_token_usage")) or _mapping(info.get("last_token_usage"))
    if not usage:
        return None
    event = LLMUsageEventPayload(
        source="codex_cli",
        phase=phase,
        provider="openai",
        model=model or str(info.get("model") or ""),
        api_surface="codex_exec",
        job_id=job_id,
        run_token=run_token,
        input_tokens=_int_mapping(usage, "input_tokens"),
        cached_input_tokens=_int_mapping(usage, "cached_input_tokens"),
        output_tokens=_int_mapping(usage, "output_tokens"),
        reasoning_output_tokens=_int_mapping(usage, "reasoning_output_tokens"),
        total_tokens=_int_mapping(usage, "total_tokens"),
        cost_source=COST_SOURCE_UNPRICED,
        raw_usage={
            "token_count": sanitized_usage_payload(
                {
                    "total_token_usage": usage,
                    "last_token_usage": info.get("last_token_usage"),
                    "model_context_window": info.get("model_context_window"),
                }
            )
        },
        external_usage_id=external_usage_id,
    )
    return price_usage_event(event, settings=settings)  # type: ignore[arg-type]


def kilo_usage_event_from_messages(
    messages: Iterable[Mapping[str, Any]],
    *,
    phase: str,
    job_id: UUID | None = None,
    run_token: UUID | None = None,
    title: str = "",
    session_id: str = "",
    settings: object | None = None,
    external_usage_id: str | None = None,
) -> LLMUsageEventPayload | None:
    total_input = 0
    total_cached = 0
    total_cache_write = 0
    total_output = 0
    total_reasoning = 0
    total_cost = Decimal("0")
    cost_seen = False
    provider = ""
    model = ""
    counted_messages = 0
    for message in messages:
        if str(message.get("role") or "") != "assistant":
            continue
        tokens = _mapping(message.get("tokens"))
        if not tokens:
            continue
        counted_messages += 1
        provider = str(message.get("providerID") or provider or "")
        model = str(message.get("modelID") or model or "")
        cache = _mapping(tokens.get("cache"))
        total_input += _int_mapping(tokens, "input")
        total_cached += _int_mapping(cache, "read")
        total_cache_write += _int_mapping(cache, "write")
        total_output += _int_mapping(tokens, "output")
        total_reasoning += _int_mapping(tokens, "reasoning")
        cost = _decimal(message.get("cost"))
        if cost is not None:
            total_cost += cost
            cost_seen = True
    if counted_messages <= 0:
        return None
    event = LLMUsageEventPayload(
        source="kilo_cli",
        phase=phase,
        provider=provider,
        model=model,
        api_surface="kilo_run",
        job_id=job_id,
        run_token=run_token,
        input_tokens=total_input,
        cached_input_tokens=total_cached,
        cache_write_tokens=total_cache_write,
        output_tokens=total_output,
        reasoning_output_tokens=total_reasoning,
        total_tokens=total_input + total_cached + total_cache_write + total_output,
        cost_usd=total_cost if cost_seen else None,
        cost_source=COST_SOURCE_PROVIDER_REPORTED if cost_seen else COST_SOURCE_UNPRICED,
        raw_usage={
            "session_id": session_id,
            "title": title,
            "message_count": counted_messages,
            "tokens": {
                "input": total_input,
                "cache_read": total_cached,
                "cache_write": total_cache_write,
                "output": total_output,
                "reasoning": total_reasoning,
            },
            "cost": str(total_cost) if cost_seen else None,
        },
        external_usage_id=external_usage_id or (f"kilo:{session_id}" if session_id else ""),
    )
    if cost_seen:
        return event
    return price_usage_event(event, settings=settings)  # type: ignore[arg-type]


def unavailable_usage_event(
    *,
    source: str,
    phase: str,
    provider: str = "",
    model: str = "",
    api_surface: str = "",
    job_id: UUID | None = None,
    run_token: UUID | None = None,
    reason: str,
    external_usage_id: str = "",
) -> LLMUsageEventPayload:
    return LLMUsageEventPayload(
        source=source,
        phase=phase,
        provider=provider,
        model=model,
        api_surface=api_surface,
        job_id=job_id,
        run_token=run_token,
        cost_source=COST_SOURCE_UNAVAILABLE,
        raw_usage={"unavailable_reason": str(reason or "usage unavailable")[:512]},
        external_usage_id=external_usage_id,
    )


def _last_codex_token_count(jsonl_text: str) -> Mapping[str, Any] | None:
    selected: Mapping[str, Any] | None = None
    for line in str(jsonl_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidate = _find_token_count(payload)
        if candidate is not None:
            selected = candidate
    return selected


def _find_token_count(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if value.get("type") == "token_count":
            return value
        for key in ("payload", "msg", "message", "event", "data"):
            nested = value.get(key)
            candidate = _find_token_count(nested)
            if candidate is not None:
                return candidate
        for nested in value.values():
            candidate = _find_token_count(nested)
            if candidate is not None:
                return candidate
    elif isinstance(value, list):
        for nested in value:
            candidate = _find_token_count(nested)
            if candidate is not None:
                return candidate
    return None


def _input_tokens(usage: object, *, api_surface: str) -> int:
    if api_surface == "chat_completions":
        return _int_attr(usage, "prompt_tokens")
    value = _int_attr(usage, "input_tokens")
    if value:
        return value
    return _int_attr(usage, "prompt_tokens")


def _output_tokens(usage: object, *, api_surface: str) -> int:
    if api_surface == "embeddings":
        return 0
    if api_surface == "chat_completions":
        return _int_attr(usage, "completion_tokens")
    value = _int_attr(usage, "output_tokens")
    if value:
        return value
    return _int_attr(usage, "completion_tokens")


def _cached_input_tokens(usage: object) -> int:
    details = _attr(usage, "input_tokens_details") or _attr(usage, "prompt_tokens_details") or {}
    return _int_attr(details, "cached_tokens")


def _cache_write_tokens(usage: object) -> int:
    details = _attr(usage, "input_tokens_details") or _attr(usage, "prompt_tokens_details") or {}
    return _int_attr(details, "cache_write_tokens")


def _reasoning_output_tokens(usage: object) -> int:
    details = _attr(usage, "output_tokens_details") or _attr(usage, "completion_tokens_details") or {}
    return _int_attr(details, "reasoning_tokens")


def _attr(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _int_attr(value: object, name: str) -> int:
    return _coerce_int(_attr(value, name))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int_mapping(value: Mapping[str, Any], name: str) -> int:
    return _coerce_int(value.get(name))


def _coerce_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _decimal(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
