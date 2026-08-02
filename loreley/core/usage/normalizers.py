from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class UsageEventMetadata:
    source: str = ""
    phase: str = ""
    provider: str = ""
    model: str = ""
    api_surface: str = ""
    job_id: UUID | str | None = None
    run_token: UUID | str | None = None
    external_usage_id: str = ""


_USAGE_METADATA_FIELDS = frozenset(
    (
        "source",
        "phase",
        "provider",
        "model",
        "api_surface",
        "job_id",
        "run_token",
        "external_usage_id",
    )
)


def _usage_metadata_values(metadata: UsageEventMetadata) -> dict[str, Any]:
    values = {
        "source": metadata.source,
        "phase": metadata.phase,
        "provider": metadata.provider,
        "model": metadata.model,
        "api_surface": metadata.api_surface,
        "job_id": metadata.job_id,
        "run_token": metadata.run_token,
        "external_usage_id": metadata.external_usage_id,
    }
    return {key: value for key, value in values.items() if value not in ("", None)}


def _usage_event_metadata(
    event_metadata: UsageEventMetadata | None,
    values: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> UsageEventMetadata:
    unknown = set(values) - _USAGE_METADATA_FIELDS
    if unknown:
        unknown_text = ", ".join(sorted(unknown))
        raise TypeError(f"unexpected usage metadata fields: {unknown_text}")

    merged: dict[str, Any] = dict(defaults or {})
    if event_metadata is not None:
        merged.update(_usage_metadata_values(event_metadata))
    merged.update({key: value for key, value in values.items() if value is not None})
    return UsageEventMetadata(
        source=str(merged.get("source") or ""),
        phase=str(merged.get("phase") or ""),
        provider=str(merged.get("provider") or ""),
        model=str(merged.get("model") or ""),
        api_surface=str(merged.get("api_surface") or ""),
        job_id=merged.get("job_id") or None,
        run_token=merged.get("run_token") or None,
        external_usage_id=str(merged.get("external_usage_id") or ""),
    )


@dataclass(slots=True)
class _KiloUsageTotals:
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    cost_usd: Decimal = Decimal("0")
    cost_seen: bool = False
    message_count: int = 0

    def add_message(self, message: Mapping[str, Any]) -> None:
        tokens = _kilo_assistant_tokens(message)
        if tokens is None:
            return
        self.message_count += 1
        self.provider = str(message.get("providerID") or self.provider or "")
        self.model = str(message.get("modelID") or self.model or "")
        cache = _mapping(tokens.get("cache"))
        self.input_tokens += _int_mapping(tokens, "input")
        self.cached_input_tokens += _int_mapping(cache, "read")
        self.cache_write_tokens += _int_mapping(cache, "write")
        self.output_tokens += _int_mapping(tokens, "output")
        self.reasoning_output_tokens += _int_mapping(tokens, "reasoning")
        cost = _decimal(message.get("cost"))
        if cost is not None:
            self.cost_usd += cost
            self.cost_seen = True

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cached_input_tokens
            + self.cache_write_tokens
            + self.output_tokens
            + self.reasoning_output_tokens
        )


def normalize_openai_usage_event(
    response: object,
    *,
    event_metadata: UsageEventMetadata | None = None,
    settings: object | None = None,
    **metadata_values: Any,
) -> LLMUsageEventPayload | None:
    usage = _attr(response, "usage")
    if usage is None:
        return None
    metadata = _usage_event_metadata(
        event_metadata,
        metadata_values,
        defaults={"source": "openai_sdk", "provider": "openai"},
    )
    context = current_usage_context()
    phase_value = metadata.phase or context.phase or ""
    api_surface = metadata.api_surface
    event = LLMUsageEventPayload(
        source=metadata.source,
        phase=phase_value,
        provider=metadata.provider,
        model=metadata.model or str(_attr(response, "model") or ""),
        api_surface=api_surface,
        job_id=metadata.job_id or context.job_id,
        run_token=metadata.run_token or context.run_token,
        input_tokens=_input_tokens(usage, api_surface=api_surface),
        cached_input_tokens=_cached_input_tokens(usage),
        cache_write_tokens=_cache_write_tokens(usage),
        output_tokens=_output_tokens(usage, api_surface=api_surface),
        reasoning_output_tokens=_reasoning_output_tokens(usage),
        total_tokens=_int_attr(usage, "total_tokens"),
        cost_source=COST_SOURCE_UNPRICED,
        raw_usage={"usage": sanitized_usage_payload(usage)},
        external_usage_id=metadata.external_usage_id or str(_attr(response, "id") or ""),
    )
    if event.total_tokens <= 0 and event.input_tokens <= 0 and event.output_tokens <= 0:
        return None
    return price_usage_event(event, settings=settings)  # type: ignore[arg-type]


def codex_usage_event_from_jsonl(
    jsonl_text: str,
    *,
    event_metadata: UsageEventMetadata | None = None,
    settings: object | None = None,
    **metadata_values: Any,
) -> LLMUsageEventPayload | None:
    token_event = _last_codex_token_count(jsonl_text)
    if token_event is None:
        return None
    metadata = _usage_event_metadata(
        event_metadata,
        metadata_values,
        defaults={"source": "codex_cli", "provider": "openai", "api_surface": "codex_exec"},
    )
    info = _mapping(_attr(token_event, "info"))
    usage = _mapping(info.get("total_token_usage")) or _mapping(info.get("last_token_usage"))
    if not usage:
        return None
    event = LLMUsageEventPayload(
        source=metadata.source,
        phase=metadata.phase,
        provider=metadata.provider,
        model=metadata.model or str(info.get("model") or ""),
        api_surface=metadata.api_surface,
        job_id=metadata.job_id,
        run_token=metadata.run_token,
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
        external_usage_id=metadata.external_usage_id,
    )
    return price_usage_event(event, settings=settings)  # type: ignore[arg-type]


def kilo_usage_event_from_messages(
    messages: Iterable[Mapping[str, Any]],
    *,
    event_metadata: UsageEventMetadata | None = None,
    title: str = "",
    session_id: str = "",
    settings: object | None = None,
    **metadata_values: Any,
) -> LLMUsageEventPayload | None:
    totals = _kilo_usage_totals(messages)
    if totals.message_count <= 0:
        return None
    metadata = _usage_event_metadata(
        event_metadata,
        metadata_values,
        defaults={"source": "kilo_cli", "api_surface": "kilo_run"},
    )
    event = LLMUsageEventPayload(
        source=metadata.source,
        phase=metadata.phase,
        provider=totals.provider,
        model=totals.model,
        api_surface=metadata.api_surface,
        job_id=metadata.job_id,
        run_token=metadata.run_token,
        input_tokens=totals.input_tokens,
        cached_input_tokens=totals.cached_input_tokens,
        cache_write_tokens=totals.cache_write_tokens,
        output_tokens=totals.output_tokens,
        reasoning_output_tokens=totals.reasoning_output_tokens,
        total_tokens=totals.total_tokens,
        cost_usd=totals.cost_usd if totals.cost_seen else None,
        cost_source=COST_SOURCE_PROVIDER_REPORTED if totals.cost_seen else COST_SOURCE_UNPRICED,
        raw_usage=_kilo_raw_usage(totals=totals, title=title, session_id=session_id),
        external_usage_id=metadata.external_usage_id or (f"kilo:{session_id}" if session_id else ""),
    )
    return price_usage_event(event, settings=settings)  # type: ignore[arg-type]


def _kilo_usage_totals(messages: Iterable[Mapping[str, Any]]) -> _KiloUsageTotals:
    totals = _KiloUsageTotals()
    for message in messages:
        totals.add_message(message)
    return totals


def _kilo_assistant_tokens(message: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if str(message.get("role") or "") != "assistant":
        return None
    tokens = _mapping(message.get("tokens"))
    return tokens or None


def _kilo_raw_usage(
    *,
    totals: _KiloUsageTotals,
    title: str,
    session_id: str,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "title": title,
        "message_count": totals.message_count,
        "tokens": {
            "input": totals.input_tokens,
            "cache_read": totals.cached_input_tokens,
            "cache_write": totals.cache_write_tokens,
            "output": totals.output_tokens,
            "reasoning": totals.reasoning_output_tokens,
        },
        "cost": str(totals.cost_usd) if totals.cost_seen else None,
    }


def unavailable_usage_event(
    *,
    reason: str,
    event_metadata: UsageEventMetadata | None = None,
    **metadata_values: Any,
) -> LLMUsageEventPayload:
    metadata = _usage_event_metadata(event_metadata, metadata_values)
    return LLMUsageEventPayload(
        source=metadata.source,
        phase=metadata.phase,
        provider=metadata.provider,
        model=metadata.model,
        api_surface=metadata.api_surface,
        job_id=metadata.job_id,
        run_token=metadata.run_token,
        cost_source=COST_SOURCE_UNAVAILABLE,
        raw_usage={"unavailable_reason": str(reason or "usage unavailable")[:512]},
        external_usage_id=metadata.external_usage_id,
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
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping) and current.get("type") == "token_count":
            return current
        stack.extend(reversed(tuple(_token_count_children(current))))
    return None


def _token_count_children(value: object) -> Iterable[object]:
    if isinstance(value, Mapping):
        for key in ("payload", "msg", "message", "event", "data"):
            if key in value:
                yield value[key]
        yield from value.values()
    elif isinstance(value, list):
        yield from value


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
