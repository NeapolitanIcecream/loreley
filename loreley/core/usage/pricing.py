from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from loguru import logger

from loreley.config import Settings, get_settings
from loreley.core.usage.events import (
    COST_SOURCE_ESTIMATED,
    COST_SOURCE_UNPRICED,
    LLMUsageEventPayload,
)

log = logger.bind(module="usage.pricing")

_ONE_MILLION = Decimal("1000000")
_COST_QUANT = Decimal("0.00000001")

_BUILTIN_PRICING: dict[str, Any] = {
    "version": "builtin-2026-05-openai-common",
    "prices": [
        {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "input_usd_per_1m": "0.40",
            "cached_input_usd_per_1m": "0.10",
            "output_usd_per_1m": "1.60",
        },
        {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_surface": "embeddings",
            "input_usd_per_1m": "0.02",
            "cached_input_usd_per_1m": "0",
            "output_usd_per_1m": "0",
        },
    ],
}


@dataclass(frozen=True, slots=True)
class PriceRule:
    provider: str
    model: str
    api_surface: str = ""
    service_tier: str = ""
    context_tier: str = ""
    input_usd_per_1m: Decimal | None = None
    cached_input_usd_per_1m: Decimal | None = None
    cache_write_usd_per_1m: Decimal | None = None
    output_usd_per_1m: Decimal | None = None


@dataclass(frozen=True, slots=True)
class PricingTable:
    version: str
    rules: tuple[PriceRule, ...]


def price_usage_event(
    event: LLMUsageEventPayload,
    *,
    settings: Settings | None = None,
) -> LLMUsageEventPayload:
    if event.cost_source != COST_SOURCE_UNPRICED or event.cost_usd is not None:
        return event
    table = load_pricing_table(settings=settings)
    rule = match_price_rule(table, event)
    if rule is None:
        return event
    cost = estimate_cost_usd(event, rule)
    if cost is None:
        return event
    return LLMUsageEventPayload(
        **{
            **event.as_record_dict(),
            "cost_usd": cost,
            "cost_source": COST_SOURCE_ESTIMATED,
            "pricing_version": table.version,
        }
    )


def load_pricing_table(*, settings: Settings | None = None) -> PricingTable:
    settings = settings or get_settings()
    payload = _pricing_payload_from_settings(settings)
    return _pricing_table_from_payload(payload)


def match_price_rule(table: PricingTable, event: LLMUsageEventPayload) -> PriceRule | None:
    provider = (event.provider or "").strip().lower()
    model = (event.model or "").strip().lower()
    api_surface = (event.api_surface or "").strip().lower()
    if not provider or not model:
        return None

    candidates: list[tuple[int, PriceRule]] = []
    for rule in table.rules:
        if rule.provider and rule.provider.lower() != provider:
            continue
        if rule.model and rule.model.lower() != model:
            continue
        if rule.api_surface and rule.api_surface.lower() != api_surface:
            continue
        score = int(bool(rule.provider)) + int(bool(rule.model)) + int(bool(rule.api_surface))
        candidates.append((score, rule))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def estimate_cost_usd(event: LLMUsageEventPayload, rule: PriceRule) -> Decimal | None:
    input_rate = rule.input_usd_per_1m
    cached_rate = rule.cached_input_usd_per_1m
    cache_write_rate = rule.cache_write_usd_per_1m
    output_rate = rule.output_usd_per_1m
    if input_rate is None and cached_rate is None and cache_write_rate is None and output_rate is None:
        return None

    billable_input = _billable_regular_input_tokens(event)
    total = Decimal("0")
    if input_rate is not None:
        total += Decimal(billable_input) * input_rate
    if cached_rate is not None:
        total += Decimal(event.cached_input_tokens) * cached_rate
    if cache_write_rate is not None:
        total += Decimal(event.cache_write_tokens) * cache_write_rate
    if output_rate is not None:
        total += Decimal(event.output_tokens) * output_rate
    return (total / _ONE_MILLION).quantize(_COST_QUANT, rounding=ROUND_HALF_UP)


def _billable_regular_input_tokens(event: LLMUsageEventPayload) -> int:
    if _cache_reads_are_separate_input_counter(event):
        return event.input_tokens
    cached_tokens = min(event.cached_input_tokens, event.input_tokens)
    return max(event.input_tokens - cached_tokens, 0)


def _cache_reads_are_separate_input_counter(event: LLMUsageEventPayload) -> bool:
    return event.source == "kilo_cli" or event.api_surface == "kilo_run"


def _pricing_payload_from_settings(settings: Settings) -> Mapping[str, Any]:
    inline = str(getattr(settings, "llm_usage_pricing_json", "") or "").strip()
    if inline:
        try:
            payload = json.loads(inline)
            if isinstance(payload, Mapping):
                return payload
        except json.JSONDecodeError as exc:
            log.warning("Invalid LLM_USAGE_PRICING_JSON: {}", exc)
    path_raw = str(getattr(settings, "llm_usage_pricing_path", "") or "").strip()
    if path_raw:
        path = Path(path_raw).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                return payload
        except FileNotFoundError:
            log.warning("LLM usage pricing file does not exist: {}", path)
        except Exception as exc:
            log.warning("Failed to load LLM usage pricing file {}: {}", path, exc)
    return _BUILTIN_PRICING


def _pricing_table_from_payload(payload: Mapping[str, Any]) -> PricingTable:
    version = str(payload.get("version") or "").strip() or "unversioned"
    raw_rules = payload.get("prices")
    if isinstance(raw_rules, Mapping):
        raw_rules = _rules_from_model_mapping(raw_rules)
    rules = tuple(
        rule
        for rule in (_price_rule_from_mapping(item) for item in _iter_mappings(raw_rules))
        if rule is not None
    )
    return PricingTable(version=version, rules=rules)


def _rules_from_model_mapping(raw_rules: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rules: list[Mapping[str, Any]] = []
    for model, value in raw_rules.items():
        if not isinstance(value, Mapping):
            continue
        rules.append({"model": str(model), **dict(value)})
    return rules


def _iter_mappings(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                yield item


def _price_rule_from_mapping(raw: Mapping[str, Any]) -> PriceRule | None:
    provider = str(raw.get("provider") or "openai").strip()
    model = str(raw.get("model") or "").strip()
    if not provider or not model:
        return None
    return PriceRule(
        provider=provider,
        model=model,
        api_surface=str(raw.get("api_surface") or raw.get("service") or "").strip(),
        service_tier=str(raw.get("service_tier") or "").strip(),
        context_tier=str(raw.get("context_tier") or "").strip(),
        input_usd_per_1m=_decimal(raw.get("input_usd_per_1m")),
        cached_input_usd_per_1m=_decimal(raw.get("cached_input_usd_per_1m")),
        cache_write_usd_per_1m=_decimal(raw.get("cache_write_usd_per_1m")),
        output_usd_per_1m=_decimal(raw.get("output_usd_per_1m")),
    )


def _decimal(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
