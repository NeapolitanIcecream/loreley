from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Iterator, Mapping
from uuid import UUID

COST_SOURCE_PROVIDER_REPORTED = "provider_reported"
COST_SOURCE_ESTIMATED = "estimated"
COST_SOURCE_UNPRICED = "unpriced"
COST_SOURCE_UNAVAILABLE = "unavailable"


def _uuid_or_none(value: UUID | str | None) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    return UUID(raw)


def _non_negative_int(value: object) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _decimal_or_none(value: Decimal | str | float | int | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def sanitized_usage_payload(value: object, *, depth: int = 0) -> Any:
    """Return a JSON-safe, bounded usage-only payload.

    Callers pass provider usage metadata, not prompts or transcripts. This
    helper still keeps the persisted raw_usage shape conservative.
    """

    if depth > 5:
        return str(type(value).__name__)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            return value[:512]
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:64]:
            out[str(key)[:128]] = sanitized_usage_payload(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [sanitized_usage_payload(item, depth=depth + 1) for item in value[:64]]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return sanitized_usage_payload(
                model_dump(mode="json", exclude_none=True),
                depth=depth + 1,
            )
        except TypeError:
            try:
                return sanitized_usage_payload(model_dump(), depth=depth + 1)
            except Exception:
                return str(type(value).__name__)
        except Exception:
            return str(type(value).__name__)
    if hasattr(value, "__dict__"):
        return sanitized_usage_payload(vars(value), depth=depth + 1)
    return str(value)[:512]


@dataclass(frozen=True, slots=True)
class UsageContext:
    job_id: UUID | None = None
    run_token: UUID | None = None
    phase: str | None = None


_CURRENT_USAGE_CONTEXT: ContextVar[UsageContext] = ContextVar(
    "loreley_llm_usage_context",
    default=UsageContext(),
)


def current_usage_context() -> UsageContext:
    return _CURRENT_USAGE_CONTEXT.get()


@contextmanager
def usage_context(
    *,
    job_id: UUID | str | None = None,
    run_token: UUID | str | None = None,
    phase: str | None = None,
) -> Iterator[UsageContext]:
    parent = current_usage_context()
    next_context = UsageContext(
        job_id=_uuid_or_none(job_id) or parent.job_id,
        run_token=_uuid_or_none(run_token) or parent.run_token,
        phase=(phase or parent.phase or None),
    )
    token = _CURRENT_USAGE_CONTEXT.set(next_context)
    try:
        yield next_context
    finally:
        _CURRENT_USAGE_CONTEXT.reset(token)


@dataclass(frozen=True, slots=True)
class LLMUsageEventPayload:
    """Provider-neutral LLM usage ledger event.

    The payload intentionally stores usage counters and bounded provider usage
    metadata only. It must not contain prompts, model outputs, secrets, or full
    CLI transcripts.
    """

    source: str
    phase: str
    provider: str = ""
    model: str = ""
    api_surface: str = ""
    job_id: UUID | None = None
    run_token: UUID | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Decimal | None = None
    cost_source: str = COST_SOURCE_UNPRICED
    pricing_version: str = ""
    raw_usage: Mapping[str, Any] = field(default_factory=dict)
    external_usage_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _uuid_or_none(self.job_id))
        object.__setattr__(self, "run_token", _uuid_or_none(self.run_token))
        for field_name in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        ):
            object.__setattr__(self, field_name, _non_negative_int(getattr(self, field_name)))
        if self.total_tokens <= 0:
            total = (
                self.input_tokens
                + self.cached_input_tokens
                + self.cache_write_tokens
                + self.output_tokens
            )
            object.__setattr__(self, "total_tokens", total)
        object.__setattr__(self, "cost_usd", _decimal_or_none(self.cost_usd))
        object.__setattr__(
            self,
            "raw_usage",
            sanitized_usage_payload(dict(self.raw_usage or {})),
        )

    def with_context(
        self,
        *,
        job_id: UUID | str | None = None,
        run_token: UUID | str | None = None,
        phase: str | None = None,
    ) -> "LLMUsageEventPayload":
        return replace(
            self,
            job_id=_uuid_or_none(job_id) or self.job_id,
            run_token=_uuid_or_none(run_token) or self.run_token,
            phase=phase or self.phase,
        )

    def with_external_usage_id(self, external_usage_id: str) -> "LLMUsageEventPayload":
        return replace(self, external_usage_id=str(external_usage_id or "").strip())

    def as_artifact_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "phase": self.phase,
            "provider": self.provider,
            "model": self.model,
            "api_surface": self.api_surface,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": str(self.cost_usd) if self.cost_usd is not None else None,
            "cost_source": self.cost_source,
            "pricing_version": self.pricing_version,
        }

    def as_record_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_token": self.run_token,
            "phase": self.phase,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "api_surface": self.api_surface,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "cost_source": self.cost_source,
            "pricing_version": self.pricing_version,
            "raw_usage": dict(self.raw_usage or {}),
            "external_usage_id": self.external_usage_id,
        }
