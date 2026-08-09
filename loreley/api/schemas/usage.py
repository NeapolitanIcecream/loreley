"""LLM usage API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from loreley.api.schemas import OrmOutModel


class UsageEventOut(OrmOutModel):
    id: UUID
    job_id: UUID | None = None
    run_token: UUID | None = None
    phase: str
    source: str
    provider: str
    model: str
    api_surface: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: str | None = None
    cost_source: str
    pricing_version: str
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    external_usage_id: str
    created_at: datetime

    @field_validator("cost_usd", mode="before")
    @classmethod
    def _cost_to_str(cls, value: object) -> str | None:
        if value is None:
            return None
        return str(value)


class UsageEventPageOut(OrmOutModel):
    items: list[UsageEventOut] = Field(default_factory=list)
    next_cursor: str | None = None


class UsageSummaryGroupOut(OrmOutModel):
    key: str
    event_count: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    cost_usd: str | None = None
    unpriced_events: int = 0
    unavailable_events: int = 0

    @field_validator("cost_usd", mode="before")
    @classmethod
    def _group_cost_to_str(cls, value: object) -> str | None:
        if value is None:
            return None
        return str(value)


class UsageSummaryOut(OrmOutModel):
    event_count: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    cost_usd: str | None = None
    unpriced_events: int = 0
    unavailable_events: int = 0
    by_source: list[UsageSummaryGroupOut] = Field(default_factory=list)
    by_phase: list[UsageSummaryGroupOut] = Field(default_factory=list)
    by_model: list[UsageSummaryGroupOut] = Field(default_factory=list)
    by_cost_source: list[UsageSummaryGroupOut] = Field(default_factory=list)

    @field_validator("cost_usd", mode="before")
    @classmethod
    def _summary_cost_to_str(cls, value: object) -> str | None:
        if value is None:
            return None
        return str(value)
