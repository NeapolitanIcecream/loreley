from __future__ import annotations

from decimal import Decimal

import httpx

from loreley.core.usage import LLMUsageEventPayload
from tools.run_v15_kilo_bakeoff import (
    conservative_input_tokens,
    optional_quota_snapshot,
    usage_counts,
)


def test_usage_counts_aggregates_all_kilo_model_calls() -> None:
    events = (
        LLMUsageEventPayload(
            source="kilo_cli",
            phase="coding",
            input_tokens=100,
            cached_input_tokens=20,
            output_tokens=30,
            reasoning_output_tokens=5,
            total_tokens=130,
            cost_usd=Decimal("0.01"),
        ),
        LLMUsageEventPayload(
            source="kilo_cli",
            phase="coding",
            input_tokens=40,
            output_tokens=10,
            total_tokens=50,
        ),
    )

    assert usage_counts(events) == {
        "input_tokens": 140,
        "cached_input_tokens": 20,
        "output_tokens": 40,
        "reasoning_output_tokens": 5,
        "total_tokens": 180,
    }
    assert conservative_input_tokens(usage_counts(events)) == 160


def test_optional_quota_snapshot_tolerates_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        snapshot = optional_quota_snapshot(
            client,
            base_url="https://proxy.example/v1",
        )

    assert snapshot == {
        "available": False,
        "error_type": "HTTPStatusError",
        "status_code": 429,
    }
