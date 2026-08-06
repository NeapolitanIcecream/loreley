from __future__ import annotations

import json
from types import SimpleNamespace

from loreley.core.usage import (
    codex_usage_event_from_jsonl,
    kilo_usage_event_from_messages,
    normalize_openai_usage_event,
)


def test_openai_responses_usage_normalizer_prices_cached_and_output_tokens(settings) -> None:
    settings.llm_usage_pricing_json = json.dumps(
        {
            "version": "test-prices",
            "prices": [
                {
                    "provider": "openai",
                    "model": "gpt-test",
                    "api_surface": "responses",
                    "input_usd_per_1m": "0.40",
                    "cached_input_usd_per_1m": "0.10",
                    "output_usd_per_1m": "1.60",
                }
            ],
        }
    )
    response = SimpleNamespace(
        id="resp-1",
        usage=SimpleNamespace(
            input_tokens=1000,
            input_tokens_details=SimpleNamespace(cached_tokens=200),
            output_tokens=300,
            output_tokens_details=SimpleNamespace(reasoning_tokens=75),
            total_tokens=1300,
        ),
    )

    event = normalize_openai_usage_event(
        response,
        phase="commit_summary",
        model="gpt-test",
        api_surface="responses",
        settings=settings,
    )

    assert event is not None
    assert event.input_tokens == 1000
    assert event.cached_input_tokens == 200
    assert event.output_tokens == 300
    assert event.reasoning_output_tokens == 75
    assert event.total_tokens == 1300
    assert str(event.cost_usd) == "0.00082000"
    assert event.cost_source == "estimated"
    assert event.pricing_version == "test-prices"


def test_openai_chat_and_embeddings_usage_normalizer_handles_missing_prices(settings) -> None:
    settings.llm_usage_pricing_json = json.dumps({"version": "empty", "prices": []})
    chat_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4),
            completion_tokens=5,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
            total_tokens=15,
        )
    )
    embedding_response = SimpleNamespace(
        usage={"prompt_tokens": 20, "total_tokens": 20},
    )

    chat_event = normalize_openai_usage_event(
        chat_response,
        phase="trajectory_summary",
        model="gpt-unpriced",
        api_surface="chat_completions",
        settings=settings,
    )
    embedding_event = normalize_openai_usage_event(
        embedding_response,
        phase="embedding",
        model="embed-unpriced",
        api_surface="embeddings",
        settings=settings,
    )

    assert chat_event is not None
    assert chat_event.input_tokens == 10
    assert chat_event.cached_input_tokens == 4
    assert chat_event.output_tokens == 5
    assert chat_event.cost_source == "unpriced"
    assert embedding_event is not None
    assert embedding_event.input_tokens == 20
    assert embedding_event.output_tokens == 0
    assert embedding_event.total_tokens == 20


def test_codex_usage_parser_uses_last_token_count_aggregate(settings) -> None:
    settings.llm_usage_pricing_json = json.dumps(
        {
            "version": "codex-test",
            "prices": [
                {
                    "provider": "openai",
                    "model": "gpt-codex",
                    "api_surface": "codex_exec",
                    "input_usd_per_1m": "1",
                    "cached_input_usd_per_1m": "0.25",
                    "output_usd_per_1m": "2",
                }
            ],
        }
    )
    first = {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "output_tokens": 5,
                    "reasoning_output_tokens": 1,
                    "total_tokens": 15,
                }
            },
        },
    }
    last = {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 30,
                    "reasoning_output_tokens": 9,
                    "total_tokens": 130,
                }
            },
        },
    }

    event = codex_usage_event_from_jsonl(
        f"{json.dumps(first)}\nnot-json\n{json.dumps(last)}\n",
        phase="coding",
        model="gpt-codex",
        settings=settings,
    )

    assert event is not None
    assert event.input_tokens == 100
    assert event.cached_input_tokens == 20
    assert event.output_tokens == 30
    assert event.reasoning_output_tokens == 9
    assert event.total_tokens == 130
    assert str(event.cost_usd) == "0.00014500"


def test_kilo_usage_parser_extracts_tokens_cache_and_provider_cost() -> None:
    messages = [
        {"role": "user", "tokens": {"input": 999}},
        {
            "role": "assistant",
            "providerID": "openrouter",
            "modelID": "openai/gpt-5.2",
            "cost": 0.01,
            "tokens": {
                "input": 100,
                "output": 40,
                "reasoning": 7,
                "cache": {"read": 1000, "write": 50},
            },
        },
        {
            "role": "assistant",
            "providerID": "openrouter",
            "modelID": "openai/gpt-5.2",
            "cost": "0.02",
            "tokens": {
                "input": 20,
                "output": 10,
                "reasoning": 1,
                "cache": {"read": 300, "write": 0},
            },
        },
    ]

    event = kilo_usage_event_from_messages(
        messages,
        phase="planning",
        title="loreley:test",
        session_id="sess-1",
    )

    assert event is not None
    assert event.provider == "openrouter"
    assert event.model == "openai/gpt-5.2"
    assert event.input_tokens == 120
    assert event.cached_input_tokens == 1300
    assert event.cache_write_tokens == 50
    assert event.output_tokens == 50
    assert event.reasoning_output_tokens == 8
    assert event.total_tokens == 1528
    assert str(event.cost_usd) == "0.03"
    assert event.cost_source == "provider_reported"


def test_kilo_usage_parser_does_not_estimate_missing_kilo_cost(settings) -> None:
    settings.llm_usage_pricing_json = json.dumps(
        {
            "version": "kilo-test",
            "prices": [
                {
                    "provider": "openrouter",
                    "model": "openai/gpt-5.2",
                    "api_surface": "kilo_run",
                    "input_usd_per_1m": "1",
                    "cached_input_usd_per_1m": "0.25",
                    "cache_write_usd_per_1m": "0.50",
                    "output_usd_per_1m": "2",
                }
            ],
        }
    )
    messages = [
        {
            "role": "assistant",
            "providerID": "openrouter",
            "modelID": "openai/gpt-5.2",
            "tokens": {
                "input": 100,
                "output": 40,
                "cache": {"read": 120, "write": 10},
            },
        },
    ]

    event = kilo_usage_event_from_messages(
        messages,
        phase="coding",
        title="loreley:test",
        session_id="sess-1",
        settings=settings,
    )

    assert event is not None
    assert event.input_tokens == 100
    assert event.cached_input_tokens == 120
    assert event.cost_source == "unpriced"
    assert event.pricing_version == ""
    assert event.cost_usd is None


def test_kilo_usage_parser_rejects_zero_cost_gateway_placeholder(settings) -> None:
    settings.llm_usage_pricing_json = json.dumps(
        {
            "version": "gateway-test",
            "prices": [
                {
                    "provider": "loreley-openai-compatible",
                    "model": "gpt-test",
                    "api_surface": "kilo_run",
                    "input_usd_per_1m": "0.9",
                    "cached_input_usd_per_1m": "0.9",
                    "output_usd_per_1m": "5.4",
                }
            ],
        }
    )
    messages = [
        {
            "role": "assistant",
            "providerID": "loreley-openai-compatible",
            "modelID": "gpt-test",
            "cost": 0,
            "tokens": {
                "input": 100,
                "output": 20,
                "reasoning": 10,
                "cache": {"read": 50, "write": 0},
            },
        },
    ]

    event = kilo_usage_event_from_messages(
        messages,
        phase="coding",
        title="loreley:test",
        session_id="sess-zero-cost",
        settings=settings,
    )

    assert event is not None
    assert event.cost_source == "unpriced"
    assert event.pricing_version == ""
    assert event.cost_usd is None


def test_kilo_usage_parser_marks_zero_cost_placeholder_unpriced_without_rule(settings) -> None:
    settings.llm_usage_pricing_json = json.dumps(
        {"version": "no-matching-price", "prices": []}
    )
    messages = [
        {
            "role": "assistant",
            "providerID": "loreley-openai-compatible",
            "modelID": "deepseek-v4-flash",
            "cost": 0,
            "tokens": {
                "input": 100,
                "output": 20,
                "reasoning": 10,
                "cache": {"read": 50, "write": 0},
            },
        },
    ]

    event = kilo_usage_event_from_messages(
        messages,
        phase="coding",
        title="loreley:test",
        session_id="sess-zero-cost-unpriced",
        settings=settings,
    )

    assert event is not None
    assert event.cost_usd is None
    assert event.cost_source == "unpriced"
    assert event.pricing_version == ""


def test_kilo_session_aggregates_are_authoritative() -> None:
    messages = [
        {
            "role": "assistant",
            "providerID": "deepseek",
            "modelID": "deepseek-v4-flash",
            "cost": "999",
            "tokens": {"input": 999, "output": 999},
        }
    ]
    sessions = [
        {
            "id": "root",
            "model": '{"providerID":"deepseek","id":"deepseek-v4-flash"}',
            "cost": 0.02,
            "tokens_input": 10,
            "tokens_output": 2,
            "tokens_reasoning": 1,
            "tokens_cache_read": 20,
            "tokens_cache_write": 0,
        }
    ]

    event = kilo_usage_event_from_messages(
        messages,
        session_rows=sessions,
        title="loreley:session-authoritative",
        session_id="root",
    )

    assert event is not None
    assert event.input_tokens == 10
    assert event.cached_input_tokens == 20
    assert event.output_tokens == 2
    assert event.reasoning_output_tokens == 1
    assert event.total_tokens == 33
    assert str(event.cost_usd) == "0.02"
    assert event.raw_usage["accounting_source"] == "kilo_session_tree"
    assert "cost_reconciliation" not in event.raw_usage
