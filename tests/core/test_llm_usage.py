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
    assert event.total_tokens == 1520
    assert str(event.cost_usd) == "0.03"
    assert event.cost_source == "provider_reported"


def test_kilo_usage_parser_estimates_cost_when_provider_cost_is_missing(settings) -> None:
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
    assert event.cost_source == "estimated"
    assert event.pricing_version == "kilo-test"
    assert str(event.cost_usd) == "0.00021500"
