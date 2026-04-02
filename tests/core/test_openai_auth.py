from __future__ import annotations

from typing import Any

import pytest

from loreley.core.openai_auth import (
    DynamicOpenAIKeyManager,
    DynamicOpenAIKeyUnavailableError,
)


class _ManualClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def test_dynamic_key_manager_fetches_initial_shared_token() -> None:
    clock = _ManualClock()
    calls: list[str] = []

    def provider() -> str:
        calls.append("called")
        return "dyn-token-1"

    manager = DynamicOpenAIKeyManager(
        provider=provider,
        provider_ref="tests.provider:token",
        ttl_seconds=30,
        refresh_skew_seconds=5,
        monotonic=clock.monotonic,
        start_refresh_thread=False,
    )

    assert manager.get_shared_token() == "dyn-token-1"
    assert calls == ["called"]


def test_dynamic_key_manager_refreshes_before_expiry() -> None:
    clock = _ManualClock()
    values = iter(["dyn-token-1", "dyn-token-2"])

    manager = DynamicOpenAIKeyManager(
        provider=lambda: next(values),
        provider_ref="tests.provider:token",
        ttl_seconds=10,
        refresh_skew_seconds=2,
        monotonic=clock.monotonic,
        start_refresh_thread=False,
    )

    assert manager.get_shared_token() == "dyn-token-1"

    clock.advance(8)
    assert manager.refresh_if_due() == "dyn-token-2"
    assert manager.get_shared_token() == "dyn-token-2"


def test_dynamic_key_manager_reuses_still_valid_token_when_refresh_fails(
    captured_logs: list[dict[str, Any]],
) -> None:
    clock = _ManualClock()
    calls = {"count": 0}

    def provider() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            return "secret-token"
        raise RuntimeError("gateway down")

    manager = DynamicOpenAIKeyManager(
        provider=provider,
        provider_ref="tests.provider:token",
        ttl_seconds=10,
        refresh_skew_seconds=2,
        monotonic=clock.monotonic,
        start_refresh_thread=False,
    )

    assert manager.get_shared_token() == "secret-token"

    clock.advance(8)
    assert manager.refresh_if_due() == "secret-token"
    assert manager.get_shared_token() == "secret-token"

    warning_logs = [
        record
        for record in captured_logs
        if record["module"] == "core.openai_auth" and record["level"] == "WARNING"
    ]
    assert warning_logs
    assert all("secret-token" not in str(record["message"]) for record in warning_logs)


def test_dynamic_key_manager_raises_after_expiry_when_refresh_fails() -> None:
    clock = _ManualClock()
    calls = {"count": 0}

    def provider() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            return "dyn-token-1"
        raise RuntimeError("provider unavailable")

    manager = DynamicOpenAIKeyManager(
        provider=provider,
        provider_ref="tests.provider:token",
        ttl_seconds=10,
        refresh_skew_seconds=2,
        monotonic=clock.monotonic,
        start_refresh_thread=False,
    )

    assert manager.get_shared_token() == "dyn-token-1"

    clock.advance(10)
    with pytest.raises(DynamicOpenAIKeyUnavailableError, match="provider unavailable"):
        manager.get_shared_token()


def test_dynamic_key_manager_agent_token_does_not_overwrite_shared_cache() -> None:
    clock = _ManualClock()
    values = iter(["shared-token", "agent-token"])

    manager = DynamicOpenAIKeyManager(
        provider=lambda: next(values),
        provider_ref="tests.provider:token",
        ttl_seconds=10,
        refresh_skew_seconds=2,
        monotonic=clock.monotonic,
        start_refresh_thread=False,
    )

    assert manager.get_shared_token() == "shared-token"
    assert manager.get_agent_token() == "agent-token"
    assert manager.get_shared_token() == "shared-token"
