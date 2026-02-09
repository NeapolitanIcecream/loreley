from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tenacity import RetryError

from loreley.core.openai_retry import openai_retrying, retry_error_details


# ---------------------------------------------------------------------------
# openai_retrying
# ---------------------------------------------------------------------------


def test_openai_retrying_rejects_empty_retry_on_types() -> None:
    """openai_retrying raises ValueError when no retryable exception types are given."""
    with pytest.raises(ValueError, match="at least one retryable exception"):
        openai_retrying(
            max_attempts=3,
            backoff_seconds=0.0,
            retry_on=(),
            log=SimpleNamespace(warning=lambda *a, **k: None),
            operation="test",
        )


def test_openai_retrying_logs_warning_before_retry() -> None:
    """Observability: the before-sleep hook emits a warning on transient failure."""
    warnings: list[str] = []

    class StubLog:
        def warning(self, fmt: str, *args: object) -> None:
            warnings.append(fmt.format(*args) if args else fmt)

    call_count = 0

    def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("transient")
        return "ok"

    retryer = openai_retrying(
        max_attempts=3,
        backoff_seconds=0.0,
        retry_on=(ValueError,),
        log=StubLog(),
        operation="StubOp",
    )
    result = None
    for attempt in retryer:
        with attempt:
            result = flaky()

    assert result == "ok"
    assert any("StubOp" in w and "transient" in w for w in warnings)


def test_openai_retrying_raises_retry_error_after_exhausting_attempts() -> None:
    """After max_attempts failures, openai_retrying raises RetryError."""

    def always_fail() -> None:
        raise ValueError("boom")

    retryer = openai_retrying(
        max_attempts=2,
        backoff_seconds=0.0,
        retry_on=(ValueError,),
        log=SimpleNamespace(warning=lambda *a, **k: None),
        operation="Fail",
    )
    with pytest.raises(RetryError):
        for attempt in retryer:
            with attempt:
                always_fail()


# ---------------------------------------------------------------------------
# retry_error_details
# ---------------------------------------------------------------------------


def test_retry_error_details_extracts_attempt_count_and_last_exception() -> None:
    """retry_error_details returns attempt number and the underlying exception."""

    class DummyAttempt:
        def __init__(self, number: int, exc: BaseException) -> None:
            self.attempt_number = number
            self._exc = exc

        def exception(self) -> BaseException:
            return self._exc

    original = ValueError("boom")
    err = RetryError(DummyAttempt(3, original))  # type: ignore[arg-type]

    attempts, last_exc = retry_error_details(err)

    assert attempts == 3
    assert last_exc is original


def test_retry_error_details_uses_default_when_no_attempt_attached() -> None:
    """When RetryError has no last_attempt, fallback to *default_attempts*."""
    err = RetryError(None)  # type: ignore[arg-type]

    attempts, last_exc = retry_error_details(err, default_attempts=5)

    assert attempts == 5
    assert last_exc is None
