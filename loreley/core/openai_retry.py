"""Shared Tenacity retry helpers for OpenAI calls.

Loreley has multiple OpenAI call sites (embeddings, summaries) that need
consistent retry/backoff behavior. This module centralizes the Tenacity
configuration and a loguru-friendly before-sleep hook.
"""

from __future__ import annotations

from typing import Any

from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_incrementing

__all__ = ["openai_retrying", "retry_error_details"]


def retry_error_details(
    exc: RetryError,
    *,
    default_attempts: int | None = None,
) -> tuple[int, BaseException | None]:
    """Extract attempt count and the last exception from a Tenacity RetryError."""

    last_attempt = getattr(exc, "last_attempt", None)
    attempts = getattr(last_attempt, "attempt_number", None) or default_attempts
    try:
        attempts_int = int(attempts or 0)
    except Exception:  # pragma: no cover - defensive
        attempts_int = int(default_attempts or 0)

    last_exc: BaseException | None = None
    try:
        last_exc = getattr(last_attempt, "exception", lambda: None)()
    except Exception:  # pragma: no cover - defensive
        last_exc = None

    return attempts_int, last_exc


def openai_retrying(
    *,
    max_attempts: int,
    backoff_seconds: float,
    retry_on: tuple[type[BaseException], ...],
    log: Any,
    operation: str,
) -> Retrying:
    """Return a configured Tenacity `Retrying` instance for OpenAI calls.

    Notes:
    - `max_attempts` maps to Tenacity's `stop_after_attempt(max_attempts)`.
    - `backoff_seconds` matches Loreley's existing linear backoff semantics:
      sleep = backoff_seconds * attempt_number (1-indexed).
    - `retry_on` controls which exception types are retried.
    """

    max_attempts = max(1, int(max_attempts))
    backoff_seconds = max(0.0, float(backoff_seconds))
    retry_on = tuple(retry_on or ())
    if not retry_on:
        raise ValueError("openai_retrying requires at least one retryable exception type.")

    operation = (operation or "OpenAI call").strip() or "OpenAI call"

    def _before_sleep(retry_state) -> None:  # type: ignore[no-untyped-def]
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        sleep = getattr(getattr(retry_state, "next_action", None), "sleep", None)
        attempt = getattr(retry_state, "attempt_number", None)
        if sleep is None:
            log.warning("{} attempt {} failed: {}. Retrying...", operation, attempt, exc)
            return
        log.warning(
            "{} attempt {} failed: {}. Retrying in {:.1f}s",
            operation,
            attempt,
            exc,
            float(sleep),
        )

    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_incrementing(start=backoff_seconds, increment=backoff_seconds),
        retry=retry_if_exception_type(retry_on),
        reraise=False,
        before_sleep=_before_sleep,
    )

