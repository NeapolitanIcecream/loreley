from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Callable, NoReturn, Sequence, TypeVar

from loreley.core.worker.agent.contracts import (
    AgentBackend,
    AgentInvocation,
    AgentTask,
)

ParsedT = TypeVar("ParsedT")
_FAILURE_REASON_CODE_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")


def run_agent_task(
    *,
    backend: AgentBackend,
    task: AgentTask,
    working_dir: Path,
    max_attempts: int,
    coerce_result: Callable[[AgentInvocation], ParsedT],
    retryable_exceptions: tuple[type[Exception], ...],
    error_cls: type[RuntimeError],
    error_message: str,
    debug_hook: Callable[[int, AgentInvocation | None, ParsedT | None, Exception | None], None]
    | None = None,
    on_attempt_start: Callable[[int, int], None] | None = None,
    on_attempt_success: Callable[[int, int, AgentInvocation, ParsedT], None] | None = None,
    on_attempt_retry: Callable[[int, int, Exception], None] | None = None,
    post_check: Callable[[AgentInvocation, ParsedT], Exception | None] | None = None,
) -> tuple[ParsedT, AgentInvocation, int]:
    """Run an agent task with retries, optional post-check, and debug hooks."""
    last_error: Exception | None = None
    attempts = max(1, int(max_attempts))
    usage_events: list[object] = []
    seen_usage_keys: set[tuple[str, str | int]] = set()
    for attempt in range(1, attempts + 1):
        if on_attempt_start is not None:
            on_attempt_start(attempt, attempts)

        attempt_task = replace(task, attempt=attempt)
        invocation: AgentInvocation | None = None
        result: ParsedT | None = None
        try:
            invocation = backend.run(attempt_task, working_dir=working_dir)
            result = coerce_result(invocation)

            if post_check is not None:
                post_error = post_check(invocation, result)
                if post_error is not None:
                    last_error = post_error
                    _append_usage_events(
                        usage_events,
                        invocation.usage_events,
                        seen_usage_keys=seen_usage_keys,
                    )
                    if debug_hook is not None:
                        debug_hook(attempt, invocation, result, post_error)
                    if on_attempt_retry is not None:
                        on_attempt_retry(attempt, attempts, post_error)
                    continue

            _append_usage_events(
                usage_events,
                invocation.usage_events,
                seen_usage_keys=seen_usage_keys,
            )
            if debug_hook is not None:
                debug_hook(attempt, invocation, result, None)
            if on_attempt_success is not None:
                on_attempt_success(attempt, attempts, invocation, result)
            return result, replace(invocation, usage_events=tuple(usage_events)), attempt
        except retryable_exceptions as exc:
            last_error = exc
            _append_usage_events(
                usage_events,
                invocation.usage_events if invocation is not None else _exception_usage_events(exc),
                seen_usage_keys=seen_usage_keys,
            )
            if debug_hook is not None:
                debug_hook(attempt, invocation, result, exc)
            if on_attempt_retry is not None:
                on_attempt_retry(attempt, attempts, exc)
            continue

    _raise_agent_task_error(
        error_cls=error_cls,
        error_message=error_message,
        last_error=last_error,
        usage_events=usage_events,
    )


def _raise_agent_task_error(
    *,
    error_cls: type[RuntimeError],
    error_message: str,
    last_error: Exception | None,
    usage_events: Sequence[object],
) -> NoReturn:
    failure_reason_code = _failure_reason_code(last_error)
    if failure_reason_code:
        error_message = f"{error_message} Failure reason: {failure_reason_code}."
    error = error_cls(error_message)
    if failure_reason_code:
        setattr(error, "failure_reason_code", failure_reason_code)
    if usage_events:
        setattr(error, "usage_events", tuple(usage_events))
    raise error from last_error


def _append_usage_events(
    collected: list[object],
    events: Sequence[object] | None,
    *,
    seen_usage_keys: set[tuple[str, str | int]],
) -> None:
    for event in events or ():
        key = _usage_event_key(event)
        if key in seen_usage_keys:
            continue
        seen_usage_keys.add(key)
        collected.append(event)


def _usage_event_key(event: object) -> tuple[str, str | int]:
    external_id = str(getattr(event, "external_usage_id", "") or "").strip()
    if external_id:
        return ("external_usage_id", external_id)
    return ("object_id", id(event))


def _exception_usage_events(exc: Exception) -> tuple[object, ...]:
    value = getattr(exc, "usage_events", ())
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _failure_reason_code(exc: Exception | None) -> str | None:
    value = str(getattr(exc, "failure_reason_code", "") or "").strip()
    if not _FAILURE_REASON_CODE_PATTERN.fullmatch(value):
        return None
    return value


__all__ = [
    "run_agent_task",
]
