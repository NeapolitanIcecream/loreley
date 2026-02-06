from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

from loreley.core.worker.agent.contracts import (
    AgentBackend,
    AgentInvocation,
    AgentTask,
)

ParsedT = TypeVar("ParsedT")


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
    for attempt in range(1, attempts + 1):
        if on_attempt_start is not None:
            on_attempt_start(attempt, attempts)

        invocation: AgentInvocation | None = None
        result: ParsedT | None = None
        try:
            invocation = backend.run(task, working_dir=working_dir)
            result = coerce_result(invocation)

            if post_check is not None:
                post_error = post_check(invocation, result)
                if post_error is not None:
                    last_error = post_error
                    if debug_hook is not None:
                        debug_hook(attempt, invocation, result, post_error)
                    if on_attempt_retry is not None:
                        on_attempt_retry(attempt, attempts, post_error)
                    continue

            if debug_hook is not None:
                debug_hook(attempt, invocation, result, None)
            if on_attempt_success is not None:
                on_attempt_success(attempt, attempts, invocation, result)
            return result, invocation, attempt
        except retryable_exceptions as exc:
            last_error = exc
            if debug_hook is not None:
                debug_hook(attempt, invocation, result, exc)
            if on_attempt_retry is not None:
                on_attempt_retry(attempt, attempts, exc)
            continue

    raise error_cls(error_message) from last_error


__all__ = [
    "run_agent_task",
]

