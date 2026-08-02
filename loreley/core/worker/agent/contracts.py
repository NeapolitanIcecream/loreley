from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from loreley.core.usage import LLMUsageEventPayload


@dataclass(slots=True, frozen=True)
class AgentInvocation:
    """Result of a single agent backend invocation."""

    command: tuple[str, ...]
    stdout: str
    stderr: str
    duration_seconds: float
    usage_events: tuple[LLMUsageEventPayload, ...] = ()
    working_directory: str | None = None


@dataclass(slots=True)
class AgentTask:
    """Backend-agnostic description of an agent call."""

    name: str
    prompt: str
    job_id: UUID | None = None
    run_token: UUID | None = None
    phase: str | None = None
    invocation: int | None = None
    attempt: int | None = None


class AgentBackend(Protocol):
    """Protocol implemented by planning/coding agent backends."""

    def run(
        self,
        task: AgentTask,
        *,
        working_dir: Path,
    ) -> AgentInvocation:
        ...


__all__ = [
    "AgentBackend",
    "AgentInvocation",
    "AgentTask",
]
