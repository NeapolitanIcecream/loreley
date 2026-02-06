from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True, frozen=True)
class AgentInvocation:
    """Result of a single agent backend invocation."""

    command: tuple[str, ...]
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass(slots=True)
class AgentTask:
    """Backend-agnostic description of an agent call."""

    name: str
    prompt: str


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

