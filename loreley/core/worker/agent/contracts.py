from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

SchemaMode = Literal["native", "prompt", "none"]
ValidationMode = Literal["strict", "lenient", "none"]


@dataclass(slots=True, frozen=True)
class AgentInvocation:
    """Result of a single agent backend invocation."""

    command: tuple[str, ...]
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass(slots=True)
class StructuredAgentTask:
    """Backend-agnostic description of a structured agent call."""

    name: str
    prompt: str
    schema: dict[str, Any] | None = None
    schema_mode: SchemaMode = "native"


class AgentBackend(Protocol):
    """Protocol implemented by planning/coding agent backends."""

    def run(
        self,
        task: StructuredAgentTask,
        *,
        working_dir: Path,
    ) -> AgentInvocation:
        ...


__all__ = [
    "AgentBackend",
    "AgentInvocation",
    "SchemaMode",
    "StructuredAgentTask",
    "ValidationMode",
]

