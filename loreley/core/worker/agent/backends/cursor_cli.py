from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

from loguru import logger

from loreley.core.worker.agent.contracts import AgentInvocation, StructuredAgentTask
from loreley.core.worker.agent.utils import validate_workdir

log = logger.bind(module="worker.agent.backends.cursor_cli")

DEFAULT_CURSOR_MODEL = "gpt-5.2-high"


@dataclass(slots=True)
class CursorCliBackend:
    """AgentBackend implementation that delegates to the Cursor Agent CLI.

    This backend runs ``cursor-agent`` in non-interactive mode, forwarding the
    structured prompt via ``-p`` and capturing plain-text output. It relies on
    prompt engineering (rather than a native JSON schema API) to obtain
    structured JSON results.
    """

    bin: str = "cursor-agent"
    model: str | None = DEFAULT_CURSOR_MODEL
    timeout_seconds: int = 1800
    extra_env: dict[str, str] = field(default_factory=dict)
    output_format: str = "text"
    force: bool = True
    error_cls: type[RuntimeError] = RuntimeError

    def run(
        self,
        task: StructuredAgentTask,
        *,
        working_dir: Path,
    ) -> AgentInvocation:
        worktree = validate_workdir(
            working_dir,
            error_cls=self.error_cls,
            agent_name=task.name or "Agent",
        )

        command: list[str] = [self.bin]

        if task.prompt:
            command.extend(["-p", task.prompt])

        if self.model:
            command.extend(["--model", self.model])

        if self.output_format:
            command.extend(["--output-format", self.output_format])

        if self.force:
            command.append("--force")

        env = os.environ.copy()
        env.update(self.extra_env or {})

        start = monotonic()
        log.debug(
            "Running Cursor CLI command: {} (cwd={}) for task={}",
            command,
            worktree,
            task.name,
        )
        try:
            result = subprocess.run(
                command,
                cwd=str(worktree),
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise self.error_cls(
                f"cursor-agent timed out after {self.timeout_seconds}s.",
            ) from exc

        duration = monotonic() - start
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        log.debug(
            "Cursor CLI finished (exit_code={}, duration={:.2f}s) for task={}",
            result.returncode,
            duration,
            task.name,
        )

        if result.returncode != 0:
            raise self.error_cls(
                f"cursor-agent failed with exit code {result.returncode}. "
                f"stderr: {stderr or 'N/A'}",
            )

        if not stdout:
            log.warning(
                "Cursor CLI produced an empty stdout payload for task={} (command={})",
                task.name,
                command,
            )

        return AgentInvocation(
            command=tuple(command),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
        )


def cursor_backend_from_settings(
    *,
    settings: Any | None = None,
    error_cls: type[RuntimeError] = RuntimeError,
) -> CursorCliBackend:
    """Factory to build a Cursor backend using configured defaults."""
    if settings is None:
        from loreley.config import get_settings

        settings = get_settings()

    model = getattr(settings, "worker_cursor_model", DEFAULT_CURSOR_MODEL)
    force = getattr(settings, "worker_cursor_force", True)
    return CursorCliBackend(
        model=model or DEFAULT_CURSOR_MODEL,
        force=force,
        error_cls=error_cls,
    )


__all__ = ["CursorCliBackend", "DEFAULT_CURSOR_MODEL", "cursor_backend_from_settings"]

