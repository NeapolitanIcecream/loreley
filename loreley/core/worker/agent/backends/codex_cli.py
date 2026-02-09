from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from loguru import logger

from loreley.config import get_settings
from loreley.core.worker.agent.contracts import AgentInvocation, AgentTask
from loreley.core.worker.agent.utils import validate_workdir

log = logger.bind(module="worker.agent.backends.codex_cli")


@dataclass(slots=True)
class CodexCliBackend:
    """AgentBackend implementation that delegates to the Codex CLI."""

    bin: str
    profile: str | None
    timeout_seconds: int
    extra_env: dict[str, str]
    error_cls: type[RuntimeError]
    full_auto: bool = False

    def run(
        self,
        task: AgentTask,
        *,
        working_dir: Path,
    ) -> AgentInvocation:
        worktree = validate_workdir(
            working_dir,
            error_cls=self.error_cls,
            agent_name=task.name or "Agent",
        )

        command: list[str] = [self.bin, "exec"]
        if self.full_auto:
            command.append("--full-auto")

        if self.profile:
            command.extend(["--profile", self.profile])

        env = os.environ.copy()
        env.update(self.extra_env or {})

        start = monotonic()
        log.debug(
            "Running Codex CLI command: {} (cwd={}) for task={}",
            command,
            worktree,
            task.name,
        )
        try:
            result = subprocess.run(
                command,
                cwd=str(worktree),
                env=env,
                input=task.prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise self.error_cls(
                f"codex exec timed out after {self.timeout_seconds}s.",
            ) from exc

        duration = monotonic() - start
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        log.debug(
            "Codex CLI finished (exit_code={}, duration={:.2f}s) for task={}",
            result.returncode,
            duration,
            task.name,
        )

        if result.returncode != 0:
            raise self.error_cls(
                f"codex exec failed with exit code {result.returncode}. "
                f"stderr: {stderr or 'N/A'}",
            )

        if not stdout:
            log.warning(
                "Codex CLI produced an empty stdout payload for task={} (command={})",
                task.name,
                command,
            )

        return AgentInvocation(
            command=tuple(command),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
        )


def codex_planning_backend() -> CodexCliBackend:
    """Factory to build a Codex backend for the planning agent."""

    from loreley.core.worker.planning import PlanningError

    settings = get_settings()
    return CodexCliBackend(
        bin=settings.worker_planning_codex_bin,
        profile=settings.worker_planning_codex_profile,
        timeout_seconds=settings.worker_planning_timeout_seconds,
        extra_env=dict(settings.worker_planning_extra_env or {}),
        error_cls=PlanningError,
        full_auto=False,
    )


def codex_coding_backend() -> CodexCliBackend:
    """Factory to build a Codex backend for the coding agent."""

    from loreley.core.worker.coding import CodingError

    settings = get_settings()
    return CodexCliBackend(
        bin=settings.worker_coding_codex_bin,
        profile=settings.worker_coding_codex_profile,
        timeout_seconds=settings.worker_coding_timeout_seconds,
        extra_env=dict(settings.worker_coding_extra_env or {}),
        error_cls=CodingError,
        full_auto=True,
    )


__all__ = [
    "CodexCliBackend",
    "codex_coding_backend",
    "codex_planning_backend",
]

