from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from loguru import logger

from loreley.config import get_settings
from loreley.core.worker.agent.contracts import AgentInvocation, AgentTask
from loreley.core.worker.agent.utils import validate_workdir

log = logger.bind(module="worker.agent.backends.kilocode_cli")


@dataclass(slots=True)
class KilocodeCliBackend:
    """AgentBackend implementation that delegates to the Kilocode CLI.

    Uses Kilocode's autonomous (non-interactive) mode via the ``--auto`` flag,
    suitable for automated CI/CD pipelines and headless agent orchestration.
    The prompt is passed as a positional argument; ``--json`` enables structured
    output for downstream parsing.
    """

    bin: str = "kilocode"
    mode: str | None = None
    timeout_seconds: int = 1800
    extra_env: dict[str, str] = field(default_factory=dict)
    json_output: bool = True
    error_cls: type[RuntimeError] = RuntimeError

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

        command: list[str] = [self.bin, "--auto"]

        if self.json_output:
            command.append("--json")

        if self.mode:
            command.extend(["--mode", self.mode])

        command.append(task.prompt)
        command_for_log = command[:-1] + [f"<prompt:{len(task.prompt)} chars>"]

        env = os.environ.copy()
        env.update(self.extra_env or {})

        start = monotonic()
        log.debug(
            "Running Kilocode CLI command: {} (cwd={}) for task={}",
            command_for_log,
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
                f"kilocode timed out after {self.timeout_seconds}s.",
            ) from exc

        duration = monotonic() - start
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        log.debug(
            "Kilocode CLI finished (exit_code={}, duration={:.2f}s) for task={}",
            result.returncode,
            duration,
            task.name,
        )

        if result.returncode != 0:
            raise self.error_cls(
                f"kilocode failed with exit code {result.returncode}. "
                f"stderr: {stderr or 'N/A'}",
            )

        if not stdout:
            log.warning(
                "Kilocode CLI produced an empty stdout payload for task={} (command={})",
                task.name,
                command_for_log,
            )

        return AgentInvocation(
            command=tuple(command),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
        )


def kilocode_backend() -> KilocodeCliBackend:
    """Factory to build a Kilocode backend using env-only settings."""

    settings = get_settings()
    bin_value = getattr(settings, "worker_kilocode_bin", "kilocode")
    mode_value = getattr(settings, "worker_kilocode_mode", None)
    json_output_value = getattr(settings, "worker_kilocode_json_output", True)
    return KilocodeCliBackend(
        bin=str(bin_value),
        mode=str(mode_value) if mode_value else None,
        json_output=bool(json_output_value),
    )


__all__ = ["KilocodeCliBackend", "kilocode_backend"]
