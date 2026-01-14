from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from loguru import logger

from loreley.core.worker.agent.contracts import AgentInvocation, StructuredAgentTask
from loreley.core.worker.agent.utils import materialise_schema_to_temp, validate_workdir

log = logger.bind(module="worker.agent.backends.codex_cli")


@dataclass(slots=True)
class CodexCliBackend:
    """AgentBackend implementation that delegates to the Codex CLI."""

    bin: str
    profile: str | None
    timeout_seconds: int
    extra_env: dict[str, str]
    schema_override: str | None
    error_cls: type[RuntimeError]
    full_auto: bool = False

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

        command: list[str] = [self.bin, "exec"]
        if self.full_auto:
            command.append("--full-auto")

        schema_path: Path | None = None
        cleanup_path: Path | None = None

        if task.schema_mode == "native":
            if self.schema_override:
                path = Path(self.schema_override).expanduser().resolve()
                if not path.exists():
                    raise self.error_cls(
                        f"Configured agent schema {path} does not exist.",
                    )
                schema_path = path
            else:
                if not task.schema:
                    raise self.error_cls(
                        "Schema mode 'native' requires an output schema definition.",
                    )
                schema_path = materialise_schema_to_temp(
                    task.schema,
                    error_cls=self.error_cls,
                )
                cleanup_path = schema_path

            command.extend(["--output-schema", str(schema_path)])

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
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

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


__all__ = ["CodexCliBackend"]

