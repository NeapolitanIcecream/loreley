from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Literal, Protocol

from loguru import logger

log = logger.bind(module="worker.agent_backend")

__all__ = [
    "AgentBackend",
    "AgentInvocation",
    "StructuredAgentTask",
    "CodexCliBackend",
    "resolve_schema_mode",
]

SchemaMode = Literal["native", "prompt", "none"]


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


def resolve_schema_mode(configured_mode: str, api_spec: str) -> SchemaMode:
    """Resolve the effective schema mode from configuration and API spec."""
    if configured_mode != "auto":
        return configured_mode
    if api_spec == "chat_completions":
        return "prompt"
    return "native"


def _validate_workdir(
    working_dir: Path,
    *,
    error_cls: type[RuntimeError],
    agent_name: str,
) -> Path:
    path = Path(working_dir).expanduser().resolve()
    if not path.exists():
        raise error_cls(f"Working directory {path} does not exist.")
    if not path.is_dir():
        raise error_cls(f"Working directory {path} is not a directory.")
    git_dir = path / ".git"
    if not git_dir.exists():
        raise error_cls(
            f"{agent_name} requires a git repository at {path} (missing .git).",
        )
    return path


def _materialise_schema_to_temp(
    schema: dict[str, Any],
    *,
    error_cls: type[RuntimeError],
) -> Path:
    """Persist the given JSON schema to a temporary file."""
    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="loreley-agent-schema-",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        with tmp:
            json.dump(schema, tmp, ensure_ascii=True, indent=2)
        return Path(tmp.name)
    except Exception as exc:  # pragma: no cover - defensive
        raise error_cls(f"Failed to materialise agent schema: {exc}") from exc


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
        worktree = _validate_workdir(
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
                schema_path = _materialise_schema_to_temp(
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
            raise self.error_cls("codex exec returned an empty response.")

        return AgentInvocation(
            command=tuple(command),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
        )


