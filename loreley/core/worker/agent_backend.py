from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
import inspect
from importlib import import_module
from pathlib import Path
from time import monotonic
from typing import Any, Literal, Protocol, cast

from loguru import logger

log = logger.bind(module="worker.agent_backend")

__all__ = [
    "AgentBackend",
    "AgentInvocation",
    "StructuredAgentTask",
    "CodexCliBackend",
    "CursorCliBackend",
    "DEFAULT_CURSOR_MODEL",
    "resolve_schema_mode",
    "load_agent_backend",
    "cursor_backend_from_settings",
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
        return cast(SchemaMode, configured_mode)
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


def _split_backend_reference(ref: str) -> tuple[str, str]:
    """Split a backend reference into module and attribute path."""
    if ":" in ref:
        module_name, attr_path = ref.split(":", 1)
        return module_name, attr_path
    module_name, _, attr_path = ref.rpartition(".")
    if not module_name or not attr_path:
        raise RuntimeError(
            f"Invalid agent backend reference {ref!r}. Use 'module:attr' or 'module.attr'.",
        )
    return module_name, attr_path


def _import_backend_target(module_name: str, attr_path: str) -> Any:
    """Import the target object for a backend reference."""
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not import agent backend module {module_name!r}.",
        ) from exc
    target: Any = module
    for part in attr_path.split("."):
        if not part:
            raise RuntimeError(
                f"Invalid agent backend attribute reference {attr_path!r}.",
            )
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise RuntimeError(
                f"Module {module_name!r} does not expose attribute {attr_path!r}.",
            ) from exc
    return target


DEFAULT_CURSOR_MODEL = "gpt-5.2-high"


def load_agent_backend(ref: str, *, label: str) -> AgentBackend:
    """Resolve and instantiate an AgentBackend from a dotted reference.

    The reference can point to:
    - an already-instantiated backend object exposing a ``run(...)`` method
    - a class implementing the ``AgentBackend`` protocol (constructed with no arguments)
    - a callable factory that returns a backend instance when called with no arguments
    """
    module_name, attr_path = _split_backend_reference(ref)
    target = _import_backend_target(module_name, attr_path)

    # Already-instantiated backend instance.
    # Avoid treating classes as instances even though they expose a callable ``run`` attribute.
    if not inspect.isclass(target) and hasattr(target, "run") and callable(
        getattr(target, "run")
    ):
        return cast(AgentBackend, target)

    # Class or factory function returning a backend instance.
    if callable(target):
        instance = target()
        if hasattr(instance, "run") and callable(getattr(instance, "run")):
            return cast(AgentBackend, instance)
        raise RuntimeError(
            f"Resolved {label} {ref!r} callable did not return a valid AgentBackend "
            "(missing callable 'run' method).",
        )

    raise RuntimeError(
        f"Resolved {label} {ref!r} is not a valid AgentBackend "
        "(object must expose a callable 'run' method).",
    )


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
        worktree = _validate_workdir(
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


