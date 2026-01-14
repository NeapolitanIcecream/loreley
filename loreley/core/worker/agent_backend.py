from __future__ import annotations

import json
import tempfile
import inspect
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TypeVar, cast

from loguru import logger

log = logger.bind(module="worker.agent_backend")

__all__ = [
    "AgentBackend",
    "AgentInvocation",
    "StructuredAgentTask",
    "TruncationMixin",
    "ValidationMode",
    "build_structured_agent_task",
    "coerce_structured_output",
    "resolve_schema_mode",
    "resolve_worker_debug_dir",
    "run_structured_agent_task",
    "truncate_text",
    "load_agent_backend",
]

SchemaMode = Literal["native", "prompt", "none"]
ValidationMode = Literal["strict", "lenient", "none"]

ParsedT = TypeVar("ParsedT")


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


def truncate_text(text: str, *, limit: int) -> str:
    """Return a whitespace-trimmed string truncated to the specified limit."""
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[:limit]}…"


class TruncationMixin:
    """Provide a consistent text truncation helper for worker agents."""

    _truncate_limit: int

    def _truncate(self, text: str, limit: int | None = None) -> str:
        active_limit = int(limit or self._truncate_limit)
        return truncate_text(text, limit=active_limit)


def resolve_worker_debug_dir(*, logs_base_dir: str | None, kind: str) -> Path:
    """Resolve directory for worker debug artifacts under logs/worker/{kind}."""
    if logs_base_dir:
        base_dir = Path(logs_base_dir).expanduser()
    else:
        base_dir = Path.cwd()
    logs_root = base_dir / "logs" / "worker" / kind
    logs_root.mkdir(parents=True, exist_ok=True)
    return logs_root


def build_structured_agent_task(
    *,
    name: str,
    prompt: str,
    schema: dict[str, Any] | None,
    schema_mode: SchemaMode,
    validation_mode: ValidationMode,
) -> StructuredAgentTask:
    """Build a StructuredAgentTask whose schema enforcement matches the validation mode."""
    if validation_mode in ("strict", "lenient"):
        return StructuredAgentTask(
            name=name,
            prompt=prompt,
            schema=schema,
            schema_mode=schema_mode,
        )
    return StructuredAgentTask(
        name=name,
        prompt=prompt,
        schema=None,
        schema_mode="none",
    )


def coerce_structured_output(
    *,
    validation_mode: ValidationMode,
    stdout: str,
    parse: Callable[[str], ParsedT],
    build_from_freeform: Callable[[str], ParsedT],
    on_parse_error: Callable[[Exception], None] | None = None,
    parse_exceptions: tuple[type[Exception], ...] = (json.JSONDecodeError,),
) -> ParsedT:
    """Coerce backend stdout into a structured value, honouring the validation mode."""
    if validation_mode == "strict":
        return parse(stdout)
    if validation_mode == "lenient":
        try:
            return parse(stdout)
        except parse_exceptions as exc:
            if on_parse_error is not None:
                on_parse_error(exc)
            return build_from_freeform(stdout)
    return build_from_freeform(stdout)


def run_structured_agent_task(
    *,
    backend: AgentBackend,
    task: StructuredAgentTask,
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
    """Run a structured agent task with retries, optional post-check, and debug hooks."""
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
