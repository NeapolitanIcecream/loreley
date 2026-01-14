from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, cast

from loreley.core.worker.agent.contracts import SchemaMode


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


def validate_workdir(
    working_dir: Path,
    *,
    error_cls: type[RuntimeError],
    agent_name: str,
) -> Path:
    """Expand and validate that the working directory is a git repository."""
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


def materialise_schema_to_temp(
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


__all__ = [
    "TruncationMixin",
    "materialise_schema_to_temp",
    "resolve_schema_mode",
    "resolve_worker_debug_dir",
    "truncate_text",
    "validate_workdir",
]

