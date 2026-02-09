from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


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


def resolve_worker_debug_dir(
    *,
    logs_base_dir: str | None,
    kind: str,
    experiment_id: uuid.UUID | str | None = None,
) -> Path:
    """Resolve directory for worker debug artifacts under logs/{experiment_namespace}/worker/{kind}."""
    if logs_base_dir:
        base_dir = Path(logs_base_dir).expanduser()
    else:
        base_dir = Path.cwd()
    from loreley.naming import safe_namespace_or_none

    exp_ns = safe_namespace_or_none(experiment_id)
    root = (base_dir / "logs" / exp_ns) if exp_ns else (base_dir / "logs")
    logs_root = root / "worker" / kind
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


__all__ = [
    "TruncationMixin",
    "coerce_agent_stdout_text",
    "resolve_worker_debug_dir",
    "truncate_text",
    "validate_workdir",
]


def coerce_agent_stdout_text(stdout: str) -> str:
    """Return best-effort plain text payload from agent stdout.

    Some agent CLIs can wrap the human-readable Markdown output in JSON (for example
    when running in a structured output mode). Loreley treats planning/coding
    backend output as plain text, so this helper unwraps common JSON shapes and
    falls back to the raw stdout when no suitable text payload is found.
    """

    raw = (stdout or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("{", "[")):
        return raw

    try:
        payload: Any = json.loads(raw)
    except Exception:
        return raw

    def _score(candidate: str) -> int:
        text = (candidate or "").strip()
        if not text:
            return -1
        score = 0
        if "##" in text:
            score += 50
        if text.lstrip().startswith("#"):
            score += 30
        if "\n" in text:
            score += 10
        lowered = text.lower()
        if "## summary" in lowered or "\n## summary" in lowered:
            score += 15
        if "- " in text:
            score += 5
        score += min(len(text), 4000) // 40
        return score

    preferred_keys = (
        "markdown",
        "output_markdown",
        "output",
        "output_text",
        "text",
        "content",
        "message",
        "result",
        "response",
        "final",
        "answer",
    )

    def _best_text(value: Any, *, depth: int) -> str:
        if depth <= 0:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in preferred_keys:
                if key in value and isinstance(value[key], str) and value[key].strip():
                    return str(value[key])
            best = ""
            best_score = -1
            for item in value.values():
                candidate = _best_text(item, depth=depth - 1)
                score = _score(candidate)
                if score > best_score:
                    best = candidate
                    best_score = score
            return best
        if isinstance(value, list):
            best = ""
            best_score = -1
            for item in value:
                candidate = _best_text(item, depth=depth - 1)
                score = _score(candidate)
                if score > best_score:
                    best = candidate
                    best_score = score
            return best
        return ""

    extracted = _best_text(payload, depth=7).strip()
    if extracted and _score(extracted) >= 0:
        return extracted
    return raw

