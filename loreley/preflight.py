from __future__ import annotations

"""Preflight checks for Loreley.

This module centralizes "fail fast" validation that is useful for:
- CLI entrypoints (`loreley ...`)
- Convenience scripts under `script/`

The goal is to surface actionable configuration errors before long-running
processes start (scheduler / worker).
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
from urllib.parse import urlparse, urlunparse

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.text import Text

from loreley.config import Settings

log = logger.bind(module="preflight")

Status = Literal["ok", "warn", "fail"]


@dataclass(slots=True)
class CheckResult:
    """Single preflight check outcome."""

    name: str
    status: Status
    details: str


def _status_text(status: Status) -> Text:
    styles = {"ok": "bold green", "warn": "bold yellow", "fail": "bold red"}
    return Text(status.upper(), style=styles.get(status, "bold"))


def _sanitize_url(raw: str) -> str:
    """Best-effort redaction for credential-bearing URLs."""
    value = (raw or "").strip()
    if not value:
        return value

    parsed = urlparse(value)
    if not parsed.scheme:
        return value

    netloc = parsed.hostname or ""
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    safe = parsed._replace(netloc=netloc)
    return urlunparse(safe)


def _sanitize_sqlalchemy_dsn(raw: str) -> str:
    """Hide passwords in SQLAlchemy DSNs when logging."""
    try:
        from sqlalchemy.engine.url import make_url
    except Exception:
        return _sanitize_url(raw)

    try:
        url = make_url(raw)
        if url.password:
            url = url.set(password="***")
        return str(url)
    except Exception:
        return _sanitize_url(raw)


def render_results(console: Console, results: Sequence[CheckResult], *, title: str = "Loreley preflight") -> None:
    """Render results as a Rich table."""
    table = Table(title=title, show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")
    for item in results:
        table.add_row(item.name, _status_text(item.status), item.details)
    console.print(table)


def summarize(results: Sequence[CheckResult]) -> tuple[int, int, int]:
    """Return (ok, warn, fail) counts."""
    ok = sum(1 for r in results if r.status == "ok")
    warn = sum(1 for r in results if r.status == "warn")
    fail = sum(1 for r in results if r.status == "fail")
    return ok, warn, fail


def has_failures(results: Sequence[CheckResult], *, treat_warnings_as_errors: bool = False) -> bool:
    """Return True when the result set should block startup."""
    _, warn, fail = summarize(results)
    if fail:
        return True
    return bool(warn and treat_warnings_as_errors)


def to_json(results: Sequence[CheckResult]) -> str:
    """Serialize results as JSON."""
    payload = [{"name": r.name, "status": r.status, "details": r.details} for r in results]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def check_binary(name: str, *, label: str) -> CheckResult:
    """Check that a binary exists on PATH."""
    resolved = shutil.which(name)
    if resolved:
        return CheckResult(label, "ok", f"found: {resolved}")
    return CheckResult(label, "fail", f"missing: {name!r} (not on PATH)")


def check_python_modules(
    modules: Sequence[str],
    *,
    label: str,
    install_hint: str,
) -> CheckResult:
    """Check that a set of Python modules are importable."""
    missing: list[str] = []
    for module in modules:
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if not missing:
        return CheckResult(label, "ok", "installed")
    return CheckResult(
        label,
        "warn",
        f"missing modules: {missing}. {install_hint}",
    )


def check_database(*, dsn: str, timeout_seconds: float) -> CheckResult:
    """Check that the configured database DSN is reachable."""
    safe = _sanitize_sqlalchemy_dsn(dsn)
    try:
        from sqlalchemy import create_engine, text

        connect_args: dict[str, object] = {"connect_timeout": int(max(1, timeout_seconds))}
        try:
            engine = create_engine(dsn, pool_pre_ping=True, connect_args=connect_args)
        except TypeError:
            # Some dialects may not accept connect_args in older versions; best-effort.
            engine = create_engine(dsn, pool_pre_ping=True)

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return CheckResult("database", "ok", f"reachable: {safe}")
    except Exception as exc:
        return CheckResult("database", "fail", f"unreachable: {safe} ({exc})")


def check_redis(
    *,
    redis_url: str | None,
    host: str,
    port: int,
    db: int,
    password: str | None,
    timeout_seconds: float,
) -> CheckResult:
    """Check that Redis is reachable using the configured broker settings."""
    safe = _sanitize_url(redis_url) if redis_url else f"redis://{host}:{int(port)}/{int(db)}"
    try:
        import redis  # type: ignore[import-not-found]

        if redis_url:
            client = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=timeout_seconds,
                socket_timeout=timeout_seconds,
                decode_responses=True,
            )
        else:
            client = redis.Redis(
                host=host,
                port=int(port),
                db=int(db),
                password=password or None,
                socket_connect_timeout=timeout_seconds,
                socket_timeout=timeout_seconds,
                decode_responses=True,
            )
        client.ping()
        return CheckResult("redis", "ok", f"reachable: {safe}")
    except Exception as exc:
        return CheckResult("redis", "fail", f"unreachable: {safe} ({exc})")


def check_git_repo(path: Path, *, label: str) -> CheckResult:
    """Check that a path is a git repository."""
    try:
        from git import Repo
        from git.exc import InvalidGitRepositoryError, NoSuchPathError
    except Exception as exc:
        return CheckResult(label, "fail", f"GitPython not available ({exc})")

    try:
        Repo(str(path))
        return CheckResult(label, "ok", f"git repo: {path}")
    except (NoSuchPathError, InvalidGitRepositoryError) as exc:
        return CheckResult(label, "fail", f"not a git repo: {path} ({exc})")
    except Exception as exc:
        return CheckResult(label, "fail", f"failed to inspect repo: {path} ({exc})")


def check_non_empty(value: str | None, *, label: str, env_name: str, help_text: str) -> CheckResult:
    """Check that a string config value is non-empty."""
    if (value or "").strip():
        return CheckResult(label, "ok", "configured")
    return CheckResult(label, "fail", f"{env_name} is not set ({help_text})")


def check_openai_api_key(value: str | None, *, required: bool) -> CheckResult:
    """Check OPENAI_API_KEY presence.

    Note:
    - Some OpenAI-compatible gateways accept dummy keys. In that case, set any
      non-empty OPENAI_API_KEY value to satisfy this check.
    """
    if (value or "").strip():
        return CheckResult("openai_api_key", "ok", "configured")
    if required:
        return CheckResult(
            "openai_api_key",
            "fail",
            "OPENAI_API_KEY is not set (required for embeddings / summaries).",
        )
    return CheckResult(
        "openai_api_key",
        "warn",
        "OPENAI_API_KEY is not set (OpenAI API calls may fail).",
    )


def check_evaluator_plugin(
    *,
    plugin_ref: str | None,
    python_paths: Sequence[str],
    settings: Settings,
) -> CheckResult:
    """Check that the evaluator plugin is configured and importable."""
    if not (plugin_ref or "").strip():
        return CheckResult(
            "evaluator_plugin",
            "fail",
            "WORKER_EVALUATOR_PLUGIN is not set (required for the worker).",
        )

    missing_paths = [
        entry
        for entry in python_paths
        if entry and not Path(entry).expanduser().resolve().exists()
    ]
    if missing_paths:
        return CheckResult("evaluator_plugin", "fail", f"python path(s) not found: {missing_paths}")

    try:
        # NOTE: We intentionally call a private method here to validate that
        # the configured plugin can be imported and resolved in this process.
        # The worker still runs the plugin in a separate subprocess at runtime.
        from loreley.core.worker.evaluator import Evaluator

        evaluator = Evaluator(settings=settings)
        evaluator._ensure_callable()  # noqa: SLF001 - preflight intentionally peeks internals
        return CheckResult("evaluator_plugin", "ok", f"importable: {plugin_ref}")
    except Exception as exc:
        return CheckResult("evaluator_plugin", "fail", f"failed to import {plugin_ref!r} ({exc})")


def _check_agent_backend(
    *,
    kind: Literal["planning", "coding"],
    settings: Settings,
) -> list[CheckResult]:
    """Validate the planning/coding backend configuration.

    Returns a list because one backend may imply additional checks (e.g. required binary).
    """
    results: list[CheckResult] = []
    if kind == "planning":
        backend_ref = (settings.worker_planning_backend or "").strip() or None
        default_bin = settings.worker_planning_codex_bin
        label = "planning_backend"
    else:
        backend_ref = (settings.worker_coding_backend or "").strip() or None
        default_bin = settings.worker_coding_codex_bin
        label = "coding_backend"

    if not backend_ref:
        # Default: Codex CLI backend.
        results.append(check_binary(default_bin, label=f"codex({kind})"))
        return results

    try:
        from loreley.core.worker.agent_backend import (
            CodexCliBackend,
            CursorCliBackend,
            load_agent_backend,
        )

        backend = load_agent_backend(backend_ref, label=label)
        results.append(CheckResult(label, "ok", f"loaded: {backend_ref!r}"))

        # Best-effort binary checks for built-in backends.
        if isinstance(backend, (CodexCliBackend, CursorCliBackend)):
            results.append(check_binary(str(backend.bin), label=f"{kind}_agent_bin"))
        else:
            bin_value = getattr(backend, "bin", None)
            if isinstance(bin_value, str) and bin_value:
                results.append(check_binary(bin_value, label=f"{kind}_agent_bin"))
            else:
                results.append(
                    CheckResult(
                        f"{kind}_agent_bin",
                        "warn",
                        "unknown backend type; skipping binary checks",
                    )
                )
        return results
    except Exception as exc:
        results.append(CheckResult(label, "fail", f"failed to load backend {backend_ref!r} ({exc})"))
        return results


def preflight_scheduler(settings: Settings, *, timeout_seconds: float = 2.0) -> list[CheckResult]:
    """Preflight checks that should pass before starting the scheduler."""
    results: list[CheckResult] = []

    results.append(check_openai_api_key(settings.openai_api_key, required=True))
    results.append(check_database(dsn=settings.database_dsn, timeout_seconds=timeout_seconds))
    results.append(
        check_redis(
            redis_url=settings.tasks_redis_url,
            host=settings.tasks_redis_host,
            port=settings.tasks_redis_port,
            db=settings.tasks_redis_db,
            password=settings.tasks_redis_password,
            timeout_seconds=timeout_seconds,
        )
    )

    candidate = settings.scheduler_repo_root or settings.worker_repo_worktree or str(Path.cwd())
    results.append(check_git_repo(Path(candidate).expanduser().resolve(), label="scheduler_repo_root"))

    goal = (settings.worker_evolution_global_goal or "").strip()
    if goal:
        results.append(CheckResult("worker_evolution_global_goal", "ok", "configured"))
    else:
        results.append(
            CheckResult(
                "worker_evolution_global_goal",
                "fail",
                "WORKER_EVOLUTION_GLOBAL_GOAL is empty (scheduler cannot create jobs).",
            )
        )
    return results


def preflight_worker(settings: Settings, *, timeout_seconds: float = 2.0) -> list[CheckResult]:
    """Preflight checks that should pass before starting the worker."""
    results: list[CheckResult] = []

    results.append(check_binary(settings.worker_repo_git_bin or "git", label="git"))
    results.append(check_openai_api_key(settings.openai_api_key, required=True))
    results.append(check_database(dsn=settings.database_dsn, timeout_seconds=timeout_seconds))
    results.append(
        check_redis(
            redis_url=settings.tasks_redis_url,
            host=settings.tasks_redis_host,
            port=settings.tasks_redis_port,
            db=settings.tasks_redis_db,
            password=settings.tasks_redis_password,
            timeout_seconds=timeout_seconds,
        )
    )

    results.append(
        check_non_empty(
            settings.worker_repo_remote_url,
            label="worker_repo_remote_url",
            env_name="WORKER_REPO_REMOTE_URL",
            help_text="required for the worker to clone/fetch/push branches",
        )
    )

    goal = (settings.worker_evolution_global_goal or "").strip()
    if goal:
        results.append(CheckResult("worker_evolution_global_goal", "ok", "configured"))
    else:
        results.append(
            CheckResult(
                "worker_evolution_global_goal",
                "fail",
                "WORKER_EVOLUTION_GLOBAL_GOAL is empty (worker may fail on jobs without per-job goals).",
            )
        )

    results.append(
        check_evaluator_plugin(
            plugin_ref=settings.worker_evaluator_plugin,
            python_paths=settings.worker_evaluator_python_paths,
            settings=settings,
        )
    )

    results.extend(_check_agent_backend(kind="planning", settings=settings))
    results.extend(_check_agent_backend(kind="coding", settings=settings))

    return results


def preflight_all(settings: Settings, *, timeout_seconds: float = 2.0) -> list[CheckResult]:
    """Union of worker + scheduler checks (deduplicated by name)."""
    combined: list[CheckResult] = []
    seen: set[str] = set()
    for item in [*preflight_scheduler(settings, timeout_seconds=timeout_seconds), *preflight_worker(settings, timeout_seconds=timeout_seconds)]:
        key = f"{item.name}:{item.status}:{item.details}"
        if key in seen:
            continue
        seen.add(key)
        combined.append(item)
    return combined


def preflight_api(settings: Settings, *, timeout_seconds: float = 2.0) -> list[CheckResult]:
    """Preflight checks before starting the read-only UI API."""
    results: list[CheckResult] = []
    results.append(check_database(dsn=settings.database_dsn, timeout_seconds=timeout_seconds))
    results.append(
        check_python_modules(
            ("fastapi", "uvicorn"),
            label="ui_api_deps",
            install_hint="Install with `uv sync --extra ui` if you want the UI/API.",
        )
    )
    return results


def preflight_ui(settings: Settings, *, timeout_seconds: float = 2.0) -> list[CheckResult]:
    """Preflight checks before starting the Streamlit UI."""
    results: list[CheckResult] = []
    results.append(
        check_python_modules(
            ("streamlit",),
            label="streamlit_deps",
            install_hint="Install with `uv sync --extra ui` if you want the UI/API.",
        )
    )
    return results


