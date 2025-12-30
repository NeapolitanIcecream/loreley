from __future__ import annotations

"""Environment doctor for Loreley.

This script performs a few fast checks to reduce onboarding friction:
- Validate that required configuration values are set.
- Check that PostgreSQL and Redis are reachable.
- Check that external binaries (git, codex, cursor-agent) are available when relevant.
- Validate evaluator plugin importability.

Usage (with uv):
    uv run python script/doctor.py --role all
"""

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence
from urllib.parse import urlparse, urlunparse

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.text import Text

from loreley.config import get_settings

console = Console()
log = logger.bind(module="script.doctor")

Status = Literal["ok", "warn", "fail"]


@dataclass(slots=True)
class CheckResult:
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

    # Hide userinfo, keep host/port/path/query.
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


def _check_binary(name: str, *, label: str) -> CheckResult:
    resolved = shutil.which(name)
    if resolved:
        return CheckResult(label, "ok", f"found: {resolved}")
    return CheckResult(label, "fail", f"missing: {name!r} (not on PATH)")


def _check_db(*, dsn: str, timeout_seconds: float) -> CheckResult:
    safe = _sanitize_sqlalchemy_dsn(dsn)
    try:
        from sqlalchemy import create_engine, text

        connect_args: dict[str, object] = {}
        # psycopg (PostgreSQL) supports connect_timeout in seconds.
        connect_args["connect_timeout"] = int(max(1, timeout_seconds))
        try:
            engine = create_engine(dsn, pool_pre_ping=True, connect_args=connect_args)
        except TypeError:
            engine = create_engine(dsn, pool_pre_ping=True)

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return CheckResult("database", "ok", f"reachable: {safe}")
    except Exception as exc:
        return CheckResult("database", "fail", f"unreachable: {safe} ({exc})")


def _check_redis(*, redis_url: str | None, host: str, port: int, db: int, password: str | None, timeout_seconds: float) -> CheckResult:
    try:
        import redis  # type: ignore[import-not-found]

        if redis_url:
            safe = _sanitize_url(redis_url)
            client = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=timeout_seconds,
                socket_timeout=timeout_seconds,
                decode_responses=True,
            )
            client.ping()
            return CheckResult("redis", "ok", f"reachable: {safe}")

        safe = f"redis://{host}:{int(port)}/{int(db)}"
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
        safe = _sanitize_url(redis_url) if redis_url else f"redis://{host}:{int(port)}/{int(db)}"
        return CheckResult("redis", "fail", f"unreachable: {safe} ({exc})")


def _check_git_repo(path: Path, *, label: str) -> CheckResult:
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


def _check_openai_api_key(value: str | None) -> CheckResult:
    if (value or "").strip():
        return CheckResult("openai_api_key", "ok", "configured")
    return CheckResult(
        "openai_api_key",
        "warn",
        "OPENAI_API_KEY is not set (required for OpenAI API calls; set a non-empty value if your gateway accepts anonymous requests).",
    )


def _check_evaluator_plugin(plugin_ref: str | None, python_paths: Sequence[str]) -> CheckResult:
    if not (plugin_ref or "").strip():
        return CheckResult(
            "evaluator_plugin",
            "fail",
            "WORKER_EVALUATOR_PLUGIN is not set (required for the worker).",
        )
    try:
        from loreley.core.worker.evaluator import Evaluator

        evaluator = Evaluator()
        # Best-effort: validate python paths exist; missing paths are often a typo.
        missing = [p for p in python_paths if p and not Path(p).expanduser().exists()]
        if missing:
            return CheckResult(
                "evaluator_plugin",
                "fail",
                f"python path(s) not found: {missing}",
            )
        evaluator._ensure_callable()  # noqa: SLF001 - intentional for doctor
        return CheckResult("evaluator_plugin", "ok", f"importable: {plugin_ref}")
    except Exception as exc:
        return CheckResult("evaluator_plugin", "fail", f"failed to import {plugin_ref!r} ({exc})")


def _check_ui_extras() -> Iterable[CheckResult]:
    missing: list[str] = []
    for module in ("fastapi", "uvicorn", "streamlit"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if not missing:
        return [CheckResult("ui_extras", "ok", "installed")]
    return [
        CheckResult(
            "ui_extras",
            "warn",
            f"missing modules: {missing}. Install with `uv sync --extra ui` if you want the UI/API.",
        )
    ]


def _render_table(results: Sequence[CheckResult]) -> None:
    table = Table(title="Loreley doctor", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")
    for item in results:
        table.add_row(item.name, _status_text(item.status), item.details)
    console.print(table)


def _summarize(results: Sequence[CheckResult]) -> tuple[int, int, int]:
    ok = sum(1 for r in results if r.status == "ok")
    warn = sum(1 for r in results if r.status == "warn")
    fail = sum(1 for r in results if r.status == "fail")
    return ok, warn, fail


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run environment checks for Loreley.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--role",
        default="all",
        choices=("all", "scheduler", "worker", "api", "ui"),
        help="Which component you want to validate.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=2.0,
        help="Network timeout used for DB/Redis connectivity checks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (non-zero exit code).",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print results as JSON (useful for CI).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)

    try:
        settings = get_settings()
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            "[bold red]Failed to load settings[/] "
            f"reason={exc}. Ensure your environment variables are valid.",
        )
        log.exception("Settings load failed")
        return 1

    timeout = float(max(0.2, args.timeout_seconds))

    results: list[CheckResult] = []

    role = str(args.role)
    if role in ("all", "scheduler", "worker"):
        results.append(_check_binary(settings.worker_repo_git_bin or "git", label="git"))

    if role in ("all", "scheduler", "worker", "api", "ui"):
        results.append(_check_db(dsn=settings.database_dsn, timeout_seconds=timeout))
        results.append(
            _check_redis(
                redis_url=settings.tasks_redis_url,
                host=settings.tasks_redis_host,
                port=settings.tasks_redis_port,
                db=settings.tasks_redis_db,
                password=settings.tasks_redis_password,
                timeout_seconds=timeout,
            )
        )

    if role in ("all", "scheduler"):
        candidate = settings.scheduler_repo_root or settings.worker_repo_worktree or str(Path.cwd())
        results.append(_check_git_repo(Path(candidate).expanduser().resolve(), label="scheduler_repo_root"))

    if role in ("all", "worker"):
        if (settings.worker_repo_remote_url or "").strip():
            results.append(
                CheckResult(
                    "worker_repo_remote_url",
                    "ok",
                    f"configured: {_sanitize_url(settings.worker_repo_remote_url or '')}",
                )
            )
        else:
            results.append(
                CheckResult(
                    "worker_repo_remote_url",
                    "fail",
                    "WORKER_REPO_REMOTE_URL is not set (required for the worker).",
                )
            )

        results.append(_check_openai_api_key(settings.openai_api_key))

        results.append(_check_evaluator_plugin(settings.worker_evaluator_plugin, settings.worker_evaluator_python_paths))

        # Planning / coding backends (only check binaries for default Codex backend).
        if settings.worker_planning_backend:
            results.append(
                CheckResult(
                    "planning_backend",
                    "warn",
                    f"custom backend configured: {settings.worker_planning_backend!r} (doctor does not validate it).",
                )
            )
        else:
            results.append(_check_binary(settings.worker_planning_codex_bin, label="codex(planning)"))

        if settings.worker_coding_backend:
            results.append(
                CheckResult(
                    "coding_backend",
                    "warn",
                    f"custom backend configured: {settings.worker_coding_backend!r} (doctor does not validate it).",
                )
            )
        else:
            results.append(_check_binary(settings.worker_coding_codex_bin, label="codex(coding)"))

        # Cursor agent is an optional alternative backend. Only warn if missing.
        cursor_bin = shutil.which("cursor-agent")
        if cursor_bin:
            results.append(CheckResult("cursor-agent", "ok", f"found: {cursor_bin}"))
        else:
            results.append(
                CheckResult(
                    "cursor-agent",
                    "warn",
                    "cursor-agent not found (only required if you use the Cursor backend).",
                )
            )

    if role in ("all", "api", "ui"):
        results.extend(_check_ui_extras())

    if args.json_output:
        payload = [{"name": r.name, "status": r.status, "details": r.details} for r in results]
        console.print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _render_table(results)

    ok, warn, fail = _summarize(results)
    summary = f"ok={ok} warn={warn} fail={fail}"
    if fail:
        console.print(f"[bold red]Doctor failed[/] {summary}")
        return 1
    if warn and args.strict:
        console.print(f"[bold yellow]Doctor warnings (strict)[/] {summary}")
        return 2
    console.print(f"[bold green]Doctor passed[/] {summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


