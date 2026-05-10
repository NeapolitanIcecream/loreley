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
from loreley.core.openai_auth import (
    DynamicOpenAIKeyConfigurationError,
    dynamic_openai_auth_enabled,
    validate_dynamic_openai_auth_settings,
)

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


def check_embedding_dimensions(settings: Settings) -> CheckResult:
    """Check MAPELITES_CODE_EMBEDDING_DIMENSIONS is configured and valid."""
    raw = getattr(settings, "mapelites_code_embedding_dimensions", None)
    if raw is None:
        return CheckResult(
            "mapelites_code_embedding_dimensions",
            "fail",
            "MAPELITES_CODE_EMBEDDING_DIMENSIONS is not set (required for repo-state embeddings).",
        )
    try:
        dims = int(raw)
    except (TypeError, ValueError):
        return CheckResult(
            "mapelites_code_embedding_dimensions",
            "fail",
            "MAPELITES_CODE_EMBEDDING_DIMENSIONS must be a positive integer.",
        )
    if dims <= 0:
        return CheckResult(
            "mapelites_code_embedding_dimensions",
            "fail",
            "MAPELITES_CODE_EMBEDDING_DIMENSIONS must be a positive integer.",
        )
    return CheckResult("mapelites_code_embedding_dimensions", "ok", "configured")


def check_scheduler_max_total_jobs(settings: Settings) -> CheckResult:
    """Check that SCHEDULER_MAX_TOTAL_JOBS is configured and positive."""
    raw = getattr(settings, "scheduler_max_total_jobs", None)
    if raw is None:
        return CheckResult(
            "scheduler_max_total_jobs",
            "fail",
            "SCHEDULER_MAX_TOTAL_JOBS is not set (required to enforce a bounded scheduler run).",
        )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return CheckResult(
            "scheduler_max_total_jobs",
            "fail",
            "SCHEDULER_MAX_TOTAL_JOBS must be a positive integer.",
        )
    if value <= 0:
        return CheckResult(
            "scheduler_max_total_jobs",
            "fail",
            "SCHEDULER_MAX_TOTAL_JOBS must be a positive integer.",
        )
    return CheckResult("scheduler_max_total_jobs", "ok", f"configured: {value}")


def check_campaign_program(settings: Settings) -> CheckResult:
    """Report the default campaign program parse status for the scheduler repo."""

    candidate = settings.scheduler_repo_root or settings.worker_repo_worktree or str(Path.cwd())
    repo_root = Path(candidate).expanduser().resolve()
    try:
        from loreley.core.campaign_program import load_campaign_program_from_repo

        loaded = load_campaign_program_from_repo(repo_root)
    except Exception as exc:
        return CheckResult(
            "campaign_program",
            "warn",
            f"failed to inspect loreley.program.md ({exc})",
        )
    if loaded.snapshot is None:
        return CheckResult(
            "campaign_program",
            "ok",
            "not found; campaign_program_hash will be null",
        )
    snapshot = loaded.snapshot
    details = (
        f"found hash={snapshot.raw_sha256[:12]} "
        f"recognized_sections={list(snapshot.recognized_sections)} "
        f"warnings={len(snapshot.parse_warnings)}"
    )
    return CheckResult("campaign_program", "ok", details)


def check_instance_marker(
    *,
    schema_version: int,
    settings: Settings | None = None,
    validate_identity: bool = True,
) -> CheckResult:
    """Check that the instance metadata marker exists and matches schema_version."""
    try:
        from loreley.db.base import get_engine
        from loreley.db.instance import InstanceMetadataError, resolve_instance_identity
        from loreley.db.migrations.runner import (
            MigrationError,
            describe_schema,
            validate_database_identity,
        )
    except Exception as exc:
        return CheckResult(
            "instance_metadata",
            "fail",
            f"failed to import DB helpers ({exc})",
        )
    try:
        engine = get_engine()
        status = describe_schema(engine=engine, target_version=schema_version)
        return _check_instance_marker_status(
            _InstanceMarkerCheckContext(
                engine=engine,
                status=status,
                settings=settings,
                validate_identity=validate_identity,
                validate_database_identity=validate_database_identity,
                resolve_instance_identity=resolve_instance_identity,
                instance_metadata_error_type=InstanceMetadataError,
            )
        )
    except MigrationError as exc:
        return CheckResult("instance_metadata", "fail", str(exc))
    except Exception as exc:
        return CheckResult(
            "instance_metadata",
            "fail",
            f"failed to validate instance metadata ({exc})",
        )


@dataclass(frozen=True)
class _InstanceMarkerCheckContext:
    engine: Any
    status: Any
    settings: Settings | None
    validate_identity: bool
    validate_database_identity: Any
    resolve_instance_identity: Any
    instance_metadata_error_type: type[Exception]


def _check_instance_marker_status(context: _InstanceMarkerCheckContext) -> CheckResult:
    handlers = {
        "current": _check_current_instance_marker,
        "migratable": _check_migratable_instance_marker,
        "fresh": _check_fresh_instance_marker,
    }
    handler = handlers.get(context.status.state)
    if handler is None:
        return CheckResult("instance_metadata", "fail", context.status.detail)
    return handler(context)


def _check_current_instance_marker(context: _InstanceMarkerCheckContext) -> CheckResult:
    if context.settings is not None and context.validate_identity:
        context.validate_database_identity(engine=context.engine, settings=context.settings)
    return CheckResult("instance_metadata", "ok", "present")


def _check_migratable_instance_marker(context: _InstanceMarkerCheckContext) -> CheckResult:
    if context.settings is not None and (
        context.validate_identity or context.settings.db_auto_migrate
    ):
        # Startup validates marker identity before applying automatic migrations.
        context.validate_database_identity(engine=context.engine, settings=context.settings)
    if context.settings is not None and not context.settings.db_auto_migrate:
        return CheckResult(
            "instance_metadata",
            "fail",
            (
                f"schema_version={context.status.schema_version} requires migration "
                f"to {context.status.target_version}; run `uv run loreley db migrate`."
            ),
        )
    return CheckResult(
        "instance_metadata",
        "ok",
        f"schema_version={context.status.schema_version} is migratable to {context.status.target_version}",
    )


def _check_fresh_instance_marker(context: _InstanceMarkerCheckContext) -> CheckResult:
    if context.settings is not None and context.settings.db_auto_migrate:
        try:
            context.resolve_instance_identity(context.settings)
        except context.instance_metadata_error_type as exc:
            return CheckResult(
                "instance_metadata",
                "fail",
                f"database is empty but startup cannot initialize the schema marker ({exc})",
            )
        return CheckResult(
            "instance_metadata",
            "ok",
            "database is empty; startup can initialize the schema",
        )
    return CheckResult(
        "instance_metadata",
        "fail",
        "instance metadata is missing; run `uv run loreley db migrate`.",
    )


def check_openai_api_key(value: str | None, *, required: bool) -> CheckResult:
    """Check OpenAI-compatible API key presence.

    Note:
    - Some OpenAI-compatible gateways accept dummy keys. In that case, set any
      non-empty OPENAI_API_KEY or LORELEY_LLM_API_KEY value to satisfy this check.
    """
    if (value or "").strip():
        return CheckResult("openai_api_key", "ok", "configured")
    if required:
        return CheckResult(
            "openai_api_key",
            "fail",
            "OPENAI_API_KEY / LORELEY_LLM_API_KEY is not set (required for embeddings / summaries).",
        )
    return CheckResult(
        "openai_api_key",
        "warn",
        "OPENAI_API_KEY / LORELEY_LLM_API_KEY is not set (OpenAI API calls may fail).",
    )


def _scheduler_requires_openai_api_key(settings: Settings) -> bool:
    model = str(getattr(settings, "mapelites_code_embedding_model", "") or "").strip().lower()
    return not model.startswith("local-hash")


def _worker_requires_openai_api_key(settings: Settings) -> bool:
    model = str(getattr(settings, "mapelites_code_embedding_model", "") or "").strip().lower()
    if not model.startswith("local-hash"):
        return True
    return int(getattr(settings, "worker_planning_trajectory_max_chunks", 0) or 0) > 0


def _check_dynamic_openai_auth(settings: Settings) -> CheckResult | None:
    provider_ref = str(getattr(settings, "openai_dynamic_api_key_provider", "") or "").strip()
    ttl = getattr(settings, "openai_dynamic_api_key_ttl_seconds", None)
    skew = getattr(settings, "openai_dynamic_api_key_refresh_skew_seconds", None)
    if not provider_ref and ttl is None and skew is None:
        return None
    try:
        validated = validate_dynamic_openai_auth_settings(settings)
    except DynamicOpenAIKeyConfigurationError as exc:
        return CheckResult("openai_api_key", "fail", str(exc))
    except Exception as exc:
        return CheckResult(
            "openai_api_key",
            "fail",
            f"failed to validate dynamic API key provider {provider_ref!r} ({exc})",
        )
    assert validated is not None
    ref, ttl_seconds, skew_seconds = validated
    return CheckResult(
        "openai_api_key",
        "ok",
        "configured via dynamic provider "
        f"{ref!r} ttl_seconds={ttl_seconds} refresh_skew_seconds={skew_seconds}",
    )


def _check_openai_api_key_for_scheduler(settings: Settings) -> CheckResult:
    dynamic_check = _check_dynamic_openai_auth(settings)
    if dynamic_check is not None:
        return dynamic_check
    if not _scheduler_requires_openai_api_key(settings) and not (settings.openai_api_key or "").strip():
        return CheckResult(
            "openai_api_key",
            "ok",
            "not required: MAP-Elites uses local-hash embeddings for this scheduler configuration.",
        )
    return check_openai_api_key(settings.openai_api_key, required=True)


def _check_openai_api_key_for_worker(settings: Settings) -> CheckResult:
    dynamic_check = _check_dynamic_openai_auth(settings)
    if dynamic_check is not None:
        return dynamic_check
    if not _worker_requires_openai_api_key(settings) and not (settings.openai_api_key or "").strip():
        return CheckResult(
            "openai_api_key",
            "ok",
            "not required: local-hash embeddings are enabled and trajectory summarization is disabled.",
        )
    return check_openai_api_key(settings.openai_api_key, required=True)


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
        default_bin = settings.worker_kilocode_bin
        label = "planning_backend"
    else:
        backend_ref = (settings.worker_coding_backend or "").strip() or None
        default_bin = settings.worker_kilocode_bin
        label = "coding_backend"

    if not backend_ref:
        # Default: Kilocode CLI backend.
        results.append(check_binary(default_bin, label=f"kilocode({kind})"))
        return results

    try:
        from loreley.core.worker.agent import load_agent_backend
        from loreley.core.worker.agent.backends import CodexCliBackend, CursorCliBackend

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


def _check_dynamic_openai_agent_ttl(settings: Settings) -> list[CheckResult]:
    if not dynamic_openai_auth_enabled(settings):
        return []
    try:
        validated = validate_dynamic_openai_auth_settings(settings)
    except DynamicOpenAIKeyConfigurationError:
        return []
    except Exception:
        return []
    if validated is None:
        return []

    from loreley.core.worker.agent import load_agent_backend
    from loreley.core.worker.agent.backends import KilocodeCliBackend

    _, ttl_seconds, _ = validated
    warnings: list[str] = []
    for kind, timeout_seconds in (
        ("planning", int(getattr(settings, "worker_planning_timeout_seconds", 0) or 0)),
        ("coding", int(getattr(settings, "worker_coding_timeout_seconds", 0) or 0)),
    ):
        backend_ref = (
            settings.worker_planning_backend if kind == "planning" else settings.worker_coding_backend
        )
        if not (backend_ref or "").strip():
            backend_is_kilocode = True
        else:
            try:
                backend = load_agent_backend(backend_ref, label=f"{kind} backend")
            except Exception:
                continue
            backend_is_kilocode = isinstance(backend, KilocodeCliBackend)
        if not backend_is_kilocode:
            continue
        if ttl_seconds <= timeout_seconds:
            warnings.append(
                f"{kind} Kilocode timeout={timeout_seconds}s is >= dynamic token TTL={ttl_seconds}s"
            )
    if not warnings:
        return []
    return [
        CheckResult(
            "openai_dynamic_api_key_ttl",
            "warn",
            "; ".join(warnings),
        )
    ]


def preflight_scheduler(settings: Settings, *, timeout_seconds: float = 2.0) -> list[CheckResult]:
    """Preflight checks that should pass before starting the scheduler."""
    results: list[CheckResult] = []

    results.append(_check_openai_api_key_for_scheduler(settings))
    _append_database_schema_checks(results, settings=settings, timeout_seconds=timeout_seconds)
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
    results.append(
        check_non_empty(
            str(settings.experiment_id) if settings.experiment_id else None,
            label="experiment_id",
            env_name="EXPERIMENT_ID",
            help_text="required to scope the scheduler to a single experiment",
        )
    )
    results.append(
        check_non_empty(
            settings.mapelites_experiment_root_commit,
            label="mapelites_experiment_root_commit",
            env_name="MAPELITES_EXPERIMENT_ROOT_COMMIT",
            help_text="required for repo-state startup approval and incremental-only ingestion",
        )
    )
    results.append(check_embedding_dimensions(settings))
    results.append(check_scheduler_max_total_jobs(settings))
    results.append(check_campaign_program(settings))

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
    results.append(_check_openai_api_key_for_worker(settings))
    _append_database_schema_checks(results, settings=settings, timeout_seconds=timeout_seconds)
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
            str(settings.experiment_id) if settings.experiment_id else None,
            label="experiment_id",
            env_name="EXPERIMENT_ID",
            help_text="required to attach this worker process to a single experiment",
        )
    )
    results.append(
        check_non_empty(
            settings.mapelites_experiment_root_commit,
            label="mapelites_experiment_root_commit",
            env_name="MAPELITES_EXPERIMENT_ROOT_COMMIT",
            help_text="required to validate the instance metadata marker",
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
    results.extend(_check_dynamic_openai_agent_ttl(settings))

    return results


def _append_database_schema_checks(
    results: list[CheckResult],
    *,
    settings: Settings,
    timeout_seconds: float,
) -> None:
    database = check_database(dsn=settings.database_dsn, timeout_seconds=timeout_seconds)
    results.append(database)
    if database.status != "ok":
        return

    try:
        from loreley.db.base import INSTANCE_SCHEMA_VERSION
    except Exception as exc:  # pragma: no cover - defensive
        results.append(
            CheckResult("instance_metadata", "fail", f"failed to load DB schema version ({exc})"),
        )
        return

    results.append(check_instance_marker(schema_version=INSTANCE_SCHEMA_VERSION, settings=settings))


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
    try:
        from loreley.db.base import INSTANCE_SCHEMA_VERSION
    except Exception as exc:  # pragma: no cover - defensive
        results.append(CheckResult("instance_metadata", "fail", f"failed to load DB schema version ({exc})"))
    else:
        results.append(
            check_instance_marker(
                schema_version=INSTANCE_SCHEMA_VERSION,
                settings=settings,
                validate_identity=False,
            )
        )
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
