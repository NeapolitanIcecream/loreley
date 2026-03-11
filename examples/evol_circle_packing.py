from __future__ import annotations

"""
Convenience launcher for the circle-packing example.

This helper standardises two experiment phases:
  - `smoke`: low-cost end-to-end validation with 4 worker processes
  - `main`: larger run for throughput and algorithm-effectiveness analysis

It supports:
  - `scheduler`: run Loreley's scheduler
  - `worker`: run one worker process
  - `workers`: supervise N independent worker OS processes
  - `api` / `ui`: inspect results
  - `report`: aggregate DB rows + artifact JSON into Markdown/JSON summaries
"""

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loguru import logger
from rich.console import Console


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
REPO_ROOT: Path = PROJECT_ROOT / "examples" / "circle-packing"
EVAL_ENV_ROOT: Path = REPO_ROOT.parent / "circle_packing_env"
LOGS_BASE_DIR: Path = PROJECT_ROOT

APP_NAME: str = "loreley-circle-packing"
APP_ENV: str = "development"
LOG_LEVEL: str = "INFO"

DB_SCHEME: str = "postgresql+psycopg"
DB_HOST: str = "localhost"
DB_PORT: int = 5432
DB_USERNAME: str = "loreley"
DB_PASSWORD: str = "loreley"
DB_NAME: str = "circle_packing"
DB_POOL_SIZE: int = 10
DB_MAX_OVERFLOW: int = 20
DB_POOL_TIMEOUT: int = 30
DB_ECHO: bool = False
DATABASE_URL: str = (
    f"{DB_SCHEME}://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

TASKS_REDIS_URL: str = "redis://localhost:6379/0"
TASKS_QUEUE_PREFETCH: int = 1
TASKS_DELAY_QUEUE_PREFETCH: int = 1

WORKER_REPO_REMOTE_URL: str = str(REPO_ROOT)
WORKER_REPO_BRANCH: str = "main"
WORKER_REPO_WORKTREE: Path = REPO_ROOT / ".cache" / "loreley" / "worker-repo"
WORKER_REPO_WORKTREE_RANDOMIZE: bool = True

SCHEDULER_REPO_ROOT: Path = REPO_ROOT
SCHEDULER_POLL_INTERVAL_SECONDS: float = 15.0

UI_API_HOST: str = "127.0.0.1"
UI_API_PORT: int = 8000
UI_API_BASE_URL: str = f"http://{UI_API_HOST}:{UI_API_PORT}"
UI_HOST: str = "127.0.0.1"
UI_PORT: int = 8501

WORKER_EVALUATOR_PYTHON_PATHS: list[str] = [str(EVAL_ENV_ROOT)]
WORKER_EVALUATOR_PLUGIN: str = "evaluate:plugin"

MAPELITES_FITNESS_METRIC: str = "sum_radii"
MAPELITES_DEFAULT_ISLAND_ID: str = "circle_packing"
MAPELITES_EXPERIMENT_ROOT_COMMIT: str = "6dab191"
MAPELITES_DIMENSION_REDUCTION_TARGET_DIMS: int = 2
MAPELITES_ARCHIVE_CELLS_PER_DIM: int = 12
MAPELITES_FEATURE_TRUNCATION_K: float = 3.0
MAPELITES_FEATURE_CLIP: bool = True

WORKER_PLANNING_BACKEND: str = (
    "loreley.core.worker.agent.backends.codex_cli:codex_planning_backend"
)
WORKER_CODING_BACKEND: str = (
    "loreley.core.worker.agent.backends.codex_cli:codex_coding_backend"
)
WORKER_PLANNING_CODEX_MODEL: str = "gpt-5.4"
WORKER_CODING_CODEX_MODEL: str = "gpt-5.4"

WORKER_EVOLUTION_COMMIT_MODEL: str = "openai/gpt-5.2"
WORKER_EVOLUTION_COMMIT_TEMPERATURE: float = 0.2
WORKER_EVOLUTION_COMMIT_MAX_OUTPUT_TOKENS: int = 128
WORKER_EVOLUTION_COMMIT_MAX_RETRIES: int = 3
WORKER_EVOLUTION_COMMIT_RETRY_BACKOFF_SECONDS: float = 2.0

WORKER_EVOLUTION_GLOBAL_GOAL: str = (
    "Evolve the circle-packing solution so that pack_circles(n=26) returns a valid, "
    "non-overlapping set of 26 circles inside the unit square with as high sum of "
    "radii as possible, while keeping the code deterministic and comfortably below "
    "250 ms per call for pack_circles(26) on local CPU. Avoid multi-second search "
    "loops or unbounded local optimization."
)
RUNTIME_BUDGET_MS: float = 250.0
RUNTIME_BUDGET_RUNS: int = 5

MAPELITES_CODE_EMBEDDING_MODEL: str = "local-hash-v1"
MAPELITES_CODE_EMBEDDING_DIMENSIONS: int = 3072
MAPELITES_CODE_EMBEDDING_BATCH_SIZE: int = 12
MAPELITES_CODE_EMBEDDING_MAX_RETRIES: int = 3
MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS: float = 2.0

LORELEY_LLM_BASE_URL: str | None = None
OPENAI_API_SPEC: str = "chat_completions"

HISTORICAL_BEST_COMMIT: str = "62d15a3"
DEFAULT_LOCAL_EVAL_RUNS: int = 50


@dataclass(frozen=True, slots=True)
class PhasePreset:
    name: str
    experiment_id: str
    seed_population_size: int
    min_fit_samples: int
    warmup_samples: int
    max_total_jobs: int
    max_unfinished_jobs: int
    schedule_batch_size: int
    dispatch_batch_size: int
    ingest_batch_size: int


@dataclass(frozen=True, slots=True)
class WorkerProcessSpec:
    instance_id: str
    command: tuple[str, ...]
    env: dict[str, str]
    log_path: Path
    manifest_entry_path: Path
    worktree_base: Path


@dataclass(frozen=True, slots=True)
class ExpansionCheck:
    phase: str
    current_max_total_jobs: int
    recommended_max_total_jobs: int
    eligible: bool
    completed_jobs_considered: int
    failed_jobs_considered: int
    median_total_duration_seconds: float | None
    failure_rate: float | None
    wall_clock_hours_remaining: float
    reasons: tuple[str, ...]


PHASE_PRESETS: dict[str, PhasePreset] = {
    "smoke": PhasePreset(
        name="smoke",
        experiment_id="circle-packing-codex-gpt54-smoke-4w",
        seed_population_size=4,
        min_fit_samples=4,
        warmup_samples=4,
        max_total_jobs=8,
        max_unfinished_jobs=4,
        schedule_batch_size=4,
        dispatch_batch_size=4,
        ingest_batch_size=4,
    ),
    "main": PhasePreset(
        name="main",
        experiment_id="circle-packing-codex-gpt54-main-4w",
        seed_population_size=12,
        min_fit_samples=12,
        warmup_samples=12,
        max_total_jobs=64,
        max_unfinished_jobs=8,
        schedule_batch_size=8,
        dispatch_batch_size=8,
        ingest_batch_size=8,
    ),
}

console = Console()
log = logger.bind(module="examples.evol_circle_packing")


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Expected a number, got {value!r}.") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return float(parsed)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got {value!r}.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return parsed


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, float(percentile))) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stats(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(item) for item in values if item is not None]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "mean": _mean(numeric),
        "p50": _percentile(numeric, 0.50),
        "p90": _percentile(numeric, 0.90),
        "min": min(numeric),
        "max": max(numeric),
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dt_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return float(value.timestamp())
    return None


def _duration_seconds(started: Any, completed: Any) -> float | None:
    started_ts = _timestamp(started)
    completed_ts = _timestamp(completed)
    if started_ts is None or completed_ts is None:
        return None
    return max(0.0, completed_ts - started_ts)


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("Failed to read JSON payload from {}: {}", path, exc)
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_env_if_unset(name: str, value: Any | None) -> None:
    if value is None:
        return
    if name in os.environ and os.environ[name]:
        return
    os.environ[name] = str(value)


def _phase_preset(phase: str) -> PhasePreset:
    try:
        return PHASE_PRESETS[phase]
    except KeyError as exc:
        raise ValueError(f"Unknown phase {phase!r}.") from exc


def _phase_logs_root(phase: str) -> Path:
    preset = _phase_preset(phase)
    return LOGS_BASE_DIR / "logs" / preset.experiment_id


def _build_env_overrides(
    *,
    phase: str,
    include_worker_repo: bool = False,
    worker_instance_id: str | None = None,
    worker_manifest_entry: Path | None = None,
    worker_worktree_base: Path | None = None,
    max_total_jobs_override: int | None = None,
) -> dict[str, str]:
    preset = _phase_preset(phase)
    max_total_jobs = int(max_total_jobs_override or preset.max_total_jobs)
    overrides: dict[str, str] = {
        "APP_NAME": APP_NAME,
        "APP_ENV": APP_ENV,
        "LOG_LEVEL": LOG_LEVEL,
        "LOGS_BASE_DIR": str(LOGS_BASE_DIR),
        "EXPERIMENT_ID": preset.experiment_id,
        "DATABASE_URL": DATABASE_URL,
        "DB_SCHEME": DB_SCHEME,
        "DB_HOST": DB_HOST,
        "DB_PORT": str(DB_PORT),
        "DB_USER": DB_USERNAME,
        "DB_PASSWORD": DB_PASSWORD,
        "DB_NAME": DB_NAME,
        "DB_POOL_SIZE": str(DB_POOL_SIZE),
        "DB_MAX_OVERFLOW": str(DB_MAX_OVERFLOW),
        "DB_POOL_TIMEOUT": str(DB_POOL_TIMEOUT),
        "DB_ECHO": str(DB_ECHO).lower(),
        "TASKS_REDIS_URL": TASKS_REDIS_URL,
        "TASKS_QUEUE_PREFETCH": str(TASKS_QUEUE_PREFETCH),
        "TASKS_DELAY_QUEUE_PREFETCH": str(TASKS_DELAY_QUEUE_PREFETCH),
        "WORKER_REPO_REMOTE_URL": WORKER_REPO_REMOTE_URL,
        "WORKER_REPO_BRANCH": WORKER_REPO_BRANCH,
        "SCHEDULER_REPO_ROOT": str(SCHEDULER_REPO_ROOT),
        "SCHEDULER_POLL_INTERVAL_SECONDS": str(SCHEDULER_POLL_INTERVAL_SECONDS),
        "SCHEDULER_MAX_UNFINISHED_JOBS": str(preset.max_unfinished_jobs),
        "SCHEDULER_MAX_TOTAL_JOBS": str(max_total_jobs),
        "SCHEDULER_SCHEDULE_BATCH_SIZE": str(preset.schedule_batch_size),
        "SCHEDULER_DISPATCH_BATCH_SIZE": str(preset.dispatch_batch_size),
        "SCHEDULER_INGEST_BATCH_SIZE": str(preset.ingest_batch_size),
        "MAPELITES_FITNESS_METRIC": MAPELITES_FITNESS_METRIC,
        "MAPELITES_DEFAULT_ISLAND_ID": MAPELITES_DEFAULT_ISLAND_ID,
        "MAPELITES_EXPERIMENT_ROOT_COMMIT": MAPELITES_EXPERIMENT_ROOT_COMMIT,
        "MAPELITES_SEED_POPULATION_SIZE": str(preset.seed_population_size),
        "MAPELITES_DIMENSION_REDUCTION_TARGET_DIMS": str(
            MAPELITES_DIMENSION_REDUCTION_TARGET_DIMS
        ),
        "MAPELITES_DIMENSION_REDUCTION_MIN_FIT_SAMPLES": str(preset.min_fit_samples),
        "MAPELITES_ARCHIVE_CELLS_PER_DIM": str(MAPELITES_ARCHIVE_CELLS_PER_DIM),
        "MAPELITES_FEATURE_TRUNCATION_K": str(MAPELITES_FEATURE_TRUNCATION_K),
        "MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES": str(preset.warmup_samples),
        "MAPELITES_FEATURE_CLIP": str(MAPELITES_FEATURE_CLIP).lower(),
        "WORKER_EVALUATOR_PYTHON_PATHS": json.dumps(WORKER_EVALUATOR_PYTHON_PATHS),
        "WORKER_EVALUATOR_PLUGIN": WORKER_EVALUATOR_PLUGIN,
        "WORKER_PLANNING_BACKEND": WORKER_PLANNING_BACKEND,
        "WORKER_CODING_BACKEND": WORKER_CODING_BACKEND,
        "WORKER_PLANNING_CODEX_MODEL": WORKER_PLANNING_CODEX_MODEL,
        "WORKER_CODING_CODEX_MODEL": WORKER_CODING_CODEX_MODEL,
        "WORKER_EVOLUTION_COMMIT_MODEL": WORKER_EVOLUTION_COMMIT_MODEL,
        "WORKER_EVOLUTION_COMMIT_TEMPERATURE": str(WORKER_EVOLUTION_COMMIT_TEMPERATURE),
        "WORKER_EVOLUTION_COMMIT_MAX_OUTPUT_TOKENS": str(
            WORKER_EVOLUTION_COMMIT_MAX_OUTPUT_TOKENS
        ),
        "WORKER_EVOLUTION_COMMIT_MAX_RETRIES": str(WORKER_EVOLUTION_COMMIT_MAX_RETRIES),
        "WORKER_EVOLUTION_COMMIT_RETRY_BACKOFF_SECONDS": str(
            WORKER_EVOLUTION_COMMIT_RETRY_BACKOFF_SECONDS
        ),
        "WORKER_EVOLUTION_GLOBAL_GOAL": WORKER_EVOLUTION_GLOBAL_GOAL,
        "CIRCLE_PACKING_RUNTIME_BUDGET_MS": str(RUNTIME_BUDGET_MS),
        "CIRCLE_PACKING_RUNTIME_RUNS": str(RUNTIME_BUDGET_RUNS),
        "WORKER_PLANNING_TRAJECTORY_MAX_CHUNKS": "0",
        "MAPELITES_CODE_EMBEDDING_MODEL": MAPELITES_CODE_EMBEDDING_MODEL,
        "MAPELITES_CODE_EMBEDDING_DIMENSIONS": str(MAPELITES_CODE_EMBEDDING_DIMENSIONS),
        "MAPELITES_CODE_EMBEDDING_BATCH_SIZE": str(MAPELITES_CODE_EMBEDDING_BATCH_SIZE),
        "MAPELITES_CODE_EMBEDDING_MAX_RETRIES": str(MAPELITES_CODE_EMBEDDING_MAX_RETRIES),
        "MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS": str(
            MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS
        ),
        "OPENAI_API_SPEC": OPENAI_API_SPEC,
    }
    if LORELEY_LLM_BASE_URL:
        overrides["LORELEY_LLM_BASE_URL"] = LORELEY_LLM_BASE_URL
    if include_worker_repo:
        overrides["WORKER_REPO_WORKTREE"] = str(worker_worktree_base or WORKER_REPO_WORKTREE)
        overrides["WORKER_REPO_WORKTREE_RANDOMIZE"] = str(
            WORKER_REPO_WORKTREE_RANDOMIZE
        ).lower()
    if worker_instance_id:
        overrides["LORELEY_WORKER_INSTANCE_ID"] = worker_instance_id
    if worker_manifest_entry is not None:
        overrides["LORELEY_WORKER_MANIFEST_ENTRY"] = str(worker_manifest_entry)
    return overrides


def _apply_base_env(
    *,
    phase: str,
    include_worker_repo: bool = False,
    worker_instance_id: str | None = None,
    worker_manifest_entry: Path | None = None,
    max_total_jobs_override: int | None = None,
) -> None:
    overrides = _build_env_overrides(
        phase=phase,
        include_worker_repo=include_worker_repo,
        worker_instance_id=worker_instance_id,
        worker_manifest_entry=worker_manifest_entry,
        max_total_jobs_override=max_total_jobs_override,
    )
    for name, value in overrides.items():
        _set_env_if_unset(name, value)


def _ensure_repo_on_sys_path() -> None:
    for path in (PROJECT_ROOT, REPO_ROOT, EVAL_ENV_ROOT):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _print_environment_summary(*, phase: str) -> None:
    preset = _phase_preset(phase)
    console.log(
        "[bold cyan]Circle-packing launcher[/] "
        f"phase={phase} experiment_id={preset.experiment_id}",
    )
    console.log(
        "[green]DB[/] DATABASE_URL={}".format(os.getenv("DATABASE_URL", "<unset>")),
    )
    console.log(
        "[green]Redis[/] TASKS_REDIS_URL={}".format(os.getenv("TASKS_REDIS_URL", "<unset>")),
    )
    console.log(
        "[green]Dramatiq[/] queue_prefetch={} delay_prefetch={}".format(
            os.getenv("TASKS_QUEUE_PREFETCH", "<unset>"),
            os.getenv("TASKS_DELAY_QUEUE_PREFETCH", "<unset>"),
        ),
    )
    console.log(
        "[green]Codex[/] planning_model={} coding_model={}".format(
            os.getenv("WORKER_PLANNING_CODEX_MODEL", "<unset>"),
            os.getenv("WORKER_CODING_CODEX_MODEL", "<unset>"),
        ),
    )
    console.log(
        "[green]Scheduler[/] unfinished={} total={} batches={}/{}/{}".format(
            os.getenv("SCHEDULER_MAX_UNFINISHED_JOBS", "<unset>"),
            os.getenv("SCHEDULER_MAX_TOTAL_JOBS", "<unset>"),
            os.getenv("SCHEDULER_SCHEDULE_BATCH_SIZE", "<unset>"),
            os.getenv("SCHEDULER_DISPATCH_BATCH_SIZE", "<unset>"),
            os.getenv("SCHEDULER_INGEST_BATCH_SIZE", "<unset>"),
        ),
    )
    console.log(
        "[green]MAP-Elites[/] seed={} min_fit={} warmup={}".format(
            os.getenv("MAPELITES_SEED_POPULATION_SIZE", "<unset>"),
            os.getenv("MAPELITES_DIMENSION_REDUCTION_MIN_FIT_SAMPLES", "<unset>"),
            os.getenv("MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES", "<unset>"),
        ),
    )
    console.log(
        "[green]Worker repo[/] remote={} branch={} worktree={}".format(
            os.getenv("WORKER_REPO_REMOTE_URL", "<unset>"),
            os.getenv("WORKER_REPO_BRANCH", "<unset>"),
            os.getenv("WORKER_REPO_WORKTREE", "<unset>"),
        ),
    )


def _reset_database(*, phase: str) -> None:
    _apply_base_env(phase=phase)
    _ensure_repo_on_sys_path()
    console.log("[bold yellow]Resetting Loreley database schema (DROP + CREATE)…[/]")

    from loreley.db.base import Base, reset_database_schema
    from loreley.tasks.broker import build_redis_broker, reset_redis_namespace

    reset_database_schema(include_console_log=False)
    console.log(
        "[bold green]Database schema reset complete[/] tables={}".format(
            ", ".join(sorted(Base.metadata.tables.keys())),
        )
    )
    redis_broker = build_redis_broker()
    deleted_keys = reset_redis_namespace()
    console.log(
        "[bold green]Redis broker reset complete[/] namespace={} deleted_keys={}".format(
            getattr(redis_broker, "namespace", "<unknown>"),
            deleted_keys,
        )
    )


def _run_scheduler(
    *,
    phase: str,
    once: bool,
    init_db: bool,
    yes: bool,
    log_level: str | None,
    no_preflight: bool,
    preflight_timeout_seconds: float,
    max_total_jobs: int | None,
) -> int:
    _apply_base_env(phase=phase, max_total_jobs_override=max_total_jobs)
    _ensure_repo_on_sys_path()
    if init_db:
        _reset_database(phase=phase)
    _print_environment_summary(phase=phase)

    from loreley.cli import main as loreley_main

    argv: list[str] = []
    if log_level:
        argv += ["--log-level", str(log_level)]
    argv.append("scheduler")
    if once:
        argv.append("--once")
    if yes:
        argv.append("--yes")
    if no_preflight:
        argv.append("--no-preflight")
    argv += ["--preflight-timeout-seconds", str(float(preflight_timeout_seconds))]
    console.log(
        "[bold green]Starting scheduler[/] phase={} once={} max_total_jobs={}".format(
            phase,
            "yes" if once else "no",
            os.getenv("SCHEDULER_MAX_TOTAL_JOBS", "<unset>"),
        )
    )
    return int(loreley_main(argv))


def _write_worker_manifest_entry(*, path: Path, phase: str, worker_instance_id: str | None) -> None:
    from loreley.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    payload = {
        "phase": phase,
        "experiment_id": str(settings.experiment_id or ""),
        "worker_instance_id": worker_instance_id,
        "pid": os.getpid(),
        "worker_repo_worktree": settings.worker_repo_worktree,
        "worker_repo_worktree_randomize": settings.worker_repo_worktree_randomize,
        "started_at": _now_utc_iso(),
    }
    _write_json(path, payload)


def _run_worker(
    *,
    phase: str,
    log_level: str | None,
    no_preflight: bool,
    preflight_timeout_seconds: float,
    manifest_entry: Path | None,
) -> int:
    worker_instance_id = (os.getenv("LORELEY_WORKER_INSTANCE_ID") or "").strip() or None
    _apply_base_env(
        phase=phase,
        include_worker_repo=True,
        worker_instance_id=worker_instance_id,
        worker_manifest_entry=manifest_entry,
    )
    _ensure_repo_on_sys_path()
    if manifest_entry is not None:
        _write_worker_manifest_entry(
            path=manifest_entry,
            phase=phase,
            worker_instance_id=worker_instance_id,
        )
    _print_environment_summary(phase=phase)

    from loreley.cli import main as loreley_main

    argv: list[str] = []
    if log_level:
        argv += ["--log-level", str(log_level)]
    argv.append("worker")
    if no_preflight:
        argv.append("--no-preflight")
    argv += ["--preflight-timeout-seconds", str(float(preflight_timeout_seconds))]
    console.log(
        "[bold green]Starting worker[/] phase={} instance={}".format(
            phase,
            worker_instance_id or "<unset>",
        )
    )
    return int(loreley_main(argv))


def _supervisor_dir(phase: str) -> Path:
    return _phase_logs_root(phase) / "worker" / "supervisor"


def _build_worker_process_specs(
    *,
    phase: str,
    count: int,
    log_level: str | None,
    no_preflight: bool,
    preflight_timeout_seconds: float,
) -> list[WorkerProcessSpec]:
    script_path = Path(__file__).resolve()
    supervisor_dir = _supervisor_dir(phase)
    supervisor_dir.mkdir(parents=True, exist_ok=True)

    specs: list[WorkerProcessSpec] = []
    for index in range(1, count + 1):
        instance_id = f"worker-{index:02d}"
        log_path = supervisor_dir / f"{instance_id}.log"
        manifest_entry_path = supervisor_dir / f"{instance_id}.json"
        worktree_base = WORKER_REPO_WORKTREE / instance_id
        command: list[str] = [
            sys.executable,
            str(script_path),
            "worker",
            "--phase",
            phase,
            "--manifest-entry",
            str(manifest_entry_path),
        ]
        if log_level:
            command += ["--log-level", str(log_level)]
        if no_preflight:
            command.append("--no-preflight")
        command += [
            "--preflight-timeout-seconds",
            str(float(preflight_timeout_seconds)),
        ]

        env = os.environ.copy()
        env.update(
            _build_env_overrides(
                phase=phase,
                include_worker_repo=True,
                worker_instance_id=instance_id,
                worker_manifest_entry=manifest_entry_path,
                worker_worktree_base=worktree_base,
            )
        )
        specs.append(
            WorkerProcessSpec(
                instance_id=instance_id,
                command=tuple(command),
                env=env,
                log_path=log_path,
                manifest_entry_path=manifest_entry_path,
                worktree_base=worktree_base,
            )
        )
    return specs


def _collect_supervisor_manifest(
    *,
    phase: str,
    specs: Sequence[WorkerProcessSpec],
    processes: Sequence[subprocess.Popen[str]],
    status: str,
    started_at: str,
    ended_at: str | None = None,
) -> dict[str, Any]:
    preset = _phase_preset(phase)
    workers: list[dict[str, Any]] = []
    for spec, proc in zip(specs, processes, strict=False):
        manifest_entry = _read_json(spec.manifest_entry_path)
        workers.append(
            {
                "instance_id": spec.instance_id,
                "pid": proc.pid,
                "returncode": proc.poll(),
                "log_path": str(spec.log_path),
                "manifest_entry_path": str(spec.manifest_entry_path),
                "worktree_base": str(spec.worktree_base),
                "worker_repo_worktree": manifest_entry.get("worker_repo_worktree"),
                "started_at": manifest_entry.get("started_at"),
            }
        )
    return {
        "phase": phase,
        "experiment_id": preset.experiment_id,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "workers": workers,
    }


def _terminate_process_group(proc: subprocess.Popen[str], *, force: bool = False) -> None:
    if proc.poll() is not None:
        return
    try:
        sig = signal.SIGKILL if force else signal.SIGTERM
        os.killpg(int(proc.pid), int(sig))
    except ProcessLookupError:
        return
    except Exception as exc:
        log.warning("Failed to terminate worker pid={} force={}: {}", proc.pid, force, exc)


def _open_supervisor_log(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _run_workers(
    *,
    phase: str,
    count: int,
    log_level: str | None,
    no_preflight: bool,
    preflight_timeout_seconds: float,
) -> int:
    specs = _build_worker_process_specs(
        phase=phase,
        count=count,
        log_level=log_level,
        no_preflight=no_preflight,
        preflight_timeout_seconds=preflight_timeout_seconds,
    )
    supervisor_dir = _supervisor_dir(phase)
    manifest_path = supervisor_dir / "workers-manifest.json"
    started_at = _now_utc_iso()

    log_handles: list[Any] = []
    processes: list[subprocess.Popen[str]] = []
    stop_requested = False
    interrupted = False
    exit_code = 0

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, interrupted, exit_code
        interrupted = True
        stop_requested = True
        exit_code = 130 if signum == signal.SIGINT else 143
        console.log(f"[yellow]Signal received[/] signum={signum}; stopping workers...")
        for proc in processes:
            _terminate_process_group(proc)

    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        for spec in specs:
            handle = _open_supervisor_log(spec.log_path)
            log_handles.append(handle)
            proc = subprocess.Popen(
                list(spec.command),
                env=spec.env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            processes.append(proc)
            console.log(
                "[bold green]Started worker process[/] instance={} pid={} log={}".format(
                    spec.instance_id,
                    proc.pid,
                    spec.log_path,
                )
            )

        _write_json(
            manifest_path,
            _collect_supervisor_manifest(
                phase=phase,
                specs=specs,
                processes=processes,
                status="running",
                started_at=started_at,
            ),
        )

        while processes:
            live_count = 0
            for proc in processes:
                returncode = proc.poll()
                if returncode is None:
                    live_count += 1
                    continue
                if returncode != 0 and not stop_requested:
                    exit_code = int(returncode)
                    stop_requested = True
                    console.log(
                        "[bold red]Worker exited unexpectedly[/] pid={} returncode={}".format(
                            proc.pid,
                            returncode,
                        )
                    )
                    for sibling in processes:
                        if sibling is not proc:
                            _terminate_process_group(sibling)
            if live_count == 0:
                break
            time.sleep(1.0)

        if stop_requested:
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if all(proc.poll() is not None for proc in processes):
                    break
                time.sleep(0.25)
            for proc in processes:
                _terminate_process_group(proc, force=True)

        final_status = "interrupted" if interrupted else ("failed" if exit_code else "completed")
        _write_json(
            manifest_path,
            _collect_supervisor_manifest(
                phase=phase,
                specs=specs,
                processes=processes,
                status=final_status,
                started_at=started_at,
                ended_at=_now_utc_iso(),
            ),
        )
        return exit_code
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for handle in log_handles:
            handle.close()


def _run_api(
    *,
    phase: str,
    host: str,
    port: int,
    log_level: str | None,
    reload: bool,
    no_preflight: bool,
    preflight_timeout_seconds: float,
) -> int:
    _apply_base_env(phase=phase)
    _ensure_repo_on_sys_path()
    _print_environment_summary(phase=phase)

    from loreley.cli import main as loreley_main

    argv: list[str] = []
    if log_level:
        argv += ["--log-level", str(log_level)]
    argv += ["api", "--host", str(host), "--port", str(int(port))]
    if reload:
        argv.append("--reload")
    if no_preflight:
        argv.append("--no-preflight")
    argv += ["--preflight-timeout-seconds", str(float(preflight_timeout_seconds))]
    return int(loreley_main(argv))


def _run_ui(
    *,
    phase: str,
    host: str,
    port: int,
    api_base_url: str,
    headless: bool,
    log_level: str | None,
    no_preflight: bool,
    preflight_timeout_seconds: float,
) -> int:
    _apply_base_env(phase=phase)
    _ensure_repo_on_sys_path()
    _print_environment_summary(phase=phase)

    from loreley.cli import main as loreley_main

    argv: list[str] = []
    if log_level:
        argv += ["--log-level", str(log_level)]
    argv += [
        "ui",
        "--api-base-url",
        str(api_base_url),
        "--host",
        str(host),
        "--port",
        str(int(port)),
    ]
    if headless:
        argv.append("--headless")
    if no_preflight:
        argv.append("--no-preflight")
    argv += ["--preflight-timeout-seconds", str(float(preflight_timeout_seconds))]
    return int(loreley_main(argv))


def _load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_eval_module() -> Any:
    return _load_module_from_path(
        "circle_packing_local_eval",
        REPO_ROOT / "scripts" / "local_eval.py",
    )


def _materialize_solution_for_commit(commit_hash: str) -> Path:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{commit_hash}:solution.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not materialize solution.py for commit {commit_hash}: {result.stderr.strip()}"
        )
    temp_dir = Path(tempfile.mkdtemp(prefix="circle-packing-report-"))
    (temp_dir / "solution.py").write_text(result.stdout, encoding="utf-8")
    return temp_dir


def _collect_reference_stats(*, best_commit_hash: str | None, runs: int) -> list[dict[str, Any]]:
    local_eval = _load_local_eval_module()
    refs: list[tuple[str, str]] = [
        ("root", MAPELITES_EXPERIMENT_ROOT_COMMIT),
        ("historical_best", HISTORICAL_BEST_COMMIT),
    ]
    if best_commit_hash:
        refs.append(("current_best", best_commit_hash))

    seen_hashes: set[str] = set()
    payloads: list[dict[str, Any]] = []
    temp_dirs: list[Path] = []
    try:
        for label, commit_hash in refs:
            if commit_hash in seen_hashes:
                continue
            seen_hashes.add(commit_hash)
            repo_root = _materialize_solution_for_commit(commit_hash)
            temp_dirs.append(repo_root)
            stats = local_eval.evaluate_repo(
                repo_root=repo_root,
                runs=runs,
                target_n=26,
            )
            stats["label"] = label
            stats["commit_hash"] = commit_hash
            payloads.append(stats)
    finally:
        for path in temp_dirs:
            try:
                for child in sorted(path.glob("**/*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()
            except Exception:
                continue
    return payloads


def _load_experiment_jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sqlalchemy import select

    from loreley.db.base import session_scope
    from loreley.db.models import CommitCard, EvolutionJob, JobArtifacts, MapElitesArchiveCell, Metric

    with session_scope() as session:
        jobs = session.execute(select(EvolutionJob).order_by(EvolutionJob.created_at.asc())).scalars().all()
        artifacts_rows = session.execute(select(JobArtifacts)).scalars().all()
        cards = session.execute(select(CommitCard).where(CommitCard.job_id.is_not(None))).scalars().all()
        archive_cells = session.execute(select(MapElitesArchiveCell)).scalars().all()

        card_by_job_id = {
            str(card.job_id): card
            for card in cards
            if getattr(card, "job_id", None) is not None
        }
        metrics_by_card_id: dict[str, dict[str, float]] = {}
        if cards:
            card_ids = [card.id for card in cards]
            metrics = session.execute(
                select(Metric).where(Metric.commit_card_id.in_(card_ids))
            ).scalars().all()
            for metric in metrics:
                metrics_by_card_id.setdefault(str(metric.commit_card_id), {})[metric.name] = float(
                    metric.value
                )

    artifacts_by_job_id = {str(row.job_id): row for row in artifacts_rows}
    job_payloads: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job.id)
        artifact_row = artifacts_by_job_id.get(job_id)
        planning_payload = _read_json(
            getattr(artifact_row, "planning_plan_json_path", None) if artifact_row else None
        )
        coding_payload = _read_json(
            getattr(artifact_row, "coding_execution_json_path", None) if artifact_row else None
        )
        evaluation_payload = _read_json(
            getattr(artifact_row, "evaluation_json_path", None) if artifact_row else None
        )
        worker = (
            evaluation_payload.get("worker")
            or coding_payload.get("worker")
            or planning_payload.get("worker")
            or {}
        )
        card = card_by_job_id.get(job_id)
        metrics = metrics_by_card_id.get(str(card.id), {}) if card is not None else {}
        status_value = getattr(job.status, "value", job.status)
        job_payloads.append(
            {
                "job_id": job_id,
                "status": str(status_value),
                "is_seed_job": bool(getattr(job, "is_seed_job", False)),
                "base_commit_hash": job.base_commit_hash,
                "result_commit_hash": job.result_commit_hash,
                "last_error": job.last_error,
                "created_at": _dt_iso(job.created_at),
                "started_at": _dt_iso(job.started_at),
                "completed_at": _dt_iso(job.completed_at),
                "_created_ts": _timestamp(job.created_at),
                "_started_ts": _timestamp(job.started_at),
                "_completed_ts": _timestamp(job.completed_at),
                "total_duration_seconds": _duration_seconds(job.started_at, job.completed_at),
                "planning_duration_seconds": (
                    planning_payload.get("backend", {}) or {}
                ).get("duration_seconds"),
                "coding_duration_seconds": (
                    coding_payload.get("backend", {}) or {}
                ).get("duration_seconds"),
                "evaluator_duration_seconds": (
                    evaluation_payload.get("extra", {}) or {}
                ).get("evaluator_duration_seconds"),
                "planning_attempts": (planning_payload.get("backend", {}) or {}).get("attempts"),
                "coding_attempts": (coding_payload.get("backend", {}) or {}).get("attempts"),
                "worker_instance_id": worker.get("instance_id"),
                "worker_pid": worker.get("pid"),
                "sum_radii": metrics.get("sum_radii"),
                "packing_density": metrics.get("packing_density"),
                "runtime_p50_ms": metrics.get("runtime_p50_ms"),
                "metrics": metrics,
            }
        )

    archive_payloads = [
        {
            "cell_index": int(cell.cell_index),
            "commit_hash": cell.commit_hash,
            "objective": float(cell.objective),
            "timestamp": float(cell.timestamp),
            "created_at": _dt_iso(cell.created_at),
            "updated_at": _dt_iso(cell.updated_at),
        }
        for cell in archive_cells
    ]
    return job_payloads, archive_payloads


def _build_report_payload(
    *,
    phase: str,
    experiment_id: str,
    jobs: Sequence[dict[str, Any]],
    archive_cells: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    total_jobs = len(jobs)
    succeeded_jobs = [job for job in jobs if job.get("status") == "succeeded"]
    failed_jobs = [job for job in jobs if job.get("status") == "failed"]
    seed_jobs = [job for job in jobs if job.get("is_seed_job")]
    non_seed_jobs = [job for job in jobs if not job.get("is_seed_job")]
    seed_success = [job for job in succeeded_jobs if job.get("is_seed_job")]
    non_seed_success = [job for job in succeeded_jobs if not job.get("is_seed_job")]

    total_durations = [
        float(job["total_duration_seconds"])
        for job in jobs
        if job.get("total_duration_seconds") is not None
    ]
    planning_durations = [
        float(job["planning_duration_seconds"])
        for job in jobs
        if job.get("planning_duration_seconds") is not None
    ]
    coding_durations = [
        float(job["coding_duration_seconds"])
        for job in jobs
        if job.get("coding_duration_seconds") is not None
    ]
    evaluator_durations = [
        float(job["evaluator_duration_seconds"])
        for job in jobs
        if job.get("evaluator_duration_seconds") is not None
    ]
    runtime_p50_values = [
        float(job["runtime_p50_ms"])
        for job in jobs
        if job.get("runtime_p50_ms") is not None
    ]
    planning_attempts = [
        float(job["planning_attempts"])
        for job in jobs
        if job.get("planning_attempts") is not None
    ]
    coding_attempts = [
        float(job["coding_attempts"])
        for job in jobs
        if job.get("coding_attempts") is not None
    ]

    ordered_success = sorted(
        [job for job in succeeded_jobs if job.get("sum_radii") is not None],
        key=lambda item: (
            item.get("_completed_ts") or item.get("_created_ts") or 0.0,
            item.get("job_id") or "",
        ),
    )
    experiment_start = min(
        (
            value
            for job in jobs
            for value in (job.get("_started_ts"), job.get("_created_ts"))
            if value is not None
        ),
        default=None,
    )
    trajectory = [
        {
            "index": index,
            "job_id": job.get("job_id"),
            "commit_hash": job.get("result_commit_hash"),
            "worker_instance_id": job.get("worker_instance_id"),
            "is_seed_job": bool(job.get("is_seed_job")),
            "sum_radii": job.get("sum_radii"),
            "packing_density": job.get("packing_density"),
            "total_duration_seconds": job.get("total_duration_seconds"),
            "completed_at": job.get("completed_at"),
            "elapsed_minutes_from_start": (
                round(
                    (
                        (job.get("_completed_ts") or job.get("_created_ts") or 0.0)
                        - experiment_start
                    )
                    / 60.0,
                    3,
                )
                if experiment_start is not None
                else None
            ),
        }
        for index, job in enumerate(ordered_success, start=1)
    ]

    best_job = None
    if ordered_success:
        best_job = max(ordered_success, key=lambda item: float(item.get("sum_radii") or 0.0))

    root_reference = next((item for item in references if item.get("label") == "root"), None)
    baseline_sum_radii = None
    if root_reference:
        baseline_sum_radii = (
            (root_reference.get("target_metrics") or {}).get("sum_radii")
        )
    first_above_baseline = None
    if baseline_sum_radii is not None:
        first_above_baseline = next(
            (
                point
                for point in trajectory
                if point.get("sum_radii") is not None
                and float(point["sum_radii"]) > float(baseline_sum_radii)
            ),
            None,
        )

    worker_distribution: dict[str, dict[str, Any]] = {}
    for job in jobs:
        worker_instance_id = job.get("worker_instance_id")
        if not worker_instance_id:
            continue
        worker_id = str(worker_instance_id)
        bucket = worker_distribution.setdefault(
            worker_id,
            {
                "worker_instance_id": worker_id,
                "jobs_total": 0,
                "jobs_succeeded": 0,
                "jobs_failed": 0,
                "best_sum_radii": None,
                "total_duration_seconds": [],
            },
        )
        bucket["jobs_total"] += 1
        if job.get("status") == "succeeded":
            bucket["jobs_succeeded"] += 1
        if job.get("status") == "failed":
            bucket["jobs_failed"] += 1
        if job.get("total_duration_seconds") is not None:
            bucket["total_duration_seconds"].append(float(job["total_duration_seconds"]))
        if job.get("sum_radii") is not None:
            candidate = float(job["sum_radii"])
            current = bucket.get("best_sum_radii")
            bucket["best_sum_radii"] = candidate if current is None else max(float(current), candidate)
    worker_throughput = []
    for bucket in worker_distribution.values():
        worker_throughput.append(
            {
                "worker_instance_id": bucket["worker_instance_id"],
                "jobs_total": bucket["jobs_total"],
                "jobs_succeeded": bucket["jobs_succeeded"],
                "jobs_failed": bucket["jobs_failed"],
                "best_sum_radii": bucket["best_sum_radii"],
                "total_duration_seconds": _stats(bucket["total_duration_seconds"]),
            }
        )
    worker_throughput.sort(key=lambda item: item["worker_instance_id"])

    failures = [
        {
            "job_id": job.get("job_id"),
            "worker_instance_id": job.get("worker_instance_id"),
            "last_error": job.get("last_error"),
        }
        for job in failed_jobs
    ]

    archive_summary = {
        "occupied_cells": len(archive_cells),
        "best_objective": max((float(cell["objective"]) for cell in archive_cells), default=None),
        "best_commit_hash": None,
    }
    if archive_cells:
        best_cell = max(archive_cells, key=lambda cell: float(cell["objective"]))
        archive_summary["best_commit_hash"] = best_cell.get("commit_hash")

    return {
        "phase": phase,
        "experiment_id": experiment_id,
        "generated_at": _now_utc_iso(),
        "jobs": {
            "total": total_jobs,
            "succeeded": len(succeeded_jobs),
            "failed": len(failed_jobs),
            "seed_total": len(seed_jobs),
            "non_seed_total": len(non_seed_jobs),
            "seed_success_rate": (
                len(seed_success) / len(seed_jobs) if seed_jobs else None
            ),
            "non_seed_success_rate": (
                len(non_seed_success) / len(non_seed_jobs) if non_seed_jobs else None
            ),
            "failure_rate": (len(failed_jobs) / total_jobs) if total_jobs else None,
        },
        "timing": {
            "total_duration_seconds": _stats(total_durations),
            "planning_duration_seconds": _stats(planning_durations),
            "coding_duration_seconds": _stats(coding_durations),
            "evaluator_duration_seconds": _stats(evaluator_durations),
            "runtime_p50_ms": _stats(runtime_p50_values),
            "planning_attempts": _stats(planning_attempts),
            "coding_attempts": _stats(coding_attempts),
        },
        "best": {
            "job_id": best_job.get("job_id") if best_job else None,
            "commit_hash": best_job.get("result_commit_hash") if best_job else None,
            "sum_radii": best_job.get("sum_radii") if best_job else None,
            "packing_density": best_job.get("packing_density") if best_job else None,
            "runtime_p50_ms": best_job.get("runtime_p50_ms") if best_job else None,
            "worker_instance_id": best_job.get("worker_instance_id") if best_job else None,
        },
        "archive": archive_summary,
        "first_above_baseline": first_above_baseline,
        "worker_throughput": worker_throughput,
        "trajectory": trajectory,
        "references": list(references),
        "failures": failures,
    }


def _evaluate_main_expansion(
    *,
    jobs: Sequence[dict[str, Any]],
    wall_clock_hours_remaining: float,
    current_max_total_jobs: int = 64,
) -> ExpansionCheck:
    if current_max_total_jobs >= 96:
        return ExpansionCheck(
            phase="main",
            current_max_total_jobs=current_max_total_jobs,
            recommended_max_total_jobs=current_max_total_jobs,
            eligible=False,
            completed_jobs_considered=0,
            failed_jobs_considered=0,
            median_total_duration_seconds=None,
            failure_rate=None,
            wall_clock_hours_remaining=float(wall_clock_hours_remaining),
            reasons=("main is already expanded to 96 jobs or beyond",),
        )

    completed = sorted(
        [
            job
            for job in jobs
            if job.get("completed_at") is not None
            and job.get("status") in {"succeeded", "failed"}
        ],
        key=lambda item: (
            item.get("_completed_ts") or item.get("_created_ts") or 0.0,
            item.get("job_id") or "",
        ),
    )
    considered = completed[:24]
    reasons: list[str] = []
    median_total_duration_seconds = _percentile(
        [
            float(job["total_duration_seconds"])
            for job in considered
            if job.get("total_duration_seconds") is not None
        ],
        0.50,
    )
    failed_jobs_considered = sum(1 for job in considered if job.get("status") == "failed")
    failure_rate = (
        float(failed_jobs_considered / len(considered))
        if considered
        else None
    )

    if len(considered) < 24:
        reasons.append("fewer than 24 completed main jobs are available")
    if median_total_duration_seconds is None:
        reasons.append("median total job duration is unavailable")
    elif median_total_duration_seconds > 15.0 * 60.0:
        reasons.append("median total job duration exceeds 15 minutes")
    if failure_rate is None:
        reasons.append("failure rate is unavailable")
    elif failure_rate > 0.15:
        reasons.append("failure rate exceeds 15%")
    if float(wall_clock_hours_remaining) < 5.0:
        reasons.append("wall-clock budget remaining is under 5 hours")

    eligible = not reasons
    return ExpansionCheck(
        phase="main",
        current_max_total_jobs=int(current_max_total_jobs),
        recommended_max_total_jobs=96 if eligible else int(current_max_total_jobs),
        eligible=eligible,
        completed_jobs_considered=len(considered),
        failed_jobs_considered=failed_jobs_considered,
        median_total_duration_seconds=median_total_duration_seconds,
        failure_rate=failure_rate,
        wall_clock_hours_remaining=float(wall_clock_hours_remaining),
        reasons=tuple(reasons),
    )


def _expansion_check_payload(check: ExpansionCheck) -> dict[str, Any]:
    return {
        "phase": check.phase,
        "current_max_total_jobs": check.current_max_total_jobs,
        "recommended_max_total_jobs": check.recommended_max_total_jobs,
        "eligible": check.eligible,
        "completed_jobs_considered": check.completed_jobs_considered,
        "failed_jobs_considered": check.failed_jobs_considered,
        "median_total_duration_seconds": check.median_total_duration_seconds,
        "failure_rate": check.failure_rate,
        "wall_clock_hours_remaining": check.wall_clock_hours_remaining,
        "reasons": list(check.reasons),
    }


def _render_report_markdown(report: dict[str, Any]) -> str:
    jobs = report["jobs"]
    timing = report["timing"]
    best = report["best"]
    archive = report["archive"]
    lines = [
        f"# Circle-Packing Report ({report['phase']})",
        "",
        f"- Experiment ID: `{report['experiment_id']}`",
        f"- Generated at: `{report['generated_at']}`",
        f"- Jobs: total={jobs['total']} succeeded={jobs['succeeded']} failed={jobs['failed']}",
        f"- Seed success rate: {jobs['seed_success_rate']!r}",
        f"- Non-seed success rate: {jobs['non_seed_success_rate']!r}",
        f"- Best sum_radii: {best['sum_radii']!r} commit={best['commit_hash']!r}",
        f"- Best runtime_p50_ms: {best['runtime_p50_ms']!r}",
        f"- Archive occupied cells: {archive['occupied_cells']} best_objective={archive['best_objective']!r}",
        "",
        "## Timing",
        "",
        "| metric | count | mean | p50 | p90 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("total_duration_seconds", "job_total_seconds"),
        ("planning_duration_seconds", "planning_seconds"),
        ("coding_duration_seconds", "coding_seconds"),
        ("evaluator_duration_seconds", "evaluator_seconds"),
        ("runtime_p50_ms", "runtime_p50_ms"),
        ("planning_attempts", "planning_attempts"),
        ("coding_attempts", "coding_attempts"),
    ):
        stat = timing[key]
        lines.append(
            f"| {label} | {stat['count']} | {stat['mean']!r} | {stat['p50']!r} | {stat['p90']!r} |"
        )

    lines.extend(
        [
            "",
            "## Worker Throughput",
            "",
            "| worker | total | succeeded | failed | best_sum_radii | mean_job_seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for worker in report["worker_throughput"]:
        lines.append(
            "| {} | {} | {} | {} | {!r} | {!r} |".format(
                worker["worker_instance_id"],
                worker["jobs_total"],
                worker["jobs_succeeded"],
                worker["jobs_failed"],
                worker["best_sum_radii"],
                worker["total_duration_seconds"]["mean"],
            )
        )

    lines.extend(
        [
            "",
            "## References",
            "",
            "| label | commit | sum_radii | density | time_p50_ms | deterministic |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for ref in report["references"]:
        target_metrics = ref.get("target_metrics") or {}
        repeated = ref.get("repeated_runs") or {}
        time_stats = repeated.get("time_ms") or {}
        lines.append(
            "| {} | `{}` | {!r} | {!r} | {!r} | {} |".format(
                ref.get("label"),
                ref.get("commit_hash"),
                target_metrics.get("sum_radii"),
                target_metrics.get("packing_density"),
                time_stats.get("p50"),
                repeated.get("deterministic"),
            )
        )

    lines.extend(
        [
            "",
            "## Objective Trajectory",
            "",
            "| idx | worker | seed | sum_radii | density | total_seconds | elapsed_minutes |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for point in report["trajectory"]:
        lines.append(
            "| {} | {} | {} | {!r} | {!r} | {!r} | {!r} |".format(
                point["index"],
                point["worker_instance_id"],
                "yes" if point["is_seed_job"] else "no",
                point["sum_radii"],
                point["packing_density"],
                point["total_duration_seconds"],
                point["elapsed_minutes_from_start"],
            )
        )
    return "\n".join(lines) + "\n"


def _run_report(*, phase: str, runs: int, output_dir: Path | None) -> int:
    _apply_base_env(phase=phase)
    _ensure_repo_on_sys_path()
    preset = _phase_preset(phase)
    jobs, archive_cells = _load_experiment_jobs()
    best_commit_hash = None
    succeeded = [job for job in jobs if job.get("status") == "succeeded" and job.get("sum_radii") is not None]
    if succeeded:
        best_commit_hash = max(succeeded, key=lambda item: float(item.get("sum_radii") or 0.0)).get(
            "result_commit_hash"
        )
    references = _collect_reference_stats(best_commit_hash=best_commit_hash, runs=runs)
    report = _build_report_payload(
        phase=phase,
        experiment_id=preset.experiment_id,
        jobs=jobs,
        archive_cells=archive_cells,
        references=references,
    )

    report_dir = output_dir or (_phase_logs_root(phase) / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{phase}-report.json"
    markdown_path = report_dir / f"{phase}-report.md"
    _write_json(json_path, report)
    markdown_path.write_text(_render_report_markdown(report), encoding="utf-8")

    console.log(f"[bold green]Report written[/] json={json_path} markdown={markdown_path}")
    return 0


def _run_expansion_check(
    *,
    phase: str,
    current_max_total_jobs: int | None,
    wall_clock_hours_remaining: float,
) -> int:
    preset = _phase_preset(phase)
    if phase != "main":
        raise SystemExit("expansion-check is only supported for the main phase.")

    _apply_base_env(phase=phase, max_total_jobs_override=current_max_total_jobs)
    _ensure_repo_on_sys_path()
    jobs, _archive_cells = _load_experiment_jobs()
    check = _evaluate_main_expansion(
        jobs=jobs,
        wall_clock_hours_remaining=float(wall_clock_hours_remaining),
        current_max_total_jobs=int(current_max_total_jobs or preset.max_total_jobs),
    )
    payload = _expansion_check_payload(check)
    console.print_json(data=payload)
    if check.eligible:
        console.log(
            "[bold green]Expansion eligible[/] phase=main recommended_max_total_jobs={}".format(
                check.recommended_max_total_jobs,
            )
        )
        return 0
    console.log(
        "[yellow]Expansion not recommended[/] reasons={}".format(
            "; ".join(check.reasons) or "unknown",
        )
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Loreley scheduler/worker configured for the circle-packing example.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scheduler_parser = subparsers.add_parser("scheduler", help="Run the evolution scheduler.")
    scheduler_parser.add_argument(
        "--phase",
        choices=sorted(PHASE_PRESETS),
        default="smoke",
        help="Experiment phase preset.",
    )
    scheduler_parser.add_argument("--once", action="store_true", help="Execute one scheduler tick.")
    scheduler_parser.add_argument(
        "--init-db",
        action="store_true",
        help="Drop and recreate all ORM-managed tables before running.",
    )
    scheduler_parser.add_argument("--yes", action="store_true", help="Auto-approve startup.")
    scheduler_parser.add_argument("--no-preflight", action="store_true", help="Skip preflight.")
    scheduler_parser.add_argument(
        "--preflight-timeout-seconds",
        type=_positive_float,
        default=2.0,
        help="Positive timeout used for DB/Redis checks.",
    )
    scheduler_parser.add_argument(
        "--max-total-jobs",
        type=_positive_int,
        default=None,
        help="Optional override for SCHEDULER_MAX_TOTAL_JOBS.",
    )
    scheduler_parser.add_argument("--log-level", dest="log_level", help="Override LOG_LEVEL.")

    worker_parser = subparsers.add_parser("worker", help="Run one evolution worker process.")
    worker_parser.add_argument(
        "--phase",
        choices=sorted(PHASE_PRESETS),
        default="smoke",
        help="Experiment phase preset.",
    )
    worker_parser.add_argument("--no-preflight", action="store_true", help="Skip preflight.")
    worker_parser.add_argument(
        "--preflight-timeout-seconds",
        type=_positive_float,
        default=2.0,
        help="Positive timeout used for DB/Redis checks.",
    )
    worker_parser.add_argument("--log-level", dest="log_level", help="Override LOG_LEVEL.")
    worker_parser.add_argument(
        "--manifest-entry",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    workers_parser = subparsers.add_parser(
        "workers",
        help="Supervise multiple independent worker OS processes.",
    )
    workers_parser.add_argument(
        "--phase",
        choices=sorted(PHASE_PRESETS),
        default="smoke",
        help="Experiment phase preset.",
    )
    workers_parser.add_argument(
        "--count",
        type=_positive_int,
        default=4,
        help="Number of worker processes to supervise.",
    )
    workers_parser.add_argument("--no-preflight", action="store_true", help="Skip preflight.")
    workers_parser.add_argument(
        "--preflight-timeout-seconds",
        type=_positive_float,
        default=2.0,
        help="Positive timeout used for DB/Redis checks.",
    )
    workers_parser.add_argument("--log-level", dest="log_level", help="Override LOG_LEVEL.")

    api_parser = subparsers.add_parser("api", help="Run the read-only UI API.")
    api_parser.add_argument(
        "--phase",
        choices=sorted(PHASE_PRESETS),
        default="main",
        help="Experiment phase preset.",
    )
    api_parser.add_argument("--host", default=UI_API_HOST, help="Bind host.")
    api_parser.add_argument("--port", type=int, default=UI_API_PORT, help="Bind port.")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    api_parser.add_argument("--no-preflight", action="store_true", help="Skip preflight.")
    api_parser.add_argument(
        "--preflight-timeout-seconds",
        type=_positive_float,
        default=2.0,
        help="Positive timeout used for DB checks.",
    )
    api_parser.add_argument("--log-level", dest="log_level", help="Override LOG_LEVEL.")

    ui_parser = subparsers.add_parser("ui", help="Run the Loreley Streamlit UI.")
    ui_parser.add_argument(
        "--phase",
        choices=sorted(PHASE_PRESETS),
        default="main",
        help="Experiment phase preset.",
    )
    ui_parser.add_argument("--api-base-url", default=UI_API_BASE_URL, help="UI API base URL.")
    ui_parser.add_argument("--host", default=UI_HOST, help="Streamlit host.")
    ui_parser.add_argument("--port", type=int, default=UI_PORT, help="Streamlit port.")
    ui_parser.add_argument("--headless", action="store_true", help="Run without browser.")
    ui_parser.add_argument("--no-preflight", action="store_true", help="Skip preflight.")
    ui_parser.add_argument(
        "--preflight-timeout-seconds",
        type=_positive_float,
        default=2.0,
        help="Positive timeout used for preflight checks.",
    )
    ui_parser.add_argument("--log-level", dest="log_level", help="Override LOG_LEVEL.")

    report_parser = subparsers.add_parser(
        "report",
        help="Aggregate DB rows + artifacts into Markdown/JSON.",
    )
    report_parser.add_argument(
        "--phase",
        choices=sorted(PHASE_PRESETS),
        default="smoke",
        help="Experiment phase preset.",
    )
    report_parser.add_argument(
        "--runs",
        type=_positive_int,
        default=DEFAULT_LOCAL_EVAL_RUNS,
        help="Repeated local-eval runs per reference commit.",
    )
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for report files.",
    )

    expansion_parser = subparsers.add_parser(
        "expansion-check",
        help="Check whether the main run should expand from 64 to 96 jobs.",
    )
    expansion_parser.add_argument(
        "--phase",
        choices=("main",),
        default="main",
        help="Experiment phase preset.",
    )
    expansion_parser.add_argument(
        "--current-max-total-jobs",
        type=_positive_int,
        default=64,
        help="Current main-phase SCHEDULER_MAX_TOTAL_JOBS.",
    )
    expansion_parser.add_argument(
        "--wall-clock-hours-remaining",
        type=_positive_float,
        required=True,
        help="Remaining wall-clock budget in hours for the overall experiment.",
    )

    args = parser.parse_args(argv)

    if args.command == "scheduler":
        return _run_scheduler(
            phase=str(args.phase),
            once=bool(args.once),
            init_db=bool(args.init_db),
            yes=bool(args.yes),
            log_level=(str(args.log_level) if args.log_level else None),
            no_preflight=bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
            max_total_jobs=(int(args.max_total_jobs) if args.max_total_jobs else None),
        )
    if args.command == "worker":
        return _run_worker(
            phase=str(args.phase),
            log_level=(str(args.log_level) if args.log_level else None),
            no_preflight=bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
            manifest_entry=(Path(args.manifest_entry) if args.manifest_entry else None),
        )
    if args.command == "workers":
        return _run_workers(
            phase=str(args.phase),
            count=int(args.count),
            log_level=(str(args.log_level) if args.log_level else None),
            no_preflight=bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        )
    if args.command == "api":
        return _run_api(
            phase=str(args.phase),
            host=str(args.host),
            port=int(args.port),
            log_level=(str(args.log_level) if args.log_level else None),
            reload=bool(args.reload),
            no_preflight=bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        )
    if args.command == "ui":
        return _run_ui(
            phase=str(args.phase),
            host=str(args.host),
            port=int(args.port),
            api_base_url=str(args.api_base_url),
            headless=bool(args.headless),
            log_level=(str(args.log_level) if args.log_level else None),
            no_preflight=bool(args.no_preflight),
            preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        )
    if args.command == "report":
        return _run_report(
            phase=str(args.phase),
            runs=int(args.runs),
            output_dir=(Path(args.output_dir) if args.output_dir else None),
        )
    if args.command == "expansion-check":
        return _run_expansion_check(
            phase=str(args.phase),
            current_max_total_jobs=int(args.current_max_total_jobs),
            wall_clock_hours_remaining=float(args.wall_clock_hours_remaining),
        )

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; exiting...")
        raise SystemExit(130)
