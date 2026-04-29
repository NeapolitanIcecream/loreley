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
from typing import Any, Callable, Sequence

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


@dataclass(slots=True)
class WorkerSupervisorState:
    stop_requested: bool = False
    interrupted: bool = False
    exit_code: int = 0


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


def _supervisor_manifest_path(phase: str) -> Path:
    return _supervisor_dir(phase) / "workers-manifest.json"


def _write_supervisor_manifest(
    *,
    path: Path,
    phase: str,
    specs: Sequence[WorkerProcessSpec],
    processes: Sequence[subprocess.Popen[str]],
    status: str,
    started_at: str,
    ended_at: str | None = None,
) -> None:
    _write_json(
        path,
        _collect_supervisor_manifest(
            phase=phase,
            specs=specs,
            processes=processes,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
        ),
    )


def _start_worker_process(
    spec: WorkerProcessSpec,
    log_handle: Any,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        list(spec.command),
        env=spec.env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _start_worker_processes(
    specs: Sequence[WorkerProcessSpec],
    *,
    processes: list[subprocess.Popen[str]],
    log_handles: list[Any],
) -> None:
    for spec in specs:
        handle = _open_supervisor_log(spec.log_path)
        log_handles.append(handle)
        proc = _start_worker_process(spec, handle)
        processes.append(proc)
        console.log(
            "[bold green]Started worker process[/] instance={} pid={} log={}".format(
                spec.instance_id,
                proc.pid,
                spec.log_path,
            )
        )


def _request_worker_supervisor_stop(
    *,
    signum: int,
    processes: Sequence[subprocess.Popen[str]],
    state: WorkerSupervisorState,
) -> None:
    state.interrupted = True
    state.stop_requested = True
    state.exit_code = 130 if signum == signal.SIGINT else 143
    console.log(f"[yellow]Signal received[/] signum={signum}; stopping workers...")
    for proc in processes:
        _terminate_process_group(proc)


def _install_worker_signal_handlers(
    *,
    processes: Sequence[subprocess.Popen[str]],
    state: WorkerSupervisorState,
) -> dict[int, Any]:
    def _request_stop(signum: int, _frame: Any) -> None:
        _request_worker_supervisor_stop(
            signum=signum,
            processes=processes,
            state=state,
        )

    previous_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    return previous_handlers


def _restore_signal_handlers(previous_handlers: dict[int, Any]) -> None:
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def _terminate_sibling_workers(
    processes: Sequence[subprocess.Popen[str]],
    failed_process: subprocess.Popen[str],
) -> None:
    for sibling in processes:
        if sibling is not failed_process:
            _terminate_process_group(sibling)


def _track_worker_exit(
    proc: subprocess.Popen[str],
    *,
    processes: Sequence[subprocess.Popen[str]],
    state: WorkerSupervisorState,
) -> bool:
    returncode = proc.poll()
    if returncode is None:
        return True
    if returncode != 0 and not state.stop_requested:
        state.exit_code = int(returncode)
        state.stop_requested = True
        console.log(
            "[bold red]Worker exited unexpectedly[/] pid={} returncode={}".format(
                proc.pid,
                returncode,
            )
        )
        _terminate_sibling_workers(processes, proc)
    return False


def _monitor_worker_processes(
    processes: Sequence[subprocess.Popen[str]],
    *,
    state: WorkerSupervisorState,
    poll_interval_seconds: float = 1.0,
) -> None:
    while processes:
        live_count = sum(
            1
            for proc in processes
            if _track_worker_exit(proc, processes=processes, state=state)
        )
        if live_count == 0:
            break
        time.sleep(poll_interval_seconds)


def _await_worker_shutdown(
    processes: Sequence[subprocess.Popen[str]],
    *,
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if all(proc.poll() is not None for proc in processes):
            break
        time.sleep(0.25)
    for proc in processes:
        _terminate_process_group(proc, force=True)


def _final_supervisor_status(state: WorkerSupervisorState) -> str:
    if state.interrupted:
        return "interrupted"
    if state.exit_code:
        return "failed"
    return "completed"


def _close_log_handles(log_handles: Sequence[Any]) -> None:
    for handle in log_handles:
        handle.close()


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
    manifest_path = _supervisor_manifest_path(phase)
    started_at = _now_utc_iso()
    processes: list[subprocess.Popen[str]] = []
    log_handles: list[Any] = []
    state = WorkerSupervisorState()
    previous_handlers = _install_worker_signal_handlers(processes=processes, state=state)

    try:
        _start_worker_processes(
            specs,
            processes=processes,
            log_handles=log_handles,
        )
        _write_supervisor_manifest(
            path=manifest_path,
            phase=phase,
            specs=specs,
            processes=processes,
            status="running",
            started_at=started_at,
        )

        _monitor_worker_processes(processes, state=state)
        if state.stop_requested:
            _await_worker_shutdown(processes)

        _write_supervisor_manifest(
            path=manifest_path,
            phase=phase,
            specs=specs,
            processes=processes,
            status=_final_supervisor_status(state),
            started_at=started_at,
            ended_at=_now_utc_iso(),
        )
        return state.exit_code
    finally:
        _restore_signal_handlers(previous_handlers)
        _close_log_handles(log_handles)


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


def _resolve_local_eval_script_path() -> Path:
    preferred = REPO_ROOT / "scripts" / "local_eval.py"
    if preferred.is_file():
        return preferred
    fallback = EVAL_ENV_ROOT / "local_eval.py"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        "Could not find local_eval.py in either the circle-packing submodule "
        f"({preferred}) or the main-repo fallback ({fallback})."
    )


def _load_local_eval_module() -> Any:
    return _load_module_from_path(
        "circle_packing_local_eval",
        _resolve_local_eval_script_path(),
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


def _load_metrics_by_card_id(
    *,
    session: Any,
    cards: Sequence[Any],
    metric_model: Any,
    select_fn: Any,
) -> dict[str, dict[str, float]]:
    metrics_by_card_id: dict[str, dict[str, float]] = {}
    if not cards:
        return metrics_by_card_id

    card_ids = [card.id for card in cards]
    metrics = session.execute(
        select_fn(metric_model).where(metric_model.commit_card_id.in_(card_ids))
    ).scalars().all()
    for metric in metrics:
        metrics_by_card_id.setdefault(str(metric.commit_card_id), {})[metric.name] = float(
            metric.value
        )
    return metrics_by_card_id


def _load_experiment_rows() -> tuple[
    list[Any],
    list[Any],
    list[Any],
    list[Any],
    dict[str, dict[str, float]],
]:
    from sqlalchemy import select

    from loreley.db.base import session_scope
    from loreley.db.models import CommitCard, EvolutionJob, JobArtifacts, MapElitesArchiveCell, Metric

    with session_scope() as session:
        jobs = session.execute(select(EvolutionJob).order_by(EvolutionJob.created_at.asc())).scalars().all()
        artifacts_rows = session.execute(select(JobArtifacts)).scalars().all()
        cards = session.execute(select(CommitCard).where(CommitCard.job_id.is_not(None))).scalars().all()
        archive_cells = session.execute(select(MapElitesArchiveCell)).scalars().all()
        metrics_by_card_id = _load_metrics_by_card_id(
            session=session,
            cards=cards,
            metric_model=Metric,
            select_fn=select,
        )

    return jobs, artifacts_rows, cards, archive_cells, metrics_by_card_id


def _cards_by_job_id(cards: Sequence[Any]) -> dict[str, Any]:
    return {
        str(card.job_id): card
        for card in cards
        if getattr(card, "job_id", None) is not None
    }


def _artifacts_by_job_id(artifacts_rows: Sequence[Any]) -> dict[str, Any]:
    return {str(row.job_id): row for row in artifacts_rows}


def _read_job_artifact_payloads(
    artifact_row: Any | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    planning_payload = _read_json(
        getattr(artifact_row, "planning_plan_json_path", None) if artifact_row else None
    )
    coding_payload = _read_json(
        getattr(artifact_row, "coding_execution_json_path", None) if artifact_row else None
    )
    evaluation_payload = _read_json(
        getattr(artifact_row, "evaluation_json_path", None) if artifact_row else None
    )
    return planning_payload, coding_payload, evaluation_payload


def _artifact_worker_payload(
    planning_payload: dict[str, Any],
    coding_payload: dict[str, Any],
    evaluation_payload: dict[str, Any],
) -> dict[str, Any]:
    return (
        evaluation_payload.get("worker")
        or coding_payload.get("worker")
        or planning_payload.get("worker")
        or {}
    )


def _payload_section(payload: dict[str, Any], section: str) -> dict[str, Any]:
    return payload.get(section, {}) or {}


def _experiment_job_metrics(
    *,
    card: Any | None,
    metrics_by_card_id: dict[str, dict[str, float]],
) -> dict[str, float]:
    return metrics_by_card_id.get(str(card.id), {}) if card is not None else {}


def _build_experiment_job_payload(
    *,
    job: Any,
    artifact_row: Any | None,
    card: Any | None,
    metrics_by_card_id: dict[str, dict[str, float]],
) -> dict[str, Any]:
    planning_payload, coding_payload, evaluation_payload = _read_job_artifact_payloads(
        artifact_row
    )
    planning_backend = _payload_section(planning_payload, "backend")
    coding_backend = _payload_section(coding_payload, "backend")
    evaluation_extra = _payload_section(evaluation_payload, "extra")
    worker = _artifact_worker_payload(
        planning_payload,
        coding_payload,
        evaluation_payload,
    )
    metrics = _experiment_job_metrics(
        card=card,
        metrics_by_card_id=metrics_by_card_id,
    )
    status_value = getattr(job.status, "value", job.status)
    return {
        "job_id": str(job.id),
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
        "planning_duration_seconds": planning_backend.get("duration_seconds"),
        "coding_duration_seconds": coding_backend.get("duration_seconds"),
        "evaluator_duration_seconds": evaluation_extra.get("evaluator_duration_seconds"),
        "planning_attempts": planning_backend.get("attempts"),
        "coding_attempts": coding_backend.get("attempts"),
        "worker_instance_id": worker.get("instance_id"),
        "worker_pid": worker.get("pid"),
        "sum_radii": metrics.get("sum_radii"),
        "packing_density": metrics.get("packing_density"),
        "runtime_p50_ms": metrics.get("runtime_p50_ms"),
        "metrics": metrics,
    }


def _build_experiment_job_payloads(
    *,
    jobs: Sequence[Any],
    artifacts_rows: Sequence[Any],
    cards: Sequence[Any],
    metrics_by_card_id: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    artifacts = _artifacts_by_job_id(artifacts_rows)
    cards_by_job = _cards_by_job_id(cards)
    return [
        _build_experiment_job_payload(
            job=job,
            artifact_row=artifacts.get(str(job.id)),
            card=cards_by_job.get(str(job.id)),
            metrics_by_card_id=metrics_by_card_id,
        )
        for job in jobs
    ]


def _build_archive_payload(cell: Any) -> dict[str, Any]:
    return {
        "cell_index": int(cell.cell_index),
        "commit_hash": cell.commit_hash,
        "objective": float(cell.objective),
        "timestamp": float(cell.timestamp),
        "created_at": _dt_iso(cell.created_at),
        "updated_at": _dt_iso(cell.updated_at),
    }


def _build_archive_payloads(archive_cells: Sequence[Any]) -> list[dict[str, Any]]:
    return [_build_archive_payload(cell) for cell in archive_cells]


def _load_experiment_jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs, artifacts_rows, cards, archive_cells, metrics_by_card_id = _load_experiment_rows()
    return (
        _build_experiment_job_payloads(
            jobs=jobs,
            artifacts_rows=artifacts_rows,
            cards=cards,
            metrics_by_card_id=metrics_by_card_id,
        ),
        _build_archive_payloads(archive_cells),
    )


def _jobs_with_status(
    jobs: Sequence[dict[str, Any]],
    status: str,
) -> list[dict[str, Any]]:
    return [job for job in jobs if job.get("status") == status]


def _build_jobs_summary(jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_jobs = len(jobs)
    succeeded_jobs = _jobs_with_status(jobs, "succeeded")
    failed_jobs = _jobs_with_status(jobs, "failed")
    seed_jobs = [job for job in jobs if job.get("is_seed_job")]
    non_seed_jobs = [job for job in jobs if not job.get("is_seed_job")]
    seed_success = [job for job in succeeded_jobs if job.get("is_seed_job")]
    non_seed_success = [job for job in succeeded_jobs if not job.get("is_seed_job")]
    return {
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
    }


def _numeric_job_samples(
    jobs: Sequence[dict[str, Any]],
    field: str,
) -> list[float]:
    return [float(job[field]) for job in jobs if job.get(field) is not None]


def _build_timing_summary(jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_duration_seconds": _stats(
            _numeric_job_samples(jobs, "total_duration_seconds")
        ),
        "planning_duration_seconds": _stats(
            _numeric_job_samples(jobs, "planning_duration_seconds")
        ),
        "coding_duration_seconds": _stats(
            _numeric_job_samples(jobs, "coding_duration_seconds")
        ),
        "evaluator_duration_seconds": _stats(
            _numeric_job_samples(jobs, "evaluator_duration_seconds")
        ),
        "runtime_p50_ms": _stats(_numeric_job_samples(jobs, "runtime_p50_ms")),
        "planning_attempts": _stats(_numeric_job_samples(jobs, "planning_attempts")),
        "coding_attempts": _stats(_numeric_job_samples(jobs, "coding_attempts")),
    }


def _job_order_timestamp(job: dict[str, Any]) -> float:
    return float(job.get("_completed_ts") or job.get("_created_ts") or 0.0)


def _ordered_success_jobs(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            job
            for job in jobs
            if job.get("status") == "succeeded" and job.get("sum_radii") is not None
        ],
        key=lambda item: (
            _job_order_timestamp(item),
            item.get("job_id") or "",
        ),
    )


def _experiment_start_timestamp(jobs: Sequence[dict[str, Any]]) -> float | None:
    return min(
        (
            value
            for job in jobs
            for value in (job.get("_started_ts"), job.get("_created_ts"))
            if value is not None
        ),
        default=None,
    )


def _build_trajectory_point(
    *,
    index: int,
    job: dict[str, Any],
    experiment_start: float | None,
) -> dict[str, Any]:
    return {
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
            round((_job_order_timestamp(job) - experiment_start) / 60.0, 3)
            if experiment_start is not None
            else None
        ),
    }


def _build_objective_trajectory(
    ordered_success: Sequence[dict[str, Any]],
    *,
    experiment_start: float | None,
) -> list[dict[str, Any]]:
    return [
        _build_trajectory_point(
            index=index,
            job=job,
            experiment_start=experiment_start,
        )
        for index, job in enumerate(ordered_success, start=1)
    ]


def _select_best_job(
    ordered_success: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not ordered_success:
        return None
    return max(ordered_success, key=lambda item: float(item.get("sum_radii") or 0.0))


def _baseline_sum_radii(references: Sequence[dict[str, Any]]) -> Any:
    root_reference = next((item for item in references if item.get("label") == "root"), None)
    if not root_reference:
        return None
    return (root_reference.get("target_metrics") or {}).get("sum_radii")


def _first_trajectory_point_above_baseline(
    trajectory: Sequence[dict[str, Any]],
    baseline_sum_radii: Any,
) -> dict[str, Any] | None:
    if baseline_sum_radii is None:
        return None
    return next(
        (
            point
            for point in trajectory
            if point.get("sum_radii") is not None
            and float(point["sum_radii"]) > float(baseline_sum_radii)
        ),
        None,
    )


def _empty_worker_distribution_bucket(worker_id: str) -> dict[str, Any]:
    return {
        "worker_instance_id": worker_id,
        "jobs_total": 0,
        "jobs_succeeded": 0,
        "jobs_failed": 0,
        "best_sum_radii": None,
        "total_duration_seconds": [],
    }


def _record_worker_job(bucket: dict[str, Any], job: dict[str, Any]) -> None:
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


def _worker_throughput_item(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "worker_instance_id": bucket["worker_instance_id"],
        "jobs_total": bucket["jobs_total"],
        "jobs_succeeded": bucket["jobs_succeeded"],
        "jobs_failed": bucket["jobs_failed"],
        "best_sum_radii": bucket["best_sum_radii"],
        "total_duration_seconds": _stats(bucket["total_duration_seconds"]),
    }


def _build_worker_throughput(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    worker_distribution: dict[str, dict[str, Any]] = {}
    for job in jobs:
        worker_instance_id = job.get("worker_instance_id")
        if not worker_instance_id:
            continue
        worker_id = str(worker_instance_id)
        bucket = worker_distribution.setdefault(
            worker_id,
            _empty_worker_distribution_bucket(worker_id),
        )
        _record_worker_job(bucket, job)

    worker_throughput = [
        _worker_throughput_item(bucket)
        for bucket in worker_distribution.values()
    ]
    worker_throughput.sort(key=lambda item: item["worker_instance_id"])
    return worker_throughput


def _failure_summaries(failed_jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": job.get("job_id"),
            "worker_instance_id": job.get("worker_instance_id"),
            "last_error": job.get("last_error"),
        }
        for job in failed_jobs
    ]


def _build_archive_summary(archive_cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    archive_summary = {
        "occupied_cells": len(archive_cells),
        "best_objective": max((float(cell["objective"]) for cell in archive_cells), default=None),
        "best_commit_hash": None,
    }
    if archive_cells:
        best_cell = max(archive_cells, key=lambda cell: float(cell["objective"]))
        archive_summary["best_commit_hash"] = best_cell.get("commit_hash")
    return archive_summary


def _best_job_summary(best_job: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "job_id": best_job.get("job_id") if best_job else None,
        "commit_hash": best_job.get("result_commit_hash") if best_job else None,
        "sum_radii": best_job.get("sum_radii") if best_job else None,
        "packing_density": best_job.get("packing_density") if best_job else None,
        "runtime_p50_ms": best_job.get("runtime_p50_ms") if best_job else None,
        "worker_instance_id": best_job.get("worker_instance_id") if best_job else None,
    }


def _build_report_payload(
    *,
    phase: str,
    experiment_id: str,
    jobs: Sequence[dict[str, Any]],
    archive_cells: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ordered_success = _ordered_success_jobs(jobs)
    trajectory = _build_objective_trajectory(
        ordered_success,
        experiment_start=_experiment_start_timestamp(jobs),
    )
    best_job = _select_best_job(ordered_success)
    first_above_baseline = _first_trajectory_point_above_baseline(
        trajectory,
        _baseline_sum_radii(references),
    )

    return {
        "phase": phase,
        "experiment_id": experiment_id,
        "generated_at": _now_utc_iso(),
        "jobs": _build_jobs_summary(jobs),
        "timing": _build_timing_summary(jobs),
        "best": _best_job_summary(best_job),
        "archive": _build_archive_summary(archive_cells),
        "first_above_baseline": first_above_baseline,
        "worker_throughput": _build_worker_throughput(jobs),
        "trajectory": trajectory,
        "references": list(references),
        "failures": _failure_summaries(_jobs_with_status(jobs, "failed")),
    }


def _completed_jobs_for_expansion(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
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


def _expansion_duration_median(jobs: Sequence[dict[str, Any]]) -> float | None:
    return _percentile(
        [
            float(job["total_duration_seconds"])
            for job in jobs
            if job.get("total_duration_seconds") is not None
        ],
        0.50,
    )


def _failed_expansion_jobs_count(jobs: Sequence[dict[str, Any]]) -> int:
    return sum(1 for job in jobs if job.get("status") == "failed")


def _expansion_failure_rate(
    *,
    failed_jobs_count: int,
    total_jobs_count: int,
) -> float | None:
    if not total_jobs_count:
        return None
    return float(failed_jobs_count / total_jobs_count)


def _main_expansion_reasons(
    *,
    considered_jobs_count: int,
    median_total_duration_seconds: float | None,
    failure_rate: float | None,
    wall_clock_hours_remaining: float,
) -> list[str]:
    reasons: list[str] = []
    if considered_jobs_count < 24:
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
    return reasons


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

    considered = _completed_jobs_for_expansion(jobs)[:24]
    median_total_duration_seconds = _expansion_duration_median(considered)
    failed_jobs_considered = _failed_expansion_jobs_count(considered)
    failure_rate = _expansion_failure_rate(
        failed_jobs_count=failed_jobs_considered,
        total_jobs_count=len(considered),
    )
    reasons = _main_expansion_reasons(
        considered_jobs_count=len(considered),
        median_total_duration_seconds=median_total_duration_seconds,
        failure_rate=failure_rate,
        wall_clock_hours_remaining=wall_clock_hours_remaining,
    )

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


def _add_phase_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str,
    choices: Sequence[str] | None = None,
) -> None:
    parser.add_argument(
        "--phase",
        choices=tuple(choices) if choices is not None else sorted(PHASE_PRESETS),
        default=default,
        help="Experiment phase preset.",
    )


def _add_preflight_arguments(
    parser: argparse.ArgumentParser,
    *,
    timeout_help: str,
) -> None:
    parser.add_argument("--no-preflight", action="store_true", help="Skip preflight.")
    parser.add_argument(
        "--preflight-timeout-seconds",
        type=_positive_float,
        default=2.0,
        help=timeout_help,
    )


def _add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", dest="log_level", help="Override LOG_LEVEL.")


def _add_scheduler_parser(subparsers: Any) -> None:
    scheduler_parser = subparsers.add_parser(
        "scheduler",
        help="Run the evolution scheduler.",
    )
    _add_phase_argument(scheduler_parser, default="smoke")
    scheduler_parser.add_argument("--once", action="store_true", help="Execute one scheduler tick.")
    scheduler_parser.add_argument(
        "--init-db",
        action="store_true",
        help="Drop and recreate all ORM-managed tables before running.",
    )
    scheduler_parser.add_argument("--yes", action="store_true", help="Auto-approve startup.")
    _add_preflight_arguments(
        scheduler_parser,
        timeout_help="Positive timeout used for DB/Redis checks.",
    )
    scheduler_parser.add_argument(
        "--max-total-jobs",
        type=_positive_int,
        default=None,
        help="Optional override for SCHEDULER_MAX_TOTAL_JOBS.",
    )
    _add_log_level_argument(scheduler_parser)


def _add_worker_parser(subparsers: Any) -> None:
    worker_parser = subparsers.add_parser("worker", help="Run one evolution worker process.")
    _add_phase_argument(worker_parser, default="smoke")
    _add_preflight_arguments(
        worker_parser,
        timeout_help="Positive timeout used for DB/Redis checks.",
    )
    _add_log_level_argument(worker_parser)
    worker_parser.add_argument(
        "--manifest-entry",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )


def _add_workers_parser(subparsers: Any) -> None:
    workers_parser = subparsers.add_parser(
        "workers",
        help="Supervise multiple independent worker OS processes.",
    )
    _add_phase_argument(workers_parser, default="smoke")
    workers_parser.add_argument(
        "--count",
        type=_positive_int,
        default=4,
        help="Number of worker processes to supervise.",
    )
    _add_preflight_arguments(
        workers_parser,
        timeout_help="Positive timeout used for DB/Redis checks.",
    )
    _add_log_level_argument(workers_parser)


def _add_api_parser(subparsers: Any) -> None:
    api_parser = subparsers.add_parser("api", help="Run the read-only UI API.")
    _add_phase_argument(api_parser, default="main")
    api_parser.add_argument("--host", default=UI_API_HOST, help="Bind host.")
    api_parser.add_argument("--port", type=int, default=UI_API_PORT, help="Bind port.")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    _add_preflight_arguments(
        api_parser,
        timeout_help="Positive timeout used for DB checks.",
    )
    _add_log_level_argument(api_parser)


def _add_ui_parser(subparsers: Any) -> None:
    ui_parser = subparsers.add_parser("ui", help="Run the Loreley Streamlit UI.")
    _add_phase_argument(ui_parser, default="main")
    ui_parser.add_argument("--api-base-url", default=UI_API_BASE_URL, help="UI API base URL.")
    ui_parser.add_argument("--host", default=UI_HOST, help="Streamlit host.")
    ui_parser.add_argument("--port", type=int, default=UI_PORT, help="Streamlit port.")
    ui_parser.add_argument("--headless", action="store_true", help="Run without browser.")
    _add_preflight_arguments(
        ui_parser,
        timeout_help="Positive timeout used for preflight checks.",
    )
    _add_log_level_argument(ui_parser)


def _add_report_parser(subparsers: Any) -> None:
    report_parser = subparsers.add_parser(
        "report",
        help="Aggregate DB rows + artifacts into Markdown/JSON.",
    )
    _add_phase_argument(report_parser, default="smoke")
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


def _add_expansion_check_parser(subparsers: Any) -> None:
    expansion_parser = subparsers.add_parser(
        "expansion-check",
        help="Check whether the main run should expand from 64 to 96 jobs.",
    )
    _add_phase_argument(expansion_parser, default="main", choices=("main",))
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


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Loreley scheduler/worker configured for the circle-packing example.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_scheduler_parser(subparsers)
    _add_worker_parser(subparsers)
    _add_workers_parser(subparsers)
    _add_api_parser(subparsers)
    _add_ui_parser(subparsers)
    _add_report_parser(subparsers)
    _add_expansion_check_parser(subparsers)
    return parser


def _arg_log_level(args: argparse.Namespace) -> str | None:
    return str(args.log_level) if args.log_level else None


def _dispatch_scheduler_command(args: argparse.Namespace) -> int:
    return _run_scheduler(
        phase=str(args.phase),
        once=bool(args.once),
        init_db=bool(args.init_db),
        yes=bool(args.yes),
        log_level=_arg_log_level(args),
        no_preflight=bool(args.no_preflight),
        preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        max_total_jobs=(int(args.max_total_jobs) if args.max_total_jobs else None),
    )


def _dispatch_worker_command(args: argparse.Namespace) -> int:
    return _run_worker(
        phase=str(args.phase),
        log_level=_arg_log_level(args),
        no_preflight=bool(args.no_preflight),
        preflight_timeout_seconds=float(args.preflight_timeout_seconds),
        manifest_entry=(Path(args.manifest_entry) if args.manifest_entry else None),
    )


def _dispatch_workers_command(args: argparse.Namespace) -> int:
    return _run_workers(
        phase=str(args.phase),
        count=int(args.count),
        log_level=_arg_log_level(args),
        no_preflight=bool(args.no_preflight),
        preflight_timeout_seconds=float(args.preflight_timeout_seconds),
    )


def _dispatch_api_command(args: argparse.Namespace) -> int:
    return _run_api(
        phase=str(args.phase),
        host=str(args.host),
        port=int(args.port),
        log_level=_arg_log_level(args),
        reload=bool(args.reload),
        no_preflight=bool(args.no_preflight),
        preflight_timeout_seconds=float(args.preflight_timeout_seconds),
    )


def _dispatch_ui_command(args: argparse.Namespace) -> int:
    return _run_ui(
        phase=str(args.phase),
        host=str(args.host),
        port=int(args.port),
        api_base_url=str(args.api_base_url),
        headless=bool(args.headless),
        log_level=_arg_log_level(args),
        no_preflight=bool(args.no_preflight),
        preflight_timeout_seconds=float(args.preflight_timeout_seconds),
    )


def _dispatch_report_command(args: argparse.Namespace) -> int:
    return _run_report(
        phase=str(args.phase),
        runs=int(args.runs),
        output_dir=(Path(args.output_dir) if args.output_dir else None),
    )


def _dispatch_expansion_check_command(args: argparse.Namespace) -> int:
    return _run_expansion_check(
        phase=str(args.phase),
        current_max_total_jobs=int(args.current_max_total_jobs),
        wall_clock_hours_remaining=float(args.wall_clock_hours_remaining),
    )


_COMMAND_DISPATCHERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "scheduler": _dispatch_scheduler_command,
    "worker": _dispatch_worker_command,
    "workers": _dispatch_workers_command,
    "api": _dispatch_api_command,
    "ui": _dispatch_ui_command,
    "report": _dispatch_report_command,
    "expansion-check": _dispatch_expansion_check_command,
}


def _dispatch_command(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    dispatcher = _COMMAND_DISPATCHERS.get(str(args.command))
    if dispatcher is not None:
        return dispatcher(args)

    parser.print_help()
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    return _dispatch_command(args=args, parser=parser)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; exiting...")
        raise SystemExit(130)
