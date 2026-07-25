#!/usr/bin/env python3
"""Run a bounded real Loreley scheduler/worker experiment and capture evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_REPO = PROJECT_ROOT / "examples" / "circle-packing"
EVALUATOR_ROOT = PROJECT_ROOT / "examples" / "circle_packing_env"
DEFAULT_DATABASE_URL = "postgresql+psycopg://loreley:loreley@localhost:5432/loreley"
DEFAULT_REDIS_URL = "redis://localhost:6379/15"
DEFAULT_LIVE_MODEL = "gpt-5.4-mini"
_PROXY_PRICING_JSON = json.dumps(
    {
        "version": "newapi-2026-07-26-conservative",
        "prices": [
            {
                "provider": "loreley-openai-compatible",
                "model": DEFAULT_LIVE_MODEL,
                "input_usd_per_1m": "0.9",
                "cached_input_usd_per_1m": "0.9",
                "output_usd_per_1m": "5.4",
            }
        ],
    },
    separators=(",", ":"),
)


def _safe_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    api_key = environment.get("LLM_API_KEY", "").strip()
    base_url = environment.get("LLM_BASE_URL", "").strip()
    for name in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LORELEY_LLM_API_KEY",
        "LORELEY_LLM_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "WORKER_KILOCODE_OPENAI_API_KEY",
        "WORKER_KILOCODE_OPENAI_BASE_URL",
        "LORELEY_KILO_OPENAI_API_KEY",
        "LORELEY_KILO_OPENAI_BASE_URL",
        "KILO_CONFIG_CONTENT",
    ):
        environment.pop(name, None)
    worktree_root = PROJECT_ROOT / ".cache" / "v15-system" / args.label / "worker"
    logs_root = args.output.parent / f"{args.label}-logs"
    environment.update(
        {
            "APP_NAME": "loreley-v15-system-experiment",
            "APP_ENV": "development",
            "LOG_LEVEL": "INFO",
            "LOGS_BASE_DIR": str(logs_root),
            "EXPERIMENT_ID": args.label,
            "DATABASE_URL": args.database_url,
            "DB_NAME": "loreley",
            "TASKS_REDIS_URL": args.redis_url,
            "TASKS_QUEUE_PREFETCH": "1",
            "TASKS_DELAY_QUEUE_PREFETCH": "1",
            "WORKER_REPO_REMOTE_URL": str(EXAMPLE_REPO),
            "WORKER_REPO_BRANCH": "main",
            "WORKER_REPO_WORKTREE": str(worktree_root),
            "WORKER_REPO_WORKTREE_RANDOMIZE": "true",
            "SCHEDULER_REPO_ROOT": str(EXAMPLE_REPO),
            "SCHEDULER_POLL_INTERVAL_SECONDS": "1",
            "SCHEDULER_MAX_UNFINISHED_JOBS": str(args.max_unfinished_jobs),
            "SCHEDULER_MAX_TOTAL_JOBS": str(args.max_total_jobs),
            "SCHEDULER_SCHEDULE_BATCH_SIZE": str(args.max_unfinished_jobs),
            "SCHEDULER_DISPATCH_BATCH_SIZE": str(args.max_unfinished_jobs),
            "SCHEDULER_INGEST_BATCH_SIZE": str(args.max_unfinished_jobs),
            "MAPELITES_OBJECTIVES": json.dumps(
                [
                    {"name": "sum_radii", "direction": "max"},
                    {"name": "runtime_p50_ms", "direction": "min"},
                ]
            ),
            "MAPELITES_ISLANDS": json.dumps(["alpha", "beta"]),
            "MAPELITES_PARETO_FRONT_MAX_SIZE": "4",
            "MAPELITES_MIGRATION_INTERVAL_JOBS": str(args.migration_interval),
            "MAPELITES_EXPERIMENT_ROOT_COMMIT": "6dab191",
            "MAPELITES_SEED_POPULATION_SIZE": "2",
            "MAPELITES_DIMENSION_REDUCTION_TARGET_DIMS": "1",
            "MAPELITES_DIMENSION_REDUCTION_MIN_FIT_SAMPLES": "2",
            "MAPELITES_DIMENSION_REDUCTION_REFIT_INTERVAL": "1",
            "MAPELITES_ARCHIVE_CELLS_PER_DIM": "4",
            "MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES": "2",
            "MAPELITES_FEATURE_TRUNCATION_K": "3",
            "MAPELITES_FEATURE_CLIP": "true",
            "MAPELITES_CODE_EMBEDDING_MODEL": "local-hash-v1",
            "MAPELITES_CODE_EMBEDDING_DIMENSIONS": "64",
            "MAPELITES_CODE_EMBEDDING_BATCH_SIZE": "8",
            "WORKER_EVALUATOR_PYTHON_PATHS": json.dumps([str(EVALUATOR_ROOT)]),
            "WORKER_EVALUATOR_PLUGIN": "evaluate:plugin",
            "WORKER_EVALUATOR_REWORK_ENABLED": "false",
            "WORKER_PLANNING_MAX_ATTEMPTS": "1",
            "WORKER_CODING_MAX_ATTEMPTS": "1",
            "WORKER_EVOLUTION_COMMIT_PROVIDER_MODE": "disabled",
            "V15_EXPERIMENT_PLANNING_DELAY_SECONDS": str(args.planning_delay),
            "V15_EXPERIMENT_CODING_DELAY_SECONDS": str(args.coding_delay),
            "V15_EXPERIMENT_TRACE_PATH": str(args.trace),
            "LLM_USAGE_TRACKING_ENABLED": "true",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if args.backend == "deterministic":
        environment.update(
            {
                "WORKER_PLANNING_BACKEND": (
                    "tools.v15_experiment_backend:planning_backend"
                ),
                "WORKER_CODING_BACKEND": (
                    "tools.v15_experiment_backend:coding_backend"
                ),
                "WORKER_EVOLUTION_GLOBAL_GOAL": (
                    "Preserve the valid deterministic circle packing while "
                    "producing one independently attributable candidate per job."
                ),
            }
        )
        return environment

    if not api_key or not base_url:
        raise RuntimeError(
            "The kilocode experiment backend requires LLM_API_KEY and "
            "LLM_BASE_URL in the parent environment."
        )
    environment.update(
        {
            "WORKER_PLANNING_BACKEND": (
                "tools.v15_experiment_backend:kilocode_planning_backend"
            ),
            "WORKER_CODING_BACKEND": (
                "tools.v15_experiment_backend:kilocode_coding_backend"
            ),
            "WORKER_PLANNING_TIMEOUT_SECONDS": str(args.planning_timeout),
            "WORKER_CODING_TIMEOUT_SECONDS": str(args.coding_timeout),
            "WORKER_KILOCODE_MODEL": f"openai/{args.model}",
            "WORKER_KILOCODE_JSON_OUTPUT": "true",
            "WORKER_KILOCODE_PROVIDER_CONFIG_MODE": "config",
            "WORKER_KILOCODE_OPENAI_API_SPEC": "chat_completions",
            "WORKER_KILOCODE_OPENAI_BASE_URL": base_url,
            "WORKER_KILOCODE_OPENAI_API_KEY": api_key,
            "WORKER_KILOCODE_OPENAI_MODEL": args.model,
            "LLM_USAGE_PRICING_JSON": _PROXY_PRICING_JSON,
            "WORKER_EVOLUTION_GLOBAL_GOAL": (
                "Improve the deterministic 26-circle packing quality while "
                "preserving exact cardinality, unit-square bounds, "
                "non-overlap, and evaluator runtime validity. Use only the "
                "standard library and verify the candidate before finishing."
            ),
        }
    )
    return environment


def _wait_for_scheduler_start(
    process: subprocess.Popen[str],
    log_path: Path,
    *,
    timeout_seconds: float = 45.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4_000:]
            raise RuntimeError(
                f"Scheduler exited before startup (exit={process.returncode}):\n{tail}"
            )
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="replace")
            if "Scheduler online" in content or "Scheduler tick" in content:
                return
        time.sleep(0.2)
    raise TimeoutError("Scheduler did not report startup before timeout.")


def _terminate(
    process: subprocess.Popen[str], *, timeout_seconds: float = 20.0
) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _query_rows(database_url: str) -> dict[str, Any]:
    engine = create_engine(database_url)
    with engine.connect() as connection:
        jobs = [
            dict(row._mapping)
            for row in connection.execute(
                text(
                    """
                    SELECT
                        id::text AS id,
                        status::text AS status,
                        island_id,
                        base_commit_hash,
                        inspiration_commit_hashes,
                        migration_source_island_id,
                        migration_commit_hash,
                        is_seed_job,
                        job_kind,
                        candidate_commit_hash,
                        result_commit_hash,
                        ingestion_status,
                        worker_id,
                        extract(epoch FROM scheduled_at) AS scheduled_at_epoch,
                        extract(epoch FROM started_at) AS started_at_epoch,
                        extract(epoch FROM completed_at) AS completed_at_epoch,
                        last_error
                    FROM evolution_jobs
                    ORDER BY created_at, id
                    """
                )
            )
        ]
        cells = [
            dict(row._mapping)
            for row in connection.execute(
                text(
                    """
                    SELECT island_id, cell_index, commit_hash, objective_values, measures
                    FROM map_elites_archive_cells
                    ORDER BY island_id, cell_index, commit_hash
                    """
                )
            )
        ]
        states = [
            dict(row._mapping)
            for row in connection.execute(
                text(
                    """
                    SELECT island_id, snapshot
                    FROM map_elites_states
                    ORDER BY island_id
                    """
                )
            )
        ]
        usage = [
            dict(row._mapping)
            for row in connection.execute(
                text(
                    """
                    SELECT phase, source, provider, model, api_surface,
                           input_tokens, cached_input_tokens, output_tokens,
                           reasoning_output_tokens, total_tokens, cost_usd,
                           cost_source, pricing_version
                    FROM llm_usage_events
                    ORDER BY created_at, id
                    """
                )
            )
        ]
    engine.dispose()
    return {
        "jobs": jobs,
        "archive_cells": cells,
        "island_states": states,
        "usage": usage,
    }


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if hasattr(value, "as_tuple"):
        return float(value)
    return value


def _trace_events_by_job(
    trace: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    phase_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in trace:
        phase_by_job[str(event.get("job_id"))].append(event)
    return phase_by_job


def _prompt_commit_hashes(events: list[dict[str, Any]]) -> set[str]:
    return {
        str(commit_hash)
        for event in events
        if event["phase"] == "planning"
        for commit_hash in event.get("prompt_commit_hashes", [])
    }


def _migration_evidence(
    jobs: list[dict[str, Any]],
    phase_by_job: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    evidence = []
    for job in jobs:
        donor = job["migration_commit_hash"]
        if not donor:
            continue
        evidence.append(
            {
                "job_id": job["id"],
                "target_island": job["island_id"],
                "source_island": job["migration_source_island_id"],
                "migration_commit_hash": donor,
                "donor_is_cross_island": (
                    job["migration_source_island_id"] != job["island_id"]
                ),
                "donor_in_persisted_inspirations": (
                    donor in tuple(job["inspiration_commit_hashes"] or ())
                ),
                "donor_reached_planning_prompt": (
                    donor in _prompt_commit_hashes(phase_by_job.get(str(job["id"]), []))
                ),
            }
        )
    return evidence


def _migration_summary(
    *,
    jobs: list[dict[str, Any]],
    phase_by_job: dict[str, list[dict[str, Any]]],
    migration_interval: int,
) -> dict[str, Any]:
    evidence = _migration_evidence(jobs, phase_by_job)
    provenance_valid = bool(evidence) and all(
        item["donor_is_cross_island"]
        and item["donor_in_persisted_inspirations"]
        and item["donor_reached_planning_prompt"]
        for item in evidence
    )
    return {
        "migration_job_count": len(evidence),
        "migration_evidence": evidence,
        "migration_provenance_valid": provenance_valid,
        "migration_disabled_has_no_donors": (migration_interval == 0 and not evidence),
    }


def _island_job_counts(
    jobs: list[dict[str, Any]],
    *,
    seed_job: bool | None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for job in jobs:
        if seed_job is None or bool(job["is_seed_job"]) == seed_job:
            counts[str(job["island_id"])] += 1
    return counts


def _job_execution_window(jobs: list[dict[str, Any]]) -> float | None:
    started = [
        float(job["started_at_epoch"])
        for job in jobs
        if job["started_at_epoch"] is not None
    ]
    completed = [
        float(job["completed_at_epoch"])
        for job in jobs
        if job["completed_at_epoch"] is not None
    ]
    return max(completed) - min(started) if started and completed else None


def _job_summary(
    jobs: list[dict[str, Any]],
    *,
    max_total_jobs: int,
) -> dict[str, Any]:
    status_counts = Counter(str(job["status"]).lower() for job in jobs)
    island_counts = _island_job_counts(jobs, seed_job=None)
    seed_by_island = _island_job_counts(jobs, seed_job=True)
    normal_by_island = _island_job_counts(jobs, seed_job=False)
    terminal = {"succeeded", "failed", "cancelled"}
    return {
        "job_execution_window_seconds": _job_execution_window(jobs),
        "jobs_total": len(jobs),
        "status_counts": dict(sorted(status_counts.items())),
        "all_jobs_terminal": (
            len(jobs) == max_total_jobs
            and all(str(job["status"]).lower() in terminal for job in jobs)
        ),
        "island_job_counts": dict(sorted(island_counts.items())),
        "seed_job_counts": dict(sorted(seed_by_island.items())),
        "normal_job_counts": dict(sorted(normal_by_island.items())),
        "fair_target_allocation": (
            len(set(island_counts.values())) == 1 and len(island_counts) == 2
        ),
    }


def _duplicate_phase_events(
    phase_by_job: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    duplicates = {}
    for job_id, events in phase_by_job.items():
        counts = Counter(str(event["phase"]) for event in events)
        if any(count != 1 for count in counts.values()):
            duplicates[job_id] = dict(counts)
    return duplicates


def _execution_summary(
    *,
    jobs: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    phase_by_job: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pids = sorted({int(event["pid"]) for event in trace})
    workspaces = sorted(
        {
            str(event["working_directory"])
            for event in trace
            if event["phase"] == "coding"
        }
    )
    expected_phases = {"planning": 1, "coding": 1}
    return {
        "worker_pids": pids,
        "distinct_worker_pid_count": len(pids),
        "coding_workspaces": workspaces,
        "distinct_coding_workspace_count": len(workspaces),
        "trace_events": len(trace),
        "duplicate_phase_events": _duplicate_phase_events(phase_by_job),
        "every_job_has_one_planning_and_coding_event": (
            len(phase_by_job) == len(jobs)
            and all(
                Counter(str(event["phase"]) for event in events) == expected_phases
                for events in phase_by_job.values()
            )
        ),
    }


def _usage_summary(usage: list[dict[str, Any]]) -> dict[str, Any]:
    usage_costs = [
        float(event["cost_usd"]) for event in usage if event["cost_usd"] is not None
    ]
    usage_models = sorted({str(event["model"]) for event in usage if event["model"]})
    usage_token_totals = {
        field: sum(max(0, int(event[field] or 0)) for event in usage)
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
    }
    return {
        "usage_event_count": len(usage),
        "usage_models": usage_models,
        "usage_token_totals": usage_token_totals,
        "unpriced_usage_event_count": sum(event["cost_usd"] is None for event in usage),
        "api_cost_usd": sum(usage_costs),
    }


def _configured_summary(args: argparse.Namespace) -> dict[str, Any]:
    backend = getattr(args, "backend", "deterministic")
    return {
        "backend": backend,
        "model": getattr(args, "model", None) if backend == "kilocode" else None,
        "processes": args.processes,
        "max_total_jobs": args.max_total_jobs,
        "max_unfinished_jobs": args.max_unfinished_jobs,
        "migration_interval_jobs": args.migration_interval,
        "planning_delay_seconds": args.planning_delay,
        "coding_delay_seconds": args.coding_delay,
        "islands": ["alpha", "beta"],
        "seed_population_per_island": 2,
    }


def _summarize(
    *,
    rows: dict[str, Any],
    trace: list[dict[str, Any]],
    wall_seconds: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    jobs = rows["jobs"]
    phase_by_job = _trace_events_by_job(trace)
    return {
        "label": args.label,
        "configured": _configured_summary(args),
        "scheduler_worker_wall_seconds": wall_seconds,
        **_job_summary(jobs, max_total_jobs=args.max_total_jobs),
        **_execution_summary(
            jobs=jobs,
            trace=trace,
            phase_by_job=phase_by_job,
        ),
        **_migration_summary(
            jobs=jobs,
            phase_by_job=phase_by_job,
            migration_interval=args.migration_interval,
        ),
        "archive_islands": sorted(
            {str(cell["island_id"]) for cell in rows["archive_cells"]}
        ),
        "island_state_ids": sorted(
            {str(state["island_id"]) for state in rows["island_states"]}
        ),
        **_usage_summary(rows["usage"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--backend",
        choices=("deterministic", "kilocode"),
        default="deterministic",
    )
    parser.add_argument("--model", default=DEFAULT_LIVE_MODEL)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--max-total-jobs", type=int, default=8)
    parser.add_argument("--max-unfinished-jobs", type=int, default=4)
    parser.add_argument("--migration-interval", type=int, default=0)
    parser.add_argument("--planning-delay", type=float, default=0.5)
    parser.add_argument("--coding-delay", type=float, default=1.5)
    parser.add_argument("--planning-timeout", type=int, default=240)
    parser.add_argument("--coding-timeout", type=int, default=360)
    parser.add_argument("--timeout-seconds", type=float, default=360.0)
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument("--redis-url", default=DEFAULT_REDIS_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--report-runs", type=int, default=7)
    return parser


def _run_report(
    *,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> dict[str, Any] | None:
    if args.report_dir is None:
        return None
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_log = args.output.with_suffix(".report.log")
    report_command = [
        sys.executable,
        "examples/evol_circle_packing.py",
        "report",
        "--phase",
        "smoke",
        "--runs",
        str(args.report_runs),
        "--output-dir",
        str(report_dir),
    ]
    report_environment = dict(environment)
    for name in (
        "WORKER_KILOCODE_OPENAI_API_KEY",
        "LORELEY_KILO_OPENAI_API_KEY",
        "KILO_CONFIG_CONTENT",
    ):
        report_environment.pop(name, None)
    with report_log.open("w", encoding="utf-8") as report_output:
        result = subprocess.run(
            report_command,
            cwd=PROJECT_ROOT,
            env=report_environment,
            stdout=report_output,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )
    if result.returncode != 0:
        tail = report_log.read_text(encoding="utf-8", errors="replace")[-5_000:]
        raise RuntimeError(f"Report failed (exit={result.returncode}):\n{tail}")
    artifacts = []
    for path in sorted(item for item in report_dir.rglob("*") if item.is_file()):
        artifacts.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "command": report_command,
        "log": str(report_log),
        "artifacts": artifacts,
    }


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.processes < 1 or args.max_total_jobs < 1:
        raise SystemExit("--processes and --max-total-jobs must be positive.")
    if args.backend == "kilocode" and args.model != DEFAULT_LIVE_MODEL:
        raise SystemExit(
            f"Only the calibrated live model {DEFAULT_LIVE_MODEL!r} has an "
            "experiment pricing rule."
        )
    if args.planning_timeout < 1 or args.coding_timeout < 1:
        raise SystemExit("--planning-timeout and --coding-timeout must be positive.")
    if args.report_runs < 1:
        raise SystemExit("--report-runs must be positive.")
    args.output = args.output.expanduser().resolve()
    args.trace = args.trace.expanduser().resolve()
    if args.report_dir is not None:
        args.report_dir = args.report_dir.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    if args.trace.exists():
        args.trace.unlink()
    return args


def _experiment_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    return {
        "scheduler": [
            sys.executable,
            "examples/evol_circle_packing.py",
            "scheduler",
            "--phase",
            "smoke",
            "--init-db",
            "--yes",
            "--preflight-timeout-seconds",
            "5",
            "--max-total-jobs",
            str(args.max_total_jobs),
        ],
        "worker": [
            sys.executable,
            "examples/evol_circle_packing.py",
            "worker",
            "--phase",
            "smoke",
            "--processes",
            str(args.processes),
            "--preflight-timeout-seconds",
            "5",
        ],
    }


def _monitor_scheduler(
    *,
    scheduler: subprocess.Popen[str],
    worker: subprocess.Popen[str],
    worker_log: Path,
    args: argparse.Namespace,
) -> None:
    deadline = time.monotonic() + args.timeout_seconds
    while scheduler.poll() is None:
        worker_status = worker.poll()
        if worker_status is not None:
            tail = worker_log.read_text(
                encoding="utf-8",
                errors="replace",
            )[-5_000:]
            raise RuntimeError(
                "Worker exited before scheduler completion "
                f"(exit={worker_status}):\n{tail}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Experiment {args.label} exceeded {args.timeout_seconds}s."
            )
        time.sleep(0.2)


def _run_scheduler_and_worker(
    *,
    args: argparse.Namespace,
    environment: dict[str, str],
    commands: dict[str, list[str]],
    logs: dict[str, Path],
) -> float:
    worker: subprocess.Popen[str] | None = None
    started = time.monotonic()
    with logs["scheduler"].open("w", encoding="utf-8") as scheduler_output:
        scheduler = subprocess.Popen(
            commands["scheduler"],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=scheduler_output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_scheduler_start(scheduler, logs["scheduler"])
            with logs["worker"].open("w", encoding="utf-8") as worker_output:
                worker = subprocess.Popen(
                    commands["worker"],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=worker_output,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                _monitor_scheduler(
                    scheduler=scheduler,
                    worker=worker,
                    worker_log=logs["worker"],
                    args=args,
                )
        finally:
            if worker is not None:
                _terminate(worker)
            _terminate(scheduler)
    wall_seconds = time.monotonic() - started
    if scheduler.returncode != 0:
        tail = logs["scheduler"].read_text(
            encoding="utf-8",
            errors="replace",
        )[-5_000:]
        raise RuntimeError(f"Scheduler failed (exit={scheduler.returncode}):\n{tail}")
    return wall_seconds


def _artifact_payload(
    *,
    args: argparse.Namespace,
    commands: dict[str, list[str]],
    logs: dict[str, Path],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "commands": commands,
        "environment": {
            "python": sys.version,
            "repository": str(EXAMPLE_REPO),
            "root_commit": "6dab191",
            "database_host": "localhost:5432",
            "redis_db": 15,
            "backend": args.backend,
            "model": args.model if args.backend == "kilocode" else None,
            "secrets_forwarded_to_worker": args.backend == "kilocode",
            "secrets_recorded": False,
        },
        "summary": evidence["summary"],
        "rows": _serialize(evidence["rows"]),
        "trace": evidence["trace"],
        "logs": {name: str(path) for name, path in logs.items()},
        "report": evidence["report"],
    }


def main() -> int:
    args = _prepare_args(build_parser().parse_args())
    environment = _safe_environment(args)
    commands = _experiment_commands(args)
    logs = {
        "scheduler": args.output.with_suffix(".scheduler.log"),
        "worker": args.output.with_suffix(".worker.log"),
    }
    wall_seconds = _run_scheduler_and_worker(
        args=args,
        environment=environment,
        commands=commands,
        logs=logs,
    )
    trace = _load_trace(args.trace)
    rows = _query_rows(args.database_url)
    summary = _summarize(
        rows=rows,
        trace=trace,
        wall_seconds=wall_seconds,
        args=args,
    )
    report = _run_report(args=args, environment=environment)
    payload = _artifact_payload(
        args=args,
        commands=commands,
        logs=logs,
        evidence={
            "summary": summary,
            "rows": rows,
            "trace": trace,
            "report": report,
        },
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"artifact={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
