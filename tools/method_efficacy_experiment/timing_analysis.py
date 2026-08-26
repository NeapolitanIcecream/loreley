"""Extract and join sanitized timing evidence for the formal policy study."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


class TimingAnalysisError(RuntimeError):
    """Raised when timing evidence is missing, inconsistent, or ambiguous."""


DEFAULT_ARMS = ("qd", "sequential-champion", "independent-root")
DEFAULT_CHECKPOINTS = (8, 16, 24, 32, 40, 48)
ALLOWED_ORCHESTRATION_ERRORS = {None, "recovered_after_controller_restart"}
NATIVE_WIDTHS = {"qd": 4, "sequential-champion": 1, "independent-root": 4}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TimingAnalysisError(f"Cannot read timing record: {path}") from exc
    if not isinstance(value, Mapping):
        raise TimingAnalysisError(f"Timing record is not an object: {path}")
    return value


def _timestamp(value: object, *, field: str, path: Path) -> datetime:
    if not isinstance(value, str) or not value:
        raise TimingAnalysisError(f"Missing {field} in {path}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TimingAnalysisError(f"Invalid {field} in {path}: {value}") from exc
    if parsed.tzinfo is None:
        raise TimingAnalysisError(f"Naive {field} in {path}: {value}")
    return parsed


def _job_path(*, arm_root: Path, arm: str, ordinal: int, jobs_per_arm: int) -> Path:
    width = max(3, len(str(jobs_per_arm)))
    if arm == "qd":
        return (
            arm_root / "campaign" / "jobs" / f"{ordinal:0{width}d}" / "job-result.json"
        )
    return (
        arm_root / "candidates" / f"{ordinal:04d}" / "jobs" / "001" / "job-result.json"
    )


def _failure_category(
    *, status: str, ingestion_status: object, last_error: object, path: Path
) -> str | None:
    error = last_error if isinstance(last_error, str) else ""
    if not error:
        if status == "FAILED":
            raise TimingAnalysisError(f"Failed job has no failure reason: {path}")
        return None
    prefixes = (
        ("Planning agent failed for job ", "planning_agent"),
        ("Coding agent failed for job ", "coding_agent"),
        ("Zstandard candidate validation failed.", "evaluator_validation"),
        (
            "Zstandard measurement precision was insufficient for archive admission.",
            "evaluator_precision",
        ),
        ("Candidate exceeded the compressed-size limit.", "evaluator_size_limit"),
    )
    for prefix, category in prefixes:
        if error.startswith(prefix):
            return category
    if (
        status == "SUCCEEDED"
        and ingestion_status == "failed"
        and error == "Request timed out."
    ):
        return "ingestion_timeout"
    raise TimingAnalysisError(f"Unknown failure reason in {path}")


def _extract_job(
    *,
    path: Path,
    scope: str,
    target: str,
    block: int,
    arm: str,
    ordinal: int,
) -> dict[str, Any]:
    record = _read_json(path)
    job = record.get("job")
    if not isinstance(job, Mapping):
        raise TimingAnalysisError(f"Missing job object in {path}")
    expected_label = f"{scope}-{target}-b{block:02d}-{arm}"
    label = record.get("label")
    if not isinstance(label, str) or not label.startswith(expected_label):
        raise TimingAnalysisError(
            f"Job label does not match {expected_label!r}: {path}"
        )
    expected_job_number = ordinal if arm == "qd" else 1
    if record.get("job_number") != expected_job_number:
        raise TimingAnalysisError(f"Job number drifted in {path}")
    scheduled = _timestamp(job.get("scheduled_at"), field="scheduled_at", path=path)
    started = _timestamp(job.get("started_at"), field="started_at", path=path)
    completed = _timestamp(job.get("completed_at"), field="completed_at", path=path)
    if not scheduled <= started <= completed:
        raise TimingAnalysisError(f"Job timestamps are out of order: {path}")
    status = job.get("status")
    if status not in {"SUCCEEDED", "FAILED"}:
        raise TimingAnalysisError(f"Job is not terminal in {path}: {status}")
    ingestion_status = job.get("ingestion_status")
    if ingestion_status not in {None, "succeeded", "skipped", "failed"}:
        raise TimingAnalysisError(f"Unknown ingestion status in {path}")
    failure_category = _failure_category(
        status=status,
        ingestion_status=ingestion_status,
        last_error=job.get("last_error"),
        path=path,
    )
    orchestration_error = record.get("orchestration_error")
    if orchestration_error not in ALLOWED_ORCHESTRATION_ERRORS:
        raise TimingAnalysisError(
            f"Unexpected orchestration error in {path}: {orchestration_error}"
        )
    return {
        "ordinal": ordinal,
        "wave": record.get("wave_number"),
        "status": status,
        "scheduled_at": scheduled.isoformat(),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "queue_seconds": (started - scheduled).total_seconds(),
        "active_seconds": (completed - started).total_seconds(),
        "ingestion_status": ingestion_status,
        "failure_category": failure_category,
        "orchestration_recovery": orchestration_error,
    }


def _checkpoint_rows(
    jobs: Sequence[Mapping[str, Any]], checkpoints: Sequence[int]
) -> list[dict[str, Any]]:
    origin = min(datetime.fromisoformat(str(row["scheduled_at"])) for row in jobs)
    result: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        eligible = [row for row in jobs if int(row["ordinal"]) <= checkpoint]
        if len(eligible) != checkpoint:
            raise TimingAnalysisError(
                f"Checkpoint {checkpoint} has {len(eligible)} timing rows"
            )
        completed = max(
            datetime.fromisoformat(str(row["completed_at"])) for row in eligible
        )
        result.append(
            {
                "checkpoint": checkpoint,
                "completed_at": completed.isoformat(),
                "elapsed_seconds": (completed - origin).total_seconds(),
            }
        )
    return result


def extract_timing(
    *,
    experiment_root: Path,
    scope: str,
    target: str,
    blocks: int,
    jobs_per_arm: int,
    checkpoints: Sequence[int],
    arms: Sequence[str] = DEFAULT_ARMS,
) -> dict[str, Any]:
    """Extract sanitized job timing from a completed formal experiment."""
    if not checkpoints or sorted(set(checkpoints)) != list(checkpoints):
        raise TimingAnalysisError("Checkpoints must be unique and increasing")
    if checkpoints[-1] != jobs_per_arm:
        raise TimingAnalysisError("Final checkpoint must equal jobs per arm")
    run_root = experiment_root.resolve(strict=True) / "state" / "runs" / scope / target
    block_rows: list[dict[str, Any]] = []
    total_jobs = 0
    for block in range(1, blocks + 1):
        for arm in arms:
            arm_root = run_root / f"block-{block:02d}" / arm
            if not arm_root.is_dir():
                raise TimingAnalysisError(f"Missing arm directory: {arm_root}")
            jobs = [
                _extract_job(
                    path=_job_path(
                        arm_root=arm_root,
                        arm=arm,
                        ordinal=ordinal,
                        jobs_per_arm=jobs_per_arm,
                    ),
                    scope=scope,
                    target=target,
                    block=block,
                    arm=arm,
                    ordinal=ordinal,
                )
                for ordinal in range(1, jobs_per_arm + 1)
            ]
            total_jobs += len(jobs)
            origin = min(
                datetime.fromisoformat(str(row["scheduled_at"])) for row in jobs
            )
            completed = max(
                datetime.fromisoformat(str(row["completed_at"])) for row in jobs
            )
            block_rows.append(
                {
                    "block": block,
                    "arm": arm,
                    "time_origin": "earliest_candidate_scheduled_at",
                    "search_makespan_seconds": (completed - origin).total_seconds(),
                    "queue_seconds_sum": sum(
                        float(row["queue_seconds"]) for row in jobs
                    ),
                    "active_seconds_sum": sum(
                        float(row["active_seconds"]) for row in jobs
                    ),
                    "successful_jobs": sum(
                        row["status"] == "SUCCEEDED" for row in jobs
                    ),
                    "unsuccessful_jobs": sum(
                        row["status"] != "SUCCEEDED" for row in jobs
                    ),
                    "recovered_orchestration_jobs": sum(
                        row["orchestration_recovery"] is not None for row in jobs
                    ),
                    "checkpoints": _checkpoint_rows(jobs, checkpoints),
                    "jobs": jobs,
                }
            )
    expected_jobs = blocks * len(arms) * jobs_per_arm
    if total_jobs != expected_jobs:
        raise TimingAnalysisError(
            f"Expected {expected_jobs} timing rows, extracted {total_jobs}"
        )
    return {
        "schema_version": 2,
        "kind": "sanitized_formal_search_timing",
        "scope": scope,
        "target": target,
        "blocks": blocks,
        "arms": list(arms),
        "jobs_per_arm": jobs_per_arm,
        "checkpoints": list(checkpoints),
        "time_scope": "candidate search only; validation and holdout excluded",
        "total_jobs": total_jobs,
        "block_arms": block_rows,
        "checks": {
            "all_expected_jobs_present": True,
            "all_jobs_terminal": True,
            "timestamps_ordered": True,
            "only_known_orchestration_recovery_markers": True,
        },
    }


def join_quality_time(
    *, timing: Mapping[str, Any], formal_records: Mapping[str, Any]
) -> dict[str, Any]:
    """Join checkpoint elapsed time to validation-selected holdout records."""
    if timing.get("kind") != "sanitized_formal_search_timing":
        raise TimingAnalysisError("Input is not a sanitized timing record")
    formal_target = str(formal_records.get("target") or "")
    timing_target = str(timing.get("target") or "")
    if formal_target.casefold() != timing_target.casefold():
        raise TimingAnalysisError("Timing and endpoint targets differ")
    timing_index: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    for row in timing.get("block_arms", ()):
        if not isinstance(row, Mapping):
            raise TimingAnalysisError("Invalid block-arm timing row")
        for checkpoint in row.get("checkpoints", ()):
            if not isinstance(checkpoint, Mapping):
                raise TimingAnalysisError("Invalid checkpoint timing row")
            key = (int(row["block"]), str(row["arm"]), int(checkpoint["checkpoint"]))
            if key in timing_index:
                raise TimingAnalysisError(f"Duplicate timing checkpoint: {key}")
            timing_index[key] = checkpoint
    quality_index: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    for row in formal_records.get("holdout_selections", ()):
        if not isinstance(row, Mapping):
            raise TimingAnalysisError("Invalid holdout selection row")
        key = (int(row["block"]), str(row["arm"]), int(row["checkpoint"]))
        if key in quality_index:
            raise TimingAnalysisError(f"Duplicate holdout selection: {key}")
        quality_index[key] = row
    if timing_index.keys() != quality_index.keys():
        missing_time = sorted(quality_index.keys() - timing_index.keys())
        missing_quality = sorted(timing_index.keys() - quality_index.keys())
        raise TimingAnalysisError(
            f"Timing/quality checkpoint mismatch: time={missing_time}, quality={missing_quality}"
        )
    rows = []
    for key in sorted(timing_index):
        checkpoint = timing_index[key]
        quality = quality_index[key]
        rows.append(
            {
                "block": key[0],
                "arm": key[1],
                "checkpoint": key[2],
                "elapsed_seconds": checkpoint["elapsed_seconds"],
                "completed_at": checkpoint["completed_at"],
                "heldout_performance_ratio": quality["heldout_performance_ratio"],
                "conservative_performance_ratio": quality[
                    "conservative_performance_ratio"
                ],
                "holdout_passed": quality["holdout_passed"],
            }
        )
    return {
        "schema_version": 1,
        "kind": "posthoc_formal_quality_time",
        "scope": timing["scope"],
        "target": formal_target,
        "selection_rule": "validation-selected winner measured on holdout",
        "inferential_status": (
            "post-hoc exploratory analysis; not evidence of quality equivalence "
            "or a confirmatory time advantage"
        ),
        "timing": timing,
        "quality_time_records": rows,
        "checks": {
            "all_timing_checkpoints_have_quality": True,
            "all_quality_checkpoints_have_timing": True,
            "record_count": len(rows),
        },
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise TimingAnalysisError("Cannot summarize an empty value set")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise TimingAnalysisError("Cannot calculate a quantile of an empty value set")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _failure_and_parallelism_summary(
    *, timing: Mapping[str, Any], arms: Sequence[str]
) -> dict[str, Any]:
    raw_block_arms = timing.get("block_arms")
    if not isinstance(raw_block_arms, Sequence):
        raise TimingAnalysisError("Timing record has no block-arm rows")
    result = []
    for arm in arms:
        if arm not in NATIVE_WIDTHS:
            raise TimingAnalysisError(f"Native width is unknown for arm: {arm}")
        selected = [
            row
            for row in raw_block_arms
            if isinstance(row, Mapping) and str(row.get("arm")) == arm
        ]
        jobs = [
            job
            for row in selected
            for job in row.get("jobs", ())
            if isinstance(job, Mapping)
        ]
        active = [float(job["active_seconds"]) for job in jobs]
        failed = [job for job in jobs if job.get("status") != "SUCCEEDED"]
        succeeded = [job for job in jobs if job.get("status") == "SUCCEEDED"]
        categories: dict[str, int] = {}
        for job in jobs:
            category = job.get("failure_category")
            if category is not None:
                categories[str(category)] = categories.get(str(category), 0) + 1
        block_parallelism = []
        for row in selected:
            makespan = float(row["search_makespan_seconds"])
            active_sum = float(row["active_seconds_sum"])
            block_parallelism.append(active_sum / makespan)
        result.append(
            {
                "arm": arm,
                "native_width": NATIVE_WIDTHS[arm],
                "jobs": len(jobs),
                "terminal_failed_jobs": len(failed),
                "nonterminal_ingestion_failures": sum(
                    job.get("status") == "SUCCEEDED"
                    and job.get("failure_category") is not None
                    for job in jobs
                ),
                "failure_categories": dict(sorted(categories.items())),
                "active_seconds": {
                    "all": {
                        **_summary(active),
                        "p95": _quantile(active, 0.95),
                    },
                    "succeeded": {
                        **_summary([float(job["active_seconds"]) for job in succeeded]),
                        "p95": _quantile(
                            [float(job["active_seconds"]) for job in succeeded], 0.95
                        ),
                    },
                    "failed": (
                        {
                            **_summary(
                                [float(job["active_seconds"]) for job in failed]
                            ),
                            "p95": _quantile(
                                [float(job["active_seconds"]) for job in failed], 0.95
                            ),
                        }
                        if failed
                        else None
                    ),
                },
                "failed_active_seconds_fraction": (
                    sum(float(job["active_seconds"]) for job in failed) / sum(active)
                ),
                "effective_parallelism_by_block": {
                    **_summary(block_parallelism),
                    "values": block_parallelism,
                },
                "median_parallel_efficiency": (
                    statistics.median(block_parallelism) / NATIVE_WIDTHS[arm]
                ),
            }
        )
    return {
        "definition": (
            "effective parallelism is summed candidate active time divided by "
            "search makespan within each block; it is descriptive, not provider "
            "service time"
        ),
        "arms": result,
    }


def summarize_quality_time(joined: Mapping[str, Any]) -> dict[str, Any]:
    """Build descriptive timing summaries without upgrading inferential status."""
    if joined.get("kind") != "posthoc_formal_quality_time":
        raise TimingAnalysisError("Input is not a joined quality-time record")
    raw_rows = joined.get("quality_time_records")
    if not isinstance(raw_rows, Sequence):
        raise TimingAnalysisError("Joined record has no quality-time rows")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    timing = joined.get("timing")
    if not isinstance(timing, Mapping):
        raise TimingAnalysisError("Joined record has no timing object")
    checkpoints = [int(value) for value in timing.get("checkpoints", ())]
    arms = [str(value) for value in timing.get("arms", ())]
    blocks = int(timing.get("blocks") or 0)
    expected = len(checkpoints) * len(arms) * blocks
    if len(rows) != expected:
        raise TimingAnalysisError(f"Expected {expected} joined rows, found {len(rows)}")
    index: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["block"]), str(row["arm"]), int(row["checkpoint"]))
        if key in index:
            raise TimingAnalysisError(f"Duplicate summary input: {key}")
        index[key] = row
    arm_checkpoints = []
    for checkpoint in checkpoints:
        for arm in arms:
            selected = [
                index[(block, arm, checkpoint)] for block in range(1, blocks + 1)
            ]
            elapsed_hours = [float(row["elapsed_seconds"]) / 3600 for row in selected]
            quality = [float(row["heldout_performance_ratio"]) for row in selected]
            arm_checkpoints.append(
                {
                    "arm": arm,
                    "checkpoint": checkpoint,
                    "elapsed_hours": _summary(elapsed_hours),
                    "heldout_performance_ratio": _summary(quality),
                    "useful_blocks_at_1_005": sum(value >= 1.005 for value in quality),
                }
            )
    final_checkpoint = checkpoints[-1]
    time_ratios = []
    qd_deadline_rows = []
    for block in range(1, blocks + 1):
        qd_final = index[(block, "qd", final_checkpoint)]
        sequential_final = index[(block, "sequential-champion", final_checkpoint)]
        ratio = float(sequential_final["elapsed_seconds"]) / float(
            qd_final["elapsed_seconds"]
        )
        time_ratios.append(ratio)
        eligible = [
            checkpoint
            for checkpoint in checkpoints
            if float(
                index[(block, "sequential-champion", checkpoint)]["elapsed_seconds"]
            )
            <= float(qd_final["elapsed_seconds"])
        ]
        if not eligible:
            raise TimingAnalysisError(
                f"Sequential has no checkpoint by QD completion in block {block}"
            )
        sequential_checkpoint = max(eligible)
        sequential_at_deadline = index[
            (block, "sequential-champion", sequential_checkpoint)
        ]
        qd_quality = float(qd_final["heldout_performance_ratio"])
        sequential_quality = float(sequential_at_deadline["heldout_performance_ratio"])
        qd_deadline_rows.append(
            {
                "block": block,
                "qd_deadline_hours": float(qd_final["elapsed_seconds"]) / 3600,
                "qd_checkpoint": final_checkpoint,
                "sequential_checkpoint_reached": sequential_checkpoint,
                "qd_heldout_performance_ratio": qd_quality,
                "sequential_heldout_performance_ratio": sequential_quality,
                "qd_over_sequential_percent": (qd_quality / sequential_quality - 1)
                * 100,
            }
        )
    return {
        "schema_version": 1,
        "kind": "posthoc_formal_quality_time_summary",
        "scope": joined["scope"],
        "target": joined["target"],
        "inferential_status": joined["inferential_status"],
        "arm_checkpoints": arm_checkpoints,
        "final_sequential_over_qd_time_ratio": {
            **_summary(time_ratios),
            "geometric_mean": math.exp(
                statistics.fmean(math.log(v) for v in time_ratios)
            ),
            "pairs": time_ratios,
        },
        "posthoc_qd_completion_deadline_comparison": {
            "deadline_definition": (
                "each block's QD 48-job search completion; policy-dependent and post-hoc"
            ),
            "rows": qd_deadline_rows,
            "qd_higher_quality_blocks": sum(
                row["qd_over_sequential_percent"] > 0 for row in qd_deadline_rows
            ),
        },
        "failure_and_parallelism": _failure_and_parallelism_summary(
            timing=timing, arms=arms
        ),
        "checks": {
            "expected_joined_record_count": expected,
            "all_blocks_have_qd_and_sequential_final_timing": True,
            "all_blocks_have_a_sequential_checkpoint_by_qd_completion": True,
        },
    }


def _load_stream_or_path(value: str) -> Mapping[str, Any]:
    if value == "-":
        raw = json.load(sys.stdin)
        if not isinstance(raw, Mapping):
            raise TimingAnalysisError("Standard input is not a JSON object")
        return raw
    return _read_json(Path(value))


def _write_json(value: Mapping[str, Any], output: str) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(text)
        return
    Path(output).write_text(text, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="extract sanitized timing")
    extract.add_argument("--experiment-root", type=Path, required=True)
    extract.add_argument("--scope", required=True)
    extract.add_argument("--target", default="zstandard")
    extract.add_argument("--blocks", type=int, default=7)
    extract.add_argument("--jobs-per-arm", type=int, default=48)
    extract.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
    )
    extract.add_argument("--output", default="-")
    join = commands.add_parser("join", help="join timing to holdout selections")
    join.add_argument("--timing", required=True)
    join.add_argument("--formal-records", required=True)
    join.add_argument("--output", default="-")
    summarize = commands.add_parser("summarize", help="summarize joined records")
    summarize.add_argument("--joined", required=True)
    summarize.add_argument("--output", default="-")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "extract":
        value = extract_timing(
            experiment_root=args.experiment_root,
            scope=args.scope,
            target=args.target,
            blocks=args.blocks,
            jobs_per_arm=args.jobs_per_arm,
            checkpoints=args.checkpoints,
        )
    elif args.command == "join":
        value = join_quality_time(
            timing=_load_stream_or_path(args.timing),
            formal_records=_read_json(Path(args.formal_records)),
        )
    else:
        value = summarize_quality_time(
            _load_stream_or_path(args.joined),
        )
    _write_json(value, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
