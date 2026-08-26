from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tools.method_efficacy_experiment.timing_analysis import (
    TimingAnalysisError,
    extract_timing,
    join_quality_time,
    summarize_quality_time,
)

SCOPE = "formal-test"
TARGET = "zstandard"
ARMS = ("qd", "sequential-champion", "independent-root")


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _write_job(
    *,
    root: Path,
    block: int,
    arm: str,
    ordinal: int,
    scheduled: datetime,
    duration_seconds: int,
    status: str = "SUCCEEDED",
    ingestion_status: str | None = None,
    last_error: str | None = None,
) -> None:
    if arm == "qd":
        path = (
            root
            / "state"
            / "runs"
            / SCOPE
            / TARGET
            / f"block-{block:02d}"
            / arm
            / "campaign"
            / "jobs"
            / f"{ordinal:03d}"
            / "job-result.json"
        )
        job_number = ordinal
        label = f"{SCOPE}-{TARGET}-b{block:02d}-{arm}:job-{ordinal:03d}"
    else:
        path = (
            root
            / "state"
            / "runs"
            / SCOPE
            / TARGET
            / f"block-{block:02d}"
            / arm
            / "candidates"
            / f"{ordinal:04d}"
            / "jobs"
            / "001"
            / "job-result.json"
        )
        job_number = 1
        label = f"{SCOPE}-{TARGET}-b{block:02d}-{arm}-c{ordinal:03d}:job-001"
    started = scheduled + timedelta(seconds=2)
    completed = started + timedelta(seconds=duration_seconds)
    record = {
        "schema_version": 1,
        "job_number": job_number,
        "wave_number": ordinal if arm != "qd" else (ordinal - 1) // 2 + 1,
        "label": label,
        "orchestration_error": None,
        "job": {
            "status": status,
            "ingestion_status": ingestion_status,
            "last_error": last_error,
            "scheduled_at": _iso(scheduled),
            "started_at": _iso(started),
            "completed_at": _iso(completed),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _experiment(tmp_path: Path) -> Path:
    origin = datetime(2026, 8, 1, tzinfo=UTC)
    for block in (1, 2):
        for arm in ARMS:
            for ordinal in range(1, 5):
                if arm == "sequential-champion":
                    offset = (ordinal - 1) * 100
                else:
                    offset = ((ordinal - 1) // 2) * 100
                _write_job(
                    root=tmp_path,
                    block=block,
                    arm=arm,
                    ordinal=ordinal,
                    scheduled=origin + timedelta(hours=block, seconds=offset),
                    duration_seconds=20 + ordinal,
                )
    return tmp_path


def _timing(tmp_path: Path) -> dict[str, object]:
    return extract_timing(
        experiment_root=_experiment(tmp_path),
        scope=SCOPE,
        target=TARGET,
        blocks=2,
        jobs_per_arm=4,
        checkpoints=(2, 4),
    )


def test_extract_timing_validates_and_summarizes_all_jobs(tmp_path: Path) -> None:
    result = _timing(tmp_path)

    assert result["total_jobs"] == 24
    assert result["checks"] == {
        "all_expected_jobs_present": True,
        "all_jobs_terminal": True,
        "timestamps_ordered": True,
        "only_known_orchestration_recovery_markers": True,
    }
    rows = result["block_arms"]
    qd = next(row for row in rows if row["block"] == 1 and row["arm"] == "qd")
    sequential = next(
        row for row in rows if row["block"] == 1 and row["arm"] == "sequential-champion"
    )
    assert qd["search_makespan_seconds"] == pytest.approx(126)
    assert sequential["search_makespan_seconds"] == pytest.approx(326)
    assert [row["elapsed_seconds"] for row in qd["checkpoints"]] == [24, 126]


def test_extract_timing_fails_on_missing_job(tmp_path: Path) -> None:
    root = _experiment(tmp_path)
    missing = (
        root
        / "state"
        / "runs"
        / SCOPE
        / TARGET
        / "block-01"
        / "qd"
        / "campaign"
        / "jobs"
        / "004"
        / "job-result.json"
    )
    missing.unlink()

    with pytest.raises(TimingAnalysisError, match="Cannot read timing record"):
        extract_timing(
            experiment_root=root,
            scope=SCOPE,
            target=TARGET,
            blocks=2,
            jobs_per_arm=4,
            checkpoints=(2, 4),
        )


def test_extract_timing_fails_on_unknown_orchestration_error(tmp_path: Path) -> None:
    root = _experiment(tmp_path)
    path = (
        root
        / "state"
        / "runs"
        / SCOPE
        / TARGET
        / "block-01"
        / "qd"
        / "campaign"
        / "jobs"
        / "001"
        / "job-result.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    record["orchestration_error"] = "unexpected"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(TimingAnalysisError, match="Unexpected orchestration error"):
        extract_timing(
            experiment_root=root,
            scope=SCOPE,
            target=TARGET,
            blocks=2,
            jobs_per_arm=4,
            checkpoints=(2, 4),
        )


def test_extract_timing_sanitizes_known_failure_reason(tmp_path: Path) -> None:
    root = _experiment(tmp_path)
    path = (
        root
        / "state"
        / "runs"
        / SCOPE
        / TARGET
        / "block-01"
        / "qd"
        / "campaign"
        / "jobs"
        / "001"
        / "job-result.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    record["job"].update(
        {
            "status": "FAILED",
            "last_error": (
                "Planning agent failed for job private-id: "
                "Planning agent could not produce a plan after 1 attempt(s)."
            ),
        }
    )
    path.write_text(json.dumps(record), encoding="utf-8")

    result = extract_timing(
        experiment_root=root,
        scope=SCOPE,
        target=TARGET,
        blocks=2,
        jobs_per_arm=4,
        checkpoints=(2, 4),
    )

    qd = next(
        row for row in result["block_arms"] if row["block"] == 1 and row["arm"] == "qd"
    )
    assert qd["jobs"][0]["failure_category"] == "planning_agent"
    assert "private-id" not in json.dumps(result)


def test_join_quality_time_requires_exact_checkpoint_keys(tmp_path: Path) -> None:
    timing = _timing(tmp_path)
    selections = []
    for row in timing["block_arms"]:
        for checkpoint in row["checkpoints"]:
            selections.append(
                {
                    "block": row["block"],
                    "arm": row["arm"],
                    "checkpoint": checkpoint["checkpoint"],
                    "heldout_performance_ratio": 1.01,
                    "conservative_performance_ratio": 1.009,
                    "holdout_passed": True,
                }
            )
    formal = {"target": "Zstandard", "holdout_selections": selections}

    joined = join_quality_time(timing=timing, formal_records=formal)

    assert joined["checks"]["record_count"] == 12
    assert joined["target"] == "Zstandard"
    assert all(
        row["heldout_performance_ratio"] == 1.01
        for row in joined["quality_time_records"]
    )
    formal["holdout_selections"] = selections[:-1]
    with pytest.raises(TimingAnalysisError, match="Timing/quality checkpoint mismatch"):
        join_quality_time(timing=timing, formal_records=formal)


def test_summarize_quality_time_keeps_posthoc_boundary(tmp_path: Path) -> None:
    timing = _timing(tmp_path)
    selections = []
    for row in timing["block_arms"]:
        for checkpoint in row["checkpoints"]:
            gain = 0.01 if row["arm"] == "qd" else 0.005
            selections.append(
                {
                    "block": row["block"],
                    "arm": row["arm"],
                    "checkpoint": checkpoint["checkpoint"],
                    "heldout_performance_ratio": 1 + gain,
                    "conservative_performance_ratio": 1 + gain,
                    "holdout_passed": True,
                }
            )
    joined = join_quality_time(
        timing=timing,
        formal_records={"target": "Zstandard", "holdout_selections": selections},
    )

    summary = summarize_quality_time(joined)

    assert summary["inferential_status"].startswith("post-hoc exploratory")
    assert summary["checks"]["expected_joined_record_count"] == 12
    assert (
        summary["posthoc_qd_completion_deadline_comparison"]["qd_higher_quality_blocks"]
        == 2
    )
    assert summary["final_sequential_over_qd_time_ratio"]["geometric_mean"] > 2
    parallelism = {
        row["arm"]: row for row in summary["failure_and_parallelism"]["arms"]
    }
    assert parallelism["qd"]["native_width"] == 4
    assert parallelism["sequential-champion"]["native_width"] == 1
    assert (
        parallelism["qd"]["effective_parallelism_by_block"]["median"]
        > parallelism["sequential-champion"]["effective_parallelism_by_block"]["median"]
    )
