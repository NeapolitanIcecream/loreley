#!/usr/bin/env python3
"""Replay the technical report's public timing records and headline metrics."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.method_efficacy_experiment.timing_analysis import (
    ALLOWED_ORCHESTRATION_ERRORS,
    DEFAULT_ARMS,
    DEFAULT_CHECKPOINTS,
    _checkpoint_rows,
    join_quality_time,
    summarize_quality_time,
)

EVIDENCE = Path(__file__).resolve().parent
FAILURE_CATEGORIES = {
    None,
    "planning_agent",
    "coding_agent",
    "evaluator_validation",
    "evaluator_precision",
    "evaluator_size_limit",
    "ingestion_timeout",
}


def close(observed: float, expected: float) -> None:
    assert math.isfinite(observed), observed
    assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-8), (
        observed,
        expected,
    )


def compare_summary(observed: object, expected: object) -> None:
    """Check every summary field, allowing only floating-point roundoff."""
    if isinstance(expected, dict):
        assert isinstance(observed, dict) and observed.keys() == expected.keys()
        for key in expected:
            compare_summary(observed[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(observed, list) and len(observed) == len(expected)
        for actual, reference in zip(observed, expected):
            compare_summary(actual, reference)
    elif isinstance(expected, float):
        # Python's summation algorithm differs between supported releases.
        assert isinstance(observed, (int, float))
        close(observed, expected)
    else:
        assert observed == expected, (observed, expected)


def validate_timing(timing: dict) -> None:
    assert timing["schema_version"] == 2
    assert timing["kind"] == "sanitized_formal_search_timing"
    assert timing["blocks"] == 7
    assert timing["arms"] == list(DEFAULT_ARMS)
    assert timing["jobs_per_arm"] == 48
    assert timing["checkpoints"] == list(DEFAULT_CHECKPOINTS)
    assert timing["total_jobs"] == 1008
    assert timing["time_scope"] == (
        "candidate search only; validation and holdout excluded"
    )
    block_arms = timing["block_arms"]
    expected_pairs = {(block, arm) for block in range(1, 8) for arm in DEFAULT_ARMS}
    assert len(block_arms) == 21
    assert {(row["block"], row["arm"]) for row in block_arms} == expected_pairs

    for row in block_arms:
        jobs = row["jobs"]
        assert row["time_origin"] == "earliest_candidate_scheduled_at"
        assert [job["ordinal"] for job in jobs] == list(range(1, 49))
        for job in jobs:
            scheduled, started, completed = (
                datetime.fromisoformat(job[field])
                for field in ("scheduled_at", "started_at", "completed_at")
            )
            assert all(
                value.tzinfo is not None for value in (scheduled, started, completed)
            )
            assert scheduled <= started <= completed
            close(job["queue_seconds"], (started - scheduled).total_seconds())
            close(job["active_seconds"], (completed - started).total_seconds())
            assert job["status"] in {"SUCCEEDED", "FAILED"}
            assert job["ingestion_status"] in {None, "succeeded", "skipped", "failed"}
            assert job["failure_category"] in FAILURE_CATEGORIES
            if job["status"] == "FAILED":
                assert job["failure_category"] is not None
            assert job["orchestration_recovery"] in ALLOWED_ORCHESTRATION_ERRORS

        checkpoints = _checkpoint_rows(jobs, DEFAULT_CHECKPOINTS)
        assert checkpoints == row["checkpoints"]
        close(row["search_makespan_seconds"], checkpoints[-1]["elapsed_seconds"])
        close(row["queue_seconds_sum"], sum(job["queue_seconds"] for job in jobs))
        close(row["active_seconds_sum"], sum(job["active_seconds"] for job in jobs))
        assert row["successful_jobs"] == sum(
            job["status"] == "SUCCEEDED" for job in jobs
        )
        assert row["unsuccessful_jobs"] == sum(
            job["status"] == "FAILED" for job in jobs
        )
        assert row["recovered_orchestration_jobs"] == sum(
            job["orchestration_recovery"] is not None for job in jobs
        )

    assert sum(len(row["jobs"]) for row in block_arms) == 1008
    assert timing["checks"] == {
        "all_expected_jobs_present": True,
        "all_jobs_terminal": True,
        "timestamps_ordered": True,
        "only_known_orchestration_recovery_markers": True,
    }


def validate_headlines(joined: dict, summary: dict) -> None:
    rows = joined["quality_time_records"]
    assert len(rows) == 126
    index = {(row["block"], row["arm"], row["checkpoint"]): row for row in rows}
    assert len(index) == 126
    threshold_times = {}
    for arm in DEFAULT_ARMS:
        times = {}
        for block in range(1, 8):
            for checkpoint in DEFAULT_CHECKPOINTS:
                row = index[(block, arm, checkpoint)]
                if row["heldout_performance_ratio"] >= 1.005:
                    times[block] = row["elapsed_seconds"] / 3600.0
                    break
        threshold_times[arm] = times
    assert {arm: len(times) for arm, times in threshold_times.items()} == {
        "qd": 6,
        "sequential-champion": 7,
        "independent-root": 5,
    }
    qd_times = threshold_times["qd"]
    sequential_times = threshold_times["sequential-champion"]
    assert f"{statistics.median(qd_times.values()):.2f}" == "0.78"
    assert f"{statistics.median(sequential_times.values()):.2f}" == "5.23"
    common_blocks = sorted(qd_times.keys() & sequential_times.keys())
    assert len(common_blocks) == 6
    assert all(qd_times[block] < sequential_times[block] for block in common_blocks)
    threshold_ratio = statistics.geometric_mean(
        sequential_times[block] / qd_times[block] for block in common_blocks
    )
    assert f"{threshold_ratio:.2f}" == "6.76"
    assert (
        f"{summary['final_sequential_over_qd_time_ratio']['geometric_mean']:.2f}"
        == "2.78"
    )

    deadline = summary["posthoc_qd_completion_deadline_comparison"]
    assert deadline["qd_higher_quality_blocks"] == 7
    assert Counter(
        row["sequential_checkpoint_reached"] for row in deadline["rows"]
    ) == {8: 4, 16: 2, 24: 1}
    margin = statistics.median(
        row["qd_over_sequential_percent"] for row in deadline["rows"]
    )
    assert f"{margin:.2f}" == "0.44"
    final = {
        row["arm"]: row for row in summary["arm_checkpoints"] if row["checkpoint"] == 48
    }
    for arm, hours, gain, useful in (
        ("qd", "3.75", "0.82", 6),
        ("sequential-champion", "11.65", "0.96", 7),
        ("independent-root", "3.66", "0.50", 2),
    ):
        assert f"{final[arm]['elapsed_hours']['median']:.2f}" == hours
        assert (
            f"{100 * (final[arm]['heldout_performance_ratio']['mean'] - 1):.2f}" == gain
        )
        assert final[arm]["useful_blocks_at_1_005"] == useful
    changes = [
        index[(block, "qd", 48)]["heldout_performance_ratio"]
        - index[(block, "qd", 24)]["heldout_performance_ratio"]
        for block in range(1, 8)
    ]
    assert sum(value > 0 for value in changes) == 5
    assert sum(value == 0 for value in changes) == 2


def main() -> None:
    joined = json.loads(
        (EVIDENCE / "zstd_quality_time.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (EVIDENCE / "zstd_quality_time_summary.json").read_text(encoding="utf-8")
    )
    formal = json.loads(
        (ROOT / "paper/evidence/zstd_formal_records.json").read_text(encoding="utf-8")
    )
    validate_timing(joined["timing"])
    assert join_quality_time(timing=joined["timing"], formal_records=formal) == joined
    compare_summary(summarize_quality_time(joined), summary)
    validate_headlines(joined, summary)
    print(
        "Evolution dynamics: 1,008 timestamp records, 126 checkpoint makespans and "
        "public holdout joins, full timing summary, and report headline metrics verified."
    )


if __name__ == "__main__":
    main()
