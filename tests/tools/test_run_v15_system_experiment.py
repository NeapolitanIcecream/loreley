from __future__ import annotations

from argparse import Namespace
from decimal import Decimal
from pathlib import Path

import pytest

from tools.run_v15_system_experiment import _safe_environment, _summarize


def test_summary_requires_cross_island_donor_in_prompt() -> None:
    jobs = [
        {
            "id": "job-1",
            "status": "succeeded",
            "island_id": "alpha",
            "inspiration_commit_hashes": ["b" * 40],
            "migration_source_island_id": "beta",
            "migration_commit_hash": "b" * 40,
            "is_seed_job": False,
            "started_at_epoch": 1.0,
            "completed_at_epoch": 3.0,
        },
        {
            "id": "job-2",
            "status": "succeeded",
            "island_id": "beta",
            "inspiration_commit_hashes": [],
            "migration_source_island_id": None,
            "migration_commit_hash": None,
            "is_seed_job": False,
            "started_at_epoch": 1.5,
            "completed_at_epoch": 3.5,
        },
    ]
    trace = [
        {
            "job_id": job_id,
            "phase": phase,
            "pid": 10 + index,
            "working_directory": f"/tmp/{job_id}",
            "prompt_commit_hashes": ["b" * 40] if job_id == "job-1" else [],
        }
        for index, (job_id, phase) in enumerate(
            [
                ("job-1", "planning"),
                ("job-1", "coding"),
                ("job-2", "planning"),
                ("job-2", "coding"),
            ]
        )
    ]
    args = Namespace(
        label="test",
        processes=2,
        max_total_jobs=2,
        max_unfinished_jobs=2,
        migration_interval=2,
        planning_delay=0.0,
        coding_delay=0.0,
        backend="deterministic",
        model="gpt-5.4-mini",
    )

    result = _summarize(
        rows={
            "jobs": jobs,
            "archive_cells": [
                {"island_id": "alpha"},
                {"island_id": "beta"},
            ],
            "island_states": [
                {"island_id": "alpha"},
                {"island_id": "beta"},
            ],
            "usage": [],
        },
        trace=trace,
        wall_seconds=4.0,
        args=args,
    )

    assert result["migration_provenance_valid"] is True
    assert result["fair_target_allocation"] is True
    assert result["every_job_has_one_planning_and_coding_event"] is True


def test_summary_accounts_priced_usage() -> None:
    args = Namespace(
        label="live",
        processes=1,
        max_total_jobs=1,
        max_unfinished_jobs=1,
        migration_interval=0,
        planning_delay=0.0,
        coding_delay=0.0,
        backend="kilocode",
        model="gpt-5.4-mini",
    )
    jobs = [
        {
            "id": "job-1",
            "status": "succeeded",
            "island_id": "alpha",
            "inspiration_commit_hashes": [],
            "migration_source_island_id": None,
            "migration_commit_hash": None,
            "is_seed_job": True,
            "started_at_epoch": 1.0,
            "completed_at_epoch": 3.0,
        }
    ]
    usage = [
        {
            "model": "gpt-5.4-mini",
            "input_tokens": 100,
            "cached_input_tokens": 50,
            "output_tokens": 20,
            "reasoning_output_tokens": 0,
            "total_tokens": 170,
            "cost_usd": Decimal("0.000243"),
        }
    ]

    result = _summarize(
        rows={
            "jobs": jobs,
            "archive_cells": [{"island_id": "alpha"}],
            "island_states": [{"island_id": "alpha"}],
            "usage": usage,
        },
        trace=[],
        wall_seconds=4.0,
        args=args,
    )

    assert result["configured"]["backend"] == "kilocode"
    assert result["usage_models"] == ["gpt-5.4-mini"]
    assert result["usage_token_totals"]["total_tokens"] == 170
    assert result["unpriced_usage_event_count"] == 0
    assert result["api_cost_usd"] == pytest.approx(0.000243)


def test_live_environment_forwards_only_worker_scoped_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example/v1")
    args = Namespace(
        label="live",
        output=tmp_path / "result.json",
        trace=tmp_path / "trace.jsonl",
        database_url="postgresql+psycopg://example",
        redis_url="redis://example/15",
        max_unfinished_jobs=2,
        max_total_jobs=4,
        migration_interval=0,
        planning_delay=0.0,
        coding_delay=0.0,
        planning_timeout=30,
        coding_timeout=60,
        backend="kilocode",
        model="gpt-5.4-mini",
    )

    environment = _safe_environment(args)

    assert "LLM_API_KEY" not in environment
    assert "LLM_BASE_URL" not in environment
    assert environment["WORKER_KILOCODE_OPENAI_API_KEY"] == "test-secret"
    assert environment["WORKER_KILOCODE_OPENAI_BASE_URL"] == "https://proxy.example/v1"
    assert "test-secret" not in environment["LLM_USAGE_PRICING_JSON"]
