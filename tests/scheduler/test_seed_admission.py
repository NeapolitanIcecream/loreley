from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from rich.console import Console

from loreley.core.map_elites.objectives import ObjectiveContract, ObjectiveSpec
from loreley.core.seed_portfolio import (
    EXPLORATORY_STEPPING_STONE_LANE,
    IMMEDIATE_EVIDENCE_LANE,
    SeedAdmissionDecision,
    classify_seed_admission,
)
from loreley.scheduler.ingestion import JobSnapshot, MapElitesIngestion


def _metric(name: str, value: float) -> dict[str, object]:
    return {"name": name, "value": value, "higher_is_better": True}


@pytest.mark.parametrize(
    ("candidate_values", "expected_lane"),
    (
        ((110.0, 98.0), IMMEDIATE_EVIDENCE_LANE),
        ((98.0, 110.0), IMMEDIATE_EVIDENCE_LANE),
        ((100.0, 98.0), EXPLORATORY_STEPPING_STONE_LANE),
    ),
)
def test_evaluator_valid_portfolio_seed_tradeoffs_reach_archive_manager(
    monkeypatch,
    settings,
    tmp_path: Path,
    candidate_values: tuple[float, float],
    expected_lane: str,
) -> None:
    contract = ObjectiveContract(
        (
            ObjectiveSpec(name="primary", direction="max"),
            ObjectiveSpec(name="secondary", direction="max"),
        )
    )
    decision = classify_seed_admission(
        objective_contract=contract,
        baseline_metrics=(
            _metric("primary", 100.0),
            _metric("secondary", 100.0),
        ),
        candidate_metrics=(
            _metric("primary", candidate_values[0]),
            _metric("secondary", candidate_values[1]),
        ),
        immediate_min_improvement_fraction=0.05,
    )
    ingestion = MapElitesIngestion(
        settings=settings,
        console=Console(record=True),
        repo_root=tmp_path,
        repo=object(),  # type: ignore[arg-type]
        manager=object(),  # type: ignore[arg-type]
    )
    snapshot = JobSnapshot(
        job_id=uuid4(),
        base_commit_hash="root",
        island_id="main",
        result_commit_hash="candidate",
        completed_at=None,
        is_seed_job=True,
        seed_portfolio_hash="a" * 64,
        seed_direction_id="cache-layout",
    )
    archive_calls: list[dict[str, object]] = []
    recorded_admission: list[SeedAdmissionDecision] = []

    monkeypatch.setattr(
        MapElitesIngestion,
        "_resolve_snapshot_commit",
        lambda _self, _snapshot: ("candidate", "candidate"),
    )
    monkeypatch.setattr(
        MapElitesIngestion,
        "_equivalent_ingested_candidate",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        MapElitesIngestion,
        "_metrics_payload_for_ingestion",
        lambda _self, **_kwargs: [
            _metric("primary", candidate_values[0]),
            _metric("secondary", candidate_values[1]),
        ],
    )
    monkeypatch.setattr(
        MapElitesIngestion,
        "_classify_portfolio_seed",
        lambda *_args, **_kwargs: decision,
    )
    monkeypatch.setattr(
        MapElitesIngestion,
        "_record_seed_admission",
        lambda _self, _snapshot, *, commit_hash, decision, session: (
            recorded_admission.append(decision)
        ),
    )
    monkeypatch.setattr(
        MapElitesIngestion,
        "_ingest_with_manager",
        lambda _self, _snapshot, **kwargs: (
            archive_calls.append(kwargs) or SimpleNamespace(record=object())
        ),
    )
    monkeypatch.setattr(
        MapElitesIngestion, "_log_ingestion_result", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        MapElitesIngestion,
        "_record_successful_ingestion",
        lambda *_args, **_kwargs: None,
    )

    inserted = ingestion._ingest_snapshot(snapshot)

    assert decision.lane == expected_lane
    assert inserted is True
    assert recorded_admission == [decision]
    assert len(archive_calls) == 1
