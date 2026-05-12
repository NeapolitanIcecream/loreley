from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import loreley.core.map_elites.sampler as sampler_module
from loreley.config import Settings
from loreley.core.campaign_program import (
    CampaignProjectionInput,
    apply_campaign_program_projection,
    campaign_program_artifact_payload,
    campaign_program_evaluator_payload,
    load_campaign_program_from_repo,
    parse_campaign_program,
    persist_campaign_program,
)
from loreley.core.map_elites.sampler import MapElitesSampler


ROOT = Path(__file__).resolve().parents[2]
CIRCLE_PACKING_PROGRAM = ROOT / "examples" / "circle-packing" / "loreley.program.md"


def test_campaign_program_parser_extracts_recognized_sections_and_keeps_unknown_metadata() -> None:
    raw = b"""# Parser campaign

## Goal
Improve parser throughput without changing public API.

## Primary metric
name: throughput
direction: higher_is_better
unit: req/s

## Correctness gates
- `uv run pytest tests/parser`

## Editable scope
- src/parser/**

## Protected scope
- docs/contracts/**

## Operator note
Human-only note.
"""

    snapshot = parse_campaign_program(raw)

    assert snapshot.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert snapshot.normalized_sha256
    assert snapshot.title == "Parser campaign"
    assert snapshot.goal == "Improve parser throughput without changing public API."
    assert snapshot.primary_metric is not None
    assert snapshot.primary_metric.as_dict() == {
        "name": "throughput",
        "direction": "higher_is_better",
        "unit": "req/s",
    }
    assert snapshot.correctness_gates == ("uv run pytest tests/parser",)
    assert snapshot.editable_scope == ("src/parser/**",)
    assert "loreley.program.md" in snapshot.protected_scope
    assert "Operator note" in [section.title for section in snapshot.unknown_sections]
    assert "Operator note" not in snapshot.recognized_sections


def test_circle_packing_campaign_program_declares_v080_contract() -> None:
    raw = CIRCLE_PACKING_PROGRAM.read_bytes()

    snapshot = parse_campaign_program(raw, source_path="loreley.program.md")
    projected = apply_campaign_program_projection(
        CampaignProjectionInput(
            snapshot=snapshot,
            goal=None,
            constraints=(),
            acceptance_criteria=(),
            notes=(),
            default_goal="Global goal",
        )
    )

    assert "LORELEY_PROFILE" not in raw.decode("utf-8")
    assert snapshot.title == "circle-packing campaign for Loreley"
    assert snapshot.goal is not None
    normalized_goal = snapshot.goal.replace("`", "")
    assert "solution.py" in snapshot.goal
    assert "pack_circles(n=26)" in snapshot.goal
    assert "pack_circles() equivalent to pack_circles(26)" in normalized_goal
    assert snapshot.primary_metric is not None
    assert snapshot.primary_metric.as_dict() == {
        "name": "sum_radii",
        "direction": "higher_is_better",
        "unit": "radius_sum",
    }
    assert snapshot.editable_scope == ("solution.py",)
    assert set(snapshot.protected_scope) == {
        "README.md",
        "pyproject.toml",
        "scripts/**",
        ".gitignore",
        "loreley.program.md",
    }
    gate_text = " | ".join(snapshot.correctness_gates).replace("`", "").lower()
    for expected in (
        "positive radii",
        "finite numeric triples",
        "exactly n circles",
        "in-bounds",
        "no overlap",
        "deterministic output",
        "p50 runtime",
        "250 ms",
        "python scripts/local_eval.py --repo-root . --runs 5 --target-n 26 --json",
    ):
        assert expected in gate_text
    assert projected.goal == snapshot.goal
    assert "Editable scope: solution.py" in projected.constraints
    assert "Protected scope: loreley.program.md" in projected.constraints
    assert any("local_eval.py --repo-root ." in item for item in projected.constraints)
    assert "Primary metric: sum_radii (higher is better, unit=radius_sum)" in (
        projected.acceptance_criteria
    )
    assert any(item.startswith("Logging policy:") and "sum_radii" in item for item in projected.notes)


def test_campaign_program_payloads_exclude_unknown_section_bodies() -> None:
    """Regression: unknown operator sections must not enter evaluator/artifact payloads."""

    snapshot = parse_campaign_program(
        b"""## Goal
Improve safely.

## Operator note
Do not put this human-only text into evaluator payloads.
"""
    )

    assert snapshot.as_dict()["unknown_sections"][0]["body"] == (
        "Do not put this human-only text into evaluator payloads."
    )

    evaluator_payload = campaign_program_evaluator_payload(snapshot)
    artifact_payload = campaign_program_artifact_payload(snapshot)

    assert evaluator_payload is not None
    assert artifact_payload is not None
    assert evaluator_payload["snapshot"]["unknown_sections"] == [
        {"title": "Operator note", "canonical_title": None}
    ]
    assert artifact_payload["normalized_snapshot"]["unknown_sections"] == [
        {"title": "Operator note", "canonical_title": None}
    ]
    assert "human-only text" not in json.dumps(evaluator_payload)
    assert "human-only text" not in json.dumps(artifact_payload)


def test_campaign_program_primary_metric_prose_records_warning() -> None:
    snapshot = parse_campaign_program(
        b"""## Goal
Improve quality.

## Primary metric
Use whatever benchmark looks best.
"""
    )

    assert snapshot.primary_metric is None
    assert any(warning["code"] == "primary_metric_prose" for warning in snapshot.parse_warnings)


def test_campaign_program_projection_fills_empty_job_fields_but_preserves_specific_goal() -> None:
    snapshot = parse_campaign_program(
        b"""## Goal
Campaign goal.

## Correctness gates
- pytest

## Complexity policy
- Prefer small diffs.
"""
    )

    projected = apply_campaign_program_projection(
        CampaignProjectionInput(
            snapshot=snapshot,
            goal="Repair the failed candidate.",
            constraints=(),
            acceptance_criteria=(),
            notes=(),
            default_goal="Global goal",
            preserve_existing_goal=True,
        )
    )

    assert projected.goal == "Repair the failed candidate."
    assert projected.constraints[0] == "Correctness gate: pytest"
    assert projected.acceptance_criteria[0] == "Correctness gate: pytest"
    assert "Complexity policy: Prefer small diffs." in projected.notes


def test_persist_campaign_program_upserts_normalized_snapshot() -> None:
    snapshot = parse_campaign_program(b"## Goal\nImprove reliability.\n")

    class FakeSession:
        def __init__(self) -> None:
            self.row: Any | None = None
            self.added: list[Any] = []

        def get(self, _model: Any, _key: str) -> Any:
            return self.row

        def add(self, row: Any) -> None:
            self.row = row
            self.added.append(row)

    session = FakeSession()

    persist_campaign_program(session=session, snapshot=snapshot, raw_markdown="raw")
    persist_campaign_program(session=session, snapshot=snapshot, raw_markdown="updated")

    assert len(session.added) == 1
    assert session.row.hash == snapshot.raw_sha256
    assert session.row.raw_markdown == "updated"
    assert session.row.normalized_snapshot["goal"] == "Improve reliability."


def test_sampler_persists_program_projection_on_evolution_job(
    monkeypatch,
    settings: Settings,
) -> None:
    snapshot = parse_campaign_program(
        b"""## Goal
Program goal.

## Correctness gates
- pytest
"""
    )
    settings.worker_evolution_global_goal = "Global goal"
    added: list[Any] = []

    class FakeSession:
        def add(self, row: Any) -> None:
            added.append(row)

        def flush(self) -> None:
            return None

    @contextmanager
    def fake_scope():
        yield FakeSession()

    monkeypatch.setattr(sampler_module, "session_scope", fake_scope)
    sampler = MapElitesSampler(
        manager=object(),  # type: ignore[arg-type]
        settings=settings,
    )

    job = sampler._persist_job(  # noqa: SLF001
        island_id="main",
        base_commit_hash="base",
        inspiration_commit_hashes=(),
        selection_stats={"initial_radius": 1, "radius_used": 1, "fallback_inspirations": 0},
        iteration_hint=None,
        priority=None,
        campaign_program=snapshot,
    )

    assert job is added[0]
    assert job.goal == "Program goal."
    assert job.constraints == ["Correctness gate: pytest", "Protected scope: loreley.program.md"]
    assert job.acceptance_criteria == ["Correctness gate: pytest"]
    assert job.campaign_program_hash == snapshot.raw_sha256


def test_missing_campaign_program_preserves_job_defaults_and_records_null_hash(
    tmp_path,
    monkeypatch,
    settings: Settings,
) -> None:
    """Absent loreley.program.md keeps existing job defaults and records a null hash."""

    load_result = load_campaign_program_from_repo(tmp_path)
    settings.worker_evolution_global_goal = "Global goal"
    added: list[Any] = []

    class FakeSession:
        def add(self, row: Any) -> None:
            added.append(row)

        def flush(self) -> None:
            return None

    @contextmanager
    def fake_scope():
        yield FakeSession()

    monkeypatch.setattr(sampler_module, "session_scope", fake_scope)
    sampler = MapElitesSampler(
        manager=object(),  # type: ignore[arg-type]
        settings=settings,
    )

    job = sampler._persist_job(  # noqa: SLF001
        island_id="main",
        base_commit_hash="base",
        inspiration_commit_hashes=(),
        selection_stats={"initial_radius": 1, "radius_used": 1, "fallback_inspirations": 0},
        iteration_hint=None,
        priority=None,
        campaign_program=load_result.snapshot,
    )

    assert load_result.found is False
    assert job is added[0]
    assert job.goal == "Global goal"
    assert job.constraints == []
    assert job.acceptance_criteria == []
    assert job.notes == []
    assert job.campaign_program_hash is None
