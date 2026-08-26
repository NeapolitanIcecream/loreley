from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from loreley.core.map_elites.objectives import ObjectiveContract, ObjectiveSpec
from loreley.core.seed_portfolio import (
    EXPLORATORY_STEPPING_STONE_LANE,
    IMMEDIATE_EVIDENCE_LANE,
    SeedPortfolioDraft,
    SeedPortfolioPlanner,
    SeedPortfolioPlanningRequest,
    SeedPortfolioValidationError,
    SeedRootEvidence,
    classify_seed_admission,
    materialize_seed_portfolio,
    validate_seed_portfolio_draft,
)
from loreley.core.worker.agent import AgentInvocation


def _direction(direction_id: str, *, intent: str, mechanism: str) -> dict[str, object]:
    return {
        "direction_id": direction_id,
        "title": f"Direction {direction_id}",
        "bottleneck": f"A measured bottleneck targeted by {direction_id}.",
        "causal_mechanism": mechanism,
        "likely_files": [f"src/{direction_id}.py"],
        "first_implementation": (
            f"Implement the smallest source change for {direction_id} and keep it isolated."
        ),
        "expected_immediate_signals": ["The configured objective may improve."],
        "acceptable_neutral_results": [
            "A neutral result preserves the next roadmap step."
        ],
        "roadmap": [
            "Measure the first isolated mechanism.",
            "Extend the mechanism only if its assumptions survive.",
        ],
        "risks": ["The optimization could alter correctness."],
        "local_checks": ["Run the focused pre-existing unit tests only."],
        "admission_intent": intent,
        "selection_reason": (
            f"This direction covers a distinct causal route for {direction_id}."
        ),
    }


def _draft_payload(*, overlap_score: float = 0.25) -> dict[str, object]:
    return {
        "directions": [
            _direction(
                "cache-layout",
                intent="immediate_evidence",
                mechanism=(
                    "Reduce repeated decoding work by changing the cache representation."
                ),
            ),
            _direction(
                "branch-shaping",
                intent="exploratory_stepping_stone",
                mechanism=(
                    "Reshape a hot control-flow path so later specialization becomes possible."
                ),
            ),
        ],
        "pairwise_overlaps": [
            {
                "direction_a": "cache-layout",
                "direction_b": "branch-shaping",
                "overlap_score": overlap_score,
                "shared_surface": "Both may touch the decoder hot path.",
                "mechanism_distinction": (
                    "One changes data reuse while the other changes control flow."
                ),
            }
        ],
        "rejected_directions": [
            {
                "title": "Cache layout variant",
                "causal_mechanism": (
                    "Use a superficially different cache container for the same reuse mechanism."
                ),
                "duplicate_of_direction_id": "cache-layout",
                "rejection_reason": (
                    "It duplicates the selected cache-representation mechanism."
                ),
            }
        ],
        "curation_summary": (
            "The selected portfolio covers data reuse and control-flow preparation."
        ),
    }


def _request() -> SeedPortfolioPlanningRequest:
    contract = ObjectiveContract(
        (ObjectiveSpec(name="compression_lower_95", direction="max"),)
    )
    return SeedPortfolioPlanningRequest(
        configured_direction_count=2,
        direction_count=2,
        root_commit_hash="root",
        campaign_program_hash="a" * 64,
        campaign_title="Zstandard",
        goal="Improve throughput without violating correctness.",
        constraints=("Preserve format compatibility.",),
        acceptance_criteria=("Correctness tests pass.",),
        notes=("Keep changes focused.",),
        objective_contract=tuple(contract.as_payload()),
        objective_contract_fingerprint=contract.fingerprint,
        root_evidence=SeedRootEvidence(
            evaluation_summary="Baseline measured.",
            metrics=(
                {
                    "name": "compression_lower_95",
                    "value": 1.0,
                    "higher_is_better": True,
                },
            ),
        ),
        input_evidence_fingerprints={"baseline_key": "b" * 64},
        model_route={
            "backend": "kilo",
            "provider": "openai",
            "model": "openai/gpt-5.6-sol",
        },
        reasoning_effort="high",
        max_pairwise_overlap=0.65,
    )


def test_seed_portfolio_validates_complete_distinct_two_lane_slate() -> None:
    draft = SeedPortfolioDraft.model_validate(_draft_payload())

    validated = validate_seed_portfolio_draft(
        draft,
        expected_direction_count=2,
        max_pairwise_overlap=0.65,
    )

    assert validated is draft
    assert {direction.admission_intent for direction in draft.directions} == {
        IMMEDIATE_EVIDENCE_LANE,
        EXPLORATORY_STEPPING_STONE_LANE,
    }


def test_seed_portfolio_rejects_overlapping_selected_briefs() -> None:
    draft = SeedPortfolioDraft.model_validate(_draft_payload(overlap_score=0.9))

    with pytest.raises(SeedPortfolioValidationError, match="overlap policy"):
        validate_seed_portfolio_draft(
            draft,
            expected_direction_count=2,
            max_pairwise_overlap=0.65,
        )


def test_seed_portfolio_rejects_incomplete_pairwise_matrix() -> None:
    payload = _draft_payload()
    payload["pairwise_overlaps"] = []
    draft = SeedPortfolioDraft.model_validate(payload)

    with pytest.raises(SeedPortfolioValidationError, match="incomplete"):
        validate_seed_portfolio_draft(
            draft,
            expected_direction_count=2,
            max_pairwise_overlap=0.65,
        )


def test_materialized_portfolio_hash_is_content_addressed_and_stable() -> None:
    request = _request()
    draft = SeedPortfolioDraft.model_validate(_draft_payload())

    first = materialize_seed_portfolio(request, draft)
    second = materialize_seed_portfolio(request, draft)

    assert first.portfolio_hash == second.portfolio_hash
    assert first.request_fingerprint == request.request_fingerprint
    assert len(first.portfolio_hash) == 64
    assert first.model_route["model"] == "openai/gpt-5.6-sol"


def test_configured_direction_count_changes_request_fingerprint_when_effective_count_does_not() -> (
    None
):
    request = _request()

    changed = replace(request, configured_direction_count=3)

    assert changed.direction_count == request.direction_count == 2
    assert changed.request_fingerprint != request.request_fingerprint


class _PortfolioBackend:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def run(self, task, *, working_dir: Path) -> AgentInvocation:
        self.calls += 1
        assert task.phase == "seed_portfolio"
        assert "campaign-level seed portfolio planner" in task.prompt
        return AgentInvocation(
            command=("fake",),
            stdout=json.dumps(self.payload),
            stderr="",
            duration_seconds=0.25,
            working_directory=str(working_dir),
        )


class _NoisyPortfolioBackend(_PortfolioBackend):
    def run(self, task, *, working_dir: Path) -> AgentInvocation:
        invocation = super().run(task, working_dir=working_dir)
        return replace(
            invocation,
            stdout=(
                "glob: lib/compress/*.{c,h}\n"
                'tool call: {"query": "ZSTD_fast"}\n'
                'tool result: [{"path": "lib/compress/zstd_fast.c"}]\n'
                f"{invocation.stdout}\n"
                'session: {"status": "complete"}'
            ),
        )


def test_seed_portfolio_planner_uses_one_structured_portfolio_call(
    tmp_path: Path,
    settings,
) -> None:
    backend = _PortfolioBackend(_draft_payload())
    planner = SeedPortfolioPlanner(settings=settings, backend=backend)

    response = planner.plan(_request(), working_dir=tmp_path)

    assert backend.calls == 1
    assert len(response.draft.directions) == 2
    assert response.attempts == 1
    assert len(response.prompt_sha256) == 64
    assert "Keep the inspection and final artifact compact" in response.prompt
    assert "no more than three rejected directions" in response.prompt


def test_seed_portfolio_planner_extracts_final_json_from_noisy_cli_output(
    tmp_path: Path,
    settings,
) -> None:
    backend = _NoisyPortfolioBackend(_draft_payload())
    planner = SeedPortfolioPlanner(settings=settings, backend=backend)

    response = planner.plan(_request(), working_dir=tmp_path)

    assert backend.calls == 1
    assert [direction.direction_id for direction in response.draft.directions] == [
        "cache-layout",
        "branch-shaping",
    ]


def _metric(name: str, value: float) -> dict[str, object]:
    return {"name": name, "value": value, "higher_is_better": True}


def test_seed_admission_classifies_immediate_evidence_without_repeat_evaluation() -> (
    None
):
    contract = ObjectiveContract((ObjectiveSpec(name="quality", direction="max"),))

    decision = classify_seed_admission(
        objective_contract=contract,
        baseline_metrics=(_metric("quality", 1.0),),
        candidate_metrics=(_metric("quality", 1.02),),
    )

    assert decision.lane == IMMEDIATE_EVIDENCE_LANE
    assert decision.admitted is True


def test_seed_admission_keeps_neutral_candidate_as_exploratory_stepping_stone() -> None:
    contract = ObjectiveContract((ObjectiveSpec(name="quality", direction="max"),))

    decision = classify_seed_admission(
        objective_contract=contract,
        baseline_metrics=(_metric("quality", 1.0),),
        candidate_metrics=(_metric("quality", 1.0),),
    )

    assert decision.lane == EXPLORATORY_STEPPING_STONE_LANE
    assert decision.admitted is True
    assert "No optimization objective" in decision.reason


@pytest.mark.parametrize(
    ("candidate", "improved_name"),
    (
        ((110.0, 98.0), "primary"),
        ((98.0, 110.0), "secondary"),
    ),
)
def test_seed_admission_labels_pareto_tradeoff_by_any_improved_objective(
    candidate: tuple[float, float],
    improved_name: str,
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
            _metric("primary", candidate[0]),
            _metric("secondary", candidate[1]),
        ),
        immediate_min_improvement_fraction=0.05,
    )

    assert decision.lane == IMMEDIATE_EVIDENCE_LANE
    assert decision.admitted is True
    assert improved_name in decision.reason


def test_seed_admission_labels_valid_all_nonpositive_candidate_exploratory() -> None:
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
            _metric("primary", 100.0),
            _metric("secondary", 98.0),
        ),
    )

    assert decision.lane == EXPLORATORY_STEPPING_STONE_LANE
    assert decision.admitted is True
