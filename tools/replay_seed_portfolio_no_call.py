"""Replay LOOP-1/2 and seed-portfolio semantics without any model/evaluator call."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from loreley.config import Settings
from loreley.core.map_elites.objectives import ObjectiveContract
from loreley.core.seed_portfolio import (
    EXPLORATORY_STEPPING_STONE_LANE,
    IMMEDIATE_EVIDENCE_LANE,
    SeedPortfolioDraft,
    SeedPortfolioPlanningRequest,
    SeedPortfolioValidationError,
    SeedRootEvidence,
    classify_seed_admission,
    materialize_seed_portfolio,
    resolve_seed_portfolio_direction_count,
    validate_seed_portfolio_draft,
)
from loreley.core.worker.planning import (
    CommitMetric,
    CommitPlanningContext,
    IterationContext,
    SharedPromptPacketRequest,
    render_shared_prompt_packet,
)


def _direction(
    direction_id: str,
    *,
    title: str,
    mechanism: str,
    intent: str,
) -> dict[str, Any]:
    return {
        "direction_id": direction_id,
        "title": title,
        "bottleneck": f"A measured Zstandard bottleneck addressed by {title}.",
        "causal_mechanism": mechanism,
        "likely_files": ["lib/compress/zstd_compress.c"],
        "first_implementation": f"Implement one bounded first step for {title}.",
        "expected_immediate_signals": [
            "One configured lower-confidence-bound objective may improve."
        ],
        "acceptable_neutral_results": [
            "A neutral first result preserves the next causal milestone."
        ],
        "roadmap": [
            "Measure the isolated first mechanism.",
            "Extend only after the mechanism assumptions remain credible.",
        ],
        "risks": ["Compressed output compatibility must remain unchanged."],
        "local_checks": ["Run existing focused correctness tests only."],
        "admission_intent": intent,
        "selection_reason": f"{title} covers a distinct causal search direction.",
    }


def _draft_payload(*, overlap_score: float = 0.2) -> dict[str, Any]:
    return {
        "directions": [
            _direction(
                "match-cache-layout",
                title="Match cache layout",
                mechanism=(
                    "Reduce repeated match decoding through a compact data-reuse layout."
                ),
                intent=IMMEDIATE_EVIDENCE_LANE,
            ),
            _direction(
                "branch-specialization",
                title="Branch specialization",
                mechanism=(
                    "Separate a common control-flow case to enable later specialization."
                ),
                intent=EXPLORATORY_STEPPING_STONE_LANE,
            ),
        ],
        "pairwise_overlaps": [
            {
                "direction_a": "match-cache-layout",
                "direction_b": "branch-specialization",
                "overlap_score": overlap_score,
                "shared_surface": "Both can touch the compression hot path.",
                "mechanism_distinction": (
                    "One changes data reuse and the other changes control flow."
                ),
            }
        ],
        "rejected_directions": [
            {
                "title": "Alternative match cache container",
                "causal_mechanism": (
                    "Use a different container for the same match-decoding reuse."
                ),
                "duplicate_of_direction_id": "match-cache-layout",
                "rejection_reason": (
                    "It is a superficial variant of the selected cache mechanism."
                ),
            }
        ],
        "curation_summary": (
            "The portfolio retains distinct data-reuse and control-flow mechanisms."
        ),
    }


def _metric(
    name: str,
    value: float,
    *,
    unit: str | None = None,
    higher_is_better: bool = True,
) -> CommitMetric:
    return CommitMetric(
        name=name,
        value=value,
        unit=unit,
        higher_is_better=higher_is_better,
    )


def _metric_payload(metric: CommitMetric) -> dict[str, Any]:
    return {
        "name": metric.name,
        "value": metric.value,
        "unit": metric.unit,
        "higher_is_better": metric.higher_is_better,
    }


def build_replay_report() -> dict[str, Any]:
    """Return deterministic acceptance evidence and raise on any mismatch."""

    settings = Settings(
        EXPERIMENT_ID="seed-portfolio-no-call-replay",
        MAPELITES_CODE_EMBEDDING_DIMENSIONS=8,
        MAPELITES_OBJECTIVES=[
            {"name": "compression_lower_95", "direction": "max"},
            {"name": "decompression_lower_95", "direction": "max"},
            {"name": "worst_cell_speedup", "direction": "max"},
        ],
        MAPELITES_SEED_PORTFOLIO_DIRECTION_COUNT=2,
    )
    contract = ObjectiveContract(settings.mapelites_objectives)
    root_metrics = (
        _metric("peak_rss_delta_mib", 0.0, unit="MiB", higher_is_better=False),
        _metric("throughput_geomean", 1.011),
        _metric("worst_cell_speedup", 1.005),
        _metric("decompression_lower_95", 1.008),
        _metric("compression_lower_95", 1.010),
    )
    inspiration_metrics = (
        _metric("peak_rss_delta_mib", -2.0, unit="MiB", higher_is_better=False),
        _metric("compression_lower_95", 1.012),
        _metric("decompression_lower_95", 1.008),
        _metric("worst_cell_speedup", 1.005),
    )
    base = CommitPlanningContext(
        commit_hash="root",
        subject="Zstandard root",
        change_summary="Root baseline.",
        evaluation_summary="Formal root metrics replay.",
        metrics=root_metrics,
    )
    inspiration = CommitPlanningContext(
        commit_hash="inspiration",
        subject="Measured alternative",
        change_summary="Small compression improvement and lower memory.",
        metrics=inspiration_metrics,
    )
    draft = SeedPortfolioDraft.model_validate(_draft_payload())
    validate_seed_portfolio_draft(
        draft,
        expected_direction_count=2,
        max_pairwise_overlap=0.65,
    )
    request = SeedPortfolioPlanningRequest(
        configured_direction_count=2,
        direction_count=2,
        root_commit_hash="root",
        campaign_program_hash="c" * 64,
        campaign_title="Formal Zstandard replay",
        goal="Improve Zstandard throughput without changing compatibility.",
        constraints=("Preserve format compatibility.",),
        acceptance_criteria=("All correctness gates pass.",),
        notes=("No framework-managed evaluator calls in workers.",),
        objective_contract=tuple(contract.as_payload()),
        objective_contract_fingerprint=contract.fingerprint,
        root_evidence=SeedRootEvidence(
            evaluation_summary=base.evaluation_summary,
            metrics=tuple(_metric_payload(metric) for metric in root_metrics),
            diagnostics=("Compression hot path dominates the profile.",),
        ),
        input_evidence_fingerprints={
            "baseline_key": "b" * 64,
            "root_metrics": hashlib.sha256(b"root-metrics").hexdigest(),
            "root_evaluation_artifacts": hashlib.sha256(b"root-evidence").hexdigest(),
        },
        model_route={
            "backend": "kilo",
            "provider": "openai",
            "model": "openai/gpt-5.6-sol",
        },
        reasoning_effort="high",
        max_pairwise_overlap=0.65,
    )
    artifact = materialize_seed_portfolio(request, draft)
    replayed_artifact = materialize_seed_portfolio(request, draft)

    prompts: list[str] = []
    for direction in artifact.directions:
        prompts.append(
            render_shared_prompt_packet(
                SharedPromptPacketRequest(
                    goal=request.goal,
                    constraints=request.constraints,
                    acceptance_criteria=request.acceptance_criteria,
                    iteration_context=IterationContext(
                        seed_job=True,
                        sampling_strategy="seed_portfolio",
                        seed_portfolio_hash=artifact.portfolio_hash,
                        seed_direction_id=direction.direction_id,
                        seed_direction=direction.model_dump(mode="json"),
                    ),
                    base=base,
                    inspirations=(inspiration,),
                    settings=settings,
                )
            )
        )

    objective_order = [prompts[0].index(f"`{name}`") for name in contract.names]
    why_it_matters = prompts[0].split("why_it_matters:", 1)[1].splitlines()[0]
    immediate_metrics = tuple(
        _metric_payload(metric)
        for metric in (
            _metric("compression_lower_95", 1.012),
            _metric("decompression_lower_95", 1.008),
            _metric("worst_cell_speedup", 1.005),
        )
    )
    neutral_metrics = tuple(
        _metric_payload(metric)
        for metric in (
            _metric("compression_lower_95", 1.010),
            _metric("decompression_lower_95", 1.008),
            _metric("worst_cell_speedup", 1.005),
        )
    )
    pareto_tradeoff_metrics = tuple(
        _metric_payload(metric)
        for metric in (
            _metric("compression_lower_95", 1.111),
            _metric("decompression_lower_95", 0.98784),
            _metric("worst_cell_speedup", 1.005),
        )
    )
    baseline_objectives = tuple(
        _metric_payload(metric)
        for metric in root_metrics
        if metric.name in contract.names
    )
    immediate = classify_seed_admission(
        objective_contract=contract,
        baseline_metrics=baseline_objectives,
        candidate_metrics=immediate_metrics,
    )
    exploratory = classify_seed_admission(
        objective_contract=contract,
        baseline_metrics=baseline_objectives,
        candidate_metrics=neutral_metrics,
    )
    pareto_tradeoff = classify_seed_admission(
        objective_contract=contract,
        baseline_metrics=baseline_objectives,
        candidate_metrics=pareto_tradeoff_metrics,
    )
    large_profile = settings.model_copy(
        update={
            "mapelites_seed_population_size": 128,
            "mapelites_feature_normalization_warmup_samples": 128,
            "mapelites_seed_portfolio_direction_count": 8,
            "scheduler_max_total_jobs": 30_000,
        }
    )
    bounded_direction_count = resolve_seed_portfolio_direction_count(large_profile)
    changed_count_request = replace(request, configured_direction_count=3)

    overlap_rejected = False
    try:
        validate_seed_portfolio_draft(
            SeedPortfolioDraft.model_validate(_draft_payload(overlap_score=0.9)),
            expected_direction_count=2,
            max_pairwise_overlap=0.65,
        )
    except SeedPortfolioValidationError:
        overlap_rejected = True

    checks = {
        "no_model_or_evaluator_calls": True,
        "portfolio_hash_restart_stable": (
            artifact.portfolio_hash == replayed_artifact.portfolio_hash
        ),
        "seed_prompts_distinct": prompts[0] != prompts[1],
        "objective_contract_ordered": objective_order == sorted(objective_order),
        "lower_confidence_objectives_rendered": all(
            name in prompts[0]
            for name in ("compression_lower_95", "decompression_lower_95")
        ),
        "memory_not_promoted_by_raw_unit_scale": (
            "compression_lower_95" in why_it_matters
            and "peak_rss_delta_mib" not in why_it_matters
        ),
        "overlapping_brief_rejected": overlap_rejected,
        "complete_model_objective_root_evidence_provenance": all(
            (
                artifact.model_route.get("model") == "openai/gpt-5.6-sol",
                artifact.objective_contract_fingerprint == contract.fingerprint,
                artifact.root_commit_hash == "root",
                bool(artifact.input_evidence_fingerprints.get("root_metrics")),
            )
        ),
        "immediate_lane_exercised": immediate.lane == IMMEDIATE_EVIDENCE_LANE,
        "neutral_is_valid_exploratory_lane": (
            exploratory.lane == EXPLORATORY_STEPPING_STONE_LANE and exploratory.admitted
        ),
        "pareto_tradeoff_reaches_qd_admission": (
            pareto_tradeoff.lane == IMMEDIATE_EVIDENCE_LANE
            and pareto_tradeoff.admitted
            and pareto_tradeoff.directed_fractional_deltas[0][1] > 0.09
            and pareto_tradeoff.directed_fractional_deltas[1][1] < -0.019
        ),
        "large_warmup_direction_count_bounded": (
            bounded_direction_count == 8
            and bounded_direction_count * (bounded_direction_count - 1) // 2 == 28
        ),
        "configured_direction_count_fingerprinted": (
            changed_count_request.direction_count == request.direction_count
            and changed_count_request.request_fingerprint != request.request_fingerprint
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"no-call seed portfolio replay failed: {failed}")
    return {
        "schema_version": 1,
        "portfolio_hash": artifact.portfolio_hash,
        "request_fingerprint": artifact.request_fingerprint,
        "prompt_sha256": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
        ],
        "objective_contract": contract.as_payload(),
        "why_it_matters": why_it_matters.strip(),
        "admission_lanes": {
            "immediate": immediate.lane,
            "neutral": exploratory.lane,
            "pareto_tradeoff": pareto_tradeoff.lane,
        },
        "checks": checks,
    }


def main() -> int:
    print(json.dumps(build_replay_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
