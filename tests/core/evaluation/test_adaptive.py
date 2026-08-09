from __future__ import annotations

import json

import pytest

from loreley.core.evaluation import (
    AdaptiveEvaluationRunner,
    AdaptiveSamplingConfig,
    HoeffdingConfidenceSequence,
    Observation,
    SampleBatchError,
    StudentTInterval,
    UnsafeIntervalMethodError,
)


def test_fixed_sample_t_interval_is_allowed_for_one_analysis_only() -> None:
    config = AdaptiveSamplingConfig(
        min_samples=4,
        max_samples=4,
        batch_size=2,
    )
    runner = AdaptiveEvaluationRunner(config, interval_method=StudentTInterval())

    result = runner.run(lambda request: [1.0] * request.requested_samples)

    assert result.stop_reason == "maximum_samples_reached"
    assert result.unsafe_fixed_sample_override is False
    assert len(result.history) == 2
    assert result.history[0].estimate is None
    assert result.history[1].look_index == 1
    assert result.estimate is not None
    assert result.inference_valid is True
    assert result.declared_target_reached is True
    assert result.decision_ready is True


def test_fixed_sample_interval_cannot_silently_drive_optional_stopping() -> None:
    config = AdaptiveSamplingConfig(min_samples=2, max_samples=4, batch_size=1)

    with pytest.raises(UnsafeIntervalMethodError, match="anytime-valid"):
        AdaptiveEvaluationRunner(config, interval_method=StudentTInterval())

    explicit_override = AdaptiveSamplingConfig(
        min_samples=2,
        max_samples=4,
        batch_size=2,
        allow_fixed_sample_optional_stopping=True,
    )
    result = AdaptiveEvaluationRunner(
        explicit_override,
        interval_method=StudentTInterval(),
    ).run(lambda request: [1.0] * request.requested_samples)
    assert result.unsafe_fixed_sample_override is True
    assert result.inference_valid is False
    assert result.decision_ready is False


def test_anytime_sequence_stops_on_effect_decision_with_full_history() -> None:
    config = AdaptiveSamplingConfig(
        min_samples=50,
        max_samples=100,
        batch_size=50,
        effect_threshold=0.5,
        indifference_zone=0.05,
    )
    requests = []

    def sample(request):
        requests.append(request)
        return [1.0] * request.requested_samples

    result = AdaptiveEvaluationRunner(
        config,
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
    ).run(sample)

    assert len(requests) == 1
    assert result.stop_reason == "effect_decision_reached"
    assert result.effect_classification == "above_indifference_zone"
    assert len(result.observations) == 50
    assert result.history[-1].stop_triggers == ("effect_decision_reached",)
    assert result.history[-1].request == requests[0]


def test_precision_target_stops_an_anytime_sequence() -> None:
    config = AdaptiveSamplingConfig(
        min_samples=50,
        max_samples=100,
        batch_size=50,
        target_ci_half_width=0.25,
    )
    result = AdaptiveEvaluationRunner(
        config,
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
    ).run(lambda request: [0.5] * request.requested_samples)

    assert result.stop_reason == "precision_target_reached"
    assert result.effect_classification == "not_evaluated"
    assert result.estimate is not None
    assert result.estimate.confidence_interval.half_width <= 0.25


def test_effect_interval_can_stop_inside_an_indifference_zone() -> None:
    config = AdaptiveSamplingConfig(
        min_samples=2,
        max_samples=10,
        batch_size=2,
        effect_threshold=0.5,
        indifference_zone=0.02,
    )
    result = AdaptiveEvaluationRunner(
        config,
        interval_method=HoeffdingConfidenceSequence(0.49, 0.51),
    ).run(lambda request: [0.5] * request.requested_samples)

    assert result.stop_reason == "effect_decision_reached"
    assert result.effect_classification == "inside_indifference_zone"


def test_wall_time_budget_records_completed_batch_and_dominates_other_triggers() -> None:
    now = [100.0]
    config = AdaptiveSamplingConfig(
        min_samples=2,
        max_samples=10,
        batch_size=2,
        max_wall_time_seconds=2.0,
        effect_threshold=0.5,
    )

    def sample(request):
        now[0] += 3.0
        return [1.0] * request.requested_samples

    result = AdaptiveEvaluationRunner(
        config,
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
        clock=lambda: now[0],
    ).run(sample)

    assert result.stop_reason == "wall_time_budget_exhausted"
    assert len(result.observations) == 2
    assert result.elapsed_seconds == 3.0
    assert result.history[-1].stop_triggers[0] == "wall_time_budget_exhausted"


def test_empty_callback_stops_without_fabricating_an_estimate() -> None:
    result = AdaptiveEvaluationRunner(
        AdaptiveSamplingConfig(min_samples=2, max_samples=10, batch_size=2),
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
    ).run(lambda request: [])

    assert result.stop_reason == "sampler_exhausted"
    assert result.observations == ()
    assert result.estimate is None
    assert result.history[-1].received_samples == 0
    assert result.history[-1].stopped is True
    assert result.declared_target_reached is False
    assert result.decision_ready is False


def test_budget_exhaustion_is_not_silently_a_completed_precision_target() -> None:
    now = [10.0]
    config = AdaptiveSamplingConfig(
        min_samples=2,
        max_samples=10,
        batch_size=2,
        max_wall_time_seconds=1.0,
        target_ci_half_width=0.01,
    )

    def sample(_request):
        now[0] += 2.0
        return [0.25, 0.75]

    result = AdaptiveEvaluationRunner(
        config,
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
        clock=lambda: now[0],
    ).run(sample)

    assert result.stop_reason == "wall_time_budget_exhausted"
    assert result.inference_valid is True
    assert result.declared_target_reached is False
    assert result.decision_ready is False
    assert result.as_dict()["decision_ready"] is False


def test_callback_cannot_overfill_a_bounded_request() -> None:
    runner = AdaptiveEvaluationRunner(
        AdaptiveSamplingConfig(min_samples=1, max_samples=2, batch_size=1),
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
    )

    with pytest.raises(SampleBatchError, match="returned 2"):
        runner.run(lambda request: [0.25, 0.75])


def test_result_is_deterministic_json_and_preserves_strata() -> None:
    config = AdaptiveSamplingConfig(
        min_samples=2,
        max_samples=2,
        batch_size=2,
        stratum_weights={"small": 0.5, "large": 0.5},
    )

    def run_once():
        return AdaptiveEvaluationRunner(
            config,
            interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
            clock=lambda: 10.0,
        ).run(
            lambda request: [
                Observation(0.25, "small"),
                Observation(0.75, "large"),
            ]
        )

    first = run_once().as_dict()
    second = run_once().as_dict()
    assert first == second
    assert first["config"]["max_samples"] == 2
    assert [item["stratum"] for item in first["observations"]] == ["small", "large"]
    assert json.loads(json.dumps(first, allow_nan=False)) == first


def test_adaptive_runner_uses_predeclared_stratified_estimand() -> None:
    config = AdaptiveSamplingConfig(
        min_samples=4,
        max_samples=4,
        batch_size=4,
        stratum_weights={"small": 0.9, "large": 0.1},
    )
    result = AdaptiveEvaluationRunner(
        config,
        interval_method=StudentTInterval(),
    ).run(
        lambda _request: [
            Observation(0.0, "small"),
            Observation(0.0, "small"),
            Observation(100.0, "large"),
            Observation(100.0, "large"),
        ]
    )

    assert result.estimate is not None
    assert result.estimate.overall.mean == pytest.approx(10.0)


def test_stratified_observations_without_weights_fail_closed() -> None:
    runner = AdaptiveEvaluationRunner(
        AdaptiveSamplingConfig(min_samples=2, max_samples=2, batch_size=2),
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
    )

    with pytest.raises(ValueError, match="predeclared stratum_weights"):
        runner.run(
            lambda _request: [
                Observation(0.25, "small"),
                Observation(0.75, "large"),
            ]
        )


def test_adaptive_checkpoint_resumes_without_reusing_look_indexes() -> None:
    config = AdaptiveSamplingConfig(min_samples=2, max_samples=6, batch_size=2)
    batches = iter(([0.5, 0.5], ()))
    first = AdaptiveEvaluationRunner(
        config,
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
        clock=lambda: 10.0,
    ).run(lambda _request: next(batches))

    resumed = AdaptiveEvaluationRunner(
        config,
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
        clock=lambda: 10.0,
    ).run(
        lambda request: [0.5] * request.requested_samples,
        checkpoint=first.checkpoint(),
    )

    assert len(resumed.observations) == 6
    assert resumed.history[-1].look_index == 3
    assert resumed.history[-1].request is not None
    assert resumed.history[-1].request.batch_index == 4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_samples": 0, "max_samples": 1}, "min_samples"),
        ({"min_samples": 1.0, "max_samples": 2}, "min_samples"),
        ({"min_samples": 2, "max_samples": 1}, "must not exceed"),
        ({"min_samples": 1, "max_samples": 2, "batch_size": 0}, "batch_size"),
        (
            {"min_samples": 1, "max_samples": 2, "indifference_zone": 0.1},
            "requires effect_threshold",
        ),
    ],
)
def test_config_rejects_ambiguous_or_unbounded_contracts(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveSamplingConfig(**kwargs)
