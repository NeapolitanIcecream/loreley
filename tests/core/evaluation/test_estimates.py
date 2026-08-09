from __future__ import annotations

import json
import math

import pytest

from loreley.core.evaluation import (
    HoeffdingConfidenceSequence,
    InsufficientSamplesError,
    IntervalCapability,
    Observation,
    StudentTInterval,
    aggregate_by_stratum,
    estimate,
    sample_moments,
)


def test_observation_requires_a_finite_value_and_named_stratum() -> None:
    assert Observation(1, "  cold ") == Observation(1.0, "cold")
    with pytest.raises(TypeError, match="not bool"):
        Observation(True)
    with pytest.raises(ValueError, match="finite"):
        Observation(math.nan)
    with pytest.raises(ValueError, match="stratum"):
        Observation(1.0, "  ")


def test_sample_moments_use_bessel_correction_and_explicit_singleton_state() -> None:
    moments = sample_moments([1.0, 2.0, 3.0, 4.0])

    assert moments.n == 4
    assert moments.mean == pytest.approx(2.5)
    assert moments.sample_variance == pytest.approx(5.0 / 3.0)
    assert moments.standard_error == pytest.approx(math.sqrt(5.0 / 12.0))

    singleton = sample_moments([7.0])
    assert singleton.mean == 7.0
    assert singleton.sample_variance is None
    assert singleton.standard_error is None


def test_sample_moments_reject_empty_or_overflowing_input() -> None:
    with pytest.raises(InsufficientSamplesError, match="At least one"):
        sample_moments([])
    with pytest.raises(ValueError, match="overflowed"):
        sample_moments([1e308, -1e308])


def test_student_t_interval_matches_known_small_sample_critical_value() -> None:
    result = estimate(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        interval_method=StudentTInterval(),
        confidence_level=0.95,
    )

    # df=4 two-sided 95% critical value is 2.776445105...
    expected_half_width = 2.776445105 * math.sqrt(2.5 / 5.0)
    assert result.mean == 3.0
    assert result.confidence_interval.lower == pytest.approx(
        3.0 - expected_half_width, rel=1e-9
    )
    assert result.confidence_interval.upper == pytest.approx(
        3.0 + expected_half_width, rel=1e-9
    )
    assert (
        result.confidence_interval.capability
        is IntervalCapability.FIXED_SAMPLE_ONLY
    )


def test_student_t_interval_rejects_single_observation() -> None:
    with pytest.raises(InsufficientSamplesError, match="at least two"):
        estimate([1.0], interval_method=StudentTInterval())


def test_hoeffding_sequence_is_anytime_valid_and_enforces_declared_bounds() -> None:
    method = HoeffdingConfidenceSequence(0.0, 1.0)
    first = estimate([0.5] * 20, interval_method=method, look_index=1)
    later_same_data = estimate([0.5] * 20, interval_method=method, look_index=3)

    assert first.confidence_interval.capability is IntervalCapability.ANYTIME_VALID
    assert later_same_data.confidence_interval.half_width > (
        first.confidence_interval.half_width
    )
    with pytest.raises(ValueError, match="outside"):
        estimate([1.01], interval_method=method)


def test_stratified_aggregation_is_sorted_and_records_bonferroni_scope() -> None:
    result = aggregate_by_stratum(
        [
            Observation(1.0, "slow"),
            Observation(3.0, "fast"),
            Observation(5.0, "slow"),
            Observation(7.0, "fast"),
        ],
        interval_method=StudentTInterval(),
        stratum_weights={"fast": 0.75, "slow": 0.25},
        confidence_level=0.95,
    )

    assert result.overall.n == 4
    assert result.overall.mean == 4.5
    assert [item.stratum for item in result.strata] == ["fast", "slow"]
    assert [item.estimate.mean for item in result.strata] == [5.0, 3.0]
    assert dict(result.stratum_weights) == {"fast": 0.75, "slow": 0.25}
    assert result.simultaneous is True
    assert result.per_interval_confidence_level == pytest.approx(1.0 - 0.05 / 2.0)
    json.dumps(result.as_dict(), allow_nan=False)


def test_stratified_aggregation_rejects_post_hoc_or_incomplete_strata() -> None:
    observations = [
        Observation(1.0, "small"),
        Observation(2.0, "small"),
        Observation(3.0, "large"),
        Observation(4.0, "large"),
    ]
    with pytest.raises(ValueError, match="do not match"):
        aggregate_by_stratum(
            observations,
            interval_method=StudentTInterval(),
            stratum_weights={"small": 1.0},
        )
    with pytest.raises(ValueError, match="sum to 1.0"):
        aggregate_by_stratum(
            observations,
            interval_method=StudentTInterval(),
            stratum_weights={"small": 0.4, "large": 0.4},
        )


def test_estimate_artifact_is_plain_json_and_names_interval_contract() -> None:
    payload = estimate(
        [0.25, 0.5, 0.75],
        interval_method=HoeffdingConfidenceSequence(0.0, 1.0),
    ).as_dict()

    assert payload["confidence_interval"]["method"] == (
        "hoeffding_confidence_sequence"
    )
    assert payload["confidence_interval"]["capability"] == "anytime_valid"
    assert payload["confidence_interval"]["method_parameters"]["lower_bound"] == 0.0
    assert payload["confidence_interval"]["method_parameters"]["upper_bound"] == 1.0
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
