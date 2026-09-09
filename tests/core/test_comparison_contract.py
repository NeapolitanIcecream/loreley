from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from loreley.core.comparison_contract import (
    ComparisonContractError,
    validate_comparison_result,
)


def _context(*, occupied: bool = True, higher_is_better: bool = False):
    return {
        "context_id": "a6b52570-6e0f-4351-905d-303d7d4cd6a4",
        "objective_name": "cpu",
        "higher_is_better": higher_is_better,
        "incumbent_commit_hash": "a" * 40 if occupied else None,
        "candidate_commit_hash": "b" * 40,
        "island_id": 0,
        "cell_index": [1, 2],
    }


def _payload(**updates):
    value = {
        "context_id": _context()["context_id"],
        "complete": True,
        "incumbent_value": 100.0,
        "improvement_lower_bound": 0.05,
        "confidence_level": 0.95,
        "sample_count": 12,
    }
    value.update(updates)
    return value


def _metrics(value=99.9, *, higher_is_better=False):
    return [{"name": "cpu", "value": value, "higher_is_better": higher_is_better}]


def test_improvement_has_oriented_log_gain_and_complete_evidence() -> None:
    payload = _payload(candidate_value=99.9, blocks=[{"sha256": "digest"}])
    decision = validate_comparison_result(payload, _context(), _metrics())

    assert decision.allowed is True
    assert decision.context_id == _context()["context_id"]
    assert decision.candidate_value == 99.9
    assert decision.incumbent_value == 100.0
    assert decision.point_gain == pytest.approx(100 * math.log(100 / 99.9))
    assert decision.improvement_lower_bound == 0.05
    assert decision.confidence_level == 0.95
    assert decision.sample_count == 12
    assert decision.evidence == payload


def test_maximized_objective_and_metric_object_are_supported() -> None:
    metric = SimpleNamespace(name="cpu", value=100.1, higher_is_better=True)
    decision = validate_comparison_result(
        _payload(), _context(higher_is_better=True), [metric]
    )

    assert decision.allowed is True
    assert decision.point_gain == pytest.approx(100 * math.log(100.1 / 100))


@pytest.mark.parametrize("bound", [0.0, -0.01])
def test_completed_uncertain_comparison_is_valid_but_not_admitted(bound) -> None:
    decision = validate_comparison_result(
        _payload(improvement_lower_bound=bound), _context(), _metrics()
    )

    assert decision.allowed is False


def test_regression_can_have_valid_negative_bound() -> None:
    decision = validate_comparison_result(
        _payload(improvement_lower_bound=-0.2), _context(), _metrics(100.1)
    )

    assert decision.allowed is False
    assert decision.point_gain < 0


def test_empty_cell_needs_no_fabricated_comparison() -> None:
    decision = validate_comparison_result(
        {"context_id": _context()["context_id"], "complete": True},
        _context(occupied=False),
        _metrics(),
    )

    assert decision.allowed is True
    assert decision.incumbent_value is None
    assert decision.point_gain is None
    assert decision.improvement_lower_bound is None
    assert decision.confidence_level is None
    assert decision.sample_count is None


@pytest.mark.parametrize(
    "field", ["incumbent_value", "improvement_lower_bound", "confidence_level", "sample_count"]
)
def test_empty_cell_rejects_comparison_fields_even_when_null(field) -> None:
    with pytest.raises(ComparisonContractError, match="Empty-cell"):
        validate_comparison_result(
            {"context_id": _context()["context_id"], "complete": True, field: None},
            _context(occupied=False),
            _metrics(),
        )


@pytest.mark.parametrize("payload", [None, [], "no improvement", 0])
def test_missing_or_nonobject_result_is_an_error_not_non_improvement(payload) -> None:
    with pytest.raises(ComparisonContractError, match="must be an object"):
        validate_comparison_result(payload, _context(), _metrics())


@pytest.mark.parametrize("complete", [False, None, 1, "true"])
def test_unfinished_result_is_an_error_not_non_improvement(complete) -> None:
    with pytest.raises(ComparisonContractError, match="complete must be true"):
        validate_comparison_result(_payload(complete=complete), _context(), _metrics())


@pytest.mark.parametrize("context_id", ["other-context", None, "", 1])
def test_result_is_bound_to_exact_context(context_id) -> None:
    with pytest.raises(ComparisonContractError, match="context_id"):
        validate_comparison_result(
            _payload(context_id=context_id), _context(), _metrics()
        )


@pytest.mark.parametrize(
    "field", ["incumbent_value", "improvement_lower_bound", "confidence_level", "sample_count"]
)
def test_occupied_cell_requires_each_comparison_field(field) -> None:
    payload = _payload()
    del payload[field]
    with pytest.raises(ComparisonContractError, match=field):
        validate_comparison_result(payload, _context(), _metrics())


@pytest.mark.parametrize(
    "field", ["incumbent_value", "candidate_value", "improvement_lower_bound", "confidence_level"]
)
@pytest.mark.parametrize("value", [True, "1.0", math.nan, math.inf, -math.inf, None])
def test_comparison_numbers_are_finite_json_numbers(field, value) -> None:
    with pytest.raises(ComparisonContractError, match=field):
        validate_comparison_result(_payload(**{field: value}), _context(), _metrics())


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("field", ["incumbent_value", "candidate_value"])
def test_comparison_values_are_positive(field, value) -> None:
    with pytest.raises(ComparisonContractError, match="positive"):
        validate_comparison_result(_payload(**{field: value}), _context(), _metrics())


@pytest.mark.parametrize("value", [True, "99.9", math.nan, math.inf, 0, -1])
def test_candidate_metric_is_positive_finite_json_number(value) -> None:
    with pytest.raises(ComparisonContractError, match="metric 'cpu' value"):
        validate_comparison_result(_payload(), _context(), _metrics(value))


def test_optional_candidate_value_cannot_override_the_metric() -> None:
    with pytest.raises(ComparisonContractError, match="does not match"):
        validate_comparison_result(_payload(candidate_value=99.8), _context(), _metrics())


@pytest.mark.parametrize("count", [True, 1, 0, -1, 2.0, "2"])
def test_sample_count_is_an_integer_at_least_two(count) -> None:
    with pytest.raises(ComparisonContractError, match="sample_count"):
        validate_comparison_result(_payload(sample_count=count), _context(), _metrics())


@pytest.mark.parametrize("confidence", [0, 0.94, 1, 1.1, -0.1])
def test_insufficient_or_invalid_confidence_is_an_error(confidence) -> None:
    with pytest.raises(ComparisonContractError, match="confidence_level"):
        validate_comparison_result(
            _payload(confidence_level=confidence), _context(), _metrics()
        )


def test_configured_confidence_is_enforced() -> None:
    with pytest.raises(ComparisonContractError, match="at least 0.99"):
        validate_comparison_result(
            _payload(), _context(), _metrics(), minimum_confidence=0.99
        )
    decision = validate_comparison_result(
        _payload(confidence_level=0.99), _context(), _metrics(), minimum_confidence=0.99
    )
    assert decision.allowed is True


@pytest.mark.parametrize("minimum", [True, None, "0.95", 0, 1, math.nan])
def test_invalid_configured_confidence_is_rejected(minimum) -> None:
    with pytest.raises(ComparisonContractError, match="minimum_confidence"):
        validate_comparison_result(
            _payload(), _context(), _metrics(), minimum_confidence=minimum
        )


@pytest.mark.parametrize("value,bound", [(99.9, 0.2), (100, 1e-16), (100.1, 0.05)])
def test_lower_bound_cannot_exceed_measured_gain(value, bound) -> None:
    with pytest.raises(ComparisonContractError, match="exceeds"):
        validate_comparison_result(
            _payload(improvement_lower_bound=bound), _context(), _metrics(value)
        )


@pytest.mark.parametrize("direction", [True, None, "false", 0])
def test_metric_direction_must_explicitly_match_context(direction) -> None:
    with pytest.raises(ComparisonContractError, match="direction"):
        validate_comparison_result(
            _payload(), _context(), _metrics(higher_is_better=direction)
        )


@pytest.mark.parametrize("metrics", [None, [], "cpu", _metrics() * 2, [{"name": "other"}]])
def test_metric_must_occur_exactly_once(metrics) -> None:
    with pytest.raises(ComparisonContractError, match="configured objective"):
        validate_comparison_result(_payload(), _context(), metrics)


def test_single_metric_mapping_is_supported() -> None:
    assert validate_comparison_result(_payload(), _context(), _metrics()[0]).allowed


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_id", None),
        ("objective_name", ""),
        ("higher_is_better", 0),
        ("incumbent_commit_hash", ""),
    ],
)
def test_context_required_fields_are_validated(field, value) -> None:
    context = _context()
    context[field] = value
    with pytest.raises(ComparisonContractError, match=f"context.{field}"):
        validate_comparison_result(_payload(), context, _metrics())


def test_missing_incumbent_context_cannot_be_mistaken_for_empty_cell() -> None:
    context = _context()
    del context["incumbent_commit_hash"]
    with pytest.raises(ComparisonContractError, match="incumbent_commit_hash is required"):
        validate_comparison_result(_payload(), context, _metrics())


@pytest.mark.parametrize("extension", [{1: "key"}, {"x": math.nan}, (1, 2), object()])
def test_extension_evidence_must_be_strict_json(extension) -> None:
    with pytest.raises(ComparisonContractError, match="fresh_comparison"):
        validate_comparison_result(
            _payload(evidence=extension), _context(), _metrics()
        )


def test_cyclic_evidence_is_a_contract_error() -> None:
    payload = _payload()
    payload["cycle"] = payload
    with pytest.raises(ComparisonContractError, match="acyclic JSON"):
        validate_comparison_result(payload, _context(), _metrics())


def test_unrepresentable_numeric_value_is_a_contract_error() -> None:
    with pytest.raises(ComparisonContractError, match="finite JSON number"):
        validate_comparison_result(
            _payload(incumbent_value=10**1000), _context(), _metrics()
        )


def test_result_and_serialized_evidence_do_not_alias_evaluator_payload() -> None:
    payload = _payload(blocks=[{"index": 1}])
    decision = validate_comparison_result(payload, _context(), _metrics())
    payload["blocks"][0]["index"] = 2
    serialized = decision.as_dict()
    serialized["evidence"]["blocks"][0]["index"] = 3

    assert decision.evidence["blocks"][0]["index"] == 1
    assert serialized["allowed"] is True


@pytest.mark.parametrize("candidate,incumbent", [(1e308, 1e-308), (1e-308, 1e308)])
def test_valid_extreme_values_do_not_overflow_log_gain(candidate, incumbent) -> None:
    gain = 100 * (math.log(candidate) - math.log(incumbent))
    decision = validate_comparison_result(
        _payload(incumbent_value=incumbent, improvement_lower_bound=gain - 1),
        _context(higher_is_better=True),
        _metrics(candidate, higher_is_better=True),
    )

    assert math.isfinite(decision.point_gain)
    assert decision.point_gain == pytest.approx(gain)


def test_contract_does_not_claim_to_verify_interval_methodology() -> None:
    # A dishonest statistical claim can be internally consistent. The evaluator
    # must be tested separately for coverage; core cannot infer it from a summary.
    decision = validate_comparison_result(
        _payload(method="not a real interval method"), _context(), _metrics()
    )
    assert decision.allowed is True
