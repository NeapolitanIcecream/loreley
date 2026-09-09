"""Validate fresh-comparison evidence before an archive replacement.

The evaluator owns sampling and interval construction. This module checks the
result's identity, shape, and internal consistency; it does not certify that a
reported confidence interval has its claimed statistical coverage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

__all__ = [
    "ComparisonContractError",
    "FreshComparisonDecision",
    "validate_comparison_result",
]


class ComparisonContractError(ValueError):
    """The evaluator did not supply a valid, completed fresh comparison."""


@dataclass(frozen=True, slots=True)
class FreshComparisonDecision:
    """Validated admission evidence, not a guarantee that its target is current.

    The caller must still compare the durable context with the current archive
    in the transaction that applies this decision. A valid non-improvement has
    ``allowed=False``; malformed or incomplete evidence raises instead.
    """

    allowed: bool
    context_id: str
    candidate_value: float
    incumbent_value: float | None
    point_gain: float | None
    improvement_lower_bound: float | None
    confidence_level: float | None
    sample_count: int | None
    evidence: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible copy for durable storage."""

        return {
            "allowed": self.allowed,
            "context_id": self.context_id,
            "candidate_value": self.candidate_value,
            "incumbent_value": self.incumbent_value,
            "point_gain": self.point_gain,
            "improvement_lower_bound": self.improvement_lower_bound,
            "confidence_level": self.confidence_level,
            "sample_count": self.sample_count,
            "evidence": _copy_json(self.evidence, "fresh_comparison"),
        }


def validate_comparison_result(
    payload: Mapping[str, Any] | None,
    context: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any] | object] | Mapping[str, Any] | None,
    minimum_confidence: float = 0.95,
) -> FreshComparisonDecision:
    """Validate ``EvaluationResult.extra['fresh_comparison']``.

    The trusted, durable context supplies ``context_id``, ``objective_name``,
    ``higher_is_better`` and ``incumbent_commit_hash``. Its other provenance is
    checked by the caller against the job, evaluator, and current archive.

    An occupied-cell result requires ``complete=True``, the exact context ID,
    positive ``incumbent_value``, finite ``improvement_lower_bound``, a
    ``confidence_level`` at least the configured minimum, and integer
    ``sample_count >= 2``. Candidate value comes from the configured metric;
    an optional repeated ``candidate_value`` must agree exactly. Gains and
    bounds are in 100 times the natural logarithm of the performance ratio,
    oriented so that positive means improvement. A positive lower bound allows
    replacement. Zero or a negative bound is a valid non-improvement.

    Empty cells require only identity, completion, and the candidate metric.
    They must not report incumbent values or comparison statistics. Additional
    JSON evidence fields are retained, but cannot override the decision.
    """

    minimum = _confidence(minimum_confidence, "minimum_confidence")
    if not isinstance(context, Mapping):
        raise ComparisonContractError("Comparison context must be an object.")
    context_id = _nonempty_string(context.get("context_id"), "context.context_id")
    objective = _nonempty_string(
        context.get("objective_name"), "context.objective_name"
    )
    direction = context.get("higher_is_better")
    if not isinstance(direction, bool):
        raise ComparisonContractError("context.higher_is_better must be boolean.")
    if "incumbent_commit_hash" not in context:
        raise ComparisonContractError("context.incumbent_commit_hash is required.")
    incumbent = context["incumbent_commit_hash"]
    if incumbent is not None:
        _nonempty_string(incumbent, "context.incumbent_commit_hash")

    if not isinstance(payload, Mapping):
        raise ComparisonContractError("fresh_comparison must be an object.")
    try:
        evidence = _copy_json(payload, "fresh_comparison")
    except RecursionError as exc:
        raise ComparisonContractError(
            "fresh_comparison must be acyclic JSON within the supported nesting depth."
        ) from exc
    observed_id = _nonempty_string(evidence.get("context_id"), "context_id")
    if observed_id != context_id:
        raise ComparisonContractError("fresh_comparison context_id does not match.")
    if evidence.get("complete") is not True:
        raise ComparisonContractError("fresh_comparison complete must be true.")

    candidate = _candidate_value(metrics, objective, direction)
    if "candidate_value" in evidence:
        repeated = _positive(evidence["candidate_value"], "candidate_value")
        if repeated != candidate:
            raise ComparisonContractError(
                "candidate_value does not match the configured objective metric."
            )

    if incumbent is None:
        forbidden = {
            "incumbent_value",
            "improvement_lower_bound",
            "confidence_level",
            "sample_count",
        }.intersection(evidence)
        if forbidden:
            raise ComparisonContractError(
                "Empty-cell evidence must not contain comparison field(s): "
                + ", ".join(sorted(forbidden))
                + "."
            )
        return FreshComparisonDecision(
            allowed=True,
            context_id=context_id,
            candidate_value=candidate,
            incumbent_value=None,
            point_gain=None,
            improvement_lower_bound=None,
            confidence_level=None,
            sample_count=None,
            evidence=evidence,
        )

    incumbent_value = _positive(evidence.get("incumbent_value"), "incumbent_value")
    lower = _number(evidence.get("improvement_lower_bound"), "improvement_lower_bound")
    confidence = _confidence(evidence.get("confidence_level"), "confidence_level")
    if confidence < minimum:
        raise ComparisonContractError(
            f"confidence_level must be at least {minimum}."
        )
    count = evidence.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ComparisonContractError("sample_count must be an integer of at least 2.")

    numerator, denominator = (
        (candidate, incumbent_value)
        if direction
        else (incumbent_value, candidate)
    )
    ratio = numerator / denominator
    # Avoid overflow/underflow in a ratio of individually valid finite metrics.
    log_ratio = (
        math.log(ratio)
        if 0 < ratio < math.inf
        else math.log(numerator) - math.log(denominator)
    )
    point_gain = 100 * log_ratio
    if lower > point_gain:
        raise ComparisonContractError(
            "improvement_lower_bound exceeds the measured oriented log gain."
        )

    return FreshComparisonDecision(
        allowed=lower > 0,
        context_id=context_id,
        candidate_value=candidate,
        incumbent_value=incumbent_value,
        point_gain=point_gain,
        improvement_lower_bound=lower,
        confidence_level=confidence,
        sample_count=count,
        evidence=evidence,
    )


def _candidate_value(
    metrics: Sequence[Mapping[str, Any] | object] | Mapping[str, Any] | None,
    objective: str,
    direction: bool,
) -> float:
    if isinstance(metrics, Mapping):
        items = (metrics,)
    elif isinstance(metrics, Sequence) and not isinstance(metrics, (str, bytes)):
        items = metrics
    else:
        raise ComparisonContractError("metrics must contain the configured objective.")
    matches: list[Mapping[str, Any] | object] = []
    for item in items:
        name = item.get("name") if isinstance(item, Mapping) else getattr(item, "name", None)
        if name == objective:
            matches.append(item)
    if len(matches) != 1:
        raise ComparisonContractError(
            f"metrics must contain exactly one configured objective {objective!r}."
        )
    item = matches[0]
    if isinstance(item, Mapping):
        raw_value = item.get("value")
        observed_direction = item.get("higher_is_better")
    else:
        raw_value = getattr(item, "value", None)
        observed_direction = getattr(item, "higher_is_better", None)
    if not isinstance(observed_direction, bool) or observed_direction != direction:
        raise ComparisonContractError(
            "Configured objective metric direction does not match comparison context."
        )
    return _positive(raw_value, f"metric {objective!r} value")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComparisonContractError(f"{label} must be a non-empty string.")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonContractError(f"{label} must be a finite JSON number.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ComparisonContractError(f"{label} must be a finite JSON number.") from exc
    if not math.isfinite(result):
        raise ComparisonContractError(f"{label} must be a finite JSON number.")
    return result


def _positive(value: Any, label: str) -> float:
    result = _number(value, label)
    if result <= 0:
        raise ComparisonContractError(f"{label} must be positive.")
    return result


def _confidence(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0 < result < 1:
        raise ComparisonContractError(f"{label} must lie strictly between 0 and 1.")
    return result


def _copy_json(value: Any, label: str) -> Any:
    """Retain extension evidence without coercing invalid JSON into valid data."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        _number(value, label)
        return value
    if isinstance(value, Mapping):
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ComparisonContractError(f"{label} keys must be strings.")
            copied[key] = _copy_json(item, f"{label}.{key}")
        return copied
    if isinstance(value, list):
        return [_copy_json(item, f"{label}[{index}]") for index, item in enumerate(value)]
    raise ComparisonContractError(f"{label} must contain only JSON values.")
