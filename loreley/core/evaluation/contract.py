"""Canonical, hashable contracts for evaluator measurement protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping

from .adaptive import AdaptiveSamplingConfig
from .estimates import HoeffdingConfidenceSequence, IntervalMethod, StudentTInterval

__all__ = ["MeasurementContract"]


@dataclass(frozen=True, slots=True)
class MeasurementContract:
    """Project-neutral description of the evidence a measurement represents.

    Evaluators pass :attr:`fingerprint` through
    ``EvaluationPreparation.measurement_contract_fingerprint``. Any change to
    workload, estimand, inference, strata, budget, or target-owned metadata then
    produces a cache miss instead of silently reusing stale measurements.
    """

    workload_fingerprint: str
    metric_name: str
    metric_unit: str
    estimand: str
    higher_is_better: bool
    interval_method: IntervalMethod
    sampling: AdaptiveSamplingConfig
    stratum_weights: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("MeasurementContract schema_version must be 1.")
        for name in ("workload_fingerprint", "metric_name", "metric_unit", "estimand"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"MeasurementContract {name} must not be empty.")
            if len(value) > 512:
                raise ValueError(f"MeasurementContract {name} cannot exceed 512 characters.")
            object.__setattr__(self, name, value)
        if not isinstance(self.higher_is_better, bool):
            raise ValueError("MeasurementContract higher_is_better must be bool.")
        sampling_weights = _canonical_weights(self.sampling.stratum_weights)
        weights = _canonical_weights(self.stratum_weights)
        if weights and weights != sampling_weights:
            raise ValueError(
                "MeasurementContract stratum_weights must match the adaptive sampling contract."
            )
        if not weights:
            weights = sampling_weights
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, label="metadata"))
        # Fail early for custom methods that do not provide stable public identity.
        _interval_payload(self.interval_method)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workload_fingerprint": self.workload_fingerprint,
            "metric_name": self.metric_name,
            "metric_unit": self.metric_unit,
            "estimand": self.estimand,
            "higher_is_better": self.higher_is_better,
            "interval": _interval_payload(self.interval_method),
            "sampling": self.sampling.as_dict(),
            "stratum_weights": dict(self.stratum_weights),
            "metadata": dict(self.metadata),
        }


def _interval_payload(method: IntervalMethod) -> dict[str, Any]:
    name = str(getattr(method, "name", "") or "").strip()
    capability = getattr(getattr(method, "capability", None), "value", None)
    if not name or capability not in {"fixed_sample_only", "anytime_valid"}:
        raise ValueError("Interval method must expose stable name and capability values.")
    parameters: dict[str, Any]
    if isinstance(method, StudentTInterval):
        parameters = {
            "assumptions": "independent_identically_distributed_normal_samples",
        }
    elif isinstance(method, HoeffdingConfidenceSequence):
        parameters = {
            "alpha_spending": "6/(pi^2*look_index^2)",
            "lower_bound": method.lower_bound,
            "upper_bound": method.upper_bound,
        }
    else:
        contract = getattr(method, "contract", None)
        if not callable(contract):
            raise ValueError(
                "Custom interval methods must expose contract() with JSON parameters."
            )
        parameters = _json_mapping(contract(), label="interval contract")
    return {
        "name": name,
        "capability": capability,
        "parameters": parameters,
    }


def _canonical_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(weights, Mapping):
        raise ValueError("MeasurementContract stratum_weights must be a mapping.")
    if not weights:
        return {}
    normalized: dict[str, float] = {}
    for raw_name, raw_weight in weights.items():
        name = str(raw_name).strip()
        if not name or isinstance(raw_weight, bool):
            raise ValueError("MeasurementContract strata need named, positive weights.")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("MeasurementContract strata need named, positive weights.")
        normalized[name] = weight
    total = math.fsum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("MeasurementContract stratum_weights must sum to 1.0.")
    return {name: normalized[name] for name in sorted(normalized)}


def _json_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"MeasurementContract {label} must be a mapping.")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MeasurementContract {label} must contain JSON values.") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - mapping round-trip invariant
        raise ValueError(f"MeasurementContract {label} must be a JSON object.")
    return decoded
