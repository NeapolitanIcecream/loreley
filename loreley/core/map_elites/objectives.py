"""Ordered multi-objective contracts and evaluator metric validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "ObjectiveContract",
    "ObjectiveContractError",
    "ObjectiveSpec",
    "ResolvedObjectives",
    "parse_higher_is_better",
]

ObjectiveDirection = Literal["max", "min"]


class ObjectiveContractError(ValueError):
    """Raised when configured objectives and evaluated metrics disagree."""


class ObjectiveSpec(BaseModel):
    """One named optimization objective in its persisted vector order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    direction: ObjectiveDirection

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("Objective name cannot be empty.")
        if len(normalized) > 128:
            raise ValueError("Objective name cannot exceed 128 characters.")
        return normalized

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "max"


@dataclass(slots=True, frozen=True)
class ResolvedObjectives:
    """Raw objective values and their all-maximized dominance scores."""

    values: tuple[float, ...]
    scores: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class ObjectiveContract:
    """Stable ordered contract used for validation, persistence, and dominance."""

    specs: tuple[ObjectiveSpec, ...]

    def __init__(self, specs: Sequence[ObjectiveSpec | Mapping[str, Any]]) -> None:
        parsed = tuple(
            spec if isinstance(spec, ObjectiveSpec) else ObjectiveSpec.model_validate(spec)
            for spec in specs
        )
        if not parsed:
            raise ObjectiveContractError("At least one objective is required.")
        names = tuple(spec.name for spec in parsed)
        if len(set(names)) != len(names):
            raise ObjectiveContractError("Objective names must be unique.")
        object.__setattr__(self, "specs", parsed)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def primary(self) -> ObjectiveSpec:
        return self.specs[0]

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.as_payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def as_payload(self) -> list[dict[str, str]]:
        return [
            {"name": spec.name, "direction": spec.direction}
            for spec in self.specs
        ]

    def resolve(
        self,
        metrics: Sequence[Mapping[str, Any] | object] | Mapping[str, Any] | None,
    ) -> ResolvedObjectives:
        """Validate evaluator metrics and return values in contract order."""

        metric_items = _coerce_metric_items(metrics)
        metrics_by_name: dict[str, tuple[float, bool]] = {}
        for item in metric_items:
            name, value, higher_is_better = _parse_metric(item)
            if name in metrics_by_name:
                raise ObjectiveContractError(f"Duplicate metric {name!r}.")
            metrics_by_name[name] = (value, higher_is_better)

        values: list[float] = []
        scores: list[float] = []
        for spec in self.specs:
            resolved = metrics_by_name.get(spec.name)
            if resolved is None:
                raise ObjectiveContractError(
                    f"Missing configured objective {spec.name!r}."
                )
            value, higher_is_better = resolved
            if higher_is_better != spec.higher_is_better:
                observed = "max" if higher_is_better else "min"
                raise ObjectiveContractError(
                    "Metric direction conflicts with the objective contract "
                    f"(name={spec.name!r} expected={spec.direction!r} observed={observed!r})."
                )
            values.append(value)
            scores.append(value if spec.higher_is_better else -value)
        return ResolvedObjectives(values=tuple(values), scores=tuple(scores))

    def resolve_values(self, values: Sequence[Any]) -> ResolvedObjectives:
        """Validate a persisted raw objective vector and derive dominance scores."""

        if len(values) != len(self.specs):
            raise ObjectiveContractError(
                "Objective vector length does not match the configured contract "
                f"(expected={len(self.specs)} observed={len(values)})."
            )
        resolved_values: list[float] = []
        scores: list[float] = []
        for spec, raw_value in zip(self.specs, values):
            if isinstance(raw_value, bool):
                raise ObjectiveContractError(
                    f"Objective {spec.name!r} value cannot be boolean."
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ObjectiveContractError(
                    f"Objective {spec.name!r} value must be numeric."
                ) from exc
            if not math.isfinite(value):
                raise ObjectiveContractError(
                    f"Objective {spec.name!r} value must be finite."
                )
            resolved_values.append(value)
            scores.append(value if spec.higher_is_better else -value)
        return ResolvedObjectives(
            values=tuple(resolved_values),
            scores=tuple(scores),
        )


def parse_higher_is_better(value: Any, *, default: bool | None = None) -> bool:
    """Parse evaluator direction without Python truthiness coercion."""

    if value is None:
        if default is not None:
            return bool(default)
        raise ObjectiveContractError("Metric direction must be explicit.")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    raise ObjectiveContractError(
        "Metric 'higher_is_better' must be a boolean or one of "
        "'true', 'false', '1', or '0'."
    )


def _coerce_metric_items(
    metrics: Sequence[Mapping[str, Any] | object] | Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | object, ...]:
    if metrics is None:
        return ()
    if isinstance(metrics, Mapping):
        if "name" in metrics or "value" in metrics:
            return (metrics,)
        return tuple(
            {
                "name": name,
                "value": value,
                "higher_is_better": None,
            }
            for name, value in metrics.items()
        )
    return tuple(metrics)


def _parse_metric(item: Mapping[str, Any] | object) -> tuple[str, float, bool]:
    if isinstance(item, Mapping):
        name_raw = item.get("name") or item.get("metric") or item.get("key")
        has_value = "value" in item
        value_raw = item.get("value")
        direction_raw = item.get("higher_is_better")
    else:
        name_raw = getattr(item, "name", None)
        has_value = hasattr(item, "value")
        value_raw = getattr(item, "value", None)
        direction_raw = getattr(item, "higher_is_better", None)

    name = str(name_raw or "").strip()
    if not name:
        raise ObjectiveContractError("Metric name cannot be empty.")
    if not has_value:
        raise ObjectiveContractError(f"Metric {name!r} must include a value.")
    if isinstance(value_raw, bool):
        raise ObjectiveContractError(f"Metric {name!r} value cannot be boolean.")
    try:
        value = float(value_raw)
    except (TypeError, ValueError) as exc:
        raise ObjectiveContractError(f"Metric {name!r} value must be numeric.") from exc
    if not math.isfinite(value):
        raise ObjectiveContractError(f"Metric {name!r} value must be finite.")
    higher_is_better = parse_higher_is_better(direction_raw)
    return name, value, higher_is_better
