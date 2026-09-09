"""Stable context and validation for fresh incumbent comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from loreley.config import Settings

from .snapshot import serialize_projection
from .types import IslandState


@dataclass(frozen=True, slots=True)
class ComparisonAdmission:
    """One candidate's metrics and the fresh comparison authorizing its fate."""

    commit_hash: str
    metrics: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None
    comparison_context: Mapping[str, Any]
    replacement_allowed: bool


class ComparisonContextError(ValueError):
    """A comparison cannot be interpreted under the current archive contract."""


class StaleComparisonError(ComparisonContextError):
    """The candidate's compared cell or incumbent has changed."""


def projection_fingerprint(state: IslandState, settings: Settings) -> str:
    """Bind the projection and coordinate system, not just its epoch number."""
    payload = {
        "projection": serialize_projection(state.projection),
        "dims": list(state.archive.dims),
        "ranges": [list(bounds) for bounds in state.archive.ranges],
        "feature_clip": bool(settings.mapelites_feature_clip),
        "feature_truncation_k": float(settings.mapelites_feature_truncation_k),
        "normalize_input": bool(settings.mapelites_dimensionality_penultimate_normalize),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_comparison_context(
    expected: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    """Reject stale or malformed contexts before archive mutation.

    Additional evaluator-protocol fields may be retained by the caller; this
    validates the archive-owned fields only.
    """
    _validate_identity(expected, current)
    _validate_target(expected, current)
    _validate_measures(expected, current)


def _validate_identity(expected: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Keep candidate identity and metric interpretation bound to the run."""
    for key in ("candidate_commit_hash", "island_id", "objective_name", "projection_fingerprint"):
        value = expected.get(key)
        if not isinstance(value, str) or not value or value != current[key]:
            raise ComparisonContextError(f"Comparison context mismatch: {key}.")
    if type(expected.get("higher_is_better")) is not bool:
        raise ComparisonContextError("Comparison metric direction must be boolean.")
    if expected["higher_is_better"] != current["higher_is_better"]:
        raise ComparisonContextError("Comparison metric direction changed.")


def _validate_target(expected: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Require the same cell and the same currently retained competitor."""
    cell = expected.get("cell_index")
    if type(cell) is not int or cell != current["cell_index"]:
        raise StaleComparisonError("Candidate no longer belongs to the compared cell.")
    if "incumbent_commit_hash" not in expected:
        raise ComparisonContextError("Comparison context is missing its incumbent.")
    incumbent = expected["incumbent_commit_hash"]
    if incumbent is not None and (not isinstance(incumbent, str) or not incumbent):
        raise ComparisonContextError("Comparison incumbent must be a commit hash or null.")
    if incumbent != current["incumbent_commit_hash"]:
        raise StaleComparisonError("Compared incumbent is no longer current.")


def _validate_measures(expected: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Reject descriptor drift even when both vectors map to the same cell."""
    try:
        measures = np.asarray(expected.get("measures"), dtype=np.float64)
        current_measures = np.asarray(current["measures"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ComparisonContextError("Comparison measures must be a numeric vector.") from exc
    if (
        measures.shape != current_measures.shape
        or not np.all(np.isfinite(measures))
        or not np.allclose(measures, current_measures, rtol=0.0, atol=1e-12)
    ):
        raise StaleComparisonError("Candidate behavior measures changed after comparison.")
