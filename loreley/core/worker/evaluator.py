from __future__ import annotations

import hashlib
import inspect
import json
import math
import multiprocessing
from multiprocessing.connection import Connection, wait as wait_for_connections
import os
import re
import signal
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, cast

from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.map_elites.objectives import (
    ObjectiveContractError,
    parse_higher_is_better,
)
from loreley.core.worker.evaluator_identity import evaluator_identity_version

console = Console()
log = logger.bind(module="worker.evaluator")

__all__ = [
    "ArtifactValidationWarning",
    "EvalFail",
    "EvalPass",
    "EvaluationArtifact",
    "EvaluationContext",
    "EvaluationDiagnostic",
    "EvaluationError",
    "EvaluationFailureResult",
    "EvaluationMetric",
    "EvaluationMeasurement",
    "EvaluationOutcome",
    "EvaluationPreparation",
    "EvaluationPlugin",
    "EvaluationResult",
    "Evaluator",
    "MeasurementEvidence",
    "MeasurementProvenance",
    "PhasedEvaluationPlugin",
    "coerce_evaluation_artifacts",
    "eval_fail_kind_from_failure_kind",
]

ArtifactVisibility = Literal["agent_visible", "human_only", "hidden"]
ArtifactAgentProjection = Literal["summary", "manifest", "path"]
ArtifactWarningAction = Literal["skipped", "downgraded", "metadata_only"]
EvaluationOutcomeKind = Literal[
    "passed",
    "candidate_failed",
    "evaluator_failed",
    "infrastructure_failed",
    "inconclusive",
]
EvaluationRepairability = Literal["repairable", "not_repairable", "unknown"]
EvaluationConcurrencyScope = Literal["whole", "measurement"]
EvalFailKind = Literal[
    "compile",
    "typecheck",
    "lint",
    "test",
    "validation",
    "benchmark",
    "other",
]
_ARTIFACT_KEY_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_VALID_VISIBILITIES: set[str] = {"agent_visible", "human_only", "hidden"}
_VALID_AGENT_PROJECTIONS: set[str] = {"summary", "manifest", "path"}
_VALID_DIAGNOSTIC_SEVERITIES: set[str] = {
    "info",
    "warning",
    "error",
    "regression",
    "improvement",
}
_VALID_OUTCOME_KINDS: set[str] = {
    "passed",
    "candidate_failed",
    "evaluator_failed",
    "infrastructure_failed",
    "inconclusive",
}
_VALID_REPAIRABILITY: set[str] = {"repairable", "not_repairable", "unknown"}
_VALID_EVAL_FAIL_KINDS: set[str] = {
    "compile",
    "typecheck",
    "lint",
    "test",
    "validation",
    "benchmark",
    "other",
}
_EVAL_FAIL_KIND_TO_FAILURE_KIND: dict[str, str] = {
    "compile": "compile_failed",
    "typecheck": "typecheck_failed",
    "lint": "lint_failed",
    "test": "test_failed",
    "validation": "validation_failed",
    "benchmark": "benchmark_failed",
    "other": "other_failed",
}


@dataclass(frozen=True, slots=True)
class ArtifactValidationWarning:
    """Sanitized warning produced while accepting evaluator artifact declarations."""

    artifact_index: int | None
    artifact_key: str | None
    code: str
    action: ArtifactWarningAction
    message: str
    input_ref: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_index": self.artifact_index,
            "artifact_key": self.artifact_key,
            "code": self.code,
            "action": self.action,
            "message": self.message,
            "input_ref": self.input_ref,
        }


@dataclass(slots=True)
class EvaluationDiagnostic:
    """Bounded structured diagnostic finding emitted by an evaluator."""

    kind: str
    message: str
    severity: str = "info"
    location: str | None = None
    metric: str | None = None
    value: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        kind = clamp_text(normalize_single_line(str(self.kind or "")), 64)
        message = clamp_text(normalize_single_line(str(self.message or "")), 512)
        if not kind:
            raise ValueError("Evaluation diagnostic kind must be provided.")
        if not message:
            raise ValueError("Evaluation diagnostic message must be provided.")
        severity = normalize_single_line(str(self.severity or "info")).lower()
        if severity not in _VALID_DIAGNOSTIC_SEVERITIES:
            severity = "info"
        self.kind = kind
        self.message = message
        self.severity = severity
        self.location = _optional_bounded_line(self.location, 256)
        self.metric = _optional_bounded_line(self.metric, 128)
        if self.value is not None:
            if isinstance(self.value, bool):
                self.value = None
            else:
                try:
                    self.value = float(self.value)
                except (TypeError, ValueError):
                    self.value = None
        self.unit = _optional_bounded_line(self.unit, 32)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "severity": self.severity,
            "location": self.location,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
        }


@dataclass(slots=True)
class EvaluationArtifact:
    """Evaluator-declared diagnostic artifact metadata or raw payload reference."""

    key: str
    kind: str
    mime_type: str
    path: Path | str | None = None
    inline_payload: str | bytes | Mapping[str, Any] | Sequence[Any] | None = None
    label: str | None = None
    summary: str | None = None
    visibility: ArtifactVisibility = "human_only"
    agent_projection: ArtifactAgentProjection = "summary"
    diagnostics: tuple[EvaluationDiagnostic, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        key = _normalise_artifact_key(self.key)
        if not key:
            raise ValueError("Evaluation artifact key must be provided.")
        kind = clamp_text(normalize_single_line(str(self.kind or "")).lower(), 64)
        if not kind:
            raise ValueError("Evaluation artifact kind must be provided.")
        mime_type = clamp_text(normalize_single_line(str(self.mime_type or "")).lower(), 128)
        if not mime_type:
            raise ValueError("Evaluation artifact mime_type must be provided.")

        visibility = normalize_single_line(str(self.visibility or "human_only")).lower()
        if visibility not in _VALID_VISIBILITIES:
            raise ValueError("Evaluation artifact visibility is invalid.")
        projection = normalize_single_line(str(self.agent_projection or "summary")).lower()
        if projection not in _VALID_AGENT_PROJECTIONS:
            raise ValueError("Evaluation artifact agent_projection is invalid.")

        self.key = key
        self.kind = kind
        self.mime_type = mime_type
        self.label = _optional_bounded_line(self.label, 128)
        self.summary = _optional_bounded_line(self.summary, 1024)
        self.visibility = cast(ArtifactVisibility, visibility)
        self.agent_projection = cast(ArtifactAgentProjection, projection)
        self.diagnostics = _coerce_diagnostics(self.diagnostics, artifact_index=None)[0]
        self.metadata = _coerce_metadata(self.metadata)

    def public_manifest(self) -> dict[str, Any]:
        """Return a safe manifest for evaluation.json without local paths."""

        return {
            "key": self.key,
            "kind": self.kind,
            "mime_type": self.mime_type,
            "label": self.label,
            "summary": self.summary,
            "visibility": self.visibility,
            "agent_projection": self.agent_projection,
            "diagnostic_count": len(self.diagnostics),
        }


class EvaluationError(RuntimeError):
    """Raised when the evaluator cannot obtain a valid result."""


@dataclass(slots=True)
class EvaluationMetric:
    """Single metric returned by the evaluation plugin."""

    name: str
    value: float
    unit: str | None = None
    higher_is_better: bool = True
    details: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable representation."""
        return {
            "name": self.name,
            "value": float(self.value),
            "unit": self.unit,
            "higher_is_better": bool(self.higher_is_better),
            "details": dict(self.details or {}),
        }


EvalPassMetricInput = EvaluationMetric | Mapping[str, Any]
EvaluatorArtifactInput = EvaluationArtifact | Mapping[str, Any]


def _coerce_public_metrics(metrics_payload: Any) -> tuple[EvaluationMetric, ...]:
    return tuple(_coerce_public_metric(item) for item in _iter_public_metrics(metrics_payload))


def _iter_public_metrics(metrics_payload: Any) -> tuple[Any, ...]:
    if metrics_payload is None:
        return tuple()
    if isinstance(metrics_payload, EvaluationMetric):
        return (metrics_payload,)
    if isinstance(metrics_payload, Mapping):
        return (metrics_payload,)
    try:
        return tuple(metrics_payload)
    except TypeError as exc:
        raise ValueError("EvalPass metrics must be iterable.") from exc


def _coerce_public_metric(item: Any) -> EvaluationMetric:
    if isinstance(item, EvaluationMetric):
        return item
    if isinstance(item, Mapping):
        return _public_metric_from_mapping(item)
    raise ValueError(f"Unsupported EvalPass metric entry type: {type(item)!r}")


def _public_metric_from_mapping(item: Mapping[str, Any]) -> EvaluationMetric:
    name = str(item.get("name") or "").strip()
    if not name:
        raise ValueError("EvalPass metric entries must include a non-empty name.")
    if "value" not in item:
        raise ValueError("EvalPass metric entries must include a value.")
    raw_value = item["value"]
    if isinstance(raw_value, bool):
        raise ValueError("EvalPass metric values cannot be boolean.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("EvalPass metric values must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError("EvalPass metric values must be finite.")
    try:
        higher_is_better = parse_higher_is_better(
            item.get("higher_is_better"),
            default=True,
        )
    except ObjectiveContractError as exc:
        raise ValueError(str(exc)) from exc
    return EvaluationMetric(
        name=name,
        value=value,
        unit=str(item["unit"]).strip() if item.get("unit") is not None else None,
        higher_is_better=higher_is_better,
        details=dict(item.get("details") or {}),
    )


def _normalise_sequence_public(values: str | Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        values_iterable: Sequence[Any] = (values,)
    else:
        values_iterable = tuple(values)
    return tuple(str(value).strip() for value in values_iterable if str(value).strip())


@dataclass(slots=True)
class EvaluationContext:
    """Information shared with the evaluation plugin."""

    worktree: Path
    base_commit_hash: str | None = None
    candidate_commit_hash: str | None = None
    job_id: str | None = None
    goal: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    plan_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.worktree = Path(self.worktree).expanduser().resolve()
        self.payload = dict(self.payload or {})
        self.metadata = dict(self.metadata or {})


@dataclass(slots=True, frozen=True)
class MeasurementEvidence:
    """Hash-linked, location-free evidence for one cacheable measurement."""

    key: str
    sha256: str
    size_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = _normalise_artifact_key(self.key)
        digest = normalize_single_line(str(self.sha256 or "")).lower()
        if not key:
            raise ValueError("Measurement evidence key must be provided.")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Measurement evidence sha256 must be 64 lowercase hex characters.")
        size = self.size_bytes
        if size is not None and (isinstance(size, bool) or int(size) < 0):
            raise ValueError("Measurement evidence size_bytes must be non-negative.")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "size_bytes", int(size) if size is not None else None)
        object.__setattr__(self, "metadata", _coerce_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class EvaluationPreparation:
    """Source-specific preparation returned before any measurement reuse decision."""

    candidate_identity: str
    measurement_contract_fingerprint: str
    state: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[EvaluationArtifact, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        identity = normalize_single_line(str(self.candidate_identity or ""))
        fingerprint = normalize_single_line(str(self.measurement_contract_fingerprint or ""))
        if not identity:
            raise ValueError("Phased evaluator preparation must provide candidate_identity.")
        if len(identity) > 512:
            raise ValueError("Phased evaluator candidate_identity cannot exceed 512 characters.")
        if not fingerprint:
            raise ValueError(
                "Phased evaluator preparation must provide measurement_contract_fingerprint."
            )
        if len(fingerprint) > 512:
            raise ValueError(
                "Phased evaluator measurement_contract_fingerprint cannot exceed 512 characters."
            )
        self.candidate_identity = identity
        self.measurement_contract_fingerprint = fingerprint
        self.state = _json_mapping(self.state, label="preparation state")
        self.artifacts = coerce_evaluation_artifacts(self.artifacts)[0]


@dataclass(slots=True)
class EvaluationMeasurement:
    """Expensive evaluator output that may be reused across source candidates."""

    data: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[MeasurementEvidence, ...] = field(default_factory=tuple)
    artifacts: tuple[EvaluationArtifact, ...] = field(default_factory=tuple)
    cacheable: bool = False

    def __post_init__(self) -> None:
        self.data = _json_mapping(self.data, label="measurement data")
        self.evidence = tuple(
            item if isinstance(item, MeasurementEvidence) else MeasurementEvidence(**dict(item))
            for item in (self.evidence or ())
        )
        keys = [item.key for item in self.evidence]
        if len(keys) != len(set(keys)):
            raise ValueError("Measurement evidence keys must be unique.")
        self.cacheable = bool(self.cacheable)
        if self.cacheable and not self.evidence:
            raise ValueError("Cacheable measurements must include hash-linked evidence.")
        self.artifacts = coerce_evaluation_artifacts(self.artifacts)[0]

    def cache_payload(self) -> dict[str, Any]:
        return {
            "data": dict(self.data),
            "evidence": [item.as_dict() for item in self.evidence],
            "cacheable": bool(self.cacheable),
        }

    @classmethod
    def from_cache_payload(cls, payload: Mapping[str, Any]) -> "EvaluationMeasurement":
        return cls(
            data=dict(payload.get("data") or {}),
            evidence=tuple(payload.get("evidence") or ()),
            cacheable=bool(payload.get("cacheable", True)),
        )


@dataclass(slots=True, frozen=True)
class MeasurementProvenance:
    """Provenance supplied to phased finalization for new or reused measurement data."""

    cache_key: str
    reused: bool
    measurement_id: str | None = None
    source_evaluation_attempt_id: str | None = None
    evidence: tuple[MeasurementEvidence, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "reused": self.reused,
            "measurement_id": self.measurement_id,
            "source_evaluation_attempt_id": self.source_evaluation_attempt_id,
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass(slots=True)
class EvaluationResult:
    """Structured evaluation output."""

    summary: str
    metrics: tuple[EvaluationMetric, ...] = field(default_factory=tuple)
    tests_executed: tuple[str, ...] = field(default_factory=tuple)
    logs: tuple[str, ...] = field(default_factory=tuple)
    extra: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[EvaluationArtifact, ...] = field(default_factory=tuple)
    artifact_validation_warnings: tuple[ArtifactValidationWarning, ...] = field(default_factory=tuple)
    candidate_identity: str | None = None

    def __post_init__(self) -> None:
        summary = (self.summary or "").strip()
        if not summary:
            raise ValueError("Evaluation summary must be provided.")
        self.summary = summary
        self.candidate_identity = _optional_bounded_line(self.candidate_identity, 512)
        self.metrics = tuple(self.metrics or ())
        self.tests_executed = tuple(self.tests_executed or ())
        self.logs = tuple(self.logs or ())
        self.extra = dict(self.extra or {})
        artifacts, warnings = coerce_evaluation_artifacts(self.artifacts)
        self.artifacts = artifacts
        self.artifact_validation_warnings = tuple(self.artifact_validation_warnings or ()) + warnings


@dataclass(slots=True)
class EvalPass:
    """Simple public evaluator success result."""

    summary: str
    metrics: EvalPassMetricInput | Sequence[EvalPassMetricInput] | None = field(default_factory=tuple)
    tests_executed: str | Sequence[str] | None = field(default_factory=tuple)
    logs: str | Sequence[str] | None = field(default_factory=tuple)
    extra: dict[str, Any] | None = None
    artifacts: EvaluatorArtifactInput | Sequence[EvaluatorArtifactInput] | None = field(default_factory=tuple)
    candidate_identity: str | None = None

    def __post_init__(self) -> None:
        result = EvaluationResult(
            summary=self.summary,
            candidate_identity=self.candidate_identity,
            metrics=_coerce_public_metrics(self.metrics),
            tests_executed=_normalise_sequence_public(self.tests_executed),
            logs=_normalise_sequence_public(self.logs),
            extra=dict(self.extra or {}),
            artifacts=self.artifacts,
        )
        self.summary = result.summary
        self.candidate_identity = result.candidate_identity
        self.metrics = result.metrics
        self.tests_executed = result.tests_executed
        self.logs = result.logs
        self.extra = result.extra
        self.artifacts = result.artifacts

    def to_result(self) -> EvaluationResult:
        return EvaluationResult(
            summary=self.summary,
            candidate_identity=self.candidate_identity,
            metrics=self.metrics,
            tests_executed=self.tests_executed,
            logs=self.logs,
            extra=dict(self.extra or {}),
            artifacts=self.artifacts,
        )


@dataclass(slots=True)
class EvalFail:
    """Simple public evaluator candidate-failure result."""

    kind: EvalFailKind
    summary: str
    details: str | None = None
    artifacts: EvaluatorArtifactInput | Sequence[EvaluatorArtifactInput] | None = field(default_factory=tuple)

    def __post_init__(self) -> None:
        kind = normalize_single_line(str(self.kind or "")).lower()
        if kind not in _VALID_EVAL_FAIL_KINDS:
            raise ValueError(
                "EvalFail kind must be one of: "
                f"{', '.join(sorted(_VALID_EVAL_FAIL_KINDS))}."
            )
        self.kind = cast(EvalFailKind, kind)
        self.summary = (
            clamp_text(normalize_single_line(str(self.summary or "")), 4096)
            or "Evaluator reported a candidate failure without a bounded summary."
        )
        self.details = _optional_bounded_line(self.details, 4096)
        artifacts, _warnings = coerce_evaluation_artifacts(self.artifacts)
        self.artifacts = artifacts


@dataclass(slots=True)
class EvaluationFailureResult:
    """Structured evaluator-owned failure evidence for a non-passing candidate."""

    failure_stage: str
    failure_kind: str
    repairability: EvaluationRepairability = "unknown"
    repairability_reason: str | None = None
    safe_failure_summary: str = ""
    agent_visible_evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    human_only_artifact_refs: tuple[str, ...] = field(default_factory=tuple)
    hidden_artifact_refs: tuple[str, ...] = field(default_factory=tuple)
    exit_code: int | None = None
    timeout_seconds: int | None = None
    failing_tests_summary: str | None = None
    compiler_errors_summary: str | None = None
    stack_trace_summary: str | None = None
    policy_version: str = "diagnostic-capsule-v1"

    def __post_init__(self) -> None:
        self.failure_stage = _bounded_token(self.failure_stage, limit=32, default="unknown")
        self.failure_kind = _bounded_token(self.failure_kind, limit=64, default="unknown")
        repairability = normalize_single_line(str(self.repairability or "unknown")).lower()
        if repairability not in _VALID_REPAIRABILITY:
            repairability = "unknown"
        self.repairability = cast(EvaluationRepairability, repairability)
        self.repairability_reason = _optional_bounded_line(self.repairability_reason, 512)
        self.safe_failure_summary = (
            clamp_text(normalize_single_line(str(self.safe_failure_summary or "")), 4096)
            or "Evaluator reported a candidate failure without a bounded summary."
        )
        self.agent_visible_evidence_refs = _bounded_string_tuple(
            self.agent_visible_evidence_refs,
            limit=256,
            max_items=32,
        )
        self.human_only_artifact_refs = _bounded_string_tuple(
            self.human_only_artifact_refs,
            limit=256,
            max_items=32,
        )
        self.hidden_artifact_refs = _bounded_string_tuple(
            self.hidden_artifact_refs,
            limit=256,
            max_items=32,
        )
        self.exit_code = _optional_int(self.exit_code)
        self.timeout_seconds = _optional_int(self.timeout_seconds)
        self.failing_tests_summary = _optional_bounded_line(self.failing_tests_summary, 2048)
        self.compiler_errors_summary = _optional_bounded_line(self.compiler_errors_summary, 2048)
        self.stack_trace_summary = _optional_bounded_line(self.stack_trace_summary, 2048)
        self.policy_version = (
            clamp_text(normalize_single_line(str(self.policy_version or "")), 64)
            or "diagnostic-capsule-v1"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_stage": self.failure_stage,
            "failure_kind": self.failure_kind,
            "repairability": self.repairability,
            "repairability_reason": self.repairability_reason,
            "safe_failure_summary": self.safe_failure_summary,
            "agent_visible_evidence_refs": list(self.agent_visible_evidence_refs),
            "human_only_artifact_refs": list(self.human_only_artifact_refs),
            "hidden_artifact_refs": list(self.hidden_artifact_refs),
            "exit_code": self.exit_code,
            "timeout_seconds": self.timeout_seconds,
            "failing_tests_summary": self.failing_tests_summary,
            "compiler_errors_summary": self.compiler_errors_summary,
            "stack_trace_summary": self.stack_trace_summary,
            "policy_version": self.policy_version,
        }


@dataclass(slots=True)
class EvaluationOutcome:
    """First-class evaluator envelope for passed and failed evaluations."""

    schema_version: int = 1
    evaluator_name: str | None = None
    evaluator_version: str | None = None
    candidate_commit_hash: str | None = None
    prepared_candidate_identity: str | None = None
    outcome_kind: EvaluationOutcomeKind = "passed"
    result: EvaluationResult | None = None
    failure: EvaluationFailureResult | None = None
    artifacts: tuple[EvaluationArtifact, ...] = field(default_factory=tuple)
    artifact_validation_warnings: tuple[ArtifactValidationWarning, ...] = field(default_factory=tuple)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    protocol: str = "one_shot"
    measurement_cache_key: str | None = None
    measurement_contract_fingerprint: str | None = None
    measurement_reused: bool = False
    measurement_executed: bool = False
    reuse_kind: str = "none"
    measurement_id: str | None = None
    reused_from_attempt_id: str | None = None
    measurement_payload: dict[str, Any] | None = None
    measurement_evidence: tuple[MeasurementEvidence, ...] = field(default_factory=tuple)
    evaluator_slot: int | None = None
    evaluator_slot_scope: str | None = None
    evaluator_slot_wait_seconds: float | None = None
    evaluator_slot_acquired_at: datetime | None = None
    evaluator_slot_released_at: datetime | None = None
    evaluator_slot_lease_id: str | None = None
    evaluator_slot_release_reason: str | None = None
    persisted_attempt_id: str | None = None
    _runtime_leases: list[Any] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        outcome_kind = normalize_single_line(str(self.outcome_kind or "")).lower()
        if outcome_kind not in _VALID_OUTCOME_KINDS:
            outcome_kind = "inconclusive"
        self.outcome_kind = cast(EvaluationOutcomeKind, outcome_kind)
        self.schema_version = int(self.schema_version or 1)
        self.evaluator_name = _optional_bounded_line(self.evaluator_name, 128)
        self.evaluator_version = _optional_bounded_line(self.evaluator_version, 128)
        self.candidate_commit_hash = _optional_bounded_line(self.candidate_commit_hash, 64)
        self.prepared_candidate_identity = _optional_bounded_line(
            self.prepared_candidate_identity,
            512,
        )
        self.protocol = _bounded_token(self.protocol, limit=32, default="one_shot")
        self.measurement_cache_key = _optional_bounded_line(self.measurement_cache_key, 64)
        self.measurement_contract_fingerprint = _optional_bounded_line(
            self.measurement_contract_fingerprint,
            512,
        )
        self.measurement_reused = bool(self.measurement_reused)
        self.measurement_executed = bool(self.measurement_executed)
        self.reuse_kind = _bounded_token(self.reuse_kind, limit=32, default="none")
        self.measurement_id = _optional_bounded_line(self.measurement_id, 64)
        self.reused_from_attempt_id = _optional_bounded_line(self.reused_from_attempt_id, 64)
        self.measurement_payload = (
            _json_mapping(self.measurement_payload, label="measurement payload")
            if self.measurement_payload is not None
            else None
        )
        self.measurement_evidence = tuple(self.measurement_evidence or ())
        self.evaluator_slot_scope = _optional_bounded_line(self.evaluator_slot_scope, 32)
        self.evaluator_slot_acquired_at = _coerce_datetime(self.evaluator_slot_acquired_at)
        self.evaluator_slot_released_at = _coerce_datetime(self.evaluator_slot_released_at)
        self.evaluator_slot_lease_id = _optional_bounded_line(self.evaluator_slot_lease_id, 64)
        self.evaluator_slot_release_reason = _optional_bounded_line(
            self.evaluator_slot_release_reason,
            64,
        )
        self.persisted_attempt_id = _optional_bounded_line(self.persisted_attempt_id, 64)
        artifacts, warnings = coerce_evaluation_artifacts(self.artifacts)
        self.artifacts = artifacts
        self.artifact_validation_warnings = tuple(self.artifact_validation_warnings or ()) + warnings

        if self.outcome_kind == "passed":
            if self.result is None:
                raise ValueError("Passed evaluation outcomes must include a result.")
            self.failure = None
            return
        if self.failure is None:
            self.failure = EvaluationFailureResult(
                failure_stage="unknown",
                failure_kind="unknown",
                repairability="unknown",
                safe_failure_summary="Evaluator did not provide structured failure details.",
            )
        self.result = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "candidate_commit_hash": self.candidate_commit_hash,
            "prepared_candidate_identity": self.prepared_candidate_identity,
            "outcome_kind": self.outcome_kind,
            "result": _evaluation_result_dict(self.result) if self.result else None,
            "failure": self.failure.as_dict() if self.failure else None,
            "artifacts": [artifact.public_manifest() for artifact in self.artifacts],
            "artifact_validation_warnings": [
                warning.as_dict() for warning in self.artifact_validation_warnings
            ],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "protocol": self.protocol,
            "measurement": {
                "cache_key": self.measurement_cache_key,
                "contract_fingerprint": self.measurement_contract_fingerprint,
                "reused": self.measurement_reused,
                "executed": self.measurement_executed,
                "reuse_kind": self.reuse_kind,
                "measurement_id": self.measurement_id,
                "reused_from_attempt_id": self.reused_from_attempt_id,
                "evidence": [item.as_dict() for item in self.measurement_evidence],
            }
            if self.protocol == "phased-v1"
            else None,
            "evaluator_slot": {
                "slot": self.evaluator_slot,
                "scope": self.evaluator_slot_scope,
                "wait_seconds": self.evaluator_slot_wait_seconds,
                "acquired_at": (
                    self.evaluator_slot_acquired_at.isoformat()
                    if self.evaluator_slot_acquired_at
                    else None
                ),
                "released_at": (
                    self.evaluator_slot_released_at.isoformat()
                    if self.evaluator_slot_released_at
                    else None
                ),
                "lease_id": self.evaluator_slot_lease_id,
                "release_reason": self.evaluator_slot_release_reason,
            }
            if self.evaluator_slot_lease_id
            else None,
        }


@dataclass(slots=True, frozen=True)
class _OutcomeMappingInput:
    payload: Mapping[str, Any]
    context: EvaluationContext
    evaluator_name: str
    started_at: datetime
    finished_at: datetime
    artifacts: tuple[EvaluationArtifact, ...]
    artifact_warnings: tuple[ArtifactValidationWarning, ...]


class EvaluationPlugin(Protocol):
    """Protocol implemented by evaluation plugins."""

    def __call__(
        self,
        context: EvaluationContext,
    ) -> EvalPass | EvalFail | EvaluationResult | EvaluationOutcome | Mapping[str, Any]:
        ...


class PhasedEvaluationPlugin(Protocol):
    """Explicit opt-in protocol for source preparation and cacheable measurement."""

    evaluation_protocol: Literal["phased-v1"]
    evaluation_concurrency_scope: EvaluationConcurrencyScope

    def prepare(
        self,
        context: EvaluationContext,
    ) -> EvaluationPreparation | EvalFail | EvaluationOutcome:
        ...

    def measure(
        self,
        context: EvaluationContext,
        preparation: EvaluationPreparation,
    ) -> EvaluationMeasurement | EvalFail | EvaluationOutcome:
        ...

    def finalize(
        self,
        context: EvaluationContext,
        preparation: EvaluationPreparation,
        measurement: EvaluationMeasurement,
        provenance: MeasurementProvenance,
    ) -> EvalPass | EvalFail | EvaluationResult | EvaluationOutcome | Mapping[str, Any]:
        ...


EvaluationCallable = Callable[
    [EvaluationContext],
    EvalPass | EvalFail | EvaluationResult | EvaluationOutcome | Mapping[str, Any],
]


def coerce_evaluation_artifacts(
    payload: Any,
) -> tuple[tuple[EvaluationArtifact, ...], tuple[ArtifactValidationWarning, ...]]:
    """Coerce evaluator artifact payloads into typed artifacts and warnings.

    Invalid artifact declarations are skipped so the evaluation can still
    succeed. Warnings are intentionally sanitized and reference only low-cardinality
    fields such as list indexes and normalized keys.
    """

    if payload is None:
        return (), ()
    if isinstance(payload, EvaluationArtifact):
        candidates: tuple[Any, ...] = (payload,)
    elif isinstance(payload, Mapping):
        candidates = (payload,)
    else:
        try:
            candidates = tuple(payload)
        except TypeError:
            return (), (
                _artifact_warning(
                    artifact_index=None,
                    artifact_key=None,
                    code="artifacts_not_iterable",
                    action="skipped",
                    message="Evaluator artifacts must be an iterable of artifact declarations.",
                    input_ref="artifacts",
                ),
            )

    artifacts: list[EvaluationArtifact] = []
    warnings: list[ArtifactValidationWarning] = []
    seen_keys: set[str] = set()
    for index, item in enumerate(candidates):
        artifact: EvaluationArtifact | None = None
        if isinstance(item, EvaluationArtifact):
            artifact = item
        elif isinstance(item, Mapping):
            artifact, item_warnings = _artifact_from_mapping(item, index)
            warnings.extend(item_warnings)
        else:
            warnings.append(
                _artifact_warning(
                    artifact_index=index,
                    artifact_key=None,
                    code="invalid_artifact_type",
                    action="skipped",
                    message="Artifact declaration must be a mapping or EvaluationArtifact.",
                    input_ref=f"artifacts[{index}]",
                )
            )
            continue
        if artifact is None:
            continue
        if artifact.key in seen_keys:
            warnings.append(
                _artifact_warning(
                    artifact_index=index,
                    artifact_key=artifact.key,
                    code="duplicate_key",
                    action="skipped",
                    message="Artifact key duplicates another accepted artifact key.",
                    input_ref=f"artifacts[{index}].key",
                )
            )
            continue
        seen_keys.add(artifact.key)
        artifacts.append(artifact)
    return tuple(artifacts), tuple(warnings)


def eval_fail_kind_from_failure_kind(value: object) -> EvalFailKind | None:
    """Return the public EvalFail kind represented by an internal failure kind."""

    normalized = normalize_single_line(str(value or "")).lower()
    if normalized in _VALID_EVAL_FAIL_KINDS:
        return cast(EvalFailKind, normalized)
    if normalized.endswith("_failed"):
        candidate = normalized[: -len("_failed")]
        if candidate in _VALID_EVAL_FAIL_KINDS:
            return cast(EvalFailKind, candidate)
    return None


def _artifact_from_mapping(
    payload: Mapping[str, Any],
    artifact_index: int,
) -> tuple[EvaluationArtifact | None, tuple[ArtifactValidationWarning, ...]]:
    warnings: list[ArtifactValidationWarning] = []
    key = _normalise_artifact_key(payload.get("key"))
    if not key:
        return None, (
            _artifact_warning(
                artifact_index=artifact_index,
                artifact_key=None,
                code="missing_key",
                action="skipped",
                message="Artifact declaration is missing a valid key.",
                input_ref=f"artifacts[{artifact_index}].key",
            ),
        )
    diagnostics, diagnostic_warnings = _coerce_diagnostics(
        payload.get("diagnostics"),
        artifact_index=artifact_index,
    )
    warnings.extend(diagnostic_warnings)

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        warnings.append(
            _artifact_warning(
                artifact_index=artifact_index,
                artifact_key=key,
                code="invalid_metadata",
                action="downgraded",
                message="Artifact metadata must be a mapping and was omitted.",
                input_ref=f"artifacts[{artifact_index}].metadata",
            )
        )
        metadata = None

    try:
        raw_visibility = normalize_single_line(str(payload.get("visibility", "human_only"))).lower()
        raw_projection = normalize_single_line(str(payload.get("agent_projection", "summary"))).lower()
        artifact = EvaluationArtifact(
            key=key,
            kind=payload.get("kind", ""),
            mime_type=payload.get("mime_type", ""),
            path=payload.get("path"),
            inline_payload=payload.get("inline_payload"),
            label=payload.get("label"),
            summary=payload.get("summary"),
            visibility=cast(ArtifactVisibility, raw_visibility),
            agent_projection=cast(
                ArtifactAgentProjection,
                raw_projection,
            ),
            diagnostics=diagnostics,
            metadata=cast(Mapping[str, Any] | None, metadata),
        )
    except ValueError as exc:
        return None, tuple(
            [
                *warnings,
                _artifact_warning(
                    artifact_index=artifact_index,
                    artifact_key=key,
                    code="invalid_artifact",
                    action="skipped",
                    message=normalize_single_line(str(exc)) or "Artifact declaration is invalid.",
                    input_ref=f"artifacts[{artifact_index}]",
                ),
            ]
        )

    if (
        artifact.path is None
        and artifact.inline_payload is None
        and not artifact.summary
        and not artifact.diagnostics
    ):
        warnings.append(
            _artifact_warning(
                artifact_index=artifact_index,
                artifact_key=artifact.key,
                code="empty_metadata_artifact",
                action="skipped",
                message="Metadata-only artifacts must include a summary or diagnostics.",
                input_ref=f"artifacts[{artifact_index}]",
            )
        )
        return None, tuple(warnings)
    return artifact, tuple(warnings)


def _coerce_diagnostics(
    payload: Any,
    *,
    artifact_index: int | None,
) -> tuple[tuple[EvaluationDiagnostic, ...], tuple[ArtifactValidationWarning, ...]]:
    if payload is None:
        return (), ()
    if isinstance(payload, EvaluationDiagnostic):
        return (payload,), ()
    if isinstance(payload, Mapping):
        candidates: tuple[Any, ...] = (payload,)
    else:
        try:
            candidates = tuple(payload)
        except TypeError:
            return (), (
                _artifact_warning(
                    artifact_index=artifact_index,
                    artifact_key=None,
                    code="diagnostics_not_iterable",
                    action="downgraded",
                    message="Artifact diagnostics must be iterable and were omitted.",
                    input_ref=_artifact_input_ref(artifact_index, "diagnostics"),
                ),
            )

    diagnostics: list[EvaluationDiagnostic] = []
    warnings: list[ArtifactValidationWarning] = []
    for index, item in enumerate(candidates[:20]):
        if isinstance(item, EvaluationDiagnostic):
            diagnostics.append(item)
            continue
        if not isinstance(item, Mapping):
            warnings.append(
                _artifact_warning(
                    artifact_index=artifact_index,
                    artifact_key=None,
                    code="invalid_diagnostic_type",
                    action="downgraded",
                    message="Artifact diagnostic entry must be a mapping and was omitted.",
                    input_ref=_artifact_input_ref(artifact_index, f"diagnostics[{index}]"),
                )
            )
            continue
        try:
            diagnostics.append(
                EvaluationDiagnostic(
                    kind=item.get("kind", ""),
                    message=item.get("message", ""),
                    severity=str(item.get("severity", "info")),
                    location=cast(str | None, item.get("location")),
                    metric=cast(str | None, item.get("metric")),
                    value=cast(float | None, item.get("value")),
                    unit=cast(str | None, item.get("unit")),
                )
            )
        except ValueError:
            warnings.append(
                _artifact_warning(
                    artifact_index=artifact_index,
                    artifact_key=None,
                    code="invalid_diagnostic",
                    action="downgraded",
                    message="Artifact diagnostic entry is invalid and was omitted.",
                    input_ref=_artifact_input_ref(artifact_index, f"diagnostics[{index}]"),
                )
            )
    if len(candidates) > 20:
        warnings.append(
            _artifact_warning(
                artifact_index=artifact_index,
                artifact_key=None,
                code="too_many_diagnostics",
                action="downgraded",
                message="Artifact diagnostics were truncated to the maximum allowed count.",
                input_ref=_artifact_input_ref(artifact_index, "diagnostics"),
            )
        )
    return tuple(diagnostics), tuple(warnings)


def _normalise_artifact_key(raw: Any) -> str:
    value = normalize_single_line(str(raw or "")).lower()
    value = _ARTIFACT_KEY_PATTERN.sub("-", value).strip("._-")
    return clamp_text(value, 128)


def _bounded_token(value: Any, *, limit: int, default: str) -> str:
    raw = normalize_single_line(str(value or "")).lower()
    token = re.sub(r"[^a-z0-9_.-]+", "_", raw).strip("._-")
    return clamp_text(token, limit) or default


def _optional_bounded_line(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = clamp_text(normalize_single_line(str(value)), limit)
    return text or None


def _bounded_string_tuple(values: Any, *, limit: int, max_items: int) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        candidates: tuple[Any, ...] = (values,)
    else:
        try:
            candidates = tuple(values)
        except TypeError:
            candidates = (values,)
    result: list[str] = []
    for item in candidates[:max_items]:
        text = clamp_text(normalize_single_line(str(item or "")), limit)
        if text:
            result.append(text)
    return tuple(result)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _outcome_identity(mapping: _OutcomeMappingInput) -> dict[str, Any]:
    payload = mapping.payload
    return {
        "schema_version": int(payload.get("schema_version") or 1),
        "evaluator_name": (
            _optional_bounded_line(payload.get("evaluator_name"), 128)
            or mapping.evaluator_name
        ),
        "evaluator_version": _optional_bounded_line(payload.get("evaluator_version"), 128),
        "candidate_commit_hash": (
            _optional_bounded_line(payload.get("candidate_commit_hash"), 64)
            or mapping.context.candidate_commit_hash
        ),
        "started_at": _coerce_datetime(payload.get("started_at")) or mapping.started_at,
        "finished_at": _coerce_datetime(payload.get("finished_at")) or mapping.finished_at,
    }


def _evaluation_result_dict(result: EvaluationResult) -> dict[str, Any]:
    return {
        "summary": result.summary,
        "candidate_identity": result.candidate_identity,
        "metrics": [metric.as_dict() for metric in result.metrics],
        "tests_executed": list(result.tests_executed),
        "logs": list(result.logs),
        "extra": dict(result.extra or {}),
        "artifacts": [artifact.public_manifest() for artifact in result.artifacts],
        "artifact_validation_warnings": [
            warning.as_dict() for warning in result.artifact_validation_warnings
        ],
    }


def _coerce_metadata(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, Any] = {}
    for raw_key, raw_value in list(payload.items())[:32]:
        key = clamp_text(normalize_single_line(str(raw_key)), 64)
        if not key:
            continue
        result[key] = _bound_metadata_value(raw_value)
    return result


def _json_mapping(payload: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Phased evaluator {label} must be a mapping.")
    try:
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Phased evaluator {label} must contain only JSON values.") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - mapping round-trip invariant
        raise ValueError(f"Phased evaluator {label} must round-trip to a JSON object.")
    return decoded


def _merge_artifact_sets(*groups: Sequence[EvaluationArtifact]) -> tuple[EvaluationArtifact, ...]:
    merged: list[EvaluationArtifact] = []
    seen: set[str] = set()
    for group in groups:
        for artifact in group:
            if artifact.key in seen:
                continue
            seen.add(artifact.key)
            merged.append(artifact)
    return tuple(merged)


def _uncached_measurement_key(
    *,
    preparation: EvaluationPreparation,
    evaluator_name: object,
    evaluator_version: object,
    campaign_program_hash: object,
) -> str:
    payload = {
        "candidate_identity": preparation.candidate_identity,
        "measurement_contract_fingerprint": preparation.measurement_contract_fingerprint,
        "evaluator_name": str(evaluator_name or ""),
        "evaluator_version": str(evaluator_version or ""),
        "campaign_program_hash": str(campaign_program_hash or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bound_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return clamp_text(normalize_single_line(value), 512)
    if isinstance(value, Mapping):
        return _bound_metadata_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_bound_metadata_value(item) for item in list(value)[:16]]
    return clamp_text(normalize_single_line(str(value)), 512)


def _bound_metadata_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for raw_key, raw_value in list(value.items())[:16]:
        key = clamp_text(normalize_single_line(str(raw_key)), 64)
        if key:
            nested[key] = _bound_metadata_value(raw_value)
    return nested


def _artifact_warning(
    *,
    artifact_index: int | None,
    artifact_key: str | None,
    code: str,
    action: ArtifactWarningAction,
    message: str,
    input_ref: str | None,
) -> ArtifactValidationWarning:
    return ArtifactValidationWarning(
        artifact_index=artifact_index,
        artifact_key=_normalise_artifact_key(artifact_key) if artifact_key else None,
        code=clamp_text(normalize_single_line(code).lower(), 64) or "invalid_artifact",
        action=action,
        message=clamp_text(normalize_single_line(message), 240),
        input_ref=clamp_text(normalize_single_line(input_ref or ""), 128) or None,
    )


def _artifact_input_ref(artifact_index: int | None, suffix: str) -> str:
    if artifact_index is None:
        return suffix
    suffix = suffix.strip(".")
    return f"artifacts[{artifact_index}].{suffix}" if suffix else f"artifacts[{artifact_index}]"


class Evaluator:
    """Adapter around user-defined evaluation plugins."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        plugin: EvaluationCallable | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.plugin_ref = self.settings.worker_evaluator_plugin
        self.python_paths = tuple(
            Path(entry).expanduser().resolve()
            for entry in self.settings.worker_evaluator_python_paths
        )
        self.timeout = max(1, self.settings.worker_evaluator_timeout_seconds)
        self.max_metrics = max(1, self.settings.worker_evaluator_max_metrics)
        self.evaluator_version = evaluator_identity_version(
            plugin_ref=self.plugin_ref,
            explicit_version=self.settings.worker_evaluator_version,
            python_paths=self.settings.worker_evaluator_python_paths,
        )
        self._plugin_callable: EvaluationCallable | None = plugin
        self._plugin_target: Any | None = plugin
        self._pythonpath_ready = False

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Execute the configured plugin and return the success-only result."""
        outcome = self.evaluate_outcome(context)
        if outcome.outcome_kind == "passed" and outcome.result is not None:
            return outcome.result
        failure = outcome.failure
        message = (
            failure.safe_failure_summary
            if failure is not None
            else f"Evaluator returned outcome_kind={outcome.outcome_kind}."
        )
        raise EvaluationError(message)

    @property
    def evaluator_name(self) -> str:
        """Return the public plugin identity used by cache and provenance contracts."""

        return self._evaluator_label()

    def evaluate_outcome(self, context: EvaluationContext) -> EvaluationOutcome:
        """Execute the configured plugin and return a first-class outcome envelope."""
        if self.supports_phased_evaluation():
            return self.evaluate_phased_uncached(context)
        started_at = datetime.now(timezone.utc)
        self._validate_context(context)
        plugin = self._ensure_callable()
        label = self.plugin_ref or getattr(plugin, "__name__", "<callable>")
        console.log(
            f"[cyan]Evaluator[/] running plugin {label} "
            f"(job={context.job_id or 'N/A'} commit={context.candidate_commit_hash or 'N/A'})",
        )

        start = monotonic()
        try:
            payload = self._execute_with_timeout(plugin, context)
            outcome = self._coerce_outcome(
                payload,
                context=context,
                evaluator_name=label,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            outcome.evaluator_version = outcome.evaluator_version or self.evaluator_version
        except EvaluationError as exc:
            outcome = self._synthetic_failure_outcome(
                context=context,
                evaluator_name=label,
                message=str(exc),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        duration = monotonic() - start
        if outcome.result is not None:
            outcome.result.extra.setdefault("evaluator_duration_seconds", float(duration))
            _record_campaign_primary_metric_warnings(outcome=outcome, context=context)
        metrics_count = len(outcome.result.metrics) if outcome.result is not None else 0
        console.log(
            f"[bold green]Evaluator[/] finished in {duration:.1f}s "
            f"outcome={outcome.outcome_kind} metrics={metrics_count}",
        )
        log.info(
            "Evaluation completed job={} commit={} outcome_kind={} evaluator={} metrics={}",
            context.job_id,
            context.candidate_commit_hash,
            outcome.outcome_kind,
            outcome.evaluator_name,
            metrics_count,
        )
        return outcome

    def supports_phased_evaluation(self) -> bool:
        """Return whether the configured plugin explicitly opts into phased-v1."""

        target = self._ensure_plugin_target()
        protocol = normalize_single_line(str(getattr(target, "evaluation_protocol", ""))).lower()
        if protocol != "phased-v1":
            return False
        missing = [
            name
            for name in ("prepare", "measure", "finalize")
            if not callable(getattr(target, name, None))
        ]
        if missing:
            raise EvaluationError(
                "phased-v1 evaluator is missing callable method(s): " + ", ".join(missing)
            )
        return True

    def evaluation_concurrency_scope(self) -> EvaluationConcurrencyScope:
        """Return the plugin-declared scope governed by evaluator capacity E."""

        if not self.supports_phased_evaluation():
            return "whole"
        raw = normalize_single_line(
            str(getattr(self._ensure_plugin_target(), "evaluation_concurrency_scope", "measurement"))
        ).lower()
        if raw not in {"whole", "measurement"}:
            raise EvaluationError(
                "phased-v1 evaluation_concurrency_scope must be 'whole' or 'measurement'."
            )
        return cast(EvaluationConcurrencyScope, raw)

    def provides_candidate_identity(self) -> bool:
        """Return whether the evaluator contract promises a stable identity."""

        if self.supports_phased_evaluation():
            return True
        return bool(getattr(self._ensure_plugin_target(), "provides_candidate_identity", False))

    def new_deadline(self) -> float:
        """Return a monotonic deadline shared by all phases and capacity waits."""

        return monotonic() + float(self.timeout)

    def prepare_phase(
        self,
        context: EvaluationContext,
        *,
        deadline: float,
    ) -> EvaluationPreparation | EvaluationOutcome:
        """Run and validate phased source preparation."""

        started_at = datetime.now(timezone.utc)
        payload = self._execute_phase_with_deadline(
            "prepare",
            context,
            (),
            deadline=deadline,
        )
        if isinstance(payload, EvaluationPreparation):
            self._validate_json_size(payload.state, label="preparation state")
            return payload
        return self._coerce_phase_failure(
            payload,
            context=context,
            phase="prepare",
            started_at=started_at,
        )

    def measure_phase(
        self,
        context: EvaluationContext,
        preparation: EvaluationPreparation,
        *,
        deadline: float,
    ) -> EvaluationMeasurement | EvaluationOutcome:
        """Run and validate the expensive measurement phase."""

        started_at = datetime.now(timezone.utc)
        payload = self._execute_phase_with_deadline(
            "measure",
            context,
            (preparation,),
            deadline=deadline,
        )
        if isinstance(payload, EvaluationMeasurement):
            self._validate_json_size(payload.cache_payload(), label="measurement payload")
            return payload
        return self._coerce_phase_failure(
            payload,
            context=context,
            phase="measure",
            started_at=started_at,
        )

    def finalize_phase(
        self,
        context: EvaluationContext,
        preparation: EvaluationPreparation,
        measurement: EvaluationMeasurement,
        provenance: MeasurementProvenance,
        *,
        deadline: float,
        started_at: datetime | None = None,
    ) -> EvaluationOutcome:
        """Run source-specific finalization and return the normal outcome envelope."""

        effective_started = started_at or datetime.now(timezone.utc)
        payload = self._execute_phase_with_deadline(
            "finalize",
            context,
            (preparation, measurement, provenance),
            deadline=deadline,
        )
        outcome = self._coerce_outcome(
            payload,
            context=context,
            evaluator_name=self._evaluator_label(),
            started_at=effective_started,
            finished_at=datetime.now(timezone.utc),
        )
        outcome.evaluator_version = outcome.evaluator_version or self.evaluator_version
        outcome.protocol = "phased-v1"
        outcome.measurement_cache_key = provenance.cache_key
        outcome.measurement_contract_fingerprint = preparation.measurement_contract_fingerprint
        outcome.measurement_reused = provenance.reused
        outcome.measurement_id = provenance.measurement_id
        outcome.reused_from_attempt_id = provenance.source_evaluation_attempt_id
        outcome.measurement_payload = measurement.cache_payload()
        outcome.measurement_evidence = tuple(measurement.evidence)
        outcome.prepared_candidate_identity = preparation.candidate_identity
        outcome.artifacts = _merge_artifact_sets(
            preparation.artifacts,
            measurement.artifacts,
            outcome.artifacts,
        )
        if outcome.result is not None:
            final_identity = normalize_single_line(str(outcome.result.candidate_identity or ""))
            if final_identity and final_identity != preparation.candidate_identity:
                raise EvaluationError(
                    "Phased evaluator finalize changed candidate_identity from prepare."
                )
            outcome.result.candidate_identity = preparation.candidate_identity
            outcome.result.extra.setdefault("measurement_provenance", provenance.as_dict())
        return outcome

    def evaluate_phased_uncached(self, context: EvaluationContext) -> EvaluationOutcome:
        """Run phased-v1 end to end without a persistent measurement cache."""

        self._validate_context(context)
        deadline = self.new_deadline()
        started_at = datetime.now(timezone.utc)
        try:
            preparation = self.prepare_phase(context, deadline=deadline)
            if isinstance(preparation, EvaluationOutcome):
                return preparation
            cache_key = _uncached_measurement_key(
                preparation=preparation,
                evaluator_name=self._evaluator_label(),
                evaluator_version=self.evaluator_version,
                campaign_program_hash=context.metadata.get("campaign_program_hash"),
            )
            measurement = self.measure_phase(context, preparation, deadline=deadline)
            if isinstance(measurement, EvaluationOutcome):
                measurement.measurement_cache_key = cache_key
                measurement.measurement_contract_fingerprint = (
                    preparation.measurement_contract_fingerprint
                )
                measurement.prepared_candidate_identity = preparation.candidate_identity
                measurement.measurement_executed = True
                measurement.reuse_kind = "none"
                return measurement
            outcome = self.finalize_phase(
                context,
                preparation,
                measurement,
                MeasurementProvenance(
                    cache_key=cache_key,
                    reused=False,
                    evidence=measurement.evidence,
                ),
                deadline=deadline,
                started_at=started_at,
            )
            outcome.measurement_executed = True
            outcome.reuse_kind = "none"
            return outcome
        except EvaluationError as exc:
            return self._synthetic_failure_outcome(
                context=context,
                evaluator_name=self._evaluator_label(),
                message=str(exc),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )

    # Internal helpers -------------------------------------------------

    def _ensure_callable(self) -> EvaluationCallable:
        if self._plugin_callable:
            return self._plugin_callable

        target = self._ensure_plugin_target()
        if normalize_single_line(str(getattr(target, "evaluation_protocol", ""))).lower() == "phased-v1":
            raise EvaluationError("phased-v1 evaluator cannot be resolved as a one-shot callable.")
        callable_plugin = self._resolve_callable(target)
        self._plugin_callable = callable_plugin
        return callable_plugin

    def _ensure_plugin_target(self) -> Any:
        if self._plugin_target is not None:
            target = self._plugin_target
        elif self._plugin_callable is not None:
            target = self._plugin_callable
            self._plugin_target = target
        else:
            if not self.plugin_ref:
                raise EvaluationError(
                    "WORKER_EVALUATOR_PLUGIN is not configured. "
                    "Provide a dotted path to an evaluator plugin.",
                )
            self._prepare_pythonpath()
            module_name, attr_path = self._split_reference(self.plugin_ref)
            target = self._import_object(module_name, attr_path)
            if inspect.isclass(target):
                target = target()
            self._plugin_target = target
        return target

    def _evaluator_label(self) -> str:
        target = self._ensure_plugin_target()
        return str(self.plugin_ref or getattr(target, "__name__", target.__class__.__name__))

    def _prepare_pythonpath(self) -> None:
        if self._pythonpath_ready:
            return
        for entry in self.python_paths:
            entry_str = str(entry)
            if entry_str not in sys.path:
                sys.path.insert(0, entry_str)
        self._pythonpath_ready = True

    @staticmethod
    def _import_object(module_name: str, attr_path: str) -> Any:
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            raise EvaluationError(
                f"Could not import evaluator module {module_name!r}.",
            ) from exc
        target: Any = module
        for part in attr_path.split("."):
            if not part:
                raise EvaluationError(
                    f"Invalid evaluator attribute reference {attr_path!r}.",
                )
            try:
                target = getattr(target, part)
            except AttributeError as exc:
                raise EvaluationError(
                    f"Module {module_name!r} does not expose attribute {attr_path!r}.",
                ) from exc
        return target

    @staticmethod
    def _split_reference(ref: str) -> tuple[str, str]:
        if ":" in ref:
            module_name, attr_name = ref.split(":", 1)
            return module_name, attr_name
        module_name, _, attr_name = ref.rpartition(".")
        if not module_name or not attr_name:
            raise EvaluationError(
                f"Invalid evaluator reference {ref!r}. "
                "Use 'module:attr' or 'module.attr'.",
            )
        return module_name, attr_name

    @staticmethod
    def _resolve_callable(target: Any) -> EvaluationCallable:
        candidate = target
        if inspect.isclass(candidate):
            instance = candidate()
            return Evaluator._resolve_callable(instance)
        if hasattr(candidate, "evaluate") and callable(candidate.evaluate):
            return candidate.evaluate  # type: ignore[return-value]
        if callable(candidate):
            return cast(EvaluationCallable, candidate)
        raise EvaluationError(
            "Evaluator plugin must be callable or expose an 'evaluate' method.",
        )

    def _execute_with_timeout(
        self,
        plugin: EvaluationCallable,
        context: EvaluationContext,
    ) -> Any:
        ctx = multiprocessing.get_context("spawn")
        result_receiver, result_sender = ctx.Pipe(duplex=False)
        child_watch, parent_watch = ctx.Pipe(duplex=False)
        inline_callable = None if self.plugin_ref else plugin
        process = ctx.Process(
            target=_plugin_subprocess_entry,
            args=(
                self.plugin_ref,
                inline_callable,
                tuple(str(path) for path in self.python_paths),
                context,
                result_sender,
                child_watch,
            ),
        )
        process.start()
        result_sender.close()
        child_watch.close()
        try:
            status, payload = _receive_plugin_process_result(
                process,
                result_receiver,
                timeout=float(self.timeout),
                timeout_message=f"Evaluation plugin timed out after {self.timeout}s.",
                no_result_message="Evaluation plugin did not return any result.",
                unreadable_message="Evaluation plugin returned an unreadable payload.",
                exit_label="Evaluation plugin",
            )
        finally:
            result_receiver.close()
            parent_watch.close()

        if status == "ok":
            return payload

        message = payload.get("message", "Evaluation plugin failed.")
        traceback_text = payload.get("traceback")
        log.error(
            "Evaluation plugin error job={} commit={}: {}",
            context.job_id,
            context.candidate_commit_hash,
            message,
        )
        if traceback_text:
            log.error("Evaluation plugin traceback:\n{}", traceback_text)
        raise EvaluationError(message)

    def _execute_phase_with_deadline(
        self,
        phase: str,
        context: EvaluationContext,
        phase_args: tuple[Any, ...],
        *,
        deadline: float,
    ) -> Any:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise EvaluationError(
                f"Phased evaluation exceeded the shared {self.timeout}s timeout before {phase}."
            )
        ctx = multiprocessing.get_context("spawn")
        result_receiver, result_sender = ctx.Pipe(duplex=False)
        child_watch, parent_watch = ctx.Pipe(duplex=False)
        inline_target = None if self.plugin_ref else self._ensure_plugin_target()
        process = ctx.Process(
            target=_plugin_phase_subprocess_entry,
            args=(
                self.plugin_ref,
                inline_target,
                tuple(str(path) for path in self.python_paths),
                phase,
                context,
                phase_args,
                result_sender,
                child_watch,
            ),
        )
        process.start()
        result_sender.close()
        child_watch.close()
        try:
            status, payload = _receive_plugin_process_result(
                process,
                result_receiver,
                timeout=remaining,
                timeout_message=(
                    f"Phased evaluator timed out during {phase} after {self.timeout}s total."
                ),
                no_result_message=f"Phased evaluator {phase} did not return a result.",
                unreadable_message=f"Phased evaluator {phase} returned an unreadable payload.",
                exit_label=f"Phased evaluator {phase}",
            )
        finally:
            result_receiver.close()
            parent_watch.close()
        if status == "ok":
            return payload
        message = payload.get("message", f"Phased evaluator {phase} failed.")
        traceback_text = payload.get("traceback")
        if traceback_text:
            log.error("Phased evaluator {} traceback:\n{}", phase, traceback_text)
        raise EvaluationError(message)

    def _coerce_phase_failure(
        self,
        payload: Any,
        *,
        context: EvaluationContext,
        phase: str,
        started_at: datetime,
    ) -> EvaluationOutcome:
        if isinstance(payload, (EvalFail, EvaluationOutcome)) or (
            isinstance(payload, Mapping) and "outcome_kind" in payload
        ):
            outcome = self._coerce_outcome(
                payload,
                context=context,
                evaluator_name=self._evaluator_label(),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            outcome.evaluator_version = outcome.evaluator_version or self.evaluator_version
            outcome.protocol = "phased-v1"
            return outcome
        raise EvaluationError(
            f"Phased evaluator {phase} must return its typed phase value or a failure outcome."
        )

    def _validate_json_size(self, payload: Mapping[str, Any], *, label: str) -> None:
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"Phased evaluator {label} must be JSON serializable.") from exc
        limit = max(1, int(self.settings.worker_evaluator_measurement_max_json_bytes))
        if len(encoded) > limit:
            raise EvaluationError(
                f"Phased evaluator {label} is {len(encoded)} bytes; configured limit is {limit}."
            )

    def _coerce_outcome(
        self,
        payload: Any,
        *,
        context: EvaluationContext,
        evaluator_name: str,
        started_at: datetime,
        finished_at: datetime,
    ) -> EvaluationOutcome:
        if isinstance(payload, EvalPass):
            return EvaluationOutcome(
                evaluator_name=evaluator_name,
                candidate_commit_hash=context.candidate_commit_hash,
                outcome_kind="passed",
                result=payload.to_result(),
                started_at=started_at,
                finished_at=finished_at,
            )

        if isinstance(payload, EvalFail):
            return self._candidate_failure_from_eval_fail(
                payload,
                context=context,
                evaluator_name=evaluator_name,
                started_at=started_at,
                finished_at=finished_at,
            )

        if isinstance(payload, EvaluationOutcome):
            outcome = payload
            outcome.evaluator_name = outcome.evaluator_name or evaluator_name
            outcome.candidate_commit_hash = (
                outcome.candidate_commit_hash or context.candidate_commit_hash
            )
            outcome.started_at = outcome.started_at or started_at
            outcome.finished_at = outcome.finished_at or finished_at
            return outcome

        if isinstance(payload, EvaluationResult):
            return EvaluationOutcome(
                evaluator_name=evaluator_name,
                candidate_commit_hash=context.candidate_commit_hash,
                outcome_kind="passed",
                result=payload,
                started_at=started_at,
                finished_at=finished_at,
            )

        if isinstance(payload, Mapping) and "outcome_kind" in payload:
            return self._outcome_from_mapping(
                payload,
                context=context,
                evaluator_name=evaluator_name,
                started_at=started_at,
                finished_at=finished_at,
            )

        return EvaluationOutcome(
            evaluator_name=evaluator_name,
            candidate_commit_hash=context.candidate_commit_hash,
            outcome_kind="passed",
            result=self._coerce_result(payload),
            started_at=started_at,
            finished_at=finished_at,
        )

    def _candidate_failure_from_eval_fail(
        self,
        payload: EvalFail,
        *,
        context: EvaluationContext,
        evaluator_name: str,
        started_at: datetime,
        finished_at: datetime,
    ) -> EvaluationOutcome:
        failure_kwargs: dict[str, Any] = {}
        if payload.details:
            if payload.kind in {"compile", "typecheck", "lint"}:
                failure_kwargs["compiler_errors_summary"] = payload.details
            elif payload.kind in {"test", "validation", "benchmark"}:
                failure_kwargs["failing_tests_summary"] = payload.details
            else:
                failure_kwargs["stack_trace_summary"] = payload.details
        return EvaluationOutcome(
            evaluator_name=evaluator_name,
            candidate_commit_hash=context.candidate_commit_hash,
            outcome_kind="candidate_failed",
            failure=EvaluationFailureResult(
                failure_stage="evaluation",
                failure_kind=_EVAL_FAIL_KIND_TO_FAILURE_KIND[payload.kind],
                repairability="repairable",
                repairability_reason=(
                    "EvalFail represents candidate-owned evaluator failure; "
                    "worker policy decides whether to rework."
                ),
                safe_failure_summary=payload.summary,
                **failure_kwargs,
            ),
            artifacts=payload.artifacts,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _outcome_from_mapping(
        self,
        payload: Mapping[str, Any],
        *,
        context: EvaluationContext,
        evaluator_name: str,
        started_at: datetime,
        finished_at: datetime,
    ) -> EvaluationOutcome:
        outcome_kind = normalize_single_line(str(payload.get("outcome_kind") or "")).lower()
        if outcome_kind not in _VALID_OUTCOME_KINDS:
            raise EvaluationError(f"Evaluator returned invalid outcome_kind={outcome_kind!r}.")

        artifacts, artifact_warnings = coerce_evaluation_artifacts(
            payload.get("artifact_records", payload.get("artifacts"))
        )
        mapping = _OutcomeMappingInput(
            payload=payload,
            context=context,
            evaluator_name=evaluator_name,
            started_at=started_at,
            finished_at=finished_at,
            artifacts=artifacts,
            artifact_warnings=artifact_warnings,
        )
        if outcome_kind == "passed":
            return self._passed_outcome_from_mapping(mapping)
        return self._failure_outcome_from_mapping(mapping, outcome_kind=outcome_kind)

    def _passed_outcome_from_mapping(
        self,
        mapping: _OutcomeMappingInput,
    ) -> EvaluationOutcome:
        payload = mapping.payload
        result_payload = payload.get("result")
        if result_payload is None:
            result_payload = {
                "summary": payload.get("summary"),
                "candidate_identity": payload.get("candidate_identity"),
                "metrics": payload.get("metrics"),
                "tests_executed": payload.get("tests_executed"),
                "logs": payload.get("logs"),
                "extra": payload.get("extra"),
            }
        return EvaluationOutcome(
            **_outcome_identity(mapping),
            outcome_kind="passed",
            result=self._coerce_result(result_payload),
            artifacts=mapping.artifacts,
            artifact_validation_warnings=mapping.artifact_warnings,
        )

    def _failure_outcome_from_mapping(
        self,
        mapping: _OutcomeMappingInput,
        *,
        outcome_kind: str,
    ) -> EvaluationOutcome:
        payload = mapping.payload
        failure_payload = payload.get("failure")
        if not isinstance(failure_payload, Mapping):
            failure_payload = payload
        return EvaluationOutcome(
            **_outcome_identity(mapping),
            outcome_kind=cast(EvaluationOutcomeKind, outcome_kind),
            failure=self._failure_from_mapping(failure_payload),
            artifacts=mapping.artifacts,
            artifact_validation_warnings=mapping.artifact_warnings,
        )

    def _failure_from_mapping(self, payload: Mapping[str, Any]) -> EvaluationFailureResult:
        return EvaluationFailureResult(
            failure_stage=str(payload.get("failure_stage") or "unknown"),
            failure_kind=str(payload.get("failure_kind") or "unknown"),
            repairability=cast(
                EvaluationRepairability,
                str(payload.get("repairability") or "unknown"),
            ),
            repairability_reason=cast(str | None, payload.get("repairability_reason")),
            safe_failure_summary=str(payload.get("safe_failure_summary") or ""),
            agent_visible_evidence_refs=_bounded_string_tuple(
                payload.get("agent_visible_evidence_refs"),
                limit=256,
                max_items=32,
            ),
            human_only_artifact_refs=_bounded_string_tuple(
                payload.get("human_only_artifact_refs"),
                limit=256,
                max_items=32,
            ),
            hidden_artifact_refs=_bounded_string_tuple(
                payload.get("hidden_artifact_refs"),
                limit=256,
                max_items=32,
            ),
            exit_code=_optional_int(payload.get("exit_code")),
            timeout_seconds=_optional_int(payload.get("timeout_seconds")),
            failing_tests_summary=cast(str | None, payload.get("failing_tests_summary")),
            compiler_errors_summary=cast(str | None, payload.get("compiler_errors_summary")),
            stack_trace_summary=cast(str | None, payload.get("stack_trace_summary")),
            policy_version=str(payload.get("policy_version") or "diagnostic-capsule-v1"),
        )

    def _synthetic_failure_outcome(
        self,
        *,
        context: EvaluationContext,
        evaluator_name: str,
        message: str,
        started_at: datetime,
        finished_at: datetime,
    ) -> EvaluationOutcome:
        bounded = clamp_text(normalize_single_line(message), 2048)
        outcome_kind: EvaluationOutcomeKind = "evaluator_failed"
        failure_kind = "evaluator_error"
        if "timed out" in bounded.lower():
            outcome_kind = "infrastructure_failed"
            failure_kind = "infrastructure_error"
        return EvaluationOutcome(
            evaluator_name=evaluator_name,
            candidate_commit_hash=context.candidate_commit_hash,
            outcome_kind=outcome_kind,
            failure=EvaluationFailureResult(
                failure_stage="evaluation",
                failure_kind=failure_kind,
                repairability="unknown",
                repairability_reason="Synthetic fallback outcomes are not repairable by default.",
                safe_failure_summary=bounded or "Evaluator failed before producing a valid outcome.",
            ),
            started_at=started_at,
            finished_at=finished_at,
        )

    def _coerce_result(self, payload: Any) -> EvaluationResult:
        if isinstance(payload, EvaluationResult):
            result = payload
        elif isinstance(payload, Mapping):
            summary = str(payload.get("summary") or "").strip()
            if not summary:
                raise EvaluationError("Evaluator plugin did not return a summary.")
            metrics = self._coerce_metrics(payload.get("metrics"))
            tests = self._normalise_sequence(payload.get("tests_executed"), "tests_executed")
            logs = self._normalise_sequence(payload.get("logs"), "logs")
            extra = self._coerce_extra(payload.get("extra"))
            artifacts, artifact_warnings = coerce_evaluation_artifacts(payload.get("artifacts"))
            try:
                result = EvaluationResult(
                    summary=summary,
                    candidate_identity=cast(str | None, payload.get("candidate_identity")),
                    metrics=metrics,
                    tests_executed=tests,
                    logs=logs,
                    extra=extra,
                    artifacts=artifacts,
                    artifact_validation_warnings=artifact_warnings,
                )
            except ValueError as exc:
                raise EvaluationError(str(exc)) from exc
        else:
            raise EvaluationError(
                "Evaluator plugin returned an unsupported payload type.",
            )

        if len(result.metrics) > self.max_metrics:
            log.warning(
                "Truncating evaluator metrics from {} to {}",
                len(result.metrics),
                self.max_metrics,
            )
            result.metrics = result.metrics[: self.max_metrics]
        return result

    def _coerce_metrics(
        self,
        metrics_payload: Any,
    ) -> tuple[EvaluationMetric, ...]:
        if metrics_payload is None:
            return tuple()
        if isinstance(metrics_payload, EvaluationMetric):
            return (metrics_payload,)

        if isinstance(metrics_payload, Mapping):
            metrics_iterable: Sequence[Any] = (metrics_payload,)
        else:
            try:
                metrics_iterable = tuple(metrics_payload)
            except TypeError as exc:
                raise EvaluationError(
                    "Evaluator metrics must be iterable.",
                ) from exc

        metrics: list[EvaluationMetric] = []
        for item in metrics_iterable:
            if isinstance(item, EvaluationMetric):
                metrics.append(item)
                continue
            if isinstance(item, Mapping):
                metrics.append(self._metric_from_mapping(item))
                continue
            raise EvaluationError(
                f"Unsupported metric entry type: {type(item)!r}",
            )
        return tuple(metrics)

    @staticmethod
    def _metric_from_mapping(payload: Mapping[str, Any]) -> EvaluationMetric:
        try:
            name = str(payload["name"]).strip()
        except KeyError as exc:
            raise EvaluationError("Metric entry must include a 'name'.") from exc
        if not name:
            raise EvaluationError("Metric name cannot be empty.")

        if "value" not in payload:
            raise EvaluationError("Metric entry must include a 'value'.")
        value = payload["value"]
        if isinstance(value, bool):
            raise EvaluationError("Metric value cannot be boolean.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise EvaluationError("Metric value must be numeric.") from exc

        unit = payload.get("unit")
        unit_str = str(unit) if unit is not None else None
        hib_bool = Evaluator._metric_higher_is_better_from_value(
            payload.get("higher_is_better"),
        )
        details = payload.get("details")
        if details is None:
            details_dict: Mapping[str, Any] | None = None
        elif isinstance(details, Mapping):
            details_dict = dict(details)
        else:
            raise EvaluationError("Metric 'details' must be a mapping.")

        return EvaluationMetric(
            name=name,
            value=numeric_value,
            unit=unit_str,
            higher_is_better=hib_bool,
            details=details_dict,
        )

    @staticmethod
    def _metric_higher_is_better_from_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return True
            if normalized in {"false", "0"}:
                return False
            raise EvaluationError(
                "Metric 'higher_is_better' must be a boolean or one of: "
                "'true', 'false', '1', '0'."
            )
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
        raise EvaluationError(
            "Metric 'higher_is_better' must be a boolean or one of: "
            "'true', 'false', '1', '0'."
        )

    @staticmethod
    def _normalise_sequence(values: Any, label: str) -> tuple[str, ...]:
        if values is None:
            return tuple()
        if isinstance(values, str):
            candidate = values.strip()
            return (candidate,) if candidate else tuple()
        try:
            iterable = tuple(values)
        except TypeError as exc:
            raise EvaluationError(
                f"Field '{label}' must be iterable or a string.",
            ) from exc
        result: list[str] = []
        for item in iterable:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return tuple(result)

    @staticmethod
    def _coerce_extra(payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
        raise EvaluationError("Field 'extra' must be a mapping if provided.")

    @staticmethod
    def _validate_context(context: EvaluationContext) -> None:
        if not context.worktree.exists():
            raise EvaluationError(
                f"Worktree path {context.worktree} does not exist.",
            )
        if not context.worktree.is_dir():
            raise EvaluationError(
                f"Worktree path {context.worktree} is not a directory.",
            )


def _record_campaign_primary_metric_warnings(
    *,
    outcome: EvaluationOutcome,
    context: EvaluationContext,
) -> None:
    if outcome.result is None:
        return

    spec = _campaign_primary_metric_spec(context)
    if spec is None:
        return

    warnings = list(outcome.result.extra.get("campaign_program_warnings") or [])
    warning = _campaign_primary_metric_warning(result=outcome.result, spec=spec)
    if warning is not None:
        warnings.append(warning)
    if not warnings:
        return

    outcome.result.extra["campaign_program_warnings"] = warnings
    log.warning(
        "Campaign primary metric warning job={} commit={} warning_count={}",
        context.job_id,
        context.candidate_commit_hash,
        len(warnings),
    )


@dataclass(frozen=True, slots=True)
class _CampaignPrimaryMetricSpec:
    program_hash: Any
    metric_name: str
    expected_higher: bool | None


def _campaign_primary_metric_spec(context: EvaluationContext) -> _CampaignPrimaryMetricSpec | None:
    campaign_program = _mapping_or_none(context.payload.get("campaign_program"))
    snapshot = _mapping_or_none(campaign_program.get("snapshot") if campaign_program else None)
    primary_metric = _mapping_or_none(snapshot.get("primary_metric") if snapshot else None)
    metric_name = _campaign_primary_metric_name(primary_metric)
    if campaign_program is None or not metric_name:
        return None
    return _CampaignPrimaryMetricSpec(
        program_hash=campaign_program.get("hash"),
        metric_name=metric_name,
        expected_higher=_campaign_primary_metric_expected_higher(primary_metric),
    )


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _campaign_primary_metric_name(primary_metric: Mapping[str, Any] | None) -> str:
    if primary_metric is None:
        return ""
    return normalize_single_line(str(primary_metric.get("name") or ""))


def _campaign_primary_metric_expected_higher(primary_metric: Mapping[str, Any] | None) -> bool | None:
    if primary_metric is None:
        return None
    direction = normalize_single_line(str(primary_metric.get("direction") or "")).lower()
    return {"higher_is_better": True, "lower_is_better": False}.get(direction)


def _campaign_primary_metric_warning(
    *,
    result: EvaluationResult,
    spec: _CampaignPrimaryMetricSpec,
) -> dict[str, Any] | None:
    matching = _matching_evaluation_metric(result=result, metric_name=spec.metric_name)
    if matching is None:
        return _missing_primary_metric_warning(spec)
    if _metric_direction_matches(metric=matching, expected_higher=spec.expected_higher):
        return None
    return _primary_metric_direction_conflict_warning(metric=matching, spec=spec)


def _matching_evaluation_metric(
    *,
    result: EvaluationResult,
    metric_name: str,
) -> EvaluationMetric | None:
    return next((metric for metric in result.metrics if metric.name == metric_name), None)


def _metric_direction_matches(
    *,
    metric: EvaluationMetric,
    expected_higher: bool | None,
) -> bool:
    return expected_higher is None or metric.higher_is_better == expected_higher


def _missing_primary_metric_warning(spec: _CampaignPrimaryMetricSpec) -> dict[str, Any]:
    return {
        "code": "primary_metric_missing",
        "campaign_program_hash": spec.program_hash,
        "metric_name": clamp_text(spec.metric_name, 128),
    }


def _primary_metric_direction_conflict_warning(
    *,
    metric: EvaluationMetric,
    spec: _CampaignPrimaryMetricSpec,
) -> dict[str, Any]:
    return {
        "code": "primary_metric_direction_conflict",
        "campaign_program_hash": spec.program_hash,
        "metric_name": clamp_text(spec.metric_name, 128),
        "campaign_higher_is_better": spec.expected_higher,
        "evaluator_higher_is_better": metric.higher_is_better,
    }


def _receive_plugin_process_result(
    process: multiprocessing.Process,
    receiver: Connection,
    *,
    timeout: float,
    timeout_message: str,
    no_result_message: str,
    unreadable_message: str,
    exit_label: str,
) -> tuple[str, Any]:
    """Drain a child result while it is running so large payloads cannot deadlock."""

    ready = wait_for_connections(
        (receiver, process.sentinel),
        timeout=max(0.0, float(timeout)),
    )
    if receiver not in ready and process.sentinel in ready:
        process.join(timeout=0)
        if not receiver.poll(0.2):
            if process.exitcode and process.exitcode != 0:
                raise EvaluationError(f"{exit_label} exited with status {process.exitcode}.")
            raise EvaluationError(no_result_message)
    elif receiver not in ready:
        _terminate_plugin_process_tree(process)
        raise EvaluationError(timeout_message)

    try:
        result = receiver.recv()
        status, payload = result
    except (EOFError, OSError, TypeError, ValueError) as exc:
        process.join(timeout=0.2)
        if process.exitcode and process.exitcode != 0:
            raise EvaluationError(f"{exit_label} exited with status {process.exitcode}.") from exc
        raise EvaluationError(unreadable_message) from exc
    process.join(timeout=1.0)
    if process.is_alive():
        _terminate_plugin_process_tree(process)
        raise EvaluationError(f"{exit_label} did not exit after returning a result.")
    return str(status), payload


def _plugin_subprocess_entry(
    plugin_ref: str | None,
    inline_callable: EvaluationCallable | None,
    python_paths: Sequence[str],
    context: EvaluationContext,
    result_connection: Connection,
    parent_watch: Connection,
) -> None:
    try:
        _start_plugin_process_group()
        _arm_parent_death_watchdog(parent_watch)
        for entry in python_paths:
            entry_str = str(entry)
            if entry_str and entry_str not in sys.path:
                sys.path.insert(0, entry_str)

        if inline_callable is not None:
            plugin = inline_callable
        else:
            if not plugin_ref:
                raise EvaluationError("Evaluator plugin reference is not configured.")
            module_name, attr_path = Evaluator._split_reference(plugin_ref)
            target = Evaluator._import_object(module_name, attr_path)
            plugin = Evaluator._resolve_callable(target)

        payload = plugin(context)
        result_connection.send(("ok", payload))
    except Exception as exc:  # pragma: no cover - defensive isolation
        result_connection.send(
            (
                "error",
                {
                    "message": f"Evaluation plugin raised an exception: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
        )


def _plugin_phase_subprocess_entry(
    plugin_ref: str | None,
    inline_target: Any | None,
    python_paths: Sequence[str],
    phase: str,
    context: EvaluationContext,
    phase_args: tuple[Any, ...],
    result_connection: Connection,
    parent_watch: Connection,
) -> None:
    try:
        _start_plugin_process_group()
        _arm_parent_death_watchdog(parent_watch)
        for entry in python_paths:
            entry_str = str(entry)
            if entry_str and entry_str not in sys.path:
                sys.path.insert(0, entry_str)

        target = inline_target
        if target is None:
            if not plugin_ref:
                raise EvaluationError("Evaluator plugin reference is not configured.")
            module_name, attr_path = Evaluator._split_reference(plugin_ref)
            target = Evaluator._import_object(module_name, attr_path)
        if inspect.isclass(target):
            target = target()
        if normalize_single_line(str(getattr(target, "evaluation_protocol", ""))).lower() != "phased-v1":
            raise EvaluationError("Evaluator plugin does not declare evaluation_protocol='phased-v1'.")
        method = getattr(target, phase, None)
        if not callable(method):
            raise EvaluationError(f"phased-v1 evaluator has no callable {phase} method.")
        payload = method(context, *phase_args)
        result_connection.send(("ok", payload))
    except Exception as exc:  # pragma: no cover - defensive isolation
        result_connection.send(
            (
                "error",
                {
                    "message": f"Phased evaluator {phase} raised an exception: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
        )


def _start_plugin_process_group() -> None:
    if os.name != "posix":
        return
    try:
        os.setsid()
    except OSError:
        return


def _arm_parent_death_watchdog(parent_watch: Connection) -> None:
    """Terminate the evaluator process group if its owning worker disappears."""

    def _watch() -> None:
        try:
            parent_watch.recv_bytes()
        except (EOFError, OSError):
            if os.name == "posix":
                try:
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                    return
                except OSError:
                    pass
            os._exit(143)

    threading.Thread(
        target=_watch,
        name="loreley-evaluator-parent-watch",
        daemon=True,
    ).start()


def _terminate_plugin_process_tree(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join(timeout=0.1)
        return
    if os.name == "posix" and process.pid:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            process.terminate()
    else:
        process.terminate()
    process.join(5)
    if process.is_alive():
        if os.name == "posix" and process.pid:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                process.kill()
        else:
            process.kill()
        process.join(5)
