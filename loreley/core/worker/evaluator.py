from __future__ import annotations

import inspect
import re
import sys
import multiprocessing
import traceback
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, cast
from queue import Empty

from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.core.contracts import clamp_text, normalize_single_line

console = Console()
log = logger.bind(module="worker.evaluator")

__all__ = [
    "ArtifactValidationWarning",
    "EvaluationArtifact",
    "EvaluationContext",
    "EvaluationDiagnostic",
    "EvaluationError",
    "EvaluationMetric",
    "EvaluationPlugin",
    "EvaluationResult",
    "Evaluator",
    "coerce_evaluation_artifacts",
]

ArtifactVisibility = Literal["agent_visible", "human_only", "hidden"]
ArtifactAgentProjection = Literal["summary", "manifest", "path"]
ArtifactWarningAction = Literal["skipped", "downgraded", "metadata_only"]

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

    def __post_init__(self) -> None:
        summary = (self.summary or "").strip()
        if not summary:
            raise ValueError("Evaluation summary must be provided.")
        self.summary = summary
        self.metrics = tuple(self.metrics or ())
        self.tests_executed = tuple(self.tests_executed or ())
        self.logs = tuple(self.logs or ())
        self.extra = dict(self.extra or {})
        artifacts, warnings = coerce_evaluation_artifacts(self.artifacts)
        self.artifacts = artifacts
        self.artifact_validation_warnings = tuple(self.artifact_validation_warnings or ()) + warnings


class EvaluationPlugin(Protocol):
    """Protocol implemented by evaluation plugins."""

    def __call__(self, context: EvaluationContext) -> EvaluationResult | Mapping[str, Any]:
        ...


EvaluationCallable = Callable[[EvaluationContext], EvaluationResult | Mapping[str, Any]]


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


def _optional_bounded_line(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = clamp_text(normalize_single_line(str(value)), limit)
    return text or None


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


def _bound_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return clamp_text(normalize_single_line(value), 512)
    if isinstance(value, Mapping):
        nested: dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:16]:
            key = clamp_text(normalize_single_line(str(raw_key)), 64)
            if key:
                nested[key] = _bound_metadata_value(raw_value)
        return nested
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_bound_metadata_value(item) for item in list(value)[:16]]
    return clamp_text(normalize_single_line(str(value)), 512)


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
        self._plugin_callable: EvaluationCallable | None = plugin
        self._pythonpath_ready = False

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Execute the configured plugin and return structured results."""
        self._validate_context(context)
        plugin = self._ensure_callable()
        label = self.plugin_ref or getattr(plugin, "__name__", "<callable>")
        console.log(
            f"[cyan]Evaluator[/] running plugin {label} "
            f"(job={context.job_id or 'N/A'} commit={context.candidate_commit_hash or 'N/A'})",
        )

        start = monotonic()
        payload = self._execute_with_timeout(plugin, context)
        result = self._coerce_result(payload)
        duration = monotonic() - start
        result.extra.setdefault("evaluator_duration_seconds", float(duration))
        console.log(
            f"[bold green]Evaluator[/] finished in {duration:.1f}s "
            f"metrics={len(result.metrics)}",
        )
        log.info(
            "Evaluation completed job={} commit={} metrics={}",
            context.job_id,
            context.candidate_commit_hash,
            len(result.metrics),
        )
        return result

    # Internal helpers -------------------------------------------------

    def _ensure_callable(self) -> EvaluationCallable:
        if self._plugin_callable:
            return self._plugin_callable

        if not self.plugin_ref:
            raise EvaluationError(
                "WORKER_EVALUATOR_PLUGIN is not configured. "
                "Provide a dotted path to a callable plugin.",
            )

        self._prepare_pythonpath()
        module_name, attr_path = self._split_reference(self.plugin_ref)
        target = self._import_object(module_name, attr_path)
        callable_plugin = self._resolve_callable(target)
        self._plugin_callable = callable_plugin
        return callable_plugin

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
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        inline_callable = None if self.plugin_ref else plugin
        process = ctx.Process(
            target=_plugin_subprocess_entry,
            args=(
                self.plugin_ref,
                inline_callable,
                tuple(str(path) for path in self.python_paths),
                context,
                result_queue,
            ),
        )
        process.start()
        process.join(self.timeout)

        if process.is_alive():
            process.terminate()
            process.join(5)
            raise EvaluationError(
                f"Evaluation plugin timed out after {self.timeout}s.",
            )

        queue_wait_seconds = max(1.0, min(5.0, self.timeout * 0.1))
        try:
            status, payload = result_queue.get(timeout=queue_wait_seconds)
        except Empty as exc:
            if process.exitcode and process.exitcode != 0:
                raise EvaluationError(
                    f"Evaluation plugin exited with status {process.exitcode}.",
                ) from exc
            raise EvaluationError(
                "Evaluation plugin did not return any result.",
            ) from exc
        except Exception as exc:
            if process.exitcode and process.exitcode != 0:
                raise EvaluationError(
                    f"Evaluation plugin exited with status {process.exitcode}.",
                ) from exc
            raise EvaluationError(
                "Evaluation plugin returned an unreadable payload.",
            ) from exc
        finally:
            result_queue.close()

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
        hib = payload.get("higher_is_better")
        hib_bool = bool(hib) if hib is not None else True
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


def _plugin_subprocess_entry(
    plugin_ref: str | None,
    inline_callable: EvaluationCallable | None,
    python_paths: Sequence[str],
    context: EvaluationContext,
    queue: multiprocessing.Queue[Any],
) -> None:
    try:
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
        queue.put(("ok", payload))
    except Exception as exc:  # pragma: no cover - defensive isolation
        queue.put(
            (
                "error",
                {
                    "message": f"Evaluation plugin raised an exception: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
        )
