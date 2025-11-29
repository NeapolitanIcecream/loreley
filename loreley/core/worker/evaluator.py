from __future__ import annotations

import inspect
import sys
import multiprocessing
import traceback
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping, Protocol, Sequence, cast
from queue import Empty

from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings

console = Console()
log = logger.bind(module="worker.evaluator")

__all__ = [
    "EvaluationContext",
    "EvaluationError",
    "EvaluationMetric",
    "EvaluationPlugin",
    "EvaluationResult",
    "Evaluator",
]


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

    def __post_init__(self) -> None:
        summary = (self.summary or "").strip()
        if not summary:
            raise ValueError("Evaluation summary must be provided.")
        self.summary = summary
        self.metrics = tuple(self.metrics or ())
        self.tests_executed = tuple(self.tests_executed or ())
        self.logs = tuple(self.logs or ())
        self.extra = dict(self.extra or {})


class EvaluationPlugin(Protocol):
    """Protocol implemented by evaluation plugins."""

    def __call__(self, context: EvaluationContext) -> EvaluationResult | Mapping[str, Any]:
        ...


EvaluationCallable = Callable[[EvaluationContext], EvaluationResult | Mapping[str, Any]]


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
            try:
                result = EvaluationResult(
                    summary=summary,
                    metrics=metrics,
                    tests_executed=tests,
                    logs=logs,
                    extra=extra,
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

