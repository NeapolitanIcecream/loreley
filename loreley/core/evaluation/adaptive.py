"""Budgeted adaptive sampling with auditable stopping decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import monotonic
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from .estimates import (
    Estimate,
    IntervalCapability,
    IntervalMethod,
    Observation,
    SampleMoments,
    StratifiedEstimate,
    aggregate_by_stratum,
    estimate,
    sample_moments,
)

__all__ = [
    "AdaptiveDecisionRecord",
    "AdaptiveEvaluationCheckpoint",
    "AdaptiveEvaluationResult",
    "AdaptiveEvaluationRunner",
    "AdaptiveSamplingConfig",
    "EffectClassification",
    "SampleBatchError",
    "SampleCallback",
    "SampleRequest",
    "StopReason",
    "UnsafeIntervalMethodError",
]

EffectClassification = Literal[
    "above_indifference_zone",
    "below_indifference_zone",
    "inside_indifference_zone",
    "undetermined",
    "not_evaluated",
]
StopReason = Literal[
    "effect_decision_reached",
    "precision_target_reached",
    "maximum_samples_reached",
    "wall_time_budget_exhausted",
    "sampler_exhausted",
]


class UnsafeIntervalMethodError(ValueError):
    """Raised when fixed-sample inference would drive optional stopping."""


class SampleBatchError(ValueError):
    """Raised when a sample callback violates its explicit batch contract."""


@dataclass(frozen=True, slots=True)
class AdaptiveSamplingConfig:
    """Budgets and stopping criteria for one adaptive evaluation."""

    min_samples: int
    max_samples: int
    batch_size: int = 1
    confidence_level: float = 0.95
    max_wall_time_seconds: float | None = None
    target_ci_half_width: float | None = None
    effect_threshold: float | None = None
    indifference_zone: float = 0.0
    allow_fixed_sample_optional_stopping: bool = False
    stratum_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_sampling_counts(self)
        _positive_optional(self.max_wall_time_seconds, "max_wall_time_seconds")
        _positive_optional(self.target_ci_half_width, "target_ci_half_width")
        confidence_level = _confidence_level(self.confidence_level)
        effect_threshold, indifference_zone = _decision_contract(self)
        if not isinstance(self.allow_fixed_sample_optional_stopping, bool):
            raise ValueError("allow_fixed_sample_optional_stopping must be bool.")
        stratum_weights = _canonical_stratum_weights(self.stratum_weights)
        object.__setattr__(self, "confidence_level", confidence_level)
        object.__setattr__(self, "effect_threshold", effect_threshold)
        object.__setattr__(self, "indifference_zone", indifference_zone)
        object.__setattr__(self, "stratum_weights", stratum_weights)
        if self.max_wall_time_seconds is not None:
            object.__setattr__(
                self,
                "max_wall_time_seconds",
                float(self.max_wall_time_seconds),
            )
        if self.target_ci_half_width is not None:
            object.__setattr__(
                self,
                "target_ci_half_width",
                float(self.target_ci_half_width),
            )

    @property
    def optional_stopping_possible(self) -> bool:
        """Whether more than one eligible analysis look can affect termination."""

        if self.min_samples == self.max_samples and self.max_wall_time_seconds is None:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "max_samples": self.max_samples,
            "batch_size": self.batch_size,
            "confidence_level": self.confidence_level,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "target_ci_half_width": self.target_ci_half_width,
            "effect_threshold": self.effect_threshold,
            "indifference_zone": self.indifference_zone,
            "allow_fixed_sample_optional_stopping": (
                self.allow_fixed_sample_optional_stopping
            ),
            "stratum_weights": dict(self.stratum_weights),
            "optional_stopping_possible": self.optional_stopping_possible,
        }


@dataclass(frozen=True, slots=True)
class SampleRequest:
    """One bounded request passed to a pluggable sample callback."""

    requested_samples: int
    collected_samples: int
    remaining_samples: int
    batch_index: int
    elapsed_seconds: float
    remaining_wall_time_seconds: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_samples": self.requested_samples,
            "collected_samples": self.collected_samples,
            "remaining_samples": self.remaining_samples,
            "batch_index": self.batch_index,
            "elapsed_seconds": self.elapsed_seconds,
            "remaining_wall_time_seconds": self.remaining_wall_time_seconds,
        }


SampleCallback = Callable[[SampleRequest], Iterable[Observation | float]]


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionRecord:
    """One callback/analysis step, including every reason considered for stop."""

    step_index: int
    look_index: int | None
    request: SampleRequest | None
    received_samples: int
    total_samples: int
    elapsed_seconds: float
    moments: SampleMoments | None
    estimate: Estimate | StratifiedEstimate | None
    effect_classification: EffectClassification
    stop_triggers: tuple[StopReason, ...]
    stopped: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "look_index": self.look_index,
            "request": self.request.as_dict() if self.request is not None else None,
            "received_samples": self.received_samples,
            "total_samples": self.total_samples,
            "elapsed_seconds": self.elapsed_seconds,
            "moments": self.moments.as_dict() if self.moments is not None else None,
            "estimate": self.estimate.as_dict() if self.estimate is not None else None,
            "effect_classification": self.effect_classification,
            "stop_triggers": list(self.stop_triggers),
            "stopped": self.stopped,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveEvaluationResult:
    """Final result plus the complete, JSON-safe decision trajectory."""

    observations: tuple[Observation, ...]
    config: AdaptiveSamplingConfig
    estimate: Estimate | StratifiedEstimate | None
    effect_classification: EffectClassification
    stop_reason: StopReason
    elapsed_seconds: float
    interval_method: str
    interval_capability: IntervalCapability
    unsafe_fixed_sample_override: bool
    history: tuple[AdaptiveDecisionRecord, ...]

    @property
    def inference_valid(self) -> bool:
        """Whether inference is valid for the recorded stopping protocol."""

        return self.estimate is not None and not self.unsafe_fixed_sample_override

    @property
    def declared_target_reached(self) -> bool:
        """Whether a declared evidence target, or the fixed design, completed."""

        targets: list[bool] = []
        if self.config.effect_threshold is not None:
            targets.append(
                self.effect_classification
                in {
                    "above_indifference_zone",
                    "below_indifference_zone",
                    "inside_indifference_zone",
                }
            )
        if self.config.target_ci_half_width is not None:
            targets.append(
                self.estimate is not None
                and _overall_estimate(self.estimate).confidence_interval.half_width
                <= self.config.target_ci_half_width
            )
        if targets:
            return any(targets)
        return (
            self.stop_reason == "maximum_samples_reached"
            and len(self.observations) == self.config.max_samples
        )

    @property
    def decision_ready(self) -> bool:
        """Safe default for a finalizer deciding whether evidence is conclusive."""

        return self.inference_valid and self.declared_target_reached

    def checkpoint(self) -> "AdaptiveEvaluationCheckpoint":
        """Return resumable state without repeating an analysis look."""

        completed_looks = max(
            (record.look_index or 0 for record in self.history),
            default=0,
        )
        completed_batches = max(
            (
                record.request.batch_index
                for record in self.history
                if record.request is not None
            ),
            default=0,
        )
        return AdaptiveEvaluationCheckpoint(
            observations=self.observations,
            completed_looks=completed_looks,
            completed_batches=completed_batches,
            elapsed_seconds=self.elapsed_seconds,
            history=self.history,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": [
                observation.as_dict() for observation in self.observations
            ],
            "config": self.config.as_dict(),
            "estimate": self.estimate.as_dict() if self.estimate is not None else None,
            "effect_classification": self.effect_classification,
            "stop_reason": self.stop_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "interval_method": self.interval_method,
            "interval_capability": self.interval_capability.value,
            "unsafe_fixed_sample_override": self.unsafe_fixed_sample_override,
            "inference_valid": self.inference_valid,
            "declared_target_reached": self.declared_target_reached,
            "decision_ready": self.decision_ready,
            "history": [record.as_dict() for record in self.history],
        }


@dataclass(frozen=True, slots=True)
class AdaptiveEvaluationCheckpoint:
    """State required to resume sampling without resetting alpha spending."""

    observations: tuple[Observation, ...] = ()
    completed_looks: int = 0
    completed_batches: int = 0
    elapsed_seconds: float = 0.0
    history: tuple[AdaptiveDecisionRecord, ...] = ()

    def __post_init__(self) -> None:
        observations = tuple(_coerce_sample(item) for item in self.observations)
        if self.completed_looks < 0 or self.completed_batches < 0:
            raise ValueError("Checkpoint look and batch counts must be non-negative.")
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(
                "Checkpoint elapsed_seconds must be finite and non-negative."
            )
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "history", tuple(self.history))

    def as_dict(self) -> dict[str, Any]:
        return {
            "observations": [item.as_dict() for item in self.observations],
            "completed_looks": self.completed_looks,
            "completed_batches": self.completed_batches,
            "elapsed_seconds": self.elapsed_seconds,
            "history": [record.as_dict() for record in self.history],
        }


@dataclass(slots=True)
class _AdaptiveRunState:
    observations: list[Observation]
    history: list[AdaptiveDecisionRecord]
    final_estimate: Estimate | StratifiedEstimate | None
    classification: EffectClassification
    look_index: int
    batch_index: int
    started_at: float


class AdaptiveEvaluationRunner:
    """Collect batches until an evidence or resource boundary is reached."""

    def __init__(
        self,
        config: AdaptiveSamplingConfig,
        *,
        interval_method: IntervalMethod,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        capability = interval_method.capability
        if not isinstance(capability, IntervalCapability):
            try:
                capability = IntervalCapability(str(capability))
            except ValueError as exc:
                raise ValueError(
                    "Interval method declares an unknown capability."
                ) from exc
        if (
            capability is IntervalCapability.FIXED_SAMPLE_ONLY
            and config.optional_stopping_possible
            and not config.allow_fixed_sample_optional_stopping
        ):
            raise UnsafeIntervalMethodError(
                "Adaptive stopping requires an anytime-valid interval method. "
                "A fixed-sample-only method may be used only for one fixed analysis "
                "or with allow_fixed_sample_optional_stopping=True, which is recorded "
                "as an unsafe override."
            )
        self.config = config
        self.interval_method = interval_method
        self.interval_capability = capability
        self.clock = clock

    def run(
        self,
        sample: SampleCallback,
        *,
        checkpoint: AdaptiveEvaluationCheckpoint | None = None,
    ) -> AdaptiveEvaluationResult:
        state = self._resume_state(checkpoint)
        while True:
            before_sample = _elapsed(self.clock(), state.started_at)
            early_result = self._result_before_sampling(state, before_sample)
            if early_result is not None:
                return early_result

            request = self._sample_request(state, before_sample)
            batch, after_sample = self._collect_batch(sample, request, state)
            if not batch:
                return self._empty_batch_result(state, request, after_sample)

            result = self._record_batch(state, request, batch, after_sample)
            if result is not None:
                return result

    def _resume_state(
        self,
        checkpoint: AdaptiveEvaluationCheckpoint | None,
    ) -> _AdaptiveRunState:
        resumed = checkpoint or AdaptiveEvaluationCheckpoint()
        if len(resumed.observations) > self.config.max_samples:
            raise ValueError("Checkpoint observations exceed max_samples.")
        if len(resumed.observations) >= self.config.min_samples:
            if resumed.completed_looks < 1:
                raise ValueError(
                    "Checkpoint with analyzable observations must include completed_looks."
                )
        elif resumed.completed_looks:
            raise ValueError("Checkpoint completed_looks precede min_samples.")

        started_at = _finite_clock_value(self.clock()) - resumed.elapsed_seconds
        observations = list(resumed.observations)
        final_estimate: Estimate | StratifiedEstimate | None = None
        classification: EffectClassification = "not_evaluated"
        if observations and len(observations) >= self.config.min_samples:
            final_estimate = self._estimate(
                observations,
                look_index=resumed.completed_looks,
            )
            classification = self._classify_effect(final_estimate)
        return _AdaptiveRunState(
            observations=observations,
            history=list(resumed.history),
            final_estimate=final_estimate,
            classification=classification,
            look_index=resumed.completed_looks,
            batch_index=resumed.completed_batches,
            started_at=started_at,
        )

    def _result_before_sampling(
        self,
        state: _AdaptiveRunState,
        elapsed_seconds: float,
    ) -> AdaptiveEvaluationResult | None:
        if self._wall_time_exhausted(elapsed_seconds):
            reasons: tuple[StopReason, ...] = ("wall_time_budget_exhausted",)
            state.history.append(self._terminal_record(state, elapsed_seconds, reasons))
            return self._result(state, reasons[0], elapsed_seconds)
        if len(state.observations) >= self.config.max_samples:
            return self._result(state, "maximum_samples_reached", elapsed_seconds)
        return None

    def _sample_request(
        self,
        state: _AdaptiveRunState,
        elapsed_seconds: float,
    ) -> SampleRequest:
        remaining = self.config.max_samples - len(state.observations)
        state.batch_index += 1
        return SampleRequest(
            requested_samples=min(self.config.batch_size, remaining),
            collected_samples=len(state.observations),
            remaining_samples=remaining,
            batch_index=state.batch_index,
            elapsed_seconds=elapsed_seconds,
            remaining_wall_time_seconds=self._remaining_wall_time(elapsed_seconds),
        )

    def _collect_batch(
        self,
        sample: SampleCallback,
        request: SampleRequest,
        state: _AdaptiveRunState,
    ) -> tuple[tuple[Observation, ...], float]:
        raw_batch = sample(request)
        if isinstance(raw_batch, (str, bytes)):
            raise SampleBatchError("Sample callback must return a sequence of samples.")
        batch = tuple(_coerce_sample(item) for item in raw_batch)
        if len(batch) > request.requested_samples:
            raise SampleBatchError(
                f"Sample callback returned {len(batch)} samples for a request of "
                f"{request.requested_samples}; no samples were accepted."
            )
        return batch, _elapsed(self.clock(), state.started_at)

    def _empty_batch_result(
        self,
        state: _AdaptiveRunState,
        request: SampleRequest,
        elapsed_seconds: float,
    ) -> AdaptiveEvaluationResult:
        reasons: tuple[StopReason, ...] = (
            ("wall_time_budget_exhausted", "sampler_exhausted")
            if self._wall_time_exhausted(elapsed_seconds)
            else ("sampler_exhausted",)
        )
        state.history.append(
            self._terminal_record(state, elapsed_seconds, reasons, request=request)
        )
        return self._result(state, reasons[0], elapsed_seconds)

    def _record_batch(
        self,
        state: _AdaptiveRunState,
        request: SampleRequest,
        batch: tuple[Observation, ...],
        elapsed_seconds: float,
    ) -> AdaptiveEvaluationResult | None:
        state.observations.extend(batch)
        moments = sample_moments(state.observations)
        if len(state.observations) >= self.config.min_samples:
            state.look_index += 1
            state.final_estimate = self._estimate(
                state.observations,
                look_index=state.look_index,
            )
            state.classification = self._classify_effect(state.final_estimate)
        triggers = self._stop_triggers(
            total_samples=len(state.observations),
            elapsed_seconds=elapsed_seconds,
            estimate=state.final_estimate,
            classification=state.classification,
        )
        state.history.append(
            AdaptiveDecisionRecord(
                step_index=len(state.history) + 1,
                look_index=state.look_index or None,
                request=request,
                received_samples=len(batch),
                total_samples=len(state.observations),
                elapsed_seconds=elapsed_seconds,
                moments=moments,
                estimate=state.final_estimate,
                effect_classification=state.classification,
                stop_triggers=triggers,
                stopped=bool(triggers),
            )
        )
        return self._result(state, triggers[0], elapsed_seconds) if triggers else None

    def _estimate(
        self,
        observations: Sequence[Observation],
        *,
        look_index: int,
    ) -> Estimate | StratifiedEstimate:
        weights = self.config.stratum_weights
        if weights:
            return aggregate_by_stratum(
                observations,
                interval_method=self.interval_method,
                stratum_weights=weights,
                confidence_level=self.config.confidence_level,
                look_index=look_index,
            )
        non_default = sorted(
            {item.stratum for item in observations if item.stratum != "all"}
        )
        if non_default:
            raise ValueError(
                "Stratified observations require predeclared stratum_weights; "
                f"observed={non_default}."
            )
        return estimate(
            observations,
            interval_method=self.interval_method,
            confidence_level=self.config.confidence_level,
            look_index=look_index,
        )

    def _classify_effect(
        self,
        current: Estimate | StratifiedEstimate,
    ) -> EffectClassification:
        threshold = self.config.effect_threshold
        if threshold is None:
            return "not_evaluated"
        lower_zone = threshold - self.config.indifference_zone
        upper_zone = threshold + self.config.indifference_zone
        interval = _overall_estimate(current).confidence_interval
        if interval.lower > upper_zone:
            return "above_indifference_zone"
        if interval.upper < lower_zone:
            return "below_indifference_zone"
        if interval.lower >= lower_zone and interval.upper <= upper_zone:
            return "inside_indifference_zone"
        return "undetermined"

    def _stop_triggers(
        self,
        *,
        total_samples: int,
        elapsed_seconds: float,
        estimate: Estimate | StratifiedEstimate | None,
        classification: EffectClassification,
    ) -> tuple[StopReason, ...]:
        triggers: list[StopReason] = []
        if self._wall_time_exhausted(elapsed_seconds):
            triggers.append("wall_time_budget_exhausted")
        if classification in {
            "above_indifference_zone",
            "below_indifference_zone",
            "inside_indifference_zone",
        }:
            triggers.append("effect_decision_reached")
        if (
            estimate is not None
            and self.config.target_ci_half_width is not None
            and _overall_estimate(estimate).confidence_interval.half_width
            <= self.config.target_ci_half_width
        ):
            triggers.append("precision_target_reached")
        if total_samples >= self.config.max_samples:
            triggers.append("maximum_samples_reached")
        return tuple(triggers)

    def _wall_time_exhausted(self, elapsed_seconds: float) -> bool:
        budget = self.config.max_wall_time_seconds
        return budget is not None and elapsed_seconds >= budget

    def _remaining_wall_time(self, elapsed_seconds: float) -> float | None:
        budget = self.config.max_wall_time_seconds
        return None if budget is None else max(0.0, budget - elapsed_seconds)

    def _terminal_record(
        self,
        state: _AdaptiveRunState,
        elapsed_seconds: float,
        reasons: tuple[StopReason, ...],
        *,
        request: SampleRequest | None = None,
    ) -> AdaptiveDecisionRecord:
        return AdaptiveDecisionRecord(
            step_index=len(state.history) + 1,
            look_index=state.look_index or None,
            request=request,
            received_samples=0,
            total_samples=len(state.observations),
            elapsed_seconds=elapsed_seconds,
            moments=_moments_or_none(state.observations),
            estimate=state.final_estimate,
            effect_classification=state.classification,
            stop_triggers=reasons,
            stopped=True,
        )

    def _result(
        self,
        state: _AdaptiveRunState,
        stop_reason: StopReason,
        elapsed_seconds: float,
    ) -> AdaptiveEvaluationResult:
        return AdaptiveEvaluationResult(
            observations=tuple(state.observations),
            config=self.config,
            estimate=state.final_estimate,
            effect_classification=state.classification,
            stop_reason=stop_reason,
            elapsed_seconds=elapsed_seconds,
            interval_method=self.interval_method.name,
            interval_capability=self.interval_capability,
            unsafe_fixed_sample_override=(
                self.interval_capability is IntervalCapability.FIXED_SAMPLE_ONLY
                and self.config.optional_stopping_possible
            ),
            history=tuple(state.history),
        )


def _positive_optional(value: float | None, field_name: str) -> None:
    if value is None:
        return
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field_name} must be a finite positive number.")


def _validate_sampling_counts(config: AdaptiveSamplingConfig) -> None:
    for field_name in ("min_samples", "max_samples", "batch_size"):
        value = getattr(config, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer.")
    if config.min_samples > config.max_samples:
        raise ValueError("min_samples must not exceed max_samples.")


def _confidence_level(value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    return parsed


def _decision_contract(config: AdaptiveSamplingConfig) -> tuple[float | None, float]:
    threshold = (
        None if config.effect_threshold is None else float(config.effect_threshold)
    )
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("effect_threshold must be finite when provided.")
    zone = float(config.indifference_zone)
    if not math.isfinite(zone) or zone < 0:
        raise ValueError("indifference_zone must be a finite non-negative number.")
    if threshold is None and zone != 0:
        raise ValueError("indifference_zone requires effect_threshold.")
    return threshold, zone


def _coerce_sample(item: Observation | float) -> Observation:
    return item if isinstance(item, Observation) else Observation(item)


def _finite_clock_value(value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("clock must return finite values.")
    return parsed


def _elapsed(current: float, started_at: float) -> float:
    elapsed = _finite_clock_value(current) - started_at
    if elapsed < 0:
        raise ValueError("clock must be monotonic.")
    return elapsed


def _moments_or_none(observations: Sequence[Observation]) -> SampleMoments | None:
    return sample_moments(observations) if observations else None


def _overall_estimate(value: Estimate | StratifiedEstimate) -> Estimate:
    return value.overall if isinstance(value, StratifiedEstimate) else value


def _canonical_stratum_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(weights, Mapping):
        raise ValueError("stratum_weights must be a mapping.")
    if not weights:
        return {}
    normalized: dict[str, float] = {}
    for raw_name, raw_weight in weights.items():
        name = str(raw_name).strip()
        if not name or isinstance(raw_weight, bool):
            raise ValueError("stratum_weights need named, positive weights.")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("stratum_weights need named, positive weights.")
        normalized[name] = weight
    total = math.fsum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("stratum_weights must sum to 1.0.")
    return {name: normalized[name] for name in sorted(normalized)}
