"""Project-neutral statistical building blocks for reliable evaluation."""

from __future__ import annotations

from .adaptive import (
    AdaptiveDecisionRecord,
    AdaptiveEvaluationCheckpoint,
    AdaptiveEvaluationResult,
    AdaptiveEvaluationRunner,
    AdaptiveSamplingConfig,
    EffectClassification,
    SampleBatchError,
    SampleCallback,
    SampleRequest,
    StopReason,
    UnsafeIntervalMethodError,
)
from .contract import MeasurementContract
from .estimates import (
    ConfidenceInterval,
    Estimate,
    HoeffdingConfidenceSequence,
    InsufficientSamplesError,
    IntervalCapability,
    IntervalMethod,
    Observation,
    SampleMoments,
    StratifiedEstimate,
    StratumEstimate,
    StudentTInterval,
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
    "ConfidenceInterval",
    "EffectClassification",
    "Estimate",
    "HoeffdingConfidenceSequence",
    "InsufficientSamplesError",
    "IntervalCapability",
    "IntervalMethod",
    "MeasurementContract",
    "Observation",
    "SampleBatchError",
    "SampleCallback",
    "SampleMoments",
    "SampleRequest",
    "StopReason",
    "StratifiedEstimate",
    "StratumEstimate",
    "StudentTInterval",
    "UnsafeIntervalMethodError",
    "aggregate_by_stratum",
    "estimate",
    "sample_moments",
]
