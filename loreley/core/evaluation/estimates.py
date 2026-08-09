"""Small, dependency-free statistical estimates for evaluator measurements.

The interval method is part of every estimate.  This is intentional: a
fixed-sample confidence interval and an anytime-valid confidence sequence make
different promises, even when their numeric fields have the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, ClassVar, Iterable, Mapping, Protocol, Sequence, runtime_checkable

__all__ = [
    "ConfidenceInterval",
    "Estimate",
    "HoeffdingConfidenceSequence",
    "InsufficientSamplesError",
    "IntervalCapability",
    "IntervalMethod",
    "Observation",
    "SampleMoments",
    "StratifiedEstimate",
    "StratumEstimate",
    "StudentTInterval",
    "aggregate_by_stratum",
    "estimate",
    "sample_moments",
]


class InsufficientSamplesError(ValueError):
    """Raised when an interval's stated assumptions need more observations."""


class IntervalCapability(str, Enum):
    """Whether an interval remains valid when its history controls stopping."""

    FIXED_SAMPLE_ONLY = "fixed_sample_only"
    ANYTIME_VALID = "anytime_valid"


@dataclass(frozen=True, slots=True)
class Observation:
    """One finite scalar observation and its optional aggregation stratum."""

    value: float
    stratum: str = "all"

    def __post_init__(self) -> None:
        if isinstance(self.value, bool):
            raise TypeError("Observation value must be a finite real number, not bool.")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("Observation value must be finite.")
        stratum = str(self.stratum).strip()
        if not stratum:
            raise ValueError("Observation stratum must not be empty.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "stratum", stratum)

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "stratum": self.stratum}


@dataclass(frozen=True, slots=True)
class SampleMoments:
    """Numerically stable sample moments.

    ``sample_variance`` uses Bessel's correction.  Variance and standard error
    are ``None`` for a single observation instead of being reported as zero.
    """

    n: int
    mean: float
    sample_variance: float | None
    standard_error: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "sample_variance": self.sample_variance,
            "standard_error": self.standard_error,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A two-sided interval with its statistical capability made explicit."""

    lower: float
    upper: float
    confidence_level: float
    method: str
    capability: IntervalCapability
    look_index: int
    method_parameters: tuple[tuple[str, str | int | float | bool | None], ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("Confidence interval bounds must be finite.")
        if self.lower > self.upper:
            raise ValueError("Confidence interval lower bound exceeds upper bound.")
        _validate_confidence_level(self.confidence_level)
        if not self.method.strip():
            raise ValueError("Confidence interval method must not be empty.")
        if self.look_index < 1:
            raise ValueError("Confidence interval look_index must be positive.")
        parameter_names = [name for name, _ in self.method_parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Confidence interval method parameter names must be unique.")

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "half_width": self.half_width,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "capability": self.capability.value,
            "look_index": self.look_index,
            "method_parameters": dict(self.method_parameters),
        }


@runtime_checkable
class IntervalMethod(Protocol):
    """Pluggable two-sided interval implementation."""

    name: str
    capability: IntervalCapability

    def interval(
        self,
        values: Sequence[float],
        moments: SampleMoments,
        confidence_level: float,
        *,
        look_index: int,
    ) -> ConfidenceInterval:
        """Return an interval for one analysis look."""


@dataclass(frozen=True, slots=True)
class StudentTInterval:
    """Classical two-sided Student-t interval for a fixed sample.

    This interval assumes independent, identically distributed normal samples.
    It explicitly rejects ``n < 2`` because sample variance is then undefined.
    It is *not* safe for a stopping rule that repeatedly inspects the interval.
    """

    name: ClassVar[str] = "student_t"
    capability: ClassVar[IntervalCapability] = IntervalCapability.FIXED_SAMPLE_ONLY

    def interval(
        self,
        values: Sequence[float],
        moments: SampleMoments,
        confidence_level: float,
        *,
        look_index: int,
    ) -> ConfidenceInterval:
        del values
        _validate_confidence_level(confidence_level)
        if moments.n < 2 or moments.standard_error is None:
            raise InsufficientSamplesError(
                "Student-t interval requires at least two independent observations; "
                "sample variance and standard error are undefined for n=1."
            )
        critical_value = _student_t_quantile(
            (1.0 + confidence_level) / 2.0,
            degrees_of_freedom=moments.n - 1,
        )
        half_width = critical_value * moments.standard_error
        return ConfidenceInterval(
            lower=moments.mean - half_width,
            upper=moments.mean + half_width,
            confidence_level=confidence_level,
            method=self.name,
            capability=self.capability,
            look_index=look_index,
            method_parameters=(("degrees_of_freedom", moments.n - 1),),
        )


@dataclass(frozen=True, slots=True)
class HoeffdingConfidenceSequence:
    """Anytime-valid confidence sequence for bounded independent observations.

    At look ``k`` this method spends ``alpha * 6 / (pi^2 * k^2)`` and applies a
    two-sided Hoeffding bound.  The union bound therefore covers all analysis
    looks with total error at most ``alpha``.  It is conservative by design.
    Adaptive callers must use each positive look index at most once and in
    increasing order; :class:`AdaptiveEvaluationRunner` enforces that contract.
    """

    lower_bound: float
    upper_bound: float
    name: ClassVar[str] = "hoeffding_confidence_sequence"
    capability: ClassVar[IntervalCapability] = IntervalCapability.ANYTIME_VALID

    def __post_init__(self) -> None:
        lower = float(self.lower_bound)
        upper = float(self.upper_bound)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("Hoeffding observation bounds must be finite.")
        if lower >= upper:
            raise ValueError("Hoeffding lower_bound must be less than upper_bound.")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)

    def interval(
        self,
        values: Sequence[float],
        moments: SampleMoments,
        confidence_level: float,
        *,
        look_index: int,
    ) -> ConfidenceInterval:
        _validate_confidence_level(confidence_level)
        if look_index < 1:
            raise ValueError("look_index must be positive.")
        outside = [
            value
            for value in values
            if value < self.lower_bound or value > self.upper_bound
        ]
        if outside:
            raise ValueError(
                "Hoeffding confidence sequence received an observation outside "
                f"[{self.lower_bound}, {self.upper_bound}]."
            )

        alpha = 1.0 - confidence_level
        look_alpha = alpha * 6.0 / (math.pi**2 * look_index**2)
        value_range = self.upper_bound - self.lower_bound
        half_width = value_range * math.sqrt(
            math.log(2.0 / look_alpha) / (2.0 * moments.n)
        )
        return ConfidenceInterval(
            lower=moments.mean - half_width,
            upper=moments.mean + half_width,
            confidence_level=confidence_level,
            method=self.name,
            capability=self.capability,
            look_index=look_index,
            method_parameters=(
                ("lower_bound", self.lower_bound),
                ("upper_bound", self.upper_bound),
                ("look_alpha", look_alpha),
                ("alpha_spending", "6/(pi^2*look_index^2)"),
            ),
        )


@dataclass(frozen=True, slots=True)
class Estimate:
    """Sample moments paired with a named two-sided interval."""

    moments: SampleMoments
    confidence_interval: ConfidenceInterval

    @property
    def n(self) -> int:
        return self.moments.n

    @property
    def mean(self) -> float:
        return self.moments.mean

    @property
    def sample_variance(self) -> float | None:
        return self.moments.sample_variance

    @property
    def standard_error(self) -> float | None:
        return self.moments.standard_error

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.moments.as_dict(),
            "confidence_interval": self.confidence_interval.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class StratumEstimate:
    stratum: str
    estimate: Estimate

    def as_dict(self) -> dict[str, Any]:
        return {"stratum": self.stratum, "estimate": self.estimate.as_dict()}


@dataclass(frozen=True, slots=True)
class StratifiedEstimate:
    """Preweighted estimand and per-stratum estimates with simultaneous coverage."""

    overall: Estimate
    strata: tuple[StratumEstimate, ...]
    stratum_weights: tuple[tuple[str, float], ...]
    family_confidence_level: float
    per_interval_confidence_level: float
    simultaneous: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.as_dict(),
            "strata": [item.as_dict() for item in self.strata],
            "stratum_weights": dict(self.stratum_weights),
            "family_confidence_level": self.family_confidence_level,
            "per_interval_confidence_level": self.per_interval_confidence_level,
            "simultaneous": self.simultaneous,
        }


def sample_moments(
    observations: Iterable[Observation | float],
) -> SampleMoments:
    """Compute mean, Bessel-corrected variance, and standard error via Welford."""

    values = _coerce_observations(observations)
    if not values:
        raise InsufficientSamplesError("At least one observation is required.")

    mean = 0.0
    sum_squared_deviations = 0.0
    for index, observation in enumerate(values, start=1):
        delta = observation.value - mean
        mean += delta / index
        sum_squared_deviations += delta * (observation.value - mean)

    if len(values) == 1:
        return SampleMoments(
            n=1,
            mean=mean,
            sample_variance=None,
            standard_error=None,
        )

    variance = max(0.0, sum_squared_deviations / (len(values) - 1))
    if not math.isfinite(mean) or not math.isfinite(variance):
        raise ValueError("Sample moments overflowed; rescale the observations.")
    return SampleMoments(
        n=len(values),
        mean=mean,
        sample_variance=variance,
        standard_error=math.sqrt(variance / len(values)),
    )


def estimate(
    observations: Iterable[Observation | float],
    *,
    interval_method: IntervalMethod,
    confidence_level: float = 0.95,
    look_index: int = 1,
) -> Estimate:
    """Estimate one scalar mean using an explicitly selected interval method."""

    values = _coerce_observations(observations)
    moments = sample_moments(values)
    interval = interval_method.interval(
        tuple(observation.value for observation in values),
        moments,
        confidence_level,
        look_index=look_index,
    )
    return Estimate(moments=moments, confidence_interval=interval)


def aggregate_by_stratum(
    observations: Iterable[Observation],
    *,
    interval_method: IntervalMethod,
    stratum_weights: Mapping[str, float],
    confidence_level: float = 0.95,
    look_index: int = 1,
) -> StratifiedEstimate:
    """Estimate a predeclared weighted mean with simultaneous stratum intervals.

    ``stratum_weights`` defines both the complete expected stratum set and the
    population estimand. Missing or unexpected strata are rejected. Bonferroni
    allocation gives simultaneous coverage for all stratum intervals; summing
    their weighted bounds yields a conservative interval for the weighted mean.
    """

    values = _coerce_observations(observations)
    if not values:
        raise InsufficientSamplesError("At least one observation is required.")
    groups: dict[str, list[Observation]] = {}
    for observation in values:
        groups.setdefault(observation.stratum, []).append(observation)

    _validate_confidence_level(confidence_level)
    weights = _validated_stratum_weights(stratum_weights)
    observed = set(groups)
    expected = set(weights)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "Observed strata do not match the predeclared estimand "
            f"(missing={missing}, unexpected={unexpected})."
        )
    per_interval_confidence = 1.0 - (1.0 - confidence_level) / len(weights)
    strata = tuple(
        StratumEstimate(
            stratum=stratum,
            estimate=estimate(
                groups[stratum],
                interval_method=interval_method,
                confidence_level=per_interval_confidence,
                look_index=look_index,
            ),
        )
        for stratum in sorted(weights)
    )
    by_stratum = {item.stratum: item.estimate for item in strata}
    mean = sum(weights[name] * by_stratum[name].mean for name in weights)
    lower = sum(
        weights[name] * by_stratum[name].confidence_interval.lower
        for name in weights
    )
    upper = sum(
        weights[name] * by_stratum[name].confidence_interval.upper
        for name in weights
    )
    standard_errors = [
        (weights[name], by_stratum[name].standard_error) for name in weights
    ]
    standard_error = (
        math.sqrt(sum(weight * weight * float(error) ** 2 for weight, error in standard_errors))
        if all(error is not None for _, error in standard_errors)
        else None
    )
    overall = Estimate(
        moments=SampleMoments(
            n=sum(item.estimate.n for item in strata),
            mean=mean,
            sample_variance=None,
            standard_error=standard_error,
        ),
        confidence_interval=ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence_level=confidence_level,
            method=f"stratified_bonferroni:{interval_method.name}",
            capability=interval_method.capability,
            look_index=look_index,
            method_parameters=(
                ("base_method", interval_method.name),
                ("strata_count", len(weights)),
                ("weighting", "predeclared_population_weights"),
            ),
        ),
    )
    return StratifiedEstimate(
        overall=overall,
        strata=strata,
        stratum_weights=tuple((name, weights[name]) for name in sorted(weights)),
        family_confidence_level=confidence_level,
        per_interval_confidence_level=per_interval_confidence,
        simultaneous=True,
    )


def _validated_stratum_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("stratum_weights must be a non-empty mapping.")
    normalized: dict[str, float] = {}
    for raw_name, raw_weight in weights.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("stratum_weights contains an empty stratum name.")
        if isinstance(raw_weight, bool):
            raise ValueError("stratum weights must be finite positive numbers.")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("stratum weights must be finite positive numbers.")
        normalized[name] = weight
    total = math.fsum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"stratum_weights must sum to 1.0; observed {total!r}.")
    return normalized


def _coerce_observations(
    observations: Iterable[Observation | float],
) -> tuple[Observation, ...]:
    coerced: list[Observation] = []
    for item in observations:
        coerced.append(item if isinstance(item, Observation) else Observation(item))
    return tuple(coerced)


def _validate_confidence_level(confidence_level: float) -> None:
    if (
        isinstance(confidence_level, bool)
        or not math.isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence_level must be a finite number between 0 and 1.")


def _student_t_quantile(probability: float, *, degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive.")
    if not 0.5 < probability < 1.0:
        raise ValueError("Only positive upper Student-t quantiles are supported.")

    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
        if upper > 1e16:  # pragma: no cover - defensive for pathological floats
            raise ArithmeticError("Student-t quantile search did not converge.")

    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        x,
        degrees_of_freedom / 2.0,
        0.5,
    )
    return 1.0 - tail if value > 0.0 else tail


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 200
    epsilon = 3e-14
    minimum_float = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum_float:
        d = minimum_float
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + even) * (a + even)
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum_float:
            d = minimum_float
        c = 1.0 + coefficient / c
        if abs(c) < minimum_float:
            c = minimum_float
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum_float:
            d = minimum_float
        c = 1.0 + coefficient / c
        if abs(c) < minimum_float:
            c = minimum_float
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < epsilon:
            return result
    raise ArithmeticError("Incomplete-beta continued fraction did not converge.")
