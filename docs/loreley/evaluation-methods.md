# Reliable evaluation methods

`loreley.core.evaluation` provides project-neutral statistical building blocks
for evaluator authors. It handles finite observations, confidence intervals,
predeclared strata, bounded adaptive budgets, and canonical measurement
fingerprints. The target evaluator still owns benchmark commands, workloads,
metric definitions, validity assumptions, and pass/fail policy.

## Fixed-sample estimates

Use `StudentTInterval` only when the sample size is fixed before measurement:

```python
from loreley.core.evaluation import StudentTInterval, estimate

result = estimate(
    [1.012, 1.009, 1.015, 1.011],
    interval_method=StudentTInterval(),
    confidence_level=0.95,
)
```

The result records sample moments, interval bounds, method, capability, and look
index. Student-t rejects fewer than two observations and is marked
`fixed_sample_only`; it assumes independent, approximately normal samples.

## Adaptive budgets

`AdaptiveEvaluationRunner` requests bounded batches until it reaches a sample,
wall-time, precision, or effect-decision boundary:

```python
from loreley.core.evaluation import (
    AdaptiveEvaluationRunner,
    AdaptiveSamplingConfig,
    HoeffdingConfidenceSequence,
)

config = AdaptiveSamplingConfig(
    min_samples=8,
    max_samples=64,
    batch_size=4,
    max_wall_time_seconds=300,
    target_ci_half_width=0.01,
    effect_threshold=1.0,
    indifference_zone=0.002,
    stratum_weights={"small": 0.4, "large": 0.6},
)
runner = AdaptiveEvaluationRunner(
    config,
    interval_method=HoeffdingConfidenceSequence(
        lower_bound=0.5,
        upper_bound=1.5,
    ),
)
result = runner.run(run_next_benchmark_batch)
```

The callback receives a `SampleRequest` with the exact remaining sample and
wall-time budget. It must not return more observations than requested. The
result contains every analysis look, trigger considered, stop reason, and the
complete JSON-safe decision history.

Before a finalizer turns this evidence into a passing evaluation, check
`result.decision_ready`. It is false when sampling stopped before a declared
effect or precision target, when too few samples produced an estimate, or when
the explicit unsafe fixed-sample override was used. `inference_valid` and
`declared_target_reached` expose the two parts separately. A target may retain
an inconclusive result as evidence, but should not mark that measurement
cacheable and passing.

Persist `result.checkpoint().as_dict()` with evaluator evidence when a target
supports recovery. Reconstructing and passing the checkpoint to `run` preserves
observations, elapsed budget, batch ordinal, and completed look index, so an
anytime-valid alpha-spending sequence does not restart at look one.

Repeatedly inspecting a conventional fixed-sample interval and stopping when it
looks favorable does not preserve its nominal coverage. Loreley therefore
rejects a fixed-sample method when optional stopping is possible. An explicit
unsafe override exists for reproducing legacy protocols and is recorded in the
result. `HoeffdingConfidenceSequence` is anytime-valid under its declared
independence, stable-mean, and finite-bound assumptions, but can be conservative.

## Stratified estimates

Use `aggregate_by_stratum` when the estimand has predeclared population strata:

```python
from loreley.core.evaluation import (
    Observation,
    StudentTInterval,
    aggregate_by_stratum,
)

result = aggregate_by_stratum(
    [
        Observation(1.02, "small"),
        Observation(1.01, "small"),
        Observation(1.00, "large"),
        Observation(1.03, "large"),
    ],
    interval_method=StudentTInterval(),
    stratum_weights=config.stratum_weights,
)
```

Weights must be positive and sum to one. Missing or unexpected strata fail
closed. Per-stratum intervals use Bonferroni allocation, and the overall bound
is the corresponding preweighted conservative interval. This avoids silently
turning an imbalanced sample into a different estimand.

## Measurement fingerprints

Create a `MeasurementContract` from everything that changes what a measurement
means:

```python
from loreley.core.evaluation import MeasurementContract

contract = MeasurementContract(
    workload_fingerprint="sha256:corpus-and-command-manifest",
    metric_name="throughput_ratio",
    metric_unit="ratio",
    estimand="weighted mean paired candidate/root throughput",
    higher_is_better=True,
    interval_method=runner.interval_method,
    sampling=config,
    stratum_weights={"small": 0.4, "large": 0.6},
    metadata={"pairing": "candidate-root", "compiler": "clang-18"},
)
```

Pass `contract.fingerprint` as
`EvaluationPreparation.measurement_contract_fingerprint` in a `phased-v1`
evaluator. Changes to the workload, metric, interval, adaptive budget, strata,
or metadata then produce a different measurement cache key. The contract does
not prove that its assumptions hold; the evaluator and experiment design remain
responsible for that evidence.
