# ADR 0054: Fresh comparisons for noisy archive admission

## Decision

Offer an opt-in `fresh_comparison` admission policy for a single positive
performance objective in one shared archive with a frozen feature projection.
Evaluate a candidate against the **current member of its target cell**, not
necessarily its parent. Use that new comparison to decide replacement. Keep
ordinary Pareto admission as the default.

The evaluator still produces the candidate's actual metric. Loreley does not
replace it with a confidence bound or a fabricated score that beats the old
member. The incumbent's historical metric remains unchanged; its fresh
measurement belongs to the current comparison's evidence.

## Why not keep comparing permanent baseline-relative scores?

A fixed baseline makes scores comparable, but does not remove selection bias.
An archive retains versions partly because their measured scores were lucky.
Reusing that same high score can obstruct later genuine improvements even when
every individual estimate was originally unbiased.

For illustration, consider one cell containing the measured winner of 100
equally good versions. With independent normal score errors of standard error
0.04 percentage points, a genuinely 0.1-point better candidate beats that old
maximum only **49.98%** of the time. Remeasuring the same incumbent at the same
precision raises ordinary correct ordering to **96.15%**:

```text
permanent winner: integral phi(z) Phi(z + 2.5)^100 dz = 0.499812
fresh comparison: Phi(2.5 / sqrt(2))                = 0.961450
```

This calculation isolates selection bias, not the confidence gate below. It
is not a benchmark result, an equal-total-cost experiment, or a prediction that
all jobs visit one cell. With a prespecified opponent and stable measurements,
old data can be just as useful as new data. Fresh comparisons prevent one old
measurement's luck from becoming a permanent admission obstacle; they do not
eliminate noise or guarantee that generation proposes useful changes.

This problem is documented in [Inference on Winners](https://academic.oup.com/qje/article/139/1/305/7276491)
and [MAP-Elites for Noisy Domains by Adaptive Sampling](https://pure.itu.dk/ws/files/84783110/map_elites_noisy_justesen.pdf).

## Cost and engineering trade-off

- If evaluation already runs candidate/baseline pairs, replacing the baseline
  with the incumbent does not add a third execution. Against candidate-only
  evaluation with a reused old score, fresh incumbent measurement costs more.
- A confidence gate reduces false replacements but also rejects some genuine
  improvements below its designed effect size. A 90% detection target at 0.1%
  is not a claim about all smaller improvements or whole-run error rates.
- The minimal core change is a guarded cell replacement. Updating both versions'
  global metrics or building a general historical-estimation database would add
  synchronization work without strengthening this fresh comparison's guarantee.
- The evaluator owns workload validity, statistical assumptions, confidence
  construction and power planning. Core checks evidence consistency and applies
  the declared decision; a reported confidence level is not proof of coverage.

## Execution contract

After normal seed bootstrap has fitted PCA, the worker prepares the candidate,
then obtains a core-issued `EvaluationContext.comparison` immediately before
measurement. It identifies the candidate, current incumbent (or `None` for an
empty cell), objective, cell, projection fingerprint and unique context ID.
Exact-tree and persistent measurement reuse cannot replace this measurement.

For an occupied cell, a phased evaluator returns the actual candidate metric
and these fields in `EvaluationResult.extra["fresh_comparison"]`:

```python
{
    "context_id": context.comparison["context_id"],
    "complete": True,
    "incumbent_value": measured_incumbent_cost,
    "improvement_lower_bound": lower_bound,
    "confidence_level": 0.95,
    "sample_count": complete_comparison_rounds,
}
```

The bound is **one-sided**, in `100 * log(incumbent/candidate)` for a
lower-is-better metric; invert the ratio for higher-is-better. Values must be
positive and finite. The bound cannot exceed the observed gain. Incomplete or
invalid evidence is an infrastructure failure, not a valid rejection. Empty
cells supply only `context_id` and `complete`, plus the actual metric; they must
not invent an incumbent or comparison statistics.

Core validates the result against its durable context and successful worker
attempt. Ingestion rechecks the current cell, incumbent and projection, then
atomically replaces the member iff the lower bound is positive. **Historical
point-score dominance does not run again after this decision.** A valid rejected
candidate retains its metrics and evidence.

One prepared cycle fences subsequent measurement until ingestion commits. The
evaluator returns first; waiting for ingestion before returning would deadlock.
Generation and preparation remain concurrent. Ledger events, archive state and
ingestion status provide restart recovery without a new database table.
Stale or missing evidence remains an explicit ingestion failure rather than
becoming a rejection after a retry limit; repair the context/measurement before
resuming. An abandoned worker run cannot publish evidence for its replacement.

## Configuration

Set `MAPELITES_ADMISSION_POLICY=fresh_comparison`, one `MAPELITES_OBJECTIVES`
entry, one `MAPELITES_ISLANDS` entry, `MAPELITES_DIMENSION_REDUCTION_REFIT_INTERVAL=0`,
`MAPELITES_MIGRATION_INTERVAL_JOBS=0` and `WORKER_EVALUATOR_MAX_CONCURRENCY=1`.
Use a phased-v1 evaluator and a timeout covering preparation, queueing and
measurement. `MAPELITES_COMPARISON_CONFIDENCE` defaults to `0.95`.
Bootstrap seeds retain normal admission; run fresh search only after their
ingestion completes. Do not change policy or projection during an active cycle.

Tests must cover a lucky old score, valid rejection, empty cells, stale contexts,
attempt/campaign identity, cache bypass, rollback, restart and duplicate ACKs.
Default-policy regression tests remain required.
