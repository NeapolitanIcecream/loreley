# Proposal: Baseline auto-retry without new user configuration

## Status

Accepted / Implemented

## Date

2026-06-04

## Context

Before a campaign starts scheduling work, Loreley creates a
`campaign_baselines` record for the root commit. Later improvement measurements
use this baseline as their reference.

The default policies are:

- `BASELINE_BOOTSTRAP_POLICY=required`: the scheduler does not dispatch, seed,
  or schedule work while the baseline is unavailable.
- `BASELINE_BOOTSTRAP_POLICY=warn`: scheduling continues in a degraded state,
  but baseline deltas are unavailable.

This gate prevents the system from consuming worker budget without a reliable
reference and keeps later results comparable.

The problem was failure recovery. `BaselineBootstrapService.ensure_or_load_baseline()`
looks up an existing baseline by `baseline_key_hash`. Unless the caller passes
`force_rerun=True`, an existing row with status `valid`, `failed`, or `degraded`
was reused without another evaluation.

The scheduler does not pass `force_rerun=True`. Under the default `required`
policy, a transient evaluator failure therefore persisted a failed row. Every
later scheduler tick read that row and stopped at the baseline gate. The
evaluator was never called again, even after the environment recovered.

The visible symptom was a repeated log message such as:

```text
Scheduler tick blocked by campaign baseline status=failed
```

The initial evaluator failure could come from Docker, a service health check,
permissions, a timeout, or another external dependency. The permanent block
was caused by Loreley's baseline reuse behavior.

## Problem

The previous implementation treated every recorded non-valid baseline as a
durable conclusion:

- Reusing a `valid` row is correct.
- Permanently reusing a `failed` or `degraded` row turns a transient failure
  into a stuck campaign.
- An operator can force a rerun, but only after recognizing and intervening in
  the internal state.

New settings could control this behavior, for example:

```text
BASELINE_BOOTSTRAP_RETRY_FAILED=true
BASELINE_BOOTSTRAP_RETRY_COOLDOWN_SECONDS=300
```

That would expand the public configuration surface for scheduler recovery that
should work safely by default. Operators would need to learn the settings,
choose a cooldown, and keep deployment environments synchronized.

## Goals

- Add no user-visible configuration.
- Preserve the meaning of `BASELINE_BOOTSTRAP_POLICY`.
- Keep the baseline gate under `required` until a valid reference exists.
- Recover automatically after transient baseline failures.
- Avoid rerunning the evaluator on every scheduler tick.
- Preserve the operator's `force_rerun=True` override for non-valid rows.
- Continue reusing valid baselines without change.

## Non-goals

- Removing the baseline-first gate.
- Combining evaluator-internal retries with scheduler-level recovery.
- Requiring a database migration or `.env` change.
- Recovering from every configuration error. Repeating an evaluation cannot
  repair a missing or incorrectly configured primary metric.

## Proposed design

Treat failed and degraded baselines as retryable states with an internal
cooldown instead of permanent conclusions.

Use a conservative code constant rather than an environment variable:

```python
_BASELINE_RETRY_COOLDOWN_SECONDS = 300
```

The behavior is:

1. Always reuse a `valid` baseline.
2. Reuse a `failed` or `degraded` baseline during the cooldown.
3. After the cooldown, rerun the baseline evaluator when the failure type is
   retryable.
4. On success, update the same `campaign_baselines` row to `valid`.
5. On failure, update the row's failure details and completion time, then wait
   for the next cooldown.
6. Keep `force_rerun=True` as the operator override for a non-valid baseline.
7. Continue reusing a `valid` baseline even when `force_rerun=True`, so an
   established campaign reference cannot be replaced accidentally.

Use existing timestamps, in this order, as the cooldown reference:

1. `finished_at`
2. `updated_at`
3. `created_at`

Database rows and test doubles may provide either timezone-aware or naive
datetimes. Normalize timestamps to UTC before comparison. Interpret a naive
datetime as UTC so SQLite, PostgreSQL, and test doubles do not trigger
aware-versus-naive comparison errors.

The scheduler already holds an experiment advisory lock, but an operator
baseline request can overlap an automatic scheduler retry. The implementation
does not hold a database row lock while the external evaluator runs. An overlap
can therefore evaluate the same baseline twice. The `baseline_key_hash` unique
constraint and `_persist_baseline_attempt()` still converge both attempts onto
one durable row. This bounded duplicate evaluation is preferable to holding a
database lock across external work.

## Retry classification

Do not retry clear campaign contract errors indefinitely. The baseline service
uses a small internal classifier.

Retry these failures by default:

- `baseline_evaluation_failed`
- `evaluator_error`
- `infrastructure_error`
- `timeout`
- `worker_timeout`
- `service_unavailable`
- `evaluation_missing_result`

Do not retry these failures by default:

- `primary_metric_not_configured`
- `primary_metric_missing`
- `primary_metric_non_finite`
- `primary_metric_direction_conflict`

The first group indicates that evaluation did not complete normally and may
recover when the environment changes. The second group indicates that the
evaluator returned a result that violates the campaign contract and requires a
configuration or evaluator change.

For an unknown failure kind:

- Treat a name ending in `_error` as retryable.
- Leave other unknown failures for the operator to rerun explicitly.

## Implementation plan

The main change is in `loreley/scheduler/baselines.py`.

Add internal helpers:

```python
_BASELINE_RETRY_COOLDOWN_SECONDS = 300

def _baseline_retry_reference_time(row: CampaignBaseline) -> datetime | None:
    ...

def _baseline_retry_cooldown_elapsed(
    row: CampaignBaseline,
    *,
    now: datetime,
) -> bool:
    ...

def _baseline_timestamp_as_utc(value: datetime | None) -> datetime | None:
    ...

def _baseline_failure_is_retryable(row: CampaignBaseline) -> bool:
    ...

def _should_retry_existing_baseline(
    row: CampaignBaseline,
    *,
    now: datetime,
) -> bool:
    ...
```

Replace the unconditional recorded-status reuse in
`ensure_or_load_baseline()`:

```python
if existing is not None and existing.status in _RECORDED_BASELINE_STATUSES:
    return self._result_from_row(existing, key_hash=key.hash, policy=policy)
```

with status-specific behavior:

```python
if existing is not None and existing.status == BASELINE_STATUS_VALID:
    return self._result_from_row(existing, key_hash=key.hash, policy=policy)

if (
    existing is not None
    and existing.status in {BASELINE_STATUS_FAILED, BASELINE_STATUS_DEGRADED}
    and not force_rerun
    and not _should_retry_existing_baseline(
        existing,
        now=datetime.now(timezone.utc),
    )
):
    return self._result_from_row(existing, key_hash=key.hash, policy=policy)

# Otherwise, evaluate and persist into the same row.
```

Log each automatic retry through the existing `scheduler.baselines` logger.
Include the key, previous status, failure kind, and cooldown:

```text
Campaign baseline retrying key=<hash> status=failed failure_kind=evaluator_error cooldown_seconds=300
```

Persist retry results through `_persist_baseline_attempt()`. It already finds
the row by key and updates its status, metric, failure summary, and timestamps.

## User-facing behavior

No configuration change is required.

`BASELINE_BOOTSTRAP_POLICY=required` continues to mean:

- A valid baseline permits scheduling.
- An invalid baseline blocks scheduling.

When the invalid state is retryable, the scheduler now evaluates it again after
the cooldown instead of remaining blocked indefinitely.

`BASELINE_BOOTSTRAP_POLICY=warn` also benefits. Scheduling can continue while
the baseline is degraded, and the service later attempts to restore a valid
baseline so baseline deltas become available again.

## Acceptance criteria

- A failed same-key baseline does not block a campaign permanently.
- The scheduler does not rerun the baseline evaluator on every tick.
- A valid baseline is never rerun.
- A failed or degraded baseline is reused during the cooldown.
- A successful retry updates the same row to `valid`.
- A failed retry updates the same row's failure details and `finished_at`.
- `force_rerun=True` immediately reruns a non-valid baseline.
- Under `required`, scheduling remains blocked until the baseline is valid.
- Under `warn`, a degraded baseline continues to permit scheduling.

## Test plan

Run the focused tests:

```bash
uv run pytest \
  tests/scheduler/test_baseline_bootstrap.py \
  tests/scheduler/test_baseline_scheduler_gate.py
```

Add or update tests for the following cases:

- A failed row does not rerun the evaluator during the cooldown.
- A failed row reruns the evaluator after the cooldown.
- A successful retry updates the same row to `valid`.
- A failed retry updates the failure summary and completion time.
- A valid row is not rerun after the cooldown.
- A non-retryable failure kind does not rerun automatically.
- `force_rerun=True` immediately reruns a non-valid row.
- The scheduler stops returning `baseline_blocked=1` after a successful retry.

Run the related suite after implementation:

```bash
uv run pytest \
  tests/scheduler \
  tests/api/test_operator_routes.py \
  tests/api/test_app.py
```

## Risks

The retry classifier can mark a persistent failure as transient. The cooldown
prevents a busy loop, and the classifier excludes known campaign contract
errors.

Periodic retries can also make a persistent environment problem less obvious.
Each attempt must therefore log the key, previous status, failure kind, and
cooldown.

A scheduler and an operator can both decide to retry the same non-valid row
before either persists its result. This overlap is rare, does not hold database
locks across evaluator work, and converges through the unique
`baseline_key_hash` row.

## Alternatives considered

### Add retry configuration

Rejected for now. It offers flexibility but expands the public configuration
surface for behavior that should have a safe default.

### Put retry logic in the scheduler main loop

Rejected. The scheduler should ask whether a baseline is ready. The decision to
reuse or evaluate a baseline belongs in `BaselineBootstrapService`, next to
baseline keying and persistence.

### Retry every failed or degraded baseline unconditionally

Rejected. This would spend evaluator budget on contract errors such as a
missing primary metric. The internal classifier provides safer default behavior
without new configuration.
