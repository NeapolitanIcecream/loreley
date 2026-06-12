# Evaluator-Guided In-Loop Rework

Status: Implemented

Date: 2026-06-12

## Summary

Loreley now treats evaluator-reported candidate failures as feedback for the
same worker job instead of scheduling a separate repair-pool job.

Evaluator authors use a small contract:

- return `EvalPass(...)` when the candidate is acceptable;
- return `EvalFail(...)` when the candidate owns the failure and another coding
  pass may fix it;
- raise an exception when the evaluator or its environment failed.

The worker runs planning once, then repeats coding/evaluation within the same
job while the rework budget allows it. Failed attempts are not published as
candidate commits. A final passing attempt follows the normal candidate
publication and persistence path.

## Motivation

The repair pool added a second asynchronous lane for failures that are often
cheap and deterministic: compile errors, typecheck failures, lint failures, and
fast validation/test failures. That lane preserved useful failed work, but it
also introduced extra scheduler state, token accounting, UI/API actions, and
candidate lineage concepts.

For high-signal evaluator failures, the lower-abstraction path is to give the
diagnostic back to the coding agent immediately while the job context is still
hot.

## Public Evaluator Contract

Evaluators import from `loreley.core.worker.evaluator`:

```python
from loreley.core.worker.evaluator import EvalFail, EvalPass, EvaluationContext


def plugin(context: EvaluationContext):
    if not (context.worktree / "pyproject.toml").exists():
        return EvalFail(
            kind="validation",
            summary="pyproject.toml is missing",
            details="The candidate removed the project configuration file.",
        )

    return EvalPass(summary="candidate passed validation")
```

Supported `EvalFail.kind` values are:

- `compile`
- `typecheck`
- `lint`
- `test`
- `validation`
- `benchmark`
- `other`

`EvalPass` can also carry metrics, tests executed, logs, extra data, and
artifacts. `EvalFail` can carry a bounded summary, optional bounded details,
and artifacts.

`EvaluationOutcome` remains available as an internal and advanced compatibility
envelope. Existing evaluators returning `EvaluationResult`, `EvaluationOutcome`,
or compatible mappings still work.

## Worker Lifecycle

For a normal evolution job:

1. Prepare the repository worktree and run planning once.
2. Run coding for attempt 1.
3. Run scope gate and create a local candidate commit.
4. Run the evaluator.
5. If the evaluator passes, clean evaluator side effects, record/push the final
   candidate, and persist success normally.
6. If the evaluator returns `EvalFail` and the worker rework policy allows it,
   store a bounded attempt artifact, clean evaluator side effects, reset back to
   the base commit with the failed diff left dirty, and run another coding pass
   with bounded diagnostic feedback.
7. If the failure is not eligible or the budget is exhausted, persist terminal
   job failure and bounded attempt history. Do not publish a failed candidate.

Planning intentionally does not rerun. The second coding pass receives the
original plan, a current diff summary, and a sanitized diagnostic capsule.

## Rework Policy

Defaults:

```text
WORKER_EVALUATOR_REWORK_ENABLED=true
WORKER_EVALUATOR_REWORK_MAX_ATTEMPTS=1
WORKER_EVALUATOR_REWORK_FAILURE_KINDS=compile,typecheck,lint,test,validation
WORKER_EVALUATOR_REWORK_MAX_SECONDS=1800
```

`MAX_ATTEMPTS` is the number of extra coding passes after the first failed
evaluation. The default is one rework pass.

The failure-kind allowlist is the main safety valve. `benchmark` and `other`
failures are representable, but they are not reworked by default.

Exceptions from the evaluator are evaluator/infrastructure failures and never
trigger rework.

## Artifacts And Safety

Failed attempts are stored only as bounded job artifacts. They are not written
as `CandidateCommit` rows, not pushed, and not eligible for repair-pool
scheduling.

The diagnostic sent back to the coding agent is bounded and sanitized through
the existing diagnostic capsule path. Raw evaluator output, arbitrary file
paths, and hidden artifacts are not projected into the prompt.

Before rework, the worker cleans evaluator side effects, verifies it is still on
the local failed-attempt commit, then performs a mixed reset to the original
base commit. This preserves the failed attempt diff as dirty worktree input for
the next coding pass without keeping evaluator side effects.

Before publishing a passing attempt, the worker cleans evaluator side effects
and resets to the evaluated commit.

## Repair Pool Deprecation

The repair pool remains in the database and legacy code paths for compatibility
with historical data and old jobs, but new scheduling is disabled:

- scheduler repair dispatch returns no jobs and logs a deprecation warning;
- `POST /api/v1/repair/schedule-one` returns a disabled/deprecated no-op
  response;
- Agent REST no longer recommends `repair_schedule_one` in safe next actions;
- the Streamlit navigation no longer links the Repair Pool page.

Failed-candidate audit and legacy candidate state actions are retained so old
data remains inspectable and operators can still quarantine or restore legacy
rows when needed.

## Non-Goals

- No schema migration removes repair-pool tables or columns.
- No new repair job kind is scheduled.
- No failed attempt commit is promoted into normal archive lineage.
- No benchmark-regression adjudication is added to the rework loop by default.

## Acceptance Notes

The implementation is covered by focused evaluator, worker, scheduler, API, and
UI navigation tests. The key behavioral assertions are:

- `EvalPass` and `EvalFail` coerce into the existing internal outcome model;
- evaluator exceptions remain non-rework failures;
- the first failed eval can lead to one in-loop coding retry and a final
  published passing candidate;
- exhausted or non-allowlisted failures persist terminal job failure without
  publishing a failed candidate;
- scope gate runs on every attempt;
- evaluator side-effect files do not enter the final candidate diff;
- repair-pool scheduling no longer creates repair jobs.
