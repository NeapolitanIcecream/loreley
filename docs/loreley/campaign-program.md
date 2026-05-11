# Campaign Program

Use `loreley.program.md` at the target repository root to define the campaign contract for a Loreley run.

The file is optional. If it is missing, jobs keep the existing behavior and store `campaign_program_hash = null`.

## Supported Sections

Loreley reads these Markdown sections:

- `Goal`
- `Primary metric`
- `Correctness gates`
- `Editable scope`
- `Protected scope`
- `Evaluation budget`
- `Complexity policy`
- `Failure policy`
- `Logging policy`

Unknown sections stay in the stored raw Markdown and snapshot metadata, but they do not affect scheduling, prompts, evaluator payloads, or scope checks.

## Example

```markdown
# Parser throughput campaign

## Goal
Improve parser throughput without changing the public API.

## Primary metric
name: throughput
direction: higher_is_better
unit: req/s

## Correctness gates
- uv run pytest tests/parser
- uv run ruff check src/parser tests/parser

## Editable scope
- src/parser/**
- tests/parser/**

## Protected scope
- docs/contracts/**

## Evaluation budget
- Keep per-candidate evaluation under 10 minutes.

## Complexity policy
- Prefer smaller diffs when metric deltas are close.

## Failure policy
- Repair typecheck and test failures only when diagnostics are bounded.

## Logging policy
- Summaries should mention metric value, gates, and notable tradeoffs.
```

`Primary metric` works best as key/value lines:

- `name`: metric name expected from the evaluator.
- `direction`: `higher_is_better` or `lower_is_better`.
- `unit`: optional display unit.

If the evaluator does not return the named metric, or returns the opposite direction, Loreley records a warning in the evaluation result. The evaluator result still wins.

## Job Projection

When the scheduler creates seed or evolution jobs, it stores the campaign program snapshot and writes the program hash to the job. Program fields fill empty job fields:

- `Goal` fills `EvolutionJob.goal` unless the caller supplied a more specific goal.
- Correctness gates, editable/protected scope, evaluation budget, and failure policy fill `constraints`.
- Primary metric and correctness gates fill `acceptance_criteria`.
- Complexity and logging policy fill `notes`.

Repair jobs inherit the source candidate's `campaign_program_hash` by default, so repair lineage stays under the original campaign contract.

## Baseline Bootstrap

The active campaign program participates in the root baseline key. Before the
scheduler dispatches or schedules mutation work, it asks
`BaselineBootstrapService` to load or create a matching row in
`campaign_baselines`.

`BASELINE_BOOTSTRAP_POLICY` controls failure handling:

- `required` (default): missing, failed, or degraded baseline state blocks
  dispatch, seed scheduling, repair scheduling, and normal sampler jobs.
- `warn`: Loreley records degraded baseline state and keeps scheduling, but
  baseline deltas stay unavailable.

Changing `loreley.program.md` changes the campaign program hash and therefore
requires a separate baseline for the new contract. Valid same-key baselines are
reused; failed or degraded same-key baselines can be rerun from the operator
console or Agent REST facade.

## Scope Gate

Workers enforce editable/protected scope after the coding agent leaves a modified worktree and before commit, push, or evaluation.

Rules:

- Patterns are repository-relative POSIX paths.
- Protected scope wins over editable scope.
- Empty editable scope allows tracked repository files except protected paths.
- Non-empty editable scope allows only matching paths except protected paths.
- `loreley.program.md` is always protected.
- Tracked modifications and untracked files are checked.
- `.git` internals and Loreley artifact stores are ignored.
- Absolute paths, path traversal, non-POSIX patterns, and unsafe symlink targets fail the gate.

Scope violations are written as structured failure artifacts with `failure_kind = campaign_scope_violation`.

## Change Policy

Set `CAMPAIGN_PROGRAM_CHANGE_POLICY` to control changes during a running scheduler:

- `locked` (default): read the program at scheduler startup and keep using that hash. If the file changes, the scheduler reports the new hash but does not use it.
- `auto`: adopt and persist the changed program automatically for future jobs. Already queued/running jobs keep their original hash.

`approve` is not accepted until an operator approval workflow is implemented;
use `locked` or `auto`.

`LORELEY_PROFILE` and other runtime settings do not belong in `loreley.program.md`. Loreley records runtime profile and a masked effective settings fingerprint with evaluator context instead.
