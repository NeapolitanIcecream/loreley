# ADR 0049: Campaign program contract

Date: 2026-05-07

Status: Draft

Related: [ADR 0032](0032-simplify-worker-prompts-freeform-default.md),
[ADR 0033](0033-remove-schema-validation-and-simplify-agent-outputs.md),
[ADR 0045](0045-config-profiles-for-large-repo-campaigns.md),
[ADR 0046](0046-agent-visible-evaluation-artifacts.md),
[ADR 0048](0048-failed-candidate-repair-pool.md)

## Context

Loreley already has prompt scaffolding and job-level task fields:

- `EvolutionJob` stores `goal`, `constraints`, `acceptance_criteria`,
  `notes`, `tags`, and `iteration_hint`.
- planning and coding agents receive a shared prompt packet with the evolution
  goal, worker contract, iteration context, base commit, and inspirations.
- the evaluator receives a structured payload containing job goal, constraints,
  acceptance criteria, notes, tags, plan summary, and guardrails.
- worker contracts already prevent agent-local commits, pushes, interactive
  clarification, and framework-managed evaluator runs.

This is enough for the worker to execute individual jobs, but it does not yet
provide a single campaign-level source of truth for an autonomous run.

Campaign rules are currently distributed across environment variables,
scheduler defaults, worker prompt constants, evaluator implementation, and
operator memory. The scheduler also creates seed and repair jobs with empty
constraints and acceptance criteria unless a caller explicitly supplies them.
As a result, later review of a candidate can answer "what prompt did this agent
receive?" but cannot always answer "what campaign contract was this candidate
created and evaluated under?"

Autonomous research loops such as `karpathy/autoresearch` demonstrate a useful
counterpoint. Their implementation is intentionally narrow, but their durable
advantage is the explicit experiment contract: one file describes what humans
edit, what agents may edit, which files are immutable, the single comparable
metric, the fixed time budget, the logging shape, and the keep/discard policy.
Loreley should not copy the single-file or single-champion search model, but it
should adopt the idea that a campaign has one clear, versioned protocol.

This ADR does not reverse ADR 0032 or ADR 0033. Planning and coding agent
outputs should remain freeform Markdown. The missing structure is on the input
side: a campaign contract that is authored once, projected into jobs, and
recorded with every result.

## Decision

Introduce a campaign program contract as the canonical operator-authored
description of a Loreley campaign.

The default source file is `loreley.program.md` in the target repository root.
Operators may provide an alternate path later, but the default should be
conventional and easy to discover.

The campaign program is a human-readable Markdown document with a small set of
recognized sections. Loreley parses the recognized sections into a normalized
program snapshot, preserves the raw Markdown, computes a content hash, and
stores that snapshot in a content-addressed `campaign_programs` table from the
first implementation.

The initial recognized sections are:

- `Goal`: the campaign objective in operator language.
- `Primary metric`: metric name, direction, and unit when known.
- `Correctness gates`: non-negotiable validation requirements.
- `Editable scope`: paths or areas the agent may modify.
- `Protected scope`: paths or areas the agent must not modify.
- `Evaluation budget`: wall-clock, resource, retry, or benchmark budget.
- `Complexity policy`: how to trade small metric deltas against diff size,
  dependency changes, churn, and maintainability.
- `Failure policy`: how to classify, retry, repair, or discard failed
  candidates.
- `Logging policy`: required human-readable result fields for campaign review.

Unknown sections remain in the raw Markdown but are not interpreted by core
control flow. This keeps the document useful to humans without making every
heading a framework feature.

The normalized snapshot has a small, stable schema:

```text
CampaignProgramSnapshot
- schema_version
- source_path
- raw_sha256
- normalized_sha256 nullable
- title nullable
- goal nullable
- primary_metric nullable
- correctness_gates[]
- editable_scope[]
- protected_scope[]
- evaluation_budget[]
- complexity_policy[]
- failure_policy[]
- logging_policy[]
- recognized_sections[]
- unknown_sections[]
- parse_warnings[]
```

`raw_sha256` is computed over the exact source bytes and is the canonical
program hash for audit. `normalized_sha256` is optional and may be used later to
detect near-equivalent formatting-only edits. Neither hash is a security
boundary.

The first persistence model is:

```text
campaign_programs
- hash primary key
- schema_version
- source_path
- title nullable
- raw_markdown or raw_artifact_path
- normalized_snapshot json
- recognized_sections[]
- parse_warnings json
- created_at

evolution_jobs.campaign_program_hash nullable indexed
evaluation_attempts.campaign_program_hash nullable indexed
candidate_commits.campaign_program_hash nullable indexed
```

Job artifacts may still write a cold-path copy of the raw Markdown and
normalized snapshot, but the table is the queryable source for cross-job
comparison.

The scheduler uses the normalized program snapshot when creating jobs:

- `goal` is filled from the program unless the enqueue path supplies a more
  specific per-job goal.
- `constraints` are derived from correctness gates, editable scope, protected
  scope, evaluation budget, and failure policy.
- `acceptance_criteria` are derived from the primary metric, correctness gates,
  and any explicit success criteria.
- `notes` may include complexity and logging policy summaries.
- `iteration_hint` remains a per-job fact, not a campaign-level rule.

Hard constraints and guidance are separated:

- correctness gates, editable scope, protected scope, and evaluation budget are
  hard campaign constraints;
- complexity policy, failure policy, and logging policy are prompt/evaluator
  guidance in the MVP unless a specific field is later promoted to structured
  enforcement.

The `Primary metric` section should use a simple key/value form when possible:

```markdown
## Primary metric
name: throughput
direction: higher_is_better
unit: req/s
```

If evaluator output does not contain the named metric, or if its
`higher_is_better` direction conflicts with the campaign program, the MVP
records a warning and evaluator truth still wins. A later ADR may choose a
stricter `inconclusive` or failure policy once operators have run enough
campaigns with program metadata.

The shared planning/coding prompt packet must explicitly render constraints
and acceptance criteria. They should be shown as bounded bullet lists near the
goal and worker contract so agents do not have to infer campaign boundaries
from environment variables or evaluator behavior. Prompts receive bounded
projections of the program, not the raw Markdown.

The evaluator receives the normalized program snapshot, raw program hash, and
program-derived constraints in `EvaluationContext.payload`. Evaluators remain
responsible for enforcing domain-specific correctness and metric semantics, but
they should be able to report which program version they evaluated against.

Every candidate should be traceable to the program version used to create it:

- job artifacts store the raw program Markdown and normalized snapshot;
- `EvolutionJob` or associated artifacts record `campaign_program_hash`;
- evaluator attempts record the same hash in their payload or metadata;
- candidate commits record the same hash to simplify UI and lineage queries;
- operator-facing summaries include the hash and a short program title when
  available.

Program changes during a running campaign are not accepted silently. Add:

```text
CAMPAIGN_PROGRAM_CHANGE_POLICY=locked | approve | auto
```

The default is `locked`: the scheduler reads the active program snapshot at
startup and continues creating new jobs with that hash. If the file changes,
the scheduler reports the new hash but does not use it until the operator
restarts or explicitly accepts it.

`approve` pauses creation of new jobs when the program hash changes and asks
the operator whether future jobs should use the new hash. Already queued and
running jobs keep their original hash. `auto` is allowed only for experimental
workflows and must make mixed program versions obvious in UI and exports.

Repair jobs inherit the `campaign_program_hash` of their source failed
candidate by default. Repairing old failed work under a new active program is a
separate explicit operator action, because otherwise failure policy, protected
scope, primary metric, and acceptance criteria can become mixed within one
repair lineage.

`LORELEY_PROFILE` and other runtime defaults remain environment/runtime
configuration, not campaign program content. Profiles from ADR 0045 affect
scaling-oriented defaults and may depend on machine or repository size. They
must be recorded alongside the program hash, not embedded in the program.

Result comparison should include at least:

```text
campaign_program_hash
runtime_profile
effective_settings_fingerprint
evaluator_name
evaluator_version
root_commit_hash
```

Protected scope is enforced in two layers:

1. the worker runs a cheap path-level diff check after the coding agent leaves a
   modified worktree and before commit, push, or evaluation;
2. the evaluator enforces semantic and domain-specific constraints that cannot
   be reduced to path patterns.

Scope rules are repo-relative and deterministic:

- all scope patterns are repository-relative POSIX paths;
- `protected_scope` wins over `editable_scope`;
- if `editable_scope` is empty, tracked repository files are editable except
  protected paths;
- if `editable_scope` is non-empty, only paths matching editable scope and not
  protected scope are editable;
- `loreley.program.md` is protected by default even when the operator omits it;
- the worker checks tracked modifications and untracked files;
- `.git` internals and Loreley artifact stores are ignored;
- absolute paths, path traversal, and unsafe symlink targets are rejected.

## Non-Goals

- Do not make planning/coding outputs schema-validated. ADR 0032 and ADR 0033
  still stand.
- Do not restrict Loreley to single-file edits. Whole-repository evolution
  remains the primary design.
- Do not replace MAP-Elites with a champion-only keep/discard branch loop.
- Do not make Markdown parsing a large configuration language.
- Do not let the campaign program override evaluator truth. It informs the
  evaluator and prompts, but the evaluator still owns structured results.
- Do not put raw evaluator artifacts or untrusted logs into the campaign
  program.
- Do not require existing campaigns to add a program file immediately.
- Do not allow agents to modify the active campaign program as part of a normal
  candidate diff.

## Consequences

Campaigns become easier to review. A candidate can be understood together with
the exact goal, metric direction, budget, scope limits, and complexity policy
active when the worker produced it.

Prompt behavior becomes less dependent on scattered implementation details.
The same campaign contract feeds scheduler job fields, planning/coding prompt
context, evaluator payloads, and operator summaries.

Evaluator and archive decisions become easier to compare across time. When
operators change the primary metric, correctness gates, or complexity policy,
the program hash changes and mixed result sets can be identified.

The implementation adds a small amount of config and schema surface. The parser
must be strict enough to produce bounded prompt/evaluator projections, but
permissive enough that the Markdown file remains readable and editable by
humans.

The campaign program table keeps provenance queryable. This is more schema than
storing everything in job artifacts, but it avoids making "which program
version produced this result?" a cold-path artifact scan.

Default `locked` program changes make long-running campaigns more predictable,
but operators who want rapid prompt iteration must opt into `approve` or
`auto`.

## Implementation Notes

The implementation should be phased but should include table-backed provenance
from the start.

Phase 1:

1. Add a parser that extracts recognized sections from `loreley.program.md`,
   clamps each projected field, and computes a stable hash over the raw
   content.
2. Add the `campaign_programs` table and nullable indexed program hash columns
   on jobs, evaluator attempts, and candidate commits.
3. Add preflight output that reports whether a program file was found and which
   recognized sections were parsed.
4. Record `campaign_program_hash=null` when no program file exists.

Phase 2:

1. Thread the normalized program snapshot into seed, evolution, and repair job
   creation.
2. Fill job `goal`, `constraints`, `acceptance_criteria`, and `notes` from the
   program projection unless the caller supplies more specific per-job values.
3. Make repair jobs inherit the source candidate's program hash by default.

Phase 3:

1. Render constraints and acceptance criteria in the shared prompt packet.
2. Pass the normalized snapshot and hash to evaluator payloads.
3. Include the program hash and snapshot in worker/evaluator artifacts.

Phase 4:

1. Add worker path-level protected/editable scope checks before commit,
   publication, and evaluation.
2. Report protected scope violations as policy failures with structured
   diagnostics.

Phase 5:

1. Add `CAMPAIGN_PROGRAM_CHANGE_POLICY`, defaulting to `locked`.
2. Add `approve` behavior that reuses the startup-approval interaction style.
3. Add operator-facing export row fields such as `campaign_program_hash`,
   `runtime_profile`, and `effective_settings_fingerprint` when campaign result
   ledgers are introduced.

The first version can be optional. If no program file exists, Loreley keeps the
current behavior and records `campaign_program_hash=null`.

## Open Questions

- Should `raw_markdown` live directly in `campaign_programs`, or should large
  programs store only `raw_artifact_path` while the normalized snapshot remains
  hot-path queryable?
- Should primary metric mismatches remain warnings, become `inconclusive`, or
  become candidate failures after enough campaigns have adopted program files?
