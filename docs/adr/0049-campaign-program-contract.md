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
stores that snapshot with campaign/job artifacts.

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

The scheduler uses the normalized program snapshot when creating jobs:

- `goal` is filled from the program unless the enqueue path supplies a more
  specific per-job goal.
- `constraints` are derived from correctness gates, editable scope, protected
  scope, evaluation budget, and failure policy.
- `acceptance_criteria` are derived from the primary metric, correctness gates,
  and any explicit success criteria.
- `notes` may include complexity and logging policy summaries.
- `iteration_hint` remains a per-job fact, not a campaign-level rule.

The shared planning/coding prompt packet must explicitly render constraints
and acceptance criteria. They should be shown as bounded bullet lists near the
goal and worker contract so agents do not have to infer campaign boundaries
from environment variables or evaluator behavior.

The evaluator receives the normalized program snapshot, raw program hash, and
program-derived constraints in `EvaluationContext.payload`. Evaluators remain
responsible for enforcing domain-specific correctness and metric semantics, but
they should be able to report which program version they evaluated against.

Every candidate should be traceable to the program version used to create it:

- job artifacts store the raw program Markdown and normalized snapshot;
- `EvolutionJob` or associated artifacts record `campaign_program_hash`;
- evaluator attempts record the same hash in their payload or metadata;
- operator-facing summaries include the hash and a short program title when
  available.

The program hash is descriptive, not a security boundary. It is used for
reproducibility, audit, and result comparison.

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

The implementation adds a small amount of config surface. The parser must be
strict enough to produce bounded prompt/evaluator projections, but permissive
enough that the Markdown file remains readable and editable by humans.

## Implementation Notes

The first implementation should be intentionally small:

1. Add a parser that extracts recognized sections from `loreley.program.md`,
   clamps each projected field, and computes a stable hash over the raw
   content.
2. Add preflight output that reports whether a program file was found and which
   recognized sections were parsed.
3. Thread the normalized program snapshot into scheduler-created jobs.
4. Render constraints and acceptance criteria in the shared prompt packet.
5. Include the program hash and snapshot in worker/evaluator artifacts.
6. Add an operator-facing export row field such as `campaign_program_hash` when
   campaign result ledgers are introduced.

The first version can be optional. If no program file exists, Loreley keeps the
current behavior and records `campaign_program_hash=null`.

## Open Questions

- Should the normalized snapshot live in a dedicated `campaign_programs` table,
  or only in job artifacts until query needs justify a hot-path table?
- Should program changes during a running campaign require explicit operator
  approval, similar to startup repo-state approval?
- Should profile selection from ADR 0045 be represented inside the program, or
  should profile remain environment configuration that is merely reported
  alongside the program hash?
- Should protected scope violations be enforced by the worker before
  evaluation, by the evaluator, or both?
