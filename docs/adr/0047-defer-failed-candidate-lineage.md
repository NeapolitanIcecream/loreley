# ADR 0047: Defer failed-candidate lineage semantics

Date: 2026-04-30

Status: Deferred

## Context

Loreley evolves repositories through git commits. The current worker flow asks an
agent to edit a worktree, creates a worker-owned commit, publishes the job
branch, and then runs the evaluator. Successful evaluations are persisted as
commit cards, metrics, and MAP-Elites archive entries that later sampling can
use as base commits.

This leaves an important design question unresolved: should commits that fail
build, validation, or evaluation become part of the evolvable lineage?

There are two valid philosophies:

- **Viable-frontier search**: only commits that pass the configured validation
  and evaluation gates are eligible as future bases. The archive represents a
  diverse frontier of viable solutions.
- **Open-lineage search**: failed or intermediate commits may still be useful
  stepping stones. A later job can start from a failed commit, repair it, and
  eventually promote the repaired descendant.

The current sampler and archive semantics are closer to viable-frontier search.
Failed jobs can leave audit evidence and candidate branch metadata, but they are
not first-class archive members and are not generally available as future base
commits.

The tension matters because agent-local build/test loops can improve output
quality, but they do not answer the larger lineage question. If failed commits
are not evolvable, Loreley should bias toward preventing bad commits from
entering the main candidate set. If failed commits are evolvable, Loreley needs
explicit modeling for failed candidates, repair jobs, and promotion back into
the viable archive.

## Decision

Do not decide this now.

For the current design, keep the existing viable-frontier behavior as the
operating assumption:

- successful evaluated commits remain the normal source of future base commits;
- failed evaluations do not become ordinary MAP-Elites archive entries;
- the evaluator remains the formal source of correctness and objective metrics;
- agent-side local checks may be encouraged by prompts or backend behavior, but
  they are not a first-class lineage mechanism.

Record the failed-candidate lineage question as a deferred architecture issue.
Any future change must be explicit, because it would alter the meaning of
candidate commits, archive membership, sampling, and job failure.

## Decision Drivers

- Keep MAP-Elites archive semantics clean: archive entries should remain
  comparable, evaluated, and viable.
- Avoid accidentally sampling broken worktrees as normal bases.
- Preserve backend-agnostic agent execution: Loreley provides input and consumes
  final worktree/report output, without depending on an agent's private repair
  loop.
- Leave room for long-horizon changes where temporarily broken commits are a
  legitimate path to a better solution.
- Avoid conflating audit history, git history, candidate metadata, and evolvable
  search state.

## Options Considered

### Option A: Viable-frontier only

Only commits that pass validation/evaluation can become future bases.

This keeps the system simpler and keeps archive entries meaningful. The downside
is that some large refactors or migrations require intermediate broken states,
which this model cannot explore directly.

### Option B: Failed commits as normal candidates

Failed commits would enter the same candidate/base-selection pool as successful
commits.

This maximizes exploration freedom but weakens the meaning of the archive and
risks wasting worker capacity on descendants of unrecoverable failures. It also
requires careful UI/API language so users do not mistake failed candidates for
viable solutions.

### Option C: Separate failed-candidate repair pool

Failed commits would be stored as first-class failed candidates, but sampled
only by repair-oriented jobs. A repaired descendant would need to pass the
normal promotion gates before entering the viable archive.

This preserves viable-frontier semantics while enabling open-lineage repair. It
is likely the most coherent future direction if Loreley needs to support
temporarily broken intermediate states.

## Consequences

Near term, no runtime behavior changes. The worker may still create and publish
a candidate commit before evaluation, but failed jobs should not be treated as
normal archive members or ordinary future bases.

Documentation and future design work should use more precise terms:

- **candidate commit** for a worker-produced git commit;
- **evaluated candidate** for a commit with evaluator results;
- **viable archive entry** for an evaluated candidate accepted into the
  MAP-Elites archive;
- **failed candidate** for a worker-produced commit whose evaluation or
  validation failed.

If Loreley later adds pre-commit smoke validation, agent repair loops, or
failed-candidate sampling, those features should be designed against this ADR
instead of being hidden behind prompt wording.

## Open Questions

- Should failed candidates be persisted in a dedicated table/state separate from
  successful commit cards and archive cells?
- Should the sampler ever select failed candidates, and if so under what mode,
  budget, and maximum failed-depth constraints?
- What diagnostics must be preserved for a repair job to be useful without
  exposing unsafe raw logs or arbitrary evaluator artifacts to agents?
- Should repair jobs have a distinct prompt contract and objective from normal
  evolution jobs?
- What promotion gates are required before a repaired descendant becomes a
  normal evaluated candidate or archive entry?
- How should the UI distinguish audit-only failed candidates from viable
  archive entries?
