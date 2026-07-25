# ADR 0051: QD keep/discard candidate semantics

Date: 2026-05-07

Status: Superseded by
[ADR 0052](0052-multi-island-pareto-archive-and-worker-processes.md).

ADR 0052 replaces the single-elite niche language below with bounded Pareto
front admission and retention. The lifecycle distinction introduced here
remains useful, but the original labels are historical terminology.

Related: [ADR 0036](0036-single-source-of-truth-for-worker-commits.md),
[ADR 0040](0040-delay-map-elites-archive-until-initial-pca-fit.md),
[ADR 0048](0048-failed-candidate-repair-pool.md),
[ADR 0049](0049-campaign-program-contract.md),
[ADR 0050](0050-baseline-first-campaign-bootstrap.md)

## Context

Small autonomous research loops such as
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) make each
experiment easy to understand: a candidate is usually `keep`, `discard`, or
`crash`.

That vocabulary works for a single-champion loop. If a new experiment improves
the current champion, the branch advances; if it does not, the branch resets.

Loreley is different. Loreley evolves whole git repositories and uses
Quality-Diversity search. MAP-Elites keeps multiple high-performing but
behaviorally different candidates instead of one global champion line. A
candidate can be worse than the global best and still be valuable if it fills
an empty archive cell or explores an underrepresented region.

Using `keep` and `discard` without qualification would hide that distinction.
It would also confuse several separate facts:

- whether the candidate commit exists for audit;
- whether evaluation passed;
- whether the candidate entered or improved the archive;
- whether the candidate is still a current archive elite;
- whether the candidate should be eligible for future sampling;
- whether a failed candidate is repairable.

Loreley needs a QD-aware way to explain candidate fate without implying a
single-champion branch policy.

## Decision

Use explicit QD-facing candidate fate labels when presenting Autoresearch-style
`keep` and `discard` concepts to operators.

The initial vocabulary is:

```text
elite_inserted:
  The candidate passed evaluation and entered an empty archive niche.

elite_replaced:
  The candidate passed evaluation, entered its archive niche's Pareto front,
  and removed one or more dominated or capacity-pruned members.

elite_retained:
  The candidate is currently retained as an archive elite.

valid_not_elite:
  The candidate passed evaluation but did not enter the archive, usually
  because it was dominated by the retained Pareto front in the same niche or
  lost the deterministic bounded-front capacity rule.

valid_not_considered:
  The candidate passed evaluation but was not considered for archive insertion,
  for example because archive warmup or ingestion was unavailable.

candidate_failed:
  The candidate failed correctness, benchmark, or candidate-owned evaluation
  checks.

repair_pending:
  The candidate failed for a candidate-owned reason and is eligible, scheduled,
  or in progress for repair.

policy_failed:
  The candidate may or may not have run successfully, but the campaign policy
  rejects it.

discarded_for_sampling:
  The candidate should not be used as a default future base or inspiration,
  even though its commit and artifacts may remain for audit.
```

These labels are operator-facing summaries. They are derived from existing
job, candidate, evaluation, repair, metric, and archive records. They are not a
new source of truth and do not replace the lower-level lifecycle fields.

### Keep and Discard

In Loreley, `keep` means "retained for the QD search process", not merely "the
git commit still exists".

A candidate is kept for QD purposes when it is a current archive member, when
it has just entered a Pareto front, or when an explicit campaign policy
retains it for repair, audit, or a labelled sampling strategy.

In Loreley, `discard` means "not eligible for default future sampling under the
active campaign policy". It does not mean the commit, branch, metrics, or
artifacts must be deleted.

Therefore:

- commit existence is not keep;
- not being globally best is not discard;
- passing evaluation is not keep by itself;
- failing to enter the archive is not necessarily a system failure;
- repair eligibility is distinct from archive retention.

### Historical and Current Archive Facts

Archive fate has two time axes:

```text
historical_archive_decision:
  What happened when this candidate was considered for archive insertion.

current_archive_membership:
  Whether this candidate is currently an archive elite.
```

For example, a candidate can historically be `elite_inserted` and later be
removed when another candidate dominates it or the bounded front prunes it.
Operator summaries should not erase either fact.

This ADR defines the vocabulary and the distinction. It does not require a
specific schema for archive ingestion events.

### Sampling Eligibility

Default inspiration and base selection should come from archive membership,
not from raw candidate branch existence.

A future sampler may intentionally draw from valid non-elites, failed
candidates, or operator-pinned candidates, but that must be an explicit,
labelled strategy. It must not happen implicitly because a candidate commit
exists.

## Consequences

Operator-facing summaries become more faithful to Loreley's QD model. A
candidate that is not globally best can still be reported as useful when it
fills or improves a niche.

Valid but non-elite candidates stop looking like generic failures. They are
successful evaluations that did not change the archive.

Failed but repairable candidates are not conflated with candidates that should
be forgotten. Repair policy remains visible as a separate lifecycle concern.

The UI, CLI, API, and future exports should share this vocabulary so different
surfaces do not explain the same candidate in incompatible ways.

## Non-Goals

- Do not replace MAP-Elites with a single-champion keep/discard loop.
- Do not restrict Loreley campaigns to single-file edits.
- Do not define a run ledger schema in this ADR.
- Do not define JSONL/TSV export behavior in this ADR.
- Do not define archive ingestion event tables in this ADR.
- Do not define complexity-cost metrics or tie-breakers in this ADR.
- Do not define non-elite shadow-pool sampling policy in this ADR.
- Do not define denormalized outcome cache columns in this ADR.
- Do not delete or reset candidate commits solely because they are not archive
  elites.

## Implementation Notes

The first implementation should be small:

1. Add one shared derivation helper that maps existing records into the
   operator-facing candidate fate labels.
2. Use the helper from UI, CLI, API, and later exports instead of duplicating
   mapping logic.
3. Preserve lower-level fields in detailed views so users can see why a label
   was assigned.
4. Treat unknown or partial historical records as `unknown` or
   `valid_not_considered`, with a bounded reason string.

## Deferred Work

Separate ADRs may cover:

- campaign run ledger JSONL/TSV schema;
- historical archive-ingestion event storage;
- current archive membership materialized views;
- versioned complexity-cost metrics;
- explicit valid-non-elite sampling strategies;
- outcome summary caching and invalidation rules.

Those are useful follow-up designs, but they are outside the scope of this ADR.
