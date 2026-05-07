# ADR 0050: Baseline-first campaign bootstrap

Date: 2026-05-07

Status: Draft

Related: [ADR 0030](0030-db-only-repo-state-embeddings-and-explicit-bootstrap.md),
[ADR 0040](0040-delay-map-elites-archive-until-initial-pca-fit.md),
[ADR 0045](0045-config-profiles-for-large-repo-campaigns.md),
[ADR 0049](0049-campaign-program-contract.md)

## Context

Autonomous experiment loops need a comparable baseline before they start
mutating code. In small loops such as `autoresearch`, the first run is always
the unmodified baseline. That makes every later result explainable as "better
or worse than the starting point" under the same evaluator and budget.

Loreley already has the ingredients for this, but the baseline semantics are
not yet strong enough to be the campaign contract:

- `MAPELITES_EXPERIMENT_ROOT_COMMIT` identifies the canonical experiment root.
- scheduler startup resolves and validates that root commit.
- root repo-state aggregate bootstrap is fatal because runtime ingestion is
  incremental-only after bootstrap.
- root commit metadata and best-effort root evaluation can populate
  `CommitCard` and `Metric` rows.
- seed jobs are created from the root commit and describe themselves as
  starting from the root baseline.
- MAP-Elites ingestion deliberately does not insert the root commit into the
  archive.

The weak point is that root evaluation is currently operationally best-effort:
if the evaluator cannot run, the scheduler can still proceed to mutation jobs.
That keeps local experimentation forgiving, but it undermines campaign review.
Without a durable root metric row produced by the same evaluator contract,
later candidates cannot reliably report deltas from the starting point, UI
summaries can only show absolute fitness, and missing primary metrics may be
masked by configured floors.

Baseline-first is therefore not just a documentation convention. It is a
campaign bootstrap invariant: before Loreley spends mutation budget, it should
prove that the target repository, evaluator, primary metric, and campaign
program can evaluate the unmodified root.

## Decision

Make baseline-first campaign bootstrap a first-class scheduler requirement.

For each campaign, Loreley must establish a durable baseline record for the
canonical `MAPELITES_EXPERIMENT_ROOT_COMMIT` before scheduling any mutation
job, seed job, repair job, or normal sampler job for that campaign.

The baseline record is keyed by:

```text
root_commit_hash
campaign_program_hash nullable
evaluator identity/version when available
primary metric name and direction
effective runtime profile/fingerprint when available
```

The MVP can persist this through existing `CommitCard` and `Metric` rows for
the root commit, but the contract should treat the root baseline as a distinct
campaign artifact, not as an archive elite and not as a normal worker-produced
candidate.

Baseline evaluation must use the same evaluator plugin and the same primary
metric semantics that candidate evaluations use. It must run with an
`EvaluationContext` that clearly marks the evaluation as baseline/root work:

```text
job_id = null
base_commit_hash = null
candidate_commit_hash = root_commit_hash
metadata.kind = "baseline"
metadata.root_commit_hash = root_commit_hash
metadata.campaign_program_hash = ...
```

The baseline evaluator output must contain the configured primary metric. If
the metric is absent, non-finite, or has a direction that conflicts with the
campaign program or `MAPELITES_FITNESS_HIGHER_IS_BETTER`, baseline bootstrap
fails closed by default.

Add an explicit policy for exceptional workflows:

```text
BASELINE_BOOTSTRAP_POLICY=required | warn
```

`required` is the default for scheduler runs. Under `required`, Loreley does
not schedule new mutation, seed, repair, or sampler jobs until the baseline is
present and valid.

`warn` preserves today's forgiving behavior for local experiments and
backfills: Loreley logs the baseline failure, marks campaign comparability as
degraded, and continues. UI and exports must make degraded baseline status
visible.

Root repo-state bootstrap and root baseline evaluation remain separate:

- repo-state bootstrap prepares incremental embedding state and is required for
  candidate ingestion;
- baseline evaluation measures the unmodified root under the campaign
  evaluator and is required for comparison;
- neither step inserts the root into MAP-Elites archive cells.

Seed jobs may still be used to create diverse initial candidates from the
root, but they are no longer a substitute for baseline evaluation. A seed job
is a mutation job whose base is the root; the baseline is an evaluator result
for the root itself.

Repair jobs inherit the comparability requirement. A repair job can run only
after its nearest viable ancestor's campaign baseline is valid. Repairing a
failed candidate from a campaign with degraded baseline status is allowed only
under `BASELINE_BOOTSTRAP_POLICY=warn`, and the repair lineage must retain that
degraded marker.

## Consequences

Campaign startup becomes stricter. A broken evaluator, missing benchmark data,
wrong primary metric name, or incompatible metric direction fails before worker
budget is spent.

The scheduler can report candidate deltas against the root baseline, not only
absolute fitness. This makes run summaries, UI graphs, and exported ledgers
much easier to interpret.

The root commit remains outside MAP-Elites occupancy. It is the reference
point for measurement and embedding bootstrap, not an elite competing for a
cell.

Baseline evaluation may duplicate some evaluator work if operators already ran
the benchmark manually. That is acceptable because the persisted baseline is
the audit record Loreley can rely on.

Campaigns that intentionally start without a working evaluator must opt into
`warn`. This should be rare and should not be the documented production path.

## Implementation Plan

1. Add configuration for `BASELINE_BOOTSTRAP_POLICY`, defaulting to `required`.
2. Represent root baseline status explicitly in scheduler startup logs, status
   output, UI overview, and machine-readable exports.
3. Tighten root baseline evaluation validation:
   - require the configured primary metric;
   - require finite numeric values;
   - validate metric direction against campaign/settings semantics;
   - record evaluator identity and version when available.
4. Move scheduler gating so `tick()` cannot dispatch or schedule any new jobs
   until baseline status is valid under the active policy.
5. Store baseline provenance with the campaign program hash from ADR 0049 and
   the effective runtime profile/settings fingerprint.
6. Add tests for:
   - successful baseline bootstrap before seed scheduling;
   - failed baseline blocking scheduler work under `required`;
   - failed baseline allowing work but marking degraded status under `warn`;
   - idempotent restart when valid baseline metrics already exist;
   - root remaining absent from MAP-Elites archive cells.
7. Include baseline fields in operator-facing run exports:

```text
root_baseline_commit
root_baseline_metric
root_baseline_value
root_baseline_direction
root_baseline_status
delta_from_root_baseline
baseline_campaign_program_hash
```

## Non-Goals

- Do not make the root commit an archive elite.
- Do not replace per-candidate evaluator results with baseline deltas.
- Do not require a baseline for repositories that have no configured
  `MAPELITES_EXPERIMENT_ROOT_COMMIT`; those remain invalid for scheduler
  campaigns for the reasons already covered by repo-state bootstrap.
- Do not add a second evaluator path. Baseline evaluation should use the same
  evaluator plugin contract as candidate evaluation.
- Do not make campaign program parsing depend on baseline evaluation. The
  program hash is provenance for the baseline, not a prerequisite for parsing
  evaluator output.

## Open Questions

- Should baseline provenance get a dedicated `campaign_baselines` table, or is
  a root `CommitCard` plus metrics sufficient for the first implementation?
- Should a changed campaign program require a new baseline even when the root
  commit and evaluator are unchanged?
- Should baseline artifacts become downloadable through the same artifact APIs
  as candidate evaluator artifacts?
- Should MAP-Elites fitness use raw candidate metrics plus direction, or should
  it eventually store both raw fitness and `delta_from_root_baseline`?
