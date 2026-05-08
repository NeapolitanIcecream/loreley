# ADR 0050: Baseline-first campaign comparability contract

Date: 2026-05-07

Status: Draft

Related: [ADR 0030](0030-db-only-repo-state-embeddings-and-explicit-bootstrap.md),
[ADR 0040](0040-delay-map-elites-archive-until-initial-pca-fit.md),
[ADR 0045](0045-config-profiles-for-large-repo-campaigns.md),
[ADR 0046](0046-agent-visible-evaluation-artifacts.md),
[ADR 0048](0048-failed-candidate-repair-pool.md),
[ADR 0049](0049-campaign-program-contract.md)

## Context

Loreley treats a git commit as the search unit, asks external agents to make
repo-wide changes, evaluates the resulting commit, and feeds the metrics into
Postgres and MAP-Elites. In that loop, the question "did this candidate improve
the project?" is only stable if the campaign has a durable root baseline under
the same evaluation contract.

Small autonomous loops such as `autoresearch` make this obvious by running the
unmodified baseline first. Loreley needs the same invariant at repository
scale, but with stronger provenance because a campaign may have multiple
program versions, evaluator versions, runtime profiles, repair lineages, and
archive states.

Loreley already has several related mechanisms:

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
- ADR 0049 records the campaign program hash on jobs, evaluation attempts, and
  candidates so results can be tied to a versioned campaign contract.

The weak point is that root evaluation is currently operationally best-effort:
if the evaluator cannot run, the scheduler can still proceed to mutation jobs.
That keeps local experimentation forgiving, but it undermines campaign review.
Without a durable baseline record produced by the same evaluator contract,
later candidates can only show absolute fitness, primary metric gaps can be
hidden by configured floors, and UI/export deltas can silently mix incompatible
campaign semantics.

Baseline-first is therefore a campaign comparability contract, not merely
"run the evaluator once before startup." Before Loreley spends mutation budget,
it must prove that the target repository, evaluator, primary metric, runtime
profile, and campaign program can evaluate the unmodified root.

## Decision

Make baseline-first campaign bootstrap a first-class scheduler requirement and
persist baseline provenance in a dedicated table.

For each campaign contract, Loreley must establish a durable baseline record
for the canonical `MAPELITES_EXPERIMENT_ROOT_COMMIT` before dispatching or
scheduling mutation, seed, repair, or normal sampler jobs.

The baseline is a campaign artifact. It is not an archive elite, not a normal
worker-produced candidate, and not merely commit metadata.

### Baseline Source of Truth

Add a dedicated `campaign_baselines` table from the first implementation.
Existing root `CommitCard` and `Metric` rows may still be maintained as
compatibility and UI projections, but they are not the source of truth for
baseline provenance.

The MVP schema is:

```text
campaign_baselines
- id uuid primary key
- baseline_key_hash varchar(64) unique not null

- root_commit_hash varchar(64) not null
- campaign_program_hash varchar(64) null
- evaluator_name varchar(128) null
- evaluator_version varchar(128) null
- primary_metric_name varchar(128) not null
- primary_metric_higher_is_better boolean not null
- runtime_profile varchar(128) null
- effective_settings_fingerprint varchar(64) null

- status varchar(32) not null
  -- valid | failed | degraded | stale

- metric_value double precision null
- metric_unit varchar(32) null
- evaluation_summary text null
- failure_kind varchar(64) null
- failure_summary text null

- commit_card_id uuid null
- metric_id uuid null
- started_at timestamptz null
- finished_at timestamptz null
- created_at timestamptz not null
- updated_at timestamptz not null
```

Use `baseline_key_hash` rather than a nullable composite unique constraint.
The key hash is computed from a canonical JSON object containing the root
commit, campaign program identity, evaluator identity, primary metric
semantics, runtime profile, and effective settings fingerprint. This avoids
Postgres nullable uniqueness edge cases and gives logs, status output, and
exports a compact stable identifier.

### Baseline Identity

The baseline key includes:

```text
root_commit_hash
campaign_program_hash
evaluator_name and evaluator_version when available
primary_metric_name
primary_metric_higher_is_better
runtime_profile
effective_settings_fingerprint
```

The MVP uses `WORKER_EVALUATOR_VERSION` as the operator-declared evaluator
version when it is set. When it is not set, the baseline key uses a best-effort
package version or source-file fingerprint derived from
`WORKER_EVALUATOR_PLUGIN`. Operators should still bump
`WORKER_EVALUATOR_VERSION` when benchmark data or scoring semantics change
without a corresponding plugin source change.

A changed campaign program hash requires a valid baseline for the new program
hash by default, even if the root commit and evaluator are unchanged. Candidate
results created under program B must not silently reuse a baseline created
under program A.

A future implementation may introduce an `evaluation_contract_fingerprint` and
allow explicit baseline reuse when formatting-only or non-evaluator program
changes are proven semantically equivalent. In that case, the reused baseline
must still record the active raw program hash for audit, for example with
`reused_from_baseline_id` or `semantic_equivalent_to_baseline_id`.

For the MVP, raw campaign program hash changes invalidate baseline reuse.

ADR 0049 permits primary metric mismatch to be recorded as a warning while the
candidate evaluator result remains authoritative. Baseline bootstrap is
stricter: if the baseline result is missing the configured primary metric,
returns a non-finite metric value, or reports a conflicting direction, the
baseline is not valid under the default policy.

### Evaluation Context

Baseline evaluation uses the same evaluator plugin contract as candidate
evaluation. It runs with an `EvaluationContext` that clearly marks root
baseline work:

```text
job_id = null
base_commit_hash = null
candidate_commit_hash = root_commit_hash
metadata.kind = "baseline"
metadata.root_commit_hash = root_commit_hash
metadata.campaign_program_hash = ...
metadata.baseline_key_hash = ...
```

The evaluator should provide `evaluator_name`, `evaluator_version`, timing, and
summary where possible. The bootstrap service persists both valid and failed
attempts so operators can inspect why a campaign is blocked or degraded.
Existing failed or degraded rows are authoritative for their baseline key until
a relevant input changes and produces a new key, or until a future explicit
retry/cooldown policy is added. This avoids spending evaluator budget on the
same failed or degraded baseline every scheduler tick.

### Bootstrap Policy

Add an explicit policy:

```text
BASELINE_BOOTSTRAP_POLICY=required | warn
```

`required` is the default for scheduler runs. Under `required`, Loreley does
not dispatch pending mutation jobs and does not schedule new seed, repair, or
sampler jobs until the active baseline key has a `valid` baseline.

`warn` is for local experiments and backfills. Loreley records a `degraded`
baseline status, logs the failure, and may continue dispatching and scheduling.
Jobs, candidate records, repair lineages, UI rows, and exports created under
this policy must make baseline degradation visible. Delta fields are null or
explicitly unavailable when there is no valid baseline.

The scheduler gate belongs before dispatch, not only before scheduling. Old
pending jobs must not be sent to workers under `required` until comparability
is established.

The scheduler tick order becomes:

```text
ingest completed jobs
reclaim stale running jobs
ensure or load baseline
if baseline cannot dispatch/schedule:
    report baseline_blocked
    return
dispatch pending jobs
schedule seed jobs
schedule repair jobs
schedule normal sampler jobs
```

### Repo-State Bootstrap Boundary

Root repo-state bootstrap and root baseline evaluation remain separate:

- repo-state bootstrap prepares incremental embedding state and is required for
  candidate ingestion;
- baseline evaluation measures the unmodified root under the campaign
  evaluator and is required for comparison;
- neither step inserts the root into MAP-Elites archive cells.

Seed jobs are not a substitute for baseline evaluation. A seed job is a
mutation job whose base is the root; the baseline is an evaluator result for
the root itself.

Repair jobs inherit the comparability requirement. A repair job can run only
after its nearest viable ancestor's campaign baseline is valid under
`required`. Repairing a failed candidate from a degraded campaign is allowed
only under `warn`, and the repair lineage must retain that degraded marker.

### Baseline Artifacts

Do not expose baseline evaluator artifacts through job artifact APIs in the
MVP. Baseline evaluation has no job/run authority, and ADR 0046's current
artifact model is job-scoped.

The first implementation stores hot-path baseline data only: status, primary
metric, summary, failure summary, evaluator identity, timing, and baseline key.
It must not create fake `EvolutionJob`, `JobArtifacts`, or
`EvaluationArtifactRecord` rows for baseline work.

If downloadable baseline benchmark JSON, logs, or profiles are needed later,
add an independent baseline artifact namespace:

```text
baseline_artifacts
- id uuid primary key
- baseline_id uuid not null references campaign_baselines(id)
- key varchar(128) not null
- kind varchar(64) not null
- mime_type varchar(128) not null
- label varchar(128) null
- summary text null
- visibility varchar(32) not null
- storage_path text null
- size_bytes bigint null
- sha256 varchar(64) null
- diagnostics jsonb not null
- metadata jsonb not null
```

The API should also be independent:

```text
GET /api/v1/campaign-baselines/{baseline_id}/artifacts
GET /api/v1/campaign-baselines/{baseline_id}/artifacts/{artifact_key}
```

That future work may reuse ADR 0046 validation, visibility, projection, MIME,
size, and hash logic, but it needs a baseline authority model rather than a job
authority model.

### Fitness and Delta Semantics

MAP-Elites archive selection continues to use the raw candidate metric
interpreted by its direction. Baseline deltas are for UI, status, exports, and
operator review.

```text
raw_value:
  evaluator output for the configured metric.

improvement_from_baseline:
  higher_is_better: candidate_value - baseline_value
  lower_is_better:  baseline_value - candidate_value

relative_improvement_from_baseline:
  optional, only when the metric supports ratio interpretation and the
  baseline value is non-zero.
```

`Metric` continues to store raw evaluator metrics. `campaign_baselines` stores
the root raw metric. Views and exports compute or cache
`delta_from_root_baseline` keyed by `campaign_baseline_id`.

Do not make `delta_from_root_baseline` the MAP-Elites primary objective unless
a later ADR explicitly introduces a cross-campaign normalized-objective mode.

### Effective Settings Fingerprint

The baseline key needs a narrow comparability fingerprint, not a hash of all
settings. It should include only fields that affect evaluator comparability:

```text
root_commit_hash
campaign_program_hash or future evaluation_contract_fingerprint
primary_metric_name
primary_metric_higher_is_better
worker_evaluator_plugin identity
WORKER_EVALUATOR_VERSION / evaluator config fingerprint
evaluation timeout or budget
LORELEY_PROFILE
MAPELITES_FITNESS_* values that affect interpretation
benchmark/runtime profile fields the evaluator actually uses
```

It must not include secrets or unrelated operational settings:

```text
DATABASE_URL
Redis settings
API keys
scheduler poll interval
worker queue prefetch
log level
UI settings
```

## Consequences

Campaign startup becomes stricter. A broken evaluator, missing benchmark data,
wrong primary metric name, incompatible metric direction, stale program hash,
or incompatible evaluator/runtime profile fails before worker budget is spent.

The scheduler can report candidate deltas against a specific
`campaign_baseline_id`, not only absolute fitness. This makes run summaries,
UI graphs, and exported ledgers easier to interpret and audit.

The root commit remains outside MAP-Elites occupancy. It is the reference point
for measurement and embedding bootstrap, not an elite competing for a cell.

Baseline evaluation may duplicate manual benchmark work. That is acceptable
because the persisted baseline is the audit record Loreley can rely on.

Campaigns that intentionally start without a working evaluator must opt into
`warn`. This should be rare and should not be the documented production path.

The first implementation has more schema than a `CommitCard`-only approach,
but it avoids forcing program/evaluator/runtime provenance into metric details
or overloading the unique `CommitCard.commit_hash` model.

## Implementation Plan

1. Add `campaign_baselines` and baseline key hashing.
2. Add `BASELINE_BOOTSTRAP_POLICY`, defaulting to `required`.
3. Implement `BaselineBootstrapService`:
   - build the active baseline key from settings, campaign program snapshot,
     evaluator identity, and narrow comparability fingerprint;
   - load the existing baseline by key hash;
   - evaluate the root if missing or stale;
   - validate primary metric presence, finiteness, and direction;
   - persist `valid`, `failed`, or `degraded` baseline rows;
   - expose `can_dispatch_or_schedule`.
4. Gate scheduler dispatch and scheduling behind the baseline service.
5. Store baseline provenance on jobs, candidates, evaluation attempts, exports,
   and repair lineage records where applicable.
6. Keep root `CommitCard` and `Metric` projection for compatibility and UI, but
   do not accept it as sufficient under `required` unless it is linked to a
   matching `campaign_baselines` row.
7. Represent root baseline status in scheduler startup logs, `status` output,
   UI overview, and machine-readable exports.
8. Include baseline fields in operator-facing run exports:

```text
campaign_baseline_id
baseline_key_hash
root_baseline_commit
root_baseline_metric
root_baseline_value
root_baseline_direction
root_baseline_status
delta_from_root_baseline
baseline_campaign_program_hash
```

## Tests

Add tests for:

- successful baseline bootstrap before seed scheduling;
- failed baseline blocking dispatch and scheduling under `required`;
- failed baseline allowing work but marking degraded status under `warn`;
- idempotent restart when a valid matching baseline row already exists;
- root remaining absent from MAP-Elites archive cells;
- same root, same metric, and different campaign program hash requiring a
  distinct baseline;
- same root, same program, same evaluator, and changed metric direction not
  reusing the old baseline;
- root `CommitCard` already having a metric but no matching
  `campaign_baselines` row not silently satisfying `required`;
- missing, `NaN`, or infinite primary metric failing under `required`;
- `warn` exports carrying degraded baseline status and null/unavailable delta;
- baseline evaluator declarations of artifacts not creating fake job artifacts;
- future semantic-equivalence reuse, if implemented, recording the active raw
  program hash and `reused_from_baseline_id`.

## Non-Goals

- Do not make the root commit an archive elite.
- Do not replace per-candidate evaluator results with baseline deltas.
- Do not expose baseline artifacts through job artifact APIs in the MVP.
- Do not create fake jobs or fake worker runs for baseline work.
- Do not add a second evaluator path. Baseline evaluation should use the same
  evaluator plugin contract as candidate evaluation.
- Do not hash full settings or secrets into baseline identity.
- Do not allow campaign program parsing to depend on baseline evaluation. The
  program hash is provenance for the baseline, not a prerequisite for parsing
  evaluator output.

## Deferred Work

- Add an `evaluation_contract_fingerprint` so semantically equivalent program
  edits can explicitly reuse an existing baseline.
- Add baseline artifact storage and API endpoints if root benchmark artifacts
  become useful enough to justify a separate authority model.
- Add an explicit baseline retry/cooldown policy if operators need automatic
  recovery from transient evaluator or environment failures without changing
  the baseline key.
- Decide whether long-term archive views should expose both raw objective and
  baseline improvement in DB materialized views.
