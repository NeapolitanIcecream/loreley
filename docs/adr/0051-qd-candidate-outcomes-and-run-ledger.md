# ADR 0051: QD candidate outcomes and campaign run ledger

Date: 2026-05-07

Status: Draft

Related: [ADR 0036](0036-single-source-of-truth-for-worker-commits.md),
[ADR 0040](0040-delay-map-elites-archive-until-initial-pca-fit.md),
[ADR 0046](0046-agent-visible-evaluation-artifacts.md),
[ADR 0048](0048-failed-candidate-repair-pool.md),
[ADR 0049](0049-campaign-program-contract.md),
[ADR 0050](0050-baseline-first-campaign-bootstrap.md)

## Context

Small autonomous research loops such as
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) make each
experiment easy to understand:

- one human-authored program document defines the experiment protocol;
- one fixed evaluator and one fixed primary metric make runs comparable;
- the first run records a baseline;
- each run appends a row to a simple result ledger;
- each candidate is described as `keep`, `discard`, or `crash`.

Loreley should not copy the single-file or single-champion search model.
Loreley evolves whole repositories and keeps multiple high-performing but
different solutions in a MAP-Elites archive. A candidate that is worse than the
global best may still be valuable if it occupies an empty or underperforming
behavioral niche.

That means Autoresearch's `keep` and `discard` terms are too coarse for
Loreley. A Loreley candidate can pass evaluation but fail to enter the archive,
enter an empty archive cell, replace an existing elite, fail correctness gates,
fail due to evaluator infrastructure, or become eligible for repair. These
outcomes affect different parts of the system:

- git branch and commit retention;
- candidate audit and repair eligibility;
- MAP-Elites archive membership;
- future sampling and inspiration selection;
- UI and operator review;
- exported campaign ledgers.

Loreley already stores several lower-level facts: job status, candidate commit
metadata, evaluation status, archive status, lifecycle status, repair state,
metrics, artifacts, and MAP-Elites cells. The missing piece is an explicit
operator-facing outcome vocabulary that explains what happened to a candidate
without collapsing Quality-Diversity semantics into a champion-loop
`keep/discard` decision.

Operators also need a durable, scan-friendly campaign ledger. The database and
UI remain the source of truth, but a JSONL/TSV export gives a compact "morning
review" view of a long-running campaign and makes archived results easier to
diff, share, and inspect when the UI or database is unavailable.

## Decision

Define a campaign candidate outcome model and a campaign run ledger projection.
The model is a derived operator-facing view over existing hot-path statuses,
not a replacement for normalized job, candidate, evaluation, repair, and
archive records.

### Outcome Dimensions

Represent candidate fate as several explicit dimensions:

```text
evaluation_outcome_kind:
  passed | candidate_failed | evaluator_failed | infrastructure_failed |
  inconclusive | cancelled | unknown

policy_outcome:
  accepted | policy_failed | not_checked | unknown

historical_archive_decision:
  inserted_empty_cell | replaced_cell_elite | retained_existing_elite |
  rejected_lower_fitness_same_cell | skipped_archive_not_ready |
  skipped_missing_metric | skipped_failed_evaluation | skipped_policy_failed |
  skipped_ingest_failed | not_considered | unknown

current_archive_status:
  current_elite | superseded | never_member | not_considered | unknown

repair_decision:
  not_applicable | eligible | scheduled | repairing | repaired |
  exhausted | audit_only | unknown

campaign_outcome:
  elite_inserted | elite_replaced | elite_retained | valid_not_elite |
  valid_not_considered | policy_failed | candidate_failed |
  evaluator_failed | infrastructure_failed | repair_pending |
  repair_exhausted | cancelled | unknown
```

`campaign_outcome` is the summary label for UI, status output, and TSV exports.
The other fields preserve the reason behind the summary. This avoids
overloading one mutable enum with every subsystem concern.

Use `historical_archive_decision` for what happened when the candidate was
ingested into the archive, and `current_archive_status` for whether the
candidate is still a current archive elite. A candidate can be
`inserted_empty_cell` historically and later become `superseded` after another
candidate replaces it. Both facts are important and must not be collapsed.

The source of truth remains the underlying rows:

- `EvolutionJob.status` says whether the job finished, failed, or was
  cancelled.
- `EvaluationOutcome` and evaluation attempt rows say whether evaluation
  passed, failed the candidate, failed the evaluator, or failed infrastructure.
- `CandidateCommit.evaluation_status`, `archive_status`, `lifecycle_status`,
  and `repair_state` say where the candidate sits in the worker, archive, and
  repair lifecycles.
- `MapElitesArchiveCell` says which candidates are current elites.
- campaign program and baseline records from ADR 0049 and ADR 0050 say which
  protocol and baseline make the result comparable.

The derived outcome can initially be computed at query/export time. If UI or
export performance later requires denormalization, a stored
`candidate_outcome_summary` or materialized view can be added without changing
the semantics.

Query-time derivation is the canonical semantic path. SQL views, materialized
views, or `candidate_commits` cache columns are permitted only as derived
caches. If a cached summary is introduced, it must carry enough invalidation
metadata to prove freshness:

```text
candidate_commits.outcome_summary_json nullable
candidate_commits.outcome_summary_version
candidate_commits.outcome_summary_derived_at
candidate_commits.outcome_summary_source_revision
```

At minimum, cached outcome summaries are invalidated by candidate evaluation
updates, policy check updates, repair-state updates, archive ingestion events,
archive rebuilds, and PCA projection epoch changes.

`unknown` is a migration and backfill escape hatch. New live-path outcomes that
derive `unknown` should emit a warning or metric with a bounded reason code.
Tests should document which legacy or partial-input combinations may produce
`unknown`.

### QD Interpretation of Keep and Discard

Use the following interpretation when explaining Autoresearch-style terms in
Loreley:

```text
keep:
  the candidate is a current archive elite, replaced an archive elite, or was
  explicitly retained for a repair/audit/sampling policy.

discard:
  the candidate is not eligible for future sampling under the active campaign
  policy. Its commit, branch, metrics, and artifacts may still be retained for
  audit.
```

Do not equate "git commit exists" with "kept". Loreley may keep candidate
branches and database rows for traceability even when the candidate is not an
archive member and should not influence future sampling.

Do not equate "not globally best" with "discarded". A candidate that fills an
empty MAP-Elites cell is a successful elite insertion even if its raw fitness
is worse than the global best commit.

Do not equate "passed evaluation" with "kept". A valid candidate can still be
`valid_not_elite` when it lands in an occupied cell and does not beat the
existing elite for that cell.

### Outcome Precedence

Use one shared derivation function for campaign outcomes. UI, CLI, API, and
exports must not each implement their own precedence rules.

The summary label precedence is:

```text
cancelled
  > infrastructure_failed
  > evaluator_failed
  > candidate_failed / repair_pending / repair_exhausted
  > policy_failed
  > evaluation passed + historical archive decision
  > valid_not_considered
  > unknown
```

`repair_pending` is not an evaluation outcome. It is the summary label for a
candidate-owned failure whose repair state is `eligible`, `scheduled`, or
`repairing`. If the same candidate exhausts repair attempts, the summary label
becomes `repair_exhausted`.

Policy failures should also expose bounded reason fields:

```text
policy_stage:
  pre_commit | pre_evaluation | post_evaluation | archive_ingestion | unknown

policy_failure_kind:
  protected_scope_violation | editable_scope_violation | budget_exceeded |
  missing_primary_metric | complexity_policy_failed |
  evaluator_contract_mismatch | unknown
```

### Archive Decisions

Archive ingestion should produce or expose enough structured information to
derive these decisions:

- empty cell insertion;
- occupied cell replacement;
- occupied cell rejection because the candidate was not fitter than the
  existing elite;
- skipped ingestion while archive initialization or PCA warmup is incomplete;
- skipped ingestion because the configured primary metric is missing or not
  finite;
- skipped ingestion because evaluation did not pass;
- skipped ingestion because campaign policy rejected the candidate;
- ingestion failure due to repository, embedding, database, or archive errors.

These are historical archive decisions. They should be persisted as ingestion
events or attempts rather than inferred later from current archive cells. The
MVP schema is:

```text
archive_ingestion_attempts
- id uuid primary key
- candidate_commit_id uuid not null
- island_id varchar(64) null
- projection_epoch integer null
- cell_key text null
- decision varchar(64) not null
- candidate_fitness double precision null
- incumbent_candidate_id uuid null
- incumbent_fitness double precision null
- reason_code varchar(64) null
- created_at timestamptz not null
```

`projection_epoch` identifies the PCA/archive projection state when the
decision was made. Archive rebuilds may create new ingestion attempts or a
separate rebuild event stream, but current archive state must still be derived
from `MapElitesArchiveCell`.

These decisions should be visible in:

- candidate detail pages;
- scheduler and worker status output;
- machine-readable API responses where candidate rows are returned;
- run ledger exports.

Archive membership, not raw candidate creation, determines default inspiration
eligibility. A later sampler policy may explicitly sample from non-elite
candidate pools, but that must be visible as a sampler strategy rather than an
implicit side effect of a candidate branch existing.

### Campaign Run Ledger

Add a campaign run ledger projection. The database remains the source of
truth; the ledger is a deterministic export and optional cold-path artifact.

The canonical machine-readable format is JSONL. TSV is a stable human review
projection with a fixed core column set and bounded text fields.

The initial JSONL fields are:

```text
schema_version
experiment_id
job_id
job_kind
base_commit_hash
candidate_commit_hash
candidate_branch_name
island_id
campaign_program_hash
campaign_baseline_id
baseline_key_hash
runtime_profile
effective_settings_fingerprint

evaluation_outcome_kind
policy_outcome
historical_archive_decision
current_archive_status
repair_decision
campaign_outcome
policy_stage
policy_failure_kind
current_inspiration_eligible
sampler_eligibility_reason

primary_metric_name
primary_metric_value
primary_metric_unit
primary_metric_higher_is_better
normalized_fitness
delta_from_root_baseline

duration_seconds
planning_duration_seconds
coding_duration_seconds
evaluation_duration_seconds
peak_memory_mb
cost_estimate

files_changed
lines_added
lines_deleted
dependency_changes
complexity_cost
complexity_formula_version
complexity_unavailable_reason

plan_summary
change_summary
evaluation_summary
failure_kind
failure_summary
created_at
completed_at
```

The initial TSV projection should include the fields operators need most often:

```text
job_id
candidate
base
outcome
archive_decision
current_archive_status
metric
value
delta_from_baseline
duration_seconds
memory_gb
files_changed
complexity_cost
description
```

In TSV, `archive_decision` is the short review-column alias for
`historical_archive_decision`. Current membership is reported separately as
`current_archive_status`.

Use TSV rather than CSV for the human projection because short descriptions
often contain commas. Long Markdown, raw logs, prompts, and evaluator artifacts
must not be embedded in TSV rows.

Ledger export is explicit by default:

```bash
loreley runs export --format jsonl
loreley runs export --format tsv
```

Automatic export is opt-in snapshot generation, not worker append:

```text
RUN_LEDGER_AUTO_EXPORT=off | jsonl | tsv | both
RUN_LEDGER_AUTO_EXPORT_INTERVAL_SECONDS=300
RUN_LEDGER_AUTO_EXPORT_ON_SHUTDOWN=true
RUN_LEDGER_AUTO_EXPORT_DEST=...
```

Automatic exports must be generated from the database, sorted stably by
`completed_at` and `job_id`, written atomically through a temporary file plus
rename, and include `schema_version`, `export_generated_at`, and
`source_db_schema_version`. File names should include the experiment id and, if
available, campaign program and baseline identifiers.

### Complexity Signals

Make complexity cost visible alongside quality, but do not change MAP-Elites
replacement semantics in the first implementation.

The first implementation should expose cheap structural signals:

```text
loreley.diff.files_changed
loreley.diff.files_added
loreley.diff.files_deleted
loreley.diff.files_modified
loreley.diff.renames
loreley.diff.lines_added
loreley.diff.lines_deleted
loreley.diff.binary_files_changed
loreley.diff.max_file_churn
loreley.diff.generated_or_vendor_files_changed

loreley.deps.manifest_files_changed
loreley.deps.lockfiles_changed
loreley.deps.dependencies_added
loreley.deps.dependencies_removed
loreley.deps.dependencies_updated

loreley.scope.protected_paths_touched
loreley.scope.editable_scope_violations

loreley.complexity.cost
loreley.complexity.formula_version
loreley.complexity.excluded_path_policy_hash
```

These can be computed by the worker from the candidate diff or emitted by the
evaluator when domain-specific complexity signals are available. Namespaced
`loreley.*` metrics avoid collisions with target project metrics.

The MVP should compute raw diff stats with git, for example
`git diff --numstat --find-renames <base>..<candidate>`. Dependency changes
can start path-based by detecting touched manifest and lock files such as
`pyproject.toml`, `uv.lock`, `package.json`, `package-lock.json`,
`pnpm-lock.yaml`, `Cargo.toml`, `Cargo.lock`, `go.mod`, and `go.sum`. Parsing
specific dependency additions, removals, and updates can come later.

Persist raw stats separately from `loreley.complexity.cost`.
`complexity_cost` is a derived score and must include a formula version so old
ledger rows remain interpretable after formula changes. If a candidate fails
before a commit exists, complexity fields are null and
`complexity_unavailable_reason` explains why:

```text
complexity_unavailable_reason:
  no_candidate_commit | checkout_failed | diff_failed | unknown
```

Use the campaign program's complexity policy from ADR 0049 to explain these
signals to planning and coding agents. For the MVP, complexity affects prompts,
operator review, ledger exports, and optional policy checks. A later ADR should
decide whether same-cell archive replacement should use a fitness epsilon plus
complexity tie-breaker.

### Valid Non-Elite Sampling

`valid_not_elite` candidates may be sampled only through an explicit shadow
pool strategy. They are not default base commits and must not enter default
inspiration selection merely because their candidate branch exists.

Expose sampler provenance with:

```text
base_selection_kind:
  root_seed | archive_elite | repaired_candidate |
  valid_non_elite_shadow_pool | operator_pinned

inspiration_selection_kind:
  archive_elite | valid_non_elite_shadow_pool |
  recent_success | operator_pinned | none
```

The first sampler labels are:

```text
valid_non_elite_shadow_inspiration
valid_non_elite_revisit
```

The shadow pool is disabled by default and must have explicit budget limits:

```text
VALID_NON_ELITE_SAMPLING_ENABLED=false
VALID_NON_ELITE_MAX_FRACTION=0.05
VALID_NON_ELITE_MAX_AGE_JOBS=200
VALID_NON_ELITE_MIN_NORMALIZED_FITNESS_PERCENTILE=0.50
VALID_NON_ELITE_REQUIRE_POLICY_ACCEPTED=true
VALID_NON_ELITE_REQUIRE_EVALUATION_PASSED=true
```

Preferred shadow-pool candidates are close same-cell losers with low
complexity, few dependency changes, useful local evaluation signals, or
lineage from under-explored archive regions.

### Pilot and Local Review Compatibility

A future single-process pilot mode should use the same outcome vocabulary and
ledger projection as distributed campaigns. Pilot mode may run without Redis or
multiple workers, but its result rows should still be comparable to full
campaign rows whenever they use the same evaluator, program hash, baseline,
and runtime profile.

This keeps "try it locally for 20 jobs" and "run a distributed campaign
overnight" on the same review path.

## Non-Goals

- Do not replace MAP-Elites with a single-champion branch loop.
- Do not require Loreley campaigns to restrict edits to a single source file.
- Do not reset, delete, or hide candidate commits solely because they are not
  archive elites.
- Do not make the ledger the source of truth. It is an export over database
  records and cold-path artifacts.
- Do not pass raw logs or unbounded evaluator output to future agents through
  the ledger.
- Do not make complexity cost a primary archive objective in this ADR.
- Do not sample valid non-elites through implicit branch existence. Shadow-pool
  sampling must be explicitly configured and labelled.
- Do not make automatic TSV/JSONL export the default scheduler behavior.
- Do not allow uncapped "run forever" behavior as the production default.
  Scheduler job caps, leases, timeouts, and baseline policy still apply.

## Consequences

Operators get a clearer answer to "what happened to this candidate?" A valid
but non-elite candidate no longer looks like a generic failure, and a niche
filling candidate no longer looks less important because it is not globally
best.

Campaign review becomes more compact. A JSONL/TSV ledger can summarize a long
campaign without requiring ad hoc database queries or reading raw worker logs.

Archive, repair, and policy behavior become easier to audit. A candidate's
outcome can be understood together with the campaign program hash, baseline
identity, runtime profile, primary metric, and repair state.

The implementation adds a small derived-model layer and one archive ingestion
event stream. The benefit is that subsystems can keep precise internal
statuses while UI, CLI, and exports share one vocabulary.

Result exports may expose incomplete data while earlier ADRs are still being
implemented. Missing program hashes, baseline ids, resource metrics, or
complexity metrics should be represented as null/empty fields rather than
silently omitted from the schema.

## Implementation Plan

Phase 1:

1. Add an `archive_ingestion_attempts` table or equivalent archive event
   record so historical archive decisions are persisted.
2. Add a pure outcome derivation module:

```text
derive_candidate_outcome(input: CandidateOutcomeInput) -> CandidateOutcomeView
```

The returned view includes all outcome dimensions, summary labels, reason
codes, source row ids, and a derivation version.

3. Add unit tests for the mapping, including passed/not-elite, inserted elite,
   replaced elite, failed candidate, evaluator failure, repair pending, and
   archive warmup cases.
4. Expose the derived fields in service-layer responses used by CLI/UI.

Phase 2:

1. Add `loreley runs export --format jsonl|tsv`.
2. Export core identifiers, primary metric, normalized fitness, archive
   decisions, current archive status, campaign outcome, branch, summaries, and
   timestamps from existing data.
3. Add baseline and campaign program fields as nullable columns in the export
   projection until ADR 0049 and ADR 0050 are fully implemented.
4. Add opt-in automatic export as atomic snapshots if operators need periodic
   review files.

Phase 3:

1. Add worker-computed diff statistics for successful and failed candidates
   when a candidate commit exists.
2. Persist raw stats and a versioned `complexity_cost` separately.
3. Add optional evaluator-provided complexity metrics under the `loreley.*`
   namespace.
4. Surface complexity signals in exports and candidate detail views.

Phase 4:

1. Add a materialized view or denormalized summary column only if export or UI
   queries become too expensive.
2. Add explicit valid-non-elite shadow-pool sampling only if campaign data
   shows a need for non-elite reuse.
3. Consider same-cell archive replacement tie-breakers based on campaign
   complexity policy in a separate ADR.
4. Reuse the same ledger schema for future pilot mode.

## Tests

Add tests for:

- a candidate inserted into an empty cell deriving `elite_inserted`;
- a candidate replacing an existing cell elite deriving `elite_replaced`;
- a valid candidate rejected by same-cell fitness deriving `valid_not_elite`;
- a passed candidate skipped during archive warmup deriving
  `valid_not_considered`;
- a candidate failing correctness deriving `candidate_failed`;
- evaluator and infrastructure failures deriving distinct outcomes;
- repair-eligible failed candidates deriving `repair_pending`;
- cancelled jobs deriving `cancelled`;
- TSV export escaping tabs and newlines in descriptions;
- JSONL export preserving null fields for unavailable program, baseline,
  memory, cost, and complexity data;
- historical archive insertion remaining visible after the candidate is later
  superseded in current archive cells;
- archive rebuild or PCA epoch changes invalidating any cached outcome summary;
- `unknown` appearing only for documented legacy or partial-input cases;
- policy stage and policy failure kind appearing in policy failures;
- archive membership, not mere candidate branch existence, controlling default
  inspiration eligibility;
- valid-non-elite shadow-pool sampling being disabled by default and labelled
  when enabled;
- automatic export, when enabled, writing atomically from database state rather
  than worker appends.

## Deferred Questions

- Which specific `complexity_cost` formula should be version 1?
- Should archive rebuilds append archive ingestion attempts, use a separate
  rebuild event table, or both?
- Which UI and API endpoints need cached outcome summaries, if any, after
  query-time derivation and SQL views are measured?
- Should same-cell replacement eventually use a fitness epsilon plus
  complexity tie-breaker, and how should that interact with MAP-Elites
  objective semantics?
