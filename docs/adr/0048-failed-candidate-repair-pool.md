# ADR 0048: Failed-candidate repair pool

Date: 2026-05-06

Status: Draft

Resolves: [ADR 0047](0047-defer-failed-candidate-lineage.md)

## Context

ADR 0047 deferred the question of whether failed candidate commits should become
part of Loreley's evolvable lineage. The current implementation is deliberately
closer to viable-frontier search:

- `EvolutionWorker` creates and publishes a worker-owned candidate commit before
  running the evaluator.
- `EvolutionJobStore.record_candidate_commit()` stores candidate hash, branch,
  and publication metadata on `EvolutionJob` before evaluation completes.
- `EvolutionJobStore.persist_success()` is the only path that creates
  `CommitCard`, `Metric`, fixed worker artifact, and evaluator artifact rows.
- `EvolutionJobStore.mark_job_failed()` records `status=FAILED` and
  `last_error`, while preserving candidate metadata already stored on the job.
- `MapElitesIngestion` only scans `SUCCEEDED` jobs with `result_commit_hash`.
  Failed jobs are not ingested into MAP-Elites.
- `MapElitesSampler` samples base and inspiration commits only from occupied
  MAP-Elites archive cells.
- repo-state ingestion is incremental-only after root bootstrap. A successful
  candidate is embedded from its git parent aggregate; runtime full-tree
  recomputation is not part of normal promotion.
- remote job branch pruning already protects failed jobs that still have
  candidate branch metadata, but manual job retry clears those fields. A repair
  design therefore needs its own durable candidate reference instead of relying
  only on the mutable job row.

The missing product capability is controlled repair of useful failed work:
large refactors, API migrations, and benchmark changes may require temporarily
broken commits, but letting those commits enter the normal MAP-Elites archive
would weaken archive meaning and waste capacity on unrecoverable failures.

The design review of this ADR agreed with the repair-pool direction and
recommended two changes that this ADR adopts:

1. Candidate validation failure must be a first-class evaluator outcome, not an
   exception shape inferred by the worker.
2. The first implementation should not make the repair result a git descendant
   of the failed commit. Repair source and git parent are distinct concepts.

## Decision

Add a separate failed-candidate repair pool, backed by a first-class candidate
commit ledger and a first-class evaluation outcome contract.

The normal MAP-Elites archive remains the only ordinary source of evolution
bases and inspirations. Failed commits are never inserted into
`map_elites_archive_cells`, never represented as `CommitCard` rows, and never
selected by `MapElitesSampler`.

Worker-produced commits are recorded in a new candidate ledger. Rows for failed
commits can become repair-pool entries only when they are durable, diagnosable,
and explicitly classified as repairable candidate failures.

The MVP repair mode is rebase/patch repair:

```text
V = nearest viable ancestor
F = failed candidate produced from V

repair worker:
  checkout V
  apply diff V..F into the worktree
  provide a safe DiagnosticCapsule
  ask the agent to repair the worktree
  commit R with git parent V
```

The resulting provenance is:

```text
archive / git parent lineage:
  V -> R

candidate provenance:
  V -> F
  R repaired_from F
```

`R` records `repair_source_candidate_id=F.id`, but `R` does not use `F` as its
git parent. This keeps incremental repo-state embedding on the viable lineage:
`R` can be embedded from `V` without requiring a failed-candidate aggregate.

True descendant repair remains a future mode:

```text
V -> F failed -> R repaired
```

That mode may be useful later, but it requires failed intermediate aggregates
and stricter safeguards. It is not part of the MVP.

The repair lane is explicit and budgeted:

- disabled by default;
- bounded by per-candidate attempts, failed-depth, scheduler tokens, active
  repair jobs, and global job limits;
- isolated from normal archive sampling by using a distinct `job_kind`,
  sampling strategy, prompts, and UI/API language;
- safe by default: failure logs and artifacts are not agent-visible unless they
  pass ADR 0046 visibility/projection policy and the DiagnosticCapsule
  sanitizer.

## Non-Goals

- Do not make failed commits normal MAP-Elites candidates.
- Do not use `CommitCard` for failed candidates. `CommitCard` remains the
  lightweight representation for evaluated viable candidates and root metadata.
- Do not let normal samplers query `CandidateCommit` directly.
- Do not make repair-produced commits git descendants of failed candidates in
  the MVP.
- Do not compute repo-state aggregates for failed candidates in the MVP.
- Do not add runtime full-tree repo-state recomputation for repair promotion.
- Do not let failed candidates influence PCA fitting history, archive
  projection state, behavior-space density, or normal sampler eligibility.
- Do not load raw evaluator logs, raw agent output, or arbitrary filesystem
  paths into repair prompts by default.
- Do not hide repair behind prompt wording or backend-local agent loops.

## Terminology

`candidate commit`
: A git commit produced by a Loreley worker.

`candidate ledger`
: The durable table of worker-produced candidate commits, including successful,
  failed, and repair-produced commits.

`failed candidate`
: A candidate commit whose evaluation outcome is `candidate_failed`.

`repair-pool entry`
: A failed candidate that passes eligibility checks and may be sampled by the
  repair scheduler.

`repair source`
: The failed candidate being repaired. The repair source is provenance; it is
  not necessarily the repair job's git parent.

`nearest viable ancestor`
: The archive-valid commit from which the failed candidate was originally
  produced and from which the repair result can be embedded.

`repair job`
: An `EvolutionJob` whose task is to repair a failed candidate, not to explore a
  new archive cell from scratch.

`DiagnosticCapsule`
: A bounded, sanitized, policy-checked summary of evaluation failure evidence
  that is safe to show to the repair agent.

`promoted repair result`
: A repair-produced candidate that passes evaluation and enters the normal
  success path. It becomes a regular evaluated candidate; it becomes a viable
  archive entry only if MAP-Elites accepts it into a cell.

## Evaluation Outcome Contract

Evaluator output should be represented as an envelope:

```text
EvaluationOutcome
- schema_version
- evaluator_name
- evaluator_version
- candidate_commit_hash
- outcome_kind
- result nullable
- failure nullable
- artifact_records
- started_at
- finished_at
```

Supported `outcome_kind` values:

- `passed`: evaluator completed and produced comparable success metrics.
- `candidate_failed`: evaluator completed and determined that the candidate
  code failed validation, tests, lint, typecheck, benchmark gates, or another
  candidate-owned gate.
- `evaluator_failed`: evaluator logic or evaluator dependencies failed.
- `infrastructure_failed`: checkout, worktree, runner, network, resource,
  artifact persistence, database, or push infrastructure failed.
- `inconclusive`: evaluator completed partially but cannot reliably assign the
  failure to the candidate, evaluator, or infrastructure.

Keep `EvaluationResult` as a success-only shape:

```text
EvaluationResult
- metrics
- quality_score
- behavior_descriptors
- evaluation_summary
- artifact_refs
```

Add a failure shape:

```text
EvaluationFailureResult
- failure_stage
- failure_kind
- repairability
- repairability_reason
- safe_failure_summary
- agent_visible_evidence_refs
- human_only_artifact_refs
- hidden_artifact_refs
- exit_code nullable
- timeout_seconds nullable
- failing_tests_summary nullable
- compiler_errors_summary nullable
- stack_trace_summary nullable
- policy_version
```

`repairability` values are `repairable`, `not_repairable`, and `unknown`.

Repair eligibility is only possible for:

```text
outcome_kind = candidate_failed
AND repairability = repairable
AND DiagnosticCapsule policy passed
```

Timeouts must be classified carefully. A candidate-owned test or program
timeout may be `candidate_failed`, but runner/container/resource timeout is
`infrastructure_failed`. MVP repair should not allow timeout failures unless
the evaluator can classify them as candidate-owned with high confidence.

Worker exception handling remains as a fallback:

- valid `EvaluationOutcome`: persist the evaluator outcome;
- evaluator process failed before a valid outcome: synthesize
  `evaluator_failed` or `infrastructure_failed`;
- checkout, runner, DB, artifact, push, or worktree error: persist
  `infrastructure_failed`.

Synthetic failure outcomes created from exceptions are not repairable by
default.

## Data Model

Add `CandidateCommit` as the source of truth for worker-produced git commits.
It is not a replacement for `CommitCard`.

```text
candidate_commits
- id UUID primary key
- commit_hash string(64) unique not null
- git_parent_commit_hash string(64) not null
- nearest_viable_ancestor_hash string(64) null
- island_id string(64) null

- produced_by_job_id UUID null references evolution_jobs(id) on delete set null
- run_token UUID null
- job_kind string(32) not null
- repair_source_candidate_id UUID null references candidate_commits(id)
- repair_mode string(32) null

- candidate_branch_name string(255) null
- candidate_published_at timestamptz null
- publication_status string(32) not null

- evaluation_status string(32) not null
- latest_evaluation_attempt_id UUID null

- archive_status string(32) not null
- lifecycle_status string(32) not null

- failure_stage string(32) null
- failure_kind string(64) null
- failure_summary text null
- failure_evidence_id UUID null

- repair_state string(32) not null
- failed_depth integer not null default 0
- repair_attempts integer not null default 0
- last_repair_job_id UUID null references evolution_jobs(id) on delete set null

- repo_state_aggregate_status string(32) not null default 'not_required'
- repo_state_aggregate_error text null

- commit_card_id UUID null references commit_cards(id) on delete set null
- created_at / updated_at
- published_at timestamptz null
- evaluated_at timestamptz null
- archived_at timestamptz null
```

Recommended low-cardinality values:

- `job_kind`: `evolution`, `seed`, `repair`.
- `repair_mode`: `rebase_from_nearest_viable`, `patch_from_source_diff`,
  `descendant`.
- `publication_status`: `created`, `published`, `publish_failed`,
  `discarded`.
- `evaluation_status`: `not_evaluated`, `passed`, `candidate_failed`,
  `evaluator_failed`, `infrastructure_failed`, `inconclusive`.
- `archive_status`: `not_considered`, `member`, `rejected`, `superseded`,
  `not_applicable`.
- `lifecycle_status`: `active`, `quarantined`, `discarded`.
- `failure_stage`: `planning`, `coding`, `commit`, `publish`, `evaluation`,
  `success_persistence`, `ingestion`, `unknown`.
- `failure_kind`: `validation_failed`, `test_failed`, `typecheck_failed`,
  `lint_failed`, `candidate_timeout`, `evaluator_error`,
  `infrastructure_error`, `repository_error`, `unknown`.
- `repair_state`: `audit_only`, `ineligible`, `eligible`, `scheduled`,
  `repairing`, `repaired`, `exhausted`, `quarantined`, `discarded`.
- `repo_state_aggregate_status`: `not_required`, `pending`, `ready`, `failed`.

Add an `EvaluationAttempt` table rather than storing all evaluation detail on
`CandidateCommit`:

```text
evaluation_attempts
- id UUID primary key
- candidate_commit_id UUID references candidate_commits(id)
- job_id UUID references evolution_jobs(id)
- evaluator_name string(128)
- evaluator_version string(128)
- outcome_kind string(32)
- failure_kind string(64) null
- failure_stage string(32) null
- repairability string(32) null
- safe_failure_summary text null
- diagnostic_capsule_id UUID null
- artifact_policy_version string(64) null
- started_at timestamptz
- finished_at timestamptz
- created_at / updated_at
```

Artifact rows remain in `EvaluationArtifactRecord`. Failed-candidate evidence
uses `commit_card_id=NULL` and links by `job_id`, `commit_hash`, and
`evaluation_attempt_id` where available.

Add these fields to `EvolutionJob`:

```text
job_kind string(32) default 'evolution'
repair_source_candidate_id UUID null references candidate_commits(id)
repair_mode string(32) null
```

Keep `EvolutionJob.candidate_commit_hash`, `candidate_branch_name`,
`candidate_published_at`, and `result_commit_hash` as denormalized job audit and
list-view fields. They remain useful for the current UI and CLI. The new
candidate ledger is the stable lineage and repair scheduler source.

Required invariants:

```text
CommitCard.commit_hash references CandidateCommit.commit_hash
AND CandidateCommit.evaluation_status = passed
```

```text
CandidateCommit.commit_card_id IS NULL
WHEN CandidateCommit.evaluation_status != passed
```

```text
Archive cell commit_hash has a CommitCard
AND CandidateCommit.evaluation_status = passed
AND CandidateCommit.archive_status = member
```

```text
Normal MapElitesSampler queries map_elites_archive_cells only,
never CandidateCommit directly.
```

```text
Repair source candidate has:
CandidateCommit.evaluation_status = candidate_failed
AND CandidateCommit.repair_state = eligible
```

```text
At most one pending, queued, or running repair job exists per
repair_source_candidate_id.
```

Indexes:

- `candidate_commits(commit_hash)` unique.
- `candidate_commits(produced_by_job_id)`.
- `candidate_commits(island_id, repair_state, evaluation_status, updated_at)`.
- `candidate_commits(repair_source_candidate_id)`.
- `candidate_commits(git_parent_commit_hash)`.
- `candidate_commits(nearest_viable_ancestor_hash)`.
- `evaluation_attempts(candidate_commit_id, started_at)`.
- `evaluation_attempts(job_id)`.
- `evolution_jobs(job_kind, status, scheduled_at)`.
- `evolution_jobs(repair_source_candidate_id, status)`.

Because Loreley uses schema reset/create-all instead of Alembic migrations, this
schema change must bump `INSTANCE_SCHEMA_VERSION` and document the reset path.

## Worker Lifecycle

`record_candidate_commit()` should create or update a `CandidateCommit` row when
the worker creates a commit. At this point the row has:

- `publication_status='created'` for a local candidate commit and
  `publication_status='published'` after remote publication succeeds;
- `evaluation_status='not_evaluated'`;
- `archive_status='not_considered'` for normal candidates and
  `archive_status='not_applicable'` for failed repair sources;
- `job_kind` from the owning job;
- `git_parent_commit_hash` from the actual commit parent;
- `nearest_viable_ancestor_hash` from the selected archive base;
- `repair_source_candidate_id` and `repair_mode` when the job is a repair job.

For MVP repair jobs, the worker should:

1. load source failed candidate `F`;
2. resolve nearest viable ancestor `V`;
3. checkout `V`;
4. apply the patch represented by `git diff V..F`;
5. render repair planning/coding context with the DiagnosticCapsule;
6. commit repair result `R` on top of `V`;
7. record `R.repair_source_candidate_id=F.id` and
   `R.repair_mode='rebase_from_nearest_viable'`.

If the patch from `V..F` does not apply cleanly, the repair job should fail with
`outcome_kind=infrastructure_failed` or `inconclusive`, not enter a nested
repair loop.

`persist_success()` updates the candidate ledger after writing `CommitCard` and
`Metric` rows:

- create an `EvaluationAttempt` with `outcome_kind='passed'`;
- set candidate `evaluation_status='passed'`;
- link `commit_card_id`;
- clear repair-only failure fields on the produced candidate;
- if the successful job was a repair job, update the source failed candidate's
  `repair_state` to `repaired` once the success row is committed.

`MapElitesIngestion` may later update the candidate row to
`archive_status='member'` when an archive record is created. If ingestion skips
the commit because it does not improve a cell, set `archive_status='rejected'`.
Archive membership should still be derived from `map_elites_archive_cells` when
exact current state matters, because PCA refits and archive replacement can move
or evict rows.

On failure, replace the best-effort-only `mark_job_failed()` path with a
structured `persist_failure()` path when the worker still owns the lease. It
should:

1. lock the active job by `job_id` and `run_token`;
2. write any available planning and coding artifacts even if evaluation failed;
3. persist the `EvaluationOutcome` or synthesize a non-repairable fallback
   outcome;
4. persist a bounded failure artifact row when no evaluator artifact exists;
5. create an `EvaluationAttempt`;
6. mark the job `FAILED`, clear lease fields, and store `last_error`;
7. update the candidate ledger if a candidate commit exists;
8. decide repair eligibility from outcome kind, failure kind, repairability,
   branch durability, nearest viable ancestor aggregate readiness, and
   DiagnosticCapsule policy.

Failures before a candidate commit exists remain ordinary failed jobs. They are
not repair-pool entries. If the failed job is itself a repair job, the source
failed candidate must still receive a terminal attempt-state update: move it
back to `eligible` when attempts remain, or to `exhausted` when the consumed
attempt reaches the configured maximum.

Failures caused by publish errors, lost leases, database errors, artifact-store
errors, or success-persistence errors should default to `repair_state='audit_only'`
or `quarantined`, because they do not prove that the candidate itself is an
invalid but useful repair source.

## Failure Evidence And DiagnosticCapsule

Repair jobs need concise failure context, but raw failure outputs can contain
secrets, prompt injection, or irrelevant noise.

Evaluator plugins should return structured failure evidence through
`EvaluationOutcome`. Exceptions remain a legacy adapter path only.

The repair agent may see only a `DiagnosticCapsule`, not raw evaluator output:

```text
DiagnosticCapsule
- schema_version
- policy_version
- failure_stage
- failure_kind
- repairability
- safe_failure_summary
- failing_test_names
- failing_test_locations
- compiler/typecheck/lint error summaries
- selected sanitized stack frames
- selected bounded stdout/stderr excerpts
- diff summary between nearest viable ancestor and failed candidate
- artifact manifest
- evaluator name/version
```

Default-deny content:

- raw stdout/stderr and raw test logs;
- raw evaluator artifacts and benchmark outputs;
- environment variables, credentials, cookies, authorization headers, and URLs
  with sensitive query params;
- absolute host paths and arbitrary artifact paths;
- paths outside the repo or artifact root;
- binary blobs, HTML reports, screenshots, network traces, and large coverage
  reports;
- human-only or hidden artifacts.

Minimum sanitization:

- enforce total and per-excerpt byte budgets;
- strip ANSI escapes, terminal OSC hyperlinks, control characters, and
  binary-looking content;
- normalize UTF-8 and newlines;
- redact common token, key, private-key, cookie, auth-header, and database URL
  patterns;
- normalize paths to repo-relative paths where possible;
- reject `..`, symlink escapes, and paths outside allowed roots;
- present diagnostic text as untrusted data, never as prompt instructions.

ADR 0046 visibility and projection rules remain authoritative:

- `agent_visible` is required but not sufficient; content still needs capsule
  projection, redaction, and budget checks.
- `human_only` and `hidden` artifacts are never rendered into repair prompts.
- `path` projection is allowed only for repo-relative source/test locations in
  the MVP.

Until the outcome contract exists, exceptions from the evaluator can produce
only a bounded synthetic diagnostic:

```text
key: evaluation_failure
kind: failure
visibility: human_only
agent_projection: summary
summary: bounded EvaluationError message
repairability: unknown
```

Synthetic diagnostics are not repairable by default.

## Repo-State Aggregate Policy

MVP repair does not require a repo-state aggregate for the failed candidate.

Reason: the repair result `R` is committed on top of the nearest viable ancestor
`V`, not on top of failed candidate `F`. `MapElitesManager.ingest(R)` can use the
existing incremental path:

```text
aggregate(R) = aggregate(V) + diff(V..R)
```

Repair eligibility therefore requires:

```text
nearest viable ancestor has MapElitesRepoStateAggregate ready
```

and sets the failed candidate's:

```text
repo_state_aggregate_status='not_required'
```

Failed candidates must not update:

- MAP-Elites archive cells;
- PCA history;
- PCA projection state;
- behavior-space density;
- candidate fitness or metrics;
- sampler state.

If a future ADR enables true descendant repair, failed intermediate aggregates
must be clearly marked and fenced:

```text
aggregate_kind='failed_intermediate'
eligible_for_archive_projection=false
eligible_for_pca_history=false
eligible_for_sampler=false
```

That future service should call the incremental repo-state API, not the full
bootstrap API, and should compute aggregates only for repair-eligible candidates
after safe evidence checks pass. It should not compute aggregates for every
failed job.

## Repair Scheduling

Introduce a `FailedCandidateRepairSampler` owned by `JobScheduler` or
`EvolutionScheduler`. It runs separately from `MapElitesSampler`.

MVP eligibility:

- repair feature is enabled;
- `publication_status='published'`;
- `evaluation_status='candidate_failed'`;
- `failure_stage='evaluation'`;
- `failure_kind` is allowlisted for repair, initially `validation_failed`,
  `test_failed`, `typecheck_failed`, and `lint_failed`;
- `repairability='repairable'`;
- DiagnosticCapsule policy passed;
- `nearest_viable_ancestor_hash` is present and has repo-state aggregate ready;
- `failed_depth <= FAILED_CANDIDATE_REPAIR_MAX_DEPTH`;
- `repair_attempts < FAILED_CANDIDATE_REPAIR_MAX_ATTEMPTS`;
- `lifecycle_status='active'`;
- no pending, queued, or running repair job references the same source;
- candidate branch is protected from pruning while eligible, scheduled, or
  repairing.

Do not repair:

- evaluator failures;
- infrastructure failures;
- repository failures;
- unknown or inconclusive failures;
- unsafe diagnostics;
- unpublished commits;
- failures without a viable ancestor aggregate;
- repair-produced failures in the MVP.

Scheduling uses a token bucket:

```text
repair_enabled=false
normal_jobs_per_repair_token=9
max_repair_tokens=3
max_active_repair_jobs=1
max_repair_jobs_per_scheduler_tick=1
```

Rules:

- completing `normal_jobs_per_repair_token` normal jobs adds one repair token;
- scheduling one repair job consumes one token;
- repair tokens are capped by `max_repair_tokens`;
- repair jobs still count against global unfinished and total job limits;
- accrued tokens reserve up to `max_repair_jobs_per_scheduler_tick` repair slots
  before normal archive sampling fills the scheduler batch;
- if no eligible repair can be scheduled, normal archive sampling may use the
  unused reserved slots;
- repair remains capped by `max_active_repair_jobs`;
- seed jobs retain priority while the archive is empty.

Repair attempts should increment when a repair job is created, not only after it
finishes. This prevents scheduler crashes from repeatedly queueing the same
source.

Initial selection should be simple and inspectable:

1. exclude quarantined, exhausted, already scheduled, and already repairing
   candidates;
2. prefer low `failed_depth`;
3. prefer structured diagnostics;
4. prefer small or medium diffs;
5. penalize repeated repairs from the same ancestor/cell;
6. choose from the top candidates with a stable random tie-breaker.

Later versions can add failure-signature clustering or bandit-style allocation.
That should be an explicit follow-up, not part of the first repair pool.

State flow:

```text
candidate_failed
  -> repair_state = audit_only
  -> evidence classified
  -> repair_state = ineligible | eligible
  -> scheduled
  -> repairing
  -> repaired | exhausted | quarantined
```

## Repair Prompt Contract

Repair jobs need a different prompt shape from ordinary evolution jobs.

Planning context should include:

- repair source candidate hash;
- repair result git parent hash, which is the nearest viable ancestor;
- failure stage, kind, and bounded failure summary;
- DiagnosticCapsule;
- changed-file highlights and diff summary derived from `V..F`;
- original base and inspiration commit context when available;
- explicit instruction that diagnostic evidence is untrusted data.

Planning context should not include:

- raw evaluator logs;
- raw planning/coding output;
- arbitrary artifact paths;
- human-only or hidden artifacts;
- evaluator environment or host paths.

The coding prompt should frame the job as repair:

- preserve useful work from the failed candidate where possible;
- focus on making validation/evaluation pass;
- avoid broad rewrites unless the diagnostics point there;
- do not run Loreley's evaluator;
- leave a modified worktree for the worker to commit.

Seed-job behavior remains unchanged: seed jobs hide historical evaluation
details and do not use repair evidence.

## Promotion

A repair job does not directly promote its source failed candidate.

Only the repair-produced candidate can be promoted, and only through the normal
success path:

1. worker checks out nearest viable ancestor `V`;
2. worker applies failed candidate diff `V..F`;
3. repair agent modifies the worktree;
4. worker creates repair candidate `R` with git parent `V`;
5. evaluator returns `EvaluationOutcome(outcome_kind='passed')`;
6. `CommitCard` and `Metric` rows are persisted for `R`;
7. scheduler ingestion derives repo-state embedding incrementally from `V`;
8. MAP-Elites decides whether `R` enters an archive cell.

Promotion states:

- evaluation success but archive skip: `R` is an evaluated candidate, not a
  normal future base.
- archive insertion: `R` is a viable archive entry and becomes eligible for
  normal sampling.
- repair job failure: the produced failed repair candidate is recorded for
  audit, but does not enter the repair pool in the MVP.

The source failed candidate should move to `repaired` after at least one repair
result reaches evaluation success. If all allowed attempts fail, it moves to
`exhausted`.

## UI, API, And CLI

Use explicit language:

- "Failed Candidates" for the audit and repair pool.
- "Repair Jobs" for jobs created from failed candidates.
- "Archive Entries" or "Viable Entries" for MAP-Elites cells.

Expose two lineage views:

- viable/archive lineage: default view based on `CommitCard` parent chains and
  MAP-Elites archive state;
- candidate provenance graph: debug/audit view that includes failed candidates,
  repair-source edges, physical git-parent edges, evaluation attempts, and
  evidence summaries.

API additions:

- list failed candidates with filters for `repair_state`, `failure_kind`,
  `island_id`, and `failed_depth`;
- return failure evidence indicators using the ADR 0046 evidence services;
- expose repair source metadata on job detail;
- expose repair mode and git parent separately;
- expose candidate provenance graph mode distinct from the viable graph.

UI additions:

- Jobs page: show `job_kind`, repair source link, and repair mode.
- Failed Candidates page or tab: status, failure summary, branch, repair
  attempts, evidence status, and nearest viable ancestor.
- Graphs page: add a candidate-provenance mode distinct from the existing
  viable `CommitCard` parent-chain graph.
- Commit detail: when a successful commit was repaired from a failed candidate,
  show repair provenance without implying the failed source was an archive
  entry.

CLI additions:

- `loreley failed-candidates list`
- `loreley failed-candidates discard <id>`
- `loreley failed-candidates schedule-repair <id>`
- `loreley failed-candidates quarantine <id>`

Manual retry of an original failed job should remain separate from scheduling a
repair job. Retrying rewinds the same job spec from the original base; repair
uses the failed candidate as diagnostic/provenance source and commits the result
on top of the nearest viable ancestor.

## Observability

Add machine-readable logs and counters for:

- candidate commit records created by job kind;
- evaluation outcomes by kind and evaluator;
- evaluation attempts created;
- failed candidate records created;
- DiagnosticCapsule projection, redaction, and omission decisions by reason;
- repair eligibility decisions by reason;
- repair tokens accrued and consumed;
- repair jobs scheduled;
- repair jobs succeeded, failed, exhausted, and promoted to archive entries;
- candidate branch protection decisions;
- patch application failures for rebase/patch repair.

Avoid high-cardinality metrics labels. Commit hashes, job IDs, candidate IDs,
and artifact IDs belong in logs or structured event payloads, not metric label
sets.

## Configuration

Suggested settings:

```text
FAILED_CANDIDATE_REPAIR_ENABLED=false
FAILED_CANDIDATE_REPAIR_MODE=rebase_from_nearest_viable
FAILED_CANDIDATE_REPAIR_MAX_DEPTH=1
FAILED_CANDIDATE_REPAIR_MAX_ATTEMPTS=1
FAILED_CANDIDATE_REPAIR_NORMAL_JOBS_PER_TOKEN=9
FAILED_CANDIDATE_REPAIR_MAX_TOKENS=3
FAILED_CANDIDATE_REPAIR_MAX_ACTIVE_JOBS=1
FAILED_CANDIDATE_REPAIR_MAX_JOBS_PER_TICK=1
FAILED_CANDIDATE_REPAIR_FAILURE_KINDS=validation_failed,test_failed,typecheck_failed,lint_failed
FAILED_CANDIDATE_REPAIR_AGENT_FEEDBACK_MODE=diagnostic_capsule
FAILED_CANDIDATE_REPAIR_MAX_DIFF_BYTES=65536
FAILED_CANDIDATE_REPAIR_MAX_DIAGNOSTIC_BYTES=16384
```

Defaults keep the feature disabled. Enabling repair should require operators to
accept that failed candidates can consume scheduler slots. In the MVP they do
not consume failed-candidate embedding budget because failed aggregates are not
computed.

## Implementation Plan

Phase 1: evaluation outcome and candidate ledger

- Add `EvaluationOutcome`, `EvaluationFailureResult`, and `EvaluationAttempt`.
- Add `CandidateCommit`, `EvolutionJob.job_kind`,
  `EvolutionJob.repair_source_candidate_id`, and `EvolutionJob.repair_mode`.
- Bump `INSTANCE_SCHEMA_VERSION`.
- Create candidate rows when candidate commits are recorded.
- Split artifact writing so planning/coding artifacts can be persisted on
  failure.
- Persist bounded failure evidence with `commit_card_id=NULL`.
- Update branch-pruning protection to read from `CandidateCommit`, not only
  mutable `EvolutionJob` candidate fields.

Phase 2: DiagnosticCapsule and repair eligibility

- Add DiagnosticCapsule projection and sanitization.
- Adapt existing evaluators to return `EvaluationOutcome`; keep exception
  fallback non-repairable.
- Classify repair eligibility from outcome kind, repairability, allowlisted
  failure kind, safe evidence, publication status, attempts, and nearest viable
  ancestor aggregate readiness.
- Add tests proving evaluator, infrastructure, unknown, unsafe, and unpublished
  failures are not repair-eligible.

Phase 3: rebase/patch repair scheduler and worker mode

- Add token-bucket repair capacity allocation.
- Add repair job creation and queueing with one unfinished job per source.
- Add worker flow that checks out nearest viable ancestor and applies `V..F`
  before repair prompting.
- Add failed-candidate planning context and repair-specific prompt rendering.
- Keep the feature disabled by default.

Phase 4: promotion, UX, and operations

- Update ingestion to annotate candidate archive outcome.
- Add API/CLI/UI surfaces for failed candidates and repair jobs.
- Add candidate-provenance graph mode.
- Add observability counters and release notes.

Future phase: true descendant repair

- Add failed-intermediate aggregate support only if rebase/patch repair proves
  insufficient.
- Mark failed aggregates with explicit non-archive, non-PCA, non-sampler flags.
- Add tests proving a successful descendant can be ingested from a failed parent
  aggregate without adding the failed parent to PCA history or archive.

## Acceptance Criteria

- Failed candidate commits are never present in `map_elites_archive_cells`.
- Failed candidate commits never create `CommitCard` rows.
- Normal archive sampling continues to select only archive cell commits.
- Evaluator candidate failures are represented as `EvaluationOutcome`, not only
  as exceptions.
- Worker-synthesized exception fallback outcomes are not repairable by default.
- A failed job with no candidate commit creates no repair-pool entry.
- A repair job that fails before producing a candidate still updates its source
  failed candidate to `eligible` or `exhausted`.
- A failed job with a candidate commit but no durable branch is audit-only by
  default.
- A repair-eligible failed candidate has safe DiagnosticCapsule evidence.
- A repair-eligible failed candidate has a nearest viable ancestor with a ready
  repo-state aggregate.
- A repair job records `job_kind='repair'`, `repair_source_candidate_id`, and
  `repair_mode='rebase_from_nearest_viable'`.
- In the MVP, repair candidate `R` has nearest viable ancestor `V` as git
  parent, not failed candidate `F`.
- A successful repair candidate follows the existing success and ingestion
  paths.
- A repair candidate can become a normal sampled base only after MAP-Elites
  inserts it into an archive cell.
- Human-only and hidden failure artifacts are never rendered into repair
  prompts.
- Manual job retry and repair scheduling remain distinct operations.

## Risks

- Repair mode can spend capacity on unrecoverable failures. Keep the feature
  disabled by default and budgeted when enabled.
- Failure diagnostics may be too weak until evaluators return structured
  `EvaluationOutcome` values.
- The rebase/patch repair mode can fail when `V..F` does not apply cleanly to
  the repair worktree. Treat that as a bounded repair failure, not as a reason
  to introduce nested repair in the MVP.
- Candidate lineage can diverge from the current `CommitCard` parent-chain
  assumptions. Graphs and trajectory rollups must either stay viable-only or
  explicitly query the candidate ledger.
- DiagnosticCapsule sanitization is a security boundary. ADR 0046 visibility
  alone is not enough.
- Failed candidates must not influence PCA axes unless a future ADR chooses
  that tradeoff. The current design keeps failed candidates out of PCA history.
- True descendant repair remains more complex because it requires failed
  intermediate aggregates and strict non-archive guards.

## Deferred Questions

These questions are deferred until the MVP has real repair data:

1. Should Loreley ever enable true descendant repair, where the repair result is
   a git child of the failed candidate?
2. If true descendant repair is enabled, what exact aggregate metadata and DB
   constraints prevent failed intermediates from affecting PCA, archive
   projection, or sampling?
3. Should timeout failures become repairable after evaluators can reliably
   classify candidate-owned timeouts?
4. Should repair allocation evolve from a token bucket to a bandit policy once
   repair success-rate data exists?
5. Should candidate provenance graphs become a primary UI, or stay an
   operations/debug surface?
