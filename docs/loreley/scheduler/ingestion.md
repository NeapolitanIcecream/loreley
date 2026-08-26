# loreley.scheduler.ingestion

Result ingestion and MAP-Elites maintenance logic extracted from the central
evolution scheduler.

The `MapElitesIngestion` class owns how succeeded jobs are **discovered**,
**mapped** to git commits, and **folded** into the MAP-Elites archives, as well
as how the configured experiment root commit is initialised.

## MapElitesIngestion

```python
from loreley.scheduler.ingestion import MapElitesIngestion
```

- **Purpose**: ingest completed evolution jobs into MAP-Elites, record rich
  ingestion state back onto the job row, and ensure the experiment's root
  commit has metadata plus repo-state bootstrap data in the database.
- **Construction**: created by `EvolutionScheduler` with:
  - the shared `Settings` instance,
  - the interactive `rich` console,
  - the scheduler `repo_root` path,
  - a `git.Repo` handle for that repository root,
  - the experiment-scoped `MapElitesManager`.

### Ingesting succeeded jobs

- **`ingest_completed_jobs() -> int`**:
  - Scans for `SUCCEEDED` `EvolutionJob` rows up to
    `SCHEDULER_INGEST_BATCH_SIZE`.
  - Filters out jobs whose ingestion status is already terminal
    (`"succeeded"` or `"skipped"`).
  - Applies an exponential retry backoff for `"failed"` jobs using
    `ingestion_attempts` and `ingestion_last_attempt_at` so transient issues do
    not trigger a tight retry loop.
  - Builds a `JobSnapshot` for each remaining job and forwards it to
    `_ingest_snapshot(...)`.
  - Commits `ingestion.started` before commit resolution, embedding, or archive
    work. A crash therefore leaves censored start evidence; every retry gets a
    new ordinal.
  - Returns the number of jobs whose commits actually updated the MAP-Elites
    archive.

Internally, `_ingest_snapshot(...)`:

1. Reads `result_commit_hash` from the job row and canonicalizes it.
2. If the evaluator supplied a `candidate_identity`, checks whether the same
   evaluator-scoped identity has already completed ingestion in the island. An
   equivalent candidate is recorded as skipped and does not enter PCA history
   or consume another archive slot.
3. Loads metrics from the `metrics` table for that commit hash.
4. Ensures the corresponding git commit is present locally, fetching from
   remotes as necessary.
5. Calls `MapElitesManager.ingest(...)` with:
   - `commit_hash`,
   - `metrics`,
   - `island_id`,
   - `repo_root`,
6. Writes ingestion state back onto the job row, including:
   - `status` (`"succeeded"` or `"skipped"`),
   - `delta`, `status_code`, and `message` from the ingest result,
   - `cell_index` when the ingest produced a record,
   - retry bookkeeping (`attempts`, `last_attempt_at`, `reason`).
7. Closes the ingestion event in the same transaction as job state and archive
   snapshot changes, and records a bounded archive-consideration outcome such
   as admission, duplicate identity, Pareto rejection, projection warmup, or
   retry exhaustion.

Atomic snapshot persistence compares archive membership before and after the
update. It emits per-member admission, movement, and removal events plus a
projection-rebuild summary when applicable. Local Pareto replacement,
projection rebuild, and explicit clearing remain distinguishable without
persisting a second full archive copy.

This state allows ingestion to be retried safely (with backoff), audited later,
and kept isolated per job so individual failures do not abort the scheduler loop.

### Root commit initialisation

When `MAPELITES_EXPERIMENT_ROOT_COMMIT` is set, `EvolutionScheduler` asks
`MapElitesIngestion` to initialise that commit via
`initialise_root_commit(commit_hash)`:

1. `_ensure_commit_available(...)` guarantees the commit exists locally,
   fetching from remotes as needed.
2. `_ensure_root_commit_metadata(...)` creates or updates a `CommitCard`
   row with:
   - the commit's parent, author, and message,
   - no archive island assignment, because the root commit is shared campaign metadata rather than an island candidate,
   - bounded commit-card fields (`subject`, `change_summary`, `highlights`).
3. `_ensure_root_commit_repo_state_bootstrap(...)` bootstraps the root
   repo-state aggregate for incremental-only ingestion by computing and
   persisting the root commit aggregate (full enumeration allowed at bootstrap).
4. Root evaluator baselines are not produced by ingestion. The scheduler calls
   `loreley.scheduler.baselines.BaselineBootstrapService` separately to create
   or load the matching row in `campaign_baselines`.

Repo-state bootstrap failures are fatal because the scheduler runs repo-state
ingestion in incremental-only mode at runtime. Campaign baseline failures are
handled by `BASELINE_BOOTSTRAP_POLICY`: `required` blocks dispatch and
scheduling, while `warn` records a degraded `campaign_baselines` row and keeps
the scheduler moving with baseline deltas unavailable.

## Interaction with EvolutionScheduler

`EvolutionScheduler.tick()` uses `MapElitesIngestion` as the first stage in the
pipeline:

1. `ingest_completed_jobs()` ingests any newly succeeded jobs and annotates
   them with ingestion state.
2. Later, when all jobs have finished and the global job limit has been
   reached, `EvolutionScheduler` uses MAP-Elites metrics and commit metadata
   to create a dedicated git branch for the best retained value of the configured
   primary objective. This is an operational projection, not a scalarization of
   Pareto admission.

Separating this logic into `MapElitesIngestion` keeps the main scheduler loop
small and clarifies the boundary between **job lifecycle** and **archive
maintenance**.
