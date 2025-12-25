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
  commit is registered and evaluated as a baseline in the database.
- **Construction**: created by `EvolutionScheduler` with:
  - the shared `Settings` instance,
  - the interactive `rich` console,
  - a `git.Repo` handle for the scheduler's repository root,
  - the experiment-scoped `MapElitesManager`,
  - the current `experiment` and its `repository`.

### Ingesting succeeded jobs

- **`ingest_completed_jobs() -> int`**:
  - Scans for `SUCCEEDED` `EvolutionJob` rows up to
    `SCHEDULER_INGEST_BATCH_SIZE`.
  - Filters out jobs whose ingestion status is already terminal
    (`"succeeded"` or `"skipped"`).
  - Builds a `JobSnapshot` for each remaining job and forwards it to
    `_ingest_snapshot(...)`.
  - Returns the number of jobs whose commits actually updated the MAP-Elites
    archive.

Internally, `_ingest_snapshot(...)`:

1. Reads `result_commit_hash` from the job row and loads metrics from the
   `metrics` table for that commit hash.
2. Ensures the corresponding git commit is present locally, fetching from
   remotes as necessary.
3. Calls `MapElitesManager.ingest(...)` with:
   - `commit_hash`,
   - `metrics`,
   - `island_id`,
   - `repo_root` and `treeish`,
4. Writes ingestion state back onto the job row, including:
   - `status` (`"succeeded"` or `"skipped"`),
   - `delta`, `status_code`, and `message` from the ingest result,
   - `cell_index` when the ingest produced a record,
   - retry bookkeeping (`attempts`, `last_attempt_at`, `reason`).

This state allows ingestion to be retried safely and audited later without
re-running the full evaluation.

### Root commit initialisation

When `MAPELITES_EXPERIMENT_ROOT_COMMIT` is set, `EvolutionScheduler` asks
`MapElitesIngestion` to initialise that commit via
`initialise_root_commit(commit_hash)`:

1. `_ensure_commit_available(...)` guarantees the commit exists locally,
   fetching from remotes as needed.
2. `_ensure_root_commit_metadata(...)` creates or updates a `CommitCard`
   row with:
   - the commit's parent, author, and message,
   - the current `experiment_id`,
   - a default island id (from `MAPELITES_DEFAULT_ISLAND_ID` or `"main"`),
   - bounded commit-card fields (`subject`, `change_summary`, `highlights`).
3. `_ensure_root_commit_evaluated(...)` runs a one-off evaluation for the root
   commit when no `Metric` rows already exist, writing baseline metrics into
   the `metrics` table. These metrics act as an experiment-wide baseline but do
   not insert the root commit into any MAP-Elites archive.

Failures during root-commit initialisation are logged but do **not** prevent
the scheduler from running; they simply mean the experiment may effectively
start from the first successfully ingested commit instead.

## Interaction with EvolutionScheduler

`EvolutionScheduler.tick()` uses `MapElitesIngestion` as the first stage in the
pipeline:

1. `ingest_completed_jobs()` ingests any newly succeeded jobs and annotates
   them with ingestion state.
2. Later, when all jobs have finished and the global job limit has been
   reached, `EvolutionScheduler` uses MAP-Elites metrics and commit metadata
   to create a dedicated git branch for the current best-fitness commit.

Separating this logic into `MapElitesIngestion` keeps the main scheduler loop
small and clarifies the boundary between **job lifecycle** and **archive
maintenance**.


