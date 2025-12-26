# loreley.scheduler.main

Central orchestration loop that keeps the Loreley evolution pipeline moving by coupling the MAP-Elites archive, the PostgreSQL job store, and the Dramatiq worker queue.

## EvolutionScheduler

- **Purpose**: continuously monitors unfinished jobs (`pending`, `queued`, `running`), schedules new work from the MAP-Elites archive when capacity allows, dispatches pending jobs to the Dramatiq `run_evolution_job` actor, and backfills the archive with freshly evaluated commits.
- **Construction**: `EvolutionScheduler(settings=None)` loads `loreley.config.Settings`, resolves the target repository root (preferring `SCHEDULER_REPO_ROOT` and falling back to `WORKER_REPO_WORKTREE`), initialises a `git` repository handle, derives a `Repository`/`Experiment` pair via `loreley.core.experiments.get_or_create_experiment()`, wires `MapElitesManager` (scoped to that `experiment_id`) plus `MapElitesSampler` with the same settings, and, when `MAPELITES_EXPERIMENT_ROOT_COMMIT` is set, delegates root-commit registration and baseline evaluation to `loreley.scheduler.ingestion.MapElitesIngestion`.
- **Lifecycle**:
  1. `tick()` runs the ingest → dispatch → measure → seed → schedule pipeline and logs a concise summary for observability. Each stage is isolated so failures are logged and do not crash the loop.
  2. `run_forever()` installs `SIGINT`/`SIGTERM` handlers, runs `tick()` at the configured poll interval, and keeps looping until interrupted.
  3. `--once` CLI flag runs a single tick and exits, useful for cron jobs or tests.
- **Job scheduling & dispatching**: the scheduler delegates all capacity calculations, MAP-Elites sampling, and Dramatiq job submission to `loreley.scheduler.job_scheduler.JobScheduler`, which:
  - counts unfinished jobs in the database,
  - enforces `SCHEDULER_MAX_UNFINISHED_JOBS` and the optional `SCHEDULER_MAX_TOTAL_JOBS` cap,
  - calls `MapElitesSampler.schedule_job(experiment_id=experiment.id)` to produce new work, and
  - marks rows as `QUEUED` and sends them to the `run_evolution_job` actor in priority order.
- **MAP-Elites maintenance**: ingestion of succeeded jobs is handled by `loreley.scheduler.ingestion.MapElitesIngestion`, which:
  - scans for `SUCCEEDED` jobs that have not yet been fully ingested,
  - reads `result_commit_hash` and ensures the corresponding git commit is available locally (fetching from remotes when needed),
  - loads metrics from the `metrics` table and calls `MapElitesManager.ingest(...)`, and
  - writes ingestion status back onto the job row (attempts, delta, placement/cell index, and any error messages).

## Configuration

The scheduler consumes the following `Settings` fields (all exposed as environment variables):

- `SCHEDULER_REPO_ROOT`: optional path to a read-only clone of the evolved repository; defaults to `WORKER_REPO_WORKTREE`.
- `SCHEDULER_POLL_INTERVAL_SECONDS`: delay between scheduler ticks (default: `30` seconds).
- `SCHEDULER_MAX_UNFINISHED_JOBS`: hard cap on the number of jobs that are not yet finished (`pending`, `queued`, `running`).
- `SCHEDULER_SCHEDULE_BATCH_SIZE`: maximum number of new jobs sampled from MAP-Elites per tick (bounded by the unused capacity).
- `SCHEDULER_DISPATCH_BATCH_SIZE`: number of pending jobs promoted to `QUEUED` and sent to Dramatiq per tick.
- `SCHEDULER_INGEST_BATCH_SIZE`: number of newly succeeded jobs ingested into MAP-Elites per tick.
- `MAPELITES_EXPERIMENT_ROOT_COMMIT`: optional git commit hash used as the logical root for the current experiment. When set, the scheduler ensures a `CommitCard` row exists for that commit and runs a one-off baseline evaluation to populate `Metric` rows, treating it as an experiment-wide baseline rather than inserting it into any MAP-Elites archive. During cold-start, when the archive is empty and no jobs exist yet, the scheduler first generates up to `MAPELITES_SEED_POPULATION_SIZE` seed evolution jobs from this root commit to form the initial population before switching to regular MAP-Elites sampling.

## CLI usage

```bash
uv run python -m loreley.scheduler.main        # continuous loop
uv run python -m loreley.scheduler.main --once # single tick (cron / smoke tests)
```

For details about the dedicated CLI wrapper script (including logging setup and
recommended usage), see `docs/script/run_scheduler.md`.

Running the module imports `loreley.tasks.workers`, so the Dramatiq broker is configured before the first dispatch. Rich console output summarises each tick, while Loguru records detailed diagnostics for ingestion, scheduling, and dispatching via the dedicated `job_scheduler` and `ingestion` helper classes. This makes the scheduler easy to supervise either interactively or under a process manager.

For more detailed information about these helper modules, see:

- `loreley.scheduler.job_scheduler.JobScheduler` — job production and dispatch pipeline.
- `loreley.scheduler.ingestion.MapElitesIngestion` — result ingestion, root-commit initialisation, and MAP-Elites maintenance.


