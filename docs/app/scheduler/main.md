# app.scheduler.main

Central orchestration loop that keeps the Loreley evolution pipeline moving by coupling the MAP-Elites archive, the PostgreSQL job store, and the Dramatiq worker queue.

## EvolutionScheduler

- **Purpose**: continuously monitors unfinished jobs (`pending`, `queued`, `running`), schedules new work from the MAP-Elites archive when capacity allows, dispatches pending jobs to the Dramatiq `run_evolution_job` actor, and backfills the archive with freshly evaluated commits.
- **Construction**: `EvolutionScheduler(settings=None)` loads `app.config.Settings`, resolves the target repository root (preferring `SCHEDULER_REPO_ROOT` and falling back to `WORKER_REPO_WORKTREE`), initialises a `git` repository handle, and wires `MapElitesManager` + `MapElitesSampler` with the same settings.
- **Lifecycle**:
  1. `tick()` runs the ingest → dispatch → measure → schedule pipeline and logs a concise summary for observability. Each stage is isolated so failures are logged and do not crash the loop.
  2. `run_forever()` installs `SIGINT`/`SIGTERM` handlers, runs `tick()` at the configured poll interval, and keeps looping until interrupted.
  3. `--once` CLI flag runs a single tick and exits, useful for cron jobs or tests.
- **Job scheduling**: enforces `SCHEDULER_MAX_UNFINISHED_JOBS` as an upper bound. When capacity exists it calls `MapElitesSampler.schedule_job()`, immediately flips new rows to `QUEUED`, and pushes them to Dramatiq.
- **Dispatching**: batches of pending jobs (ordered by priority, then schedule time) are sent to Dramatiq according to `SCHEDULER_DISPATCH_BATCH_SIZE`, ensuring legacy jobs drain before creating more.
- **MAP-Elites maintenance**: after jobs succeed, the scheduler gathers their resulting commit hash, fetches the diff from git, and calls `MapElitesManager.ingest(...)`. Ingestion results (status, delta, placement) are stored back into each job's JSON payload under `payload["ingestion"]["map_elites"]` for auditability and retry tracking.

## Configuration

The scheduler consumes the following `Settings` fields (all exposed as environment variables):

- `SCHEDULER_REPO_ROOT`: optional path to a read-only clone of the evolved repository; defaults to `WORKER_REPO_WORKTREE`.
- `SCHEDULER_POLL_INTERVAL_SECONDS`: delay between scheduler ticks (default: `30` seconds).
- `SCHEDULER_MAX_UNFINISHED_JOBS`: hard cap on the number of jobs that are not yet finished (`pending`, `queued`, `running`).
- `SCHEDULER_SCHEDULE_BATCH_SIZE`: maximum number of new jobs sampled from MAP-Elites per tick (bounded by the unused capacity).
- `SCHEDULER_DISPATCH_BATCH_SIZE`: number of pending jobs promoted to `QUEUED` and sent to Dramatiq per tick.
- `SCHEDULER_INGEST_BATCH_SIZE`: number of newly succeeded jobs ingested into MAP-Elites per tick.

## CLI usage

```bash
uv run python -m app.scheduler.main        # continuous loop
uv run python -m app.scheduler.main --once # single tick (cron / smoke tests)
```

Running the module imports `app.tasks.workers`, so the Dramatiq broker is configured before the first dispatch. Rich console output summarises each tick, while Loguru records detailed diagnostics for ingestion, scheduling, and dispatching. This makes the scheduler easy to supervise either interactively or under a process manager. 


