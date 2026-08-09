# loreley.scheduler.job_scheduler

Job production and dispatch logic extracted from the central evolution scheduler.

The `JobScheduler` class keeps all concerns related to **how many** jobs can be
scheduled, **which** jobs should be dispatched next, and **when** they are
submitted to the Dramatiq worker queue.

## JobScheduler

```python
from loreley.scheduler.job_scheduler import JobScheduler
```

- **Purpose**: encapsulate database interaction and Dramatiq calls for
  scheduling and dispatching evolution jobs, so that the main
  `EvolutionScheduler` can focus on orchestration.
- **Construction**: created by `EvolutionScheduler` with:
  - a shared `Settings` instance,
  - the interactive `rich` console,
  - the `MapElitesSampler`.

### Measuring unfinished work

- **`count_unfinished_jobs()`**:
  - Counts all jobs whose status is one of `PENDING`, `QUEUED`, or `RUNNING`.
  - Used by the main scheduler loop to decide how much new work (if any) can
    safely be created this tick.

- **`count_total_jobs()`**:
  - Counts all `EvolutionJob` rows in the database.
  - Used by `EvolutionScheduler` to enforce `SCHEDULER_MAX_TOTAL_JOBS`
    without rescanning unrelated state.

### Reclaiming stale running jobs

- **`reclaim_stale_running_jobs(now=None) -> JobLeaseReclaimResult`**:
  - Selects up to `SCHEDULER_STALE_RUNNING_RECLAIM_BATCH_SIZE` jobs with
    status `RUNNING` whose lease is stale or malformed.
  - Treats a row as reclaimable when `run_token`, `worker_id`, or
    `lease_expires_at` is missing, or when `lease_expires_at < now`.
  - Uses `SELECT ... FOR UPDATE SKIP LOCKED` so one scheduler tick does not
    block on another transaction touching the same row.
  - Clears the active lease fields, increments `recovery_count`, and records a
    concise `last_error` message describing whether the reclaim was caused by a
    missing lease or a missed heartbeat.
  - Requeues the row as `PENDING` when the job is still within the configured
    recovery budget.
  - Marks the row `FAILED` once `recovery_count` exceeds
    `SCHEDULER_STALE_RUNNING_MAX_RECOVERY_ATTEMPTS`.
  - Returns a `JobLeaseReclaimResult` with separate `requeued` and `failed`
    counters for tick-level observability.

### Scheduling new jobs

- **`schedule_jobs(unfinished_jobs: int, *, total_jobs: int) -> int`**:
  - Enforces `SCHEDULER_MAX_UNFINISHED_JOBS` as an upper bound across
    `PENDING`/`QUEUED`/`RUNNING` jobs.
  - Enforces the required `SCHEDULER_MAX_TOTAL_JOBS` global cap using the
    current database job count supplied by `EvolutionScheduler`.
  - Requests new work from MAP-Elites via `MapElitesSampler.schedule_job()`.
  - Uses the durable per-island job count as the next sampling ordinal and loads
    the most recent `MAPELITES_SAMPLER_RECIPE_COOLDOWN_JOBS` recipe hashes. Newly
    scheduled recipes join the same exclusion set immediately within the tick.
  - Enqueues job messages first and then marks the successfully sent jobs as
    `QUEUED` using the private `_enqueue_jobs(...)` helper.
  - Returns the number of jobs scheduled during this tick.

- **`create_seed_jobs(base_commit_hash, count, island_id=None) -> int`**:
  - Creates cold-start seed jobs from the root commit while the archive is still
    warming up.
  - Requires a non-empty global optimization objective. `Settings` provides a
    default, and operators can override it via `WORKER_EVOLUTION_GLOBAL_GOAL`.
  - Uses the same send-first queueing flow as regular jobs.

If the sampler indicates that no archive cell currently wants new work, the
console logs a short `[yellow]Sampler returned no job[/]` message and no rows
are touched in the database.

### Dispatching pending jobs

- **`dispatch_pending_jobs() -> int`**:
  - Selects up to `SCHEDULER_DISPATCH_BATCH_SIZE` jobs with status `PENDING`
    plus stale `QUEUED` jobs older than `WORKER_JOB_LEASE_TTL_SECONDS`,
    ordered by:
    1. `priority` (descending),
    2. `scheduled_at` (ascending),
    3. `created_at` (ascending),
    so that higher-priority and older jobs drain first.
  - Uses a `SELECT ... FOR UPDATE` window to safely select eligible jobs.
  - Redispatching stale `QUEUED` rows repairs broker-loss or restart cases where
    a row was marked queued but no worker ever started it.
  - Sends each selected job id to the Dramatiq `run_evolution_job` actor, then
    marks successfully submitted `PENDING` rows as `QUEUED` and stamps `scheduled_at`.
  - Returns the number of jobs successfully dispatched this tick.

Any failures while enqueuing individual jobs are logged with Loguru and
surfaced on the Rich console, but do not prevent other jobs from being
dispatched.

### Identity endpoint drain

- **`cancel_pending_for_identity_endpoint()`** locks and cancels only
  `PENDING` rows after shared campaign progress reaches
  `SCHEDULER_MAX_UNIQUE_EVALUATION_IDENTITIES`.
- It does not cancel `QUEUED` or `RUNNING` work and does not alter terminal
  evidence. The main scheduler waits for those jobs and pending ingestion to
  drain.
- `SCHEDULER_MAX_TOTAL_JOBS` remains an independent physical safety ceiling.

## Interaction with EvolutionScheduler

`EvolutionScheduler.tick()` coordinates with `JobScheduler` as follows:

1. `reclaim_stale_running_jobs()` to repair stale or malformed `RUNNING` rows.
2. Check the identity endpoint before dispatch or scheduling; cancel only
   `PENDING` rows and drain when reached.
3. `promote_staged_jobs()` to admit imported manual seeds up to the remaining
   unfinished capacity and physical endpoint.
4. `dispatch_pending_jobs()` to move already-`PENDING` jobs into the worker queue.
5. `count_unfinished_jobs()` to measure current load after reclaim + dispatch.
6. `create_seed_jobs(...)` when the archive is still warming up and capacity remains.
7. `schedule_jobs(...)` to request new work from MAP-Elites, honouring both
   capacity and the cached global job limit maintained by `EvolutionScheduler`.

`count_total_jobs()` is called during scheduler bootstrap to initialise that
cached total-job counter, which `EvolutionScheduler` then adjusts as it creates
seed and regular jobs.

This separation keeps the scheduler loop simple and makes it easier to test
and evolve the job pipeline independently of the rest of the orchestration
logic.
