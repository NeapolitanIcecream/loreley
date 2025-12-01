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
  - the experiment-scoped `MapElitesSampler`,
  - the current `experiment_id`.

### Measuring unfinished work

- **`count_unfinished_jobs()`**:
  - Counts all jobs whose status is one of `PENDING`, `QUEUED`, or `RUNNING`.
  - Used by the main scheduler loop to decide how much new work (if any) can
    safely be created this tick.

### Scheduling new jobs

- **`schedule_jobs(unfinished_jobs: int, *, total_scheduled_jobs: int) -> int`**:
  - Enforces `SCHEDULER_MAX_UNFINISHED_JOBS` as an upper bound across
    `PENDING`/`QUEUED`/`RUNNING` jobs.
  - Respects the optional `SCHEDULER_MAX_TOTAL_JOBS` global cap using the
    `total_scheduled_jobs` counter maintained by `EvolutionScheduler`.
  - Requests new work from MAP-Elites via
    `MapElitesSampler.schedule_job(experiment_id=experiment_id)`.
  - Immediately transitions any newly created jobs to `QUEUED` and pushes them
    to Dramatiq using the private `_enqueue_jobs(...)` helper.
  - Returns the number of jobs scheduled during this tick.

If the sampler indicates that no archive cell currently wants new work, the
console logs a short `[yellow]Sampler returned no job[/]` message and no rows
are touched in the database.

### Dispatching pending jobs

- **`dispatch_pending_jobs() -> int`**:
  - Selects up to `SCHEDULER_DISPATCH_BATCH_SIZE` jobs with status `PENDING`,
    ordered by:
    1. `priority` (descending),
    2. `scheduled_at` (ascending),
    3. `created_at` (ascending),
    so that higher-priority and older jobs drain first.
  - Uses a `SELECT ... FOR UPDATE` window to safely promote eligible jobs to
    `QUEUED` and stamp their `scheduled_at` time.
  - Sends each queued job id to the Dramatiq `run_evolution_job` actor.
  - Returns the number of jobs successfully dispatched this tick.

Any failures while enqueuing individual jobs are logged with Loguru and
surfaced on the Rich console, but do not prevent other jobs from being
dispatched.

## Interaction with EvolutionScheduler

`EvolutionScheduler.tick()` calls into `JobScheduler` as follows:

1. `count_unfinished_jobs()` to measure current load.
2. `schedule_jobs(...)` to request new work from MAP-Elites, honouring both
   capacity and global job limits.
3. `dispatch_pending_jobs()` to move ready jobs into the worker queue.

This separation keeps the scheduler loop simple and makes it easier to test
and evolve the job pipeline independently of the rest of the orchestration
logic.


