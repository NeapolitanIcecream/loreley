# loreley.tasks.workers

Dramatiq task actors that drive the Loreley evolution worker.

## Evolution worker

- **`run_evolution_job(job_id: str) -> None`**  
  A Dramatiq actor that runs a single evolution job via `loreley.core.worker.evolution.EvolutionWorker`. The queue name, retry policy, and time limit are derived from the task-related settings in `loreley.config.Settings` (`TASKS_QUEUE_NAME`, `TASKS_WORKER_MAX_RETRIES`, and `TASKS_WORKER_TIME_LIMIT_SECONDS`). The time limit is configured in milliseconds at the actor level: positive `TASKS_WORKER_TIME_LIMIT_SECONDS` values set a hard wall-clock limit, while values `<= 0` disable the time limit (no hard cap).

  On execution, the actor:

  - Validates and normalises the `job_id` argument.
  - Logs a “job started” event to both the rich console and `loguru`.
  - Delegates execution to `EvolutionWorker.run(...)`.
  - Handles worker-specific exceptions with distinct behaviours:
    - `JobLockConflict`: logs that the job was skipped due to a lock conflict and returns without raising.
    - `JobPreconditionError`: logs a warning and skips the job without raising (treating it as a non-retriable business error).
    - `EvolutionWorkerError`: logs an error and re-raises so Dramatiq can apply its retry policy.
    - Any other unexpected exception: logs with a full stack trace and re-raises as a defensive fallback.
  - Logs a “job complete” event including the resulting candidate commit hash on success.

Importing `loreley.tasks.workers` also imports `loreley.tasks.broker`, which configures the global Dramatiq broker using the Redis settings in `loreley.config.Settings`.

For details about the dedicated worker CLI wrapper script (including how it
starts a single-process, single-threaded Dramatiq worker), see
`docs/script/run_worker.md`.


