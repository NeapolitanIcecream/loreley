# loreley.tasks.workers

Dramatiq task actor builders that drive the Loreley evolution worker.

## Evolution worker

- **`build_evolution_job_worker_actor(settings: Settings, experiment_id: UUID | str) -> dramatiq.Actor`**  
  Builds an **experiment-attached** Dramatiq actor that runs a single evolution job via
  `loreley.core.worker.evolution.EvolutionWorker`. The returned actor is bound to the
  per-experiment queue derived from `TASKS_QUEUE_NAME` (prefix) and the experiment UUID hex:
  `"{TASKS_QUEUE_NAME}.{experiment_id.hex}"`.

  The actor reuses a single `EvolutionWorker` instance for the lifetime of the worker process
  (no per-job config reloads / no dynamic rebuilding). On execution, it:

  - Validates and normalises the `job_id` argument.
  - Logs a “job started” event to both the rich console and `loguru`.
  - Delegates execution to `EvolutionWorker.run(...)`.
  - Handles worker-specific exceptions with distinct behaviours:
    - `JobLockConflict`: logs that the job was skipped due to a lock conflict and returns without raising.
    - `JobPreconditionError`: logs a warning and skips the job without raising (treating it as a non-retriable business error).
    - `EvolutionWorkerError`: logs an error and re-raises so Dramatiq can apply its retry policy.
    - Any other unexpected exception: logs with a full stack trace and re-raises as a defensive fallback.
  - Logs a “job complete” event including the resulting candidate commit hash on success.

- **`build_evolution_job_sender_actor(settings: Settings, experiment_id: UUID | str) -> dramatiq.Actor`**  
  Builds a scheduler-side sender stub used only for enqueueing messages via `.send(...)`. The
  callable body is not expected to run in the scheduler process.

## Broker side effect

Importing `loreley.tasks.workers` imports `loreley.tasks.broker`, which configures the global
Dramatiq broker using the Redis settings in `loreley.config.Settings`.

For usage and operational details, see `docs/script/run_worker.md`.


