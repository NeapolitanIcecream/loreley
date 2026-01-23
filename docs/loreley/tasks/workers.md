# loreley.tasks.workers

Dramatiq task actor builders that drive the Loreley evolution worker.

## Evolution worker

- **`build_evolution_job_worker_actor(settings: Settings) -> dramatiq.Actor`**  
  Builds an **experiment-attached** Dramatiq actor that runs a single evolution job via
  `loreley.core.worker.evolution.EvolutionWorker`. The returned actor is bound to the
  per-experiment queue derived from `EXPERIMENT_ID`:
  `"loreley.evolution.{experiment_namespace}"`.

  `EXPERIMENT_ID` can be a UUID or a short slug; slugs are mapped to stable UUIDs (uuid5) and
  yield a stable `experiment_namespace` for routing and filesystem naming.

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

- **`build_evolution_job_sender_actor(settings: Settings) -> dramatiq.Actor`**  
  Builds a scheduler-side sender stub used only for enqueueing messages via `.send(...)`. The
  callable body is not expected to run in the scheduler process.

## Broker configuration

Both actor builders call `setup_broker(settings)` to ensure Dramatiq is configured with the
experiment-scoped Redis namespace before any actors are registered. The broker is no longer
configured implicitly at import time.

For usage and operational details, see `docs/script/run_worker.md`.


