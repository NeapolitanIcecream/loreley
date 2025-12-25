# loreley.core.worker.job_store

Persistence adapter for the evolution worker, responsible for locking jobs, storing results, and recording job failures in the database.

## Domain types and errors

- **`EvolutionWorkerError`**: base runtime error used when the worker cannot complete or persist a job due to configuration, database, or repository issues.
- **`JobLockConflict`**: raised when `start_job()` fails to obtain a NOWAIT lock on a job row, indicating that another worker is already processing the same job.
- **`JobPreconditionError`**: raised when a job cannot start because preconditions are not satisfied (missing row, unsupported status, missing `base_commit_hash`, etc.).
- **`LockedJob`**: dataclass snapshot of the locked `EvolutionJob` row containing the `job_id`, `base_commit_hash`, optional `island_id`, optional `experiment_id` and `repository_id`, the bounded job spec fields, and the tuple of `inspiration_commit_hashes`. This is used by `EvolutionWorker` to build its `JobContext`.

## Artifacts

Large, audit/debug oriented payloads (prompts, raw outputs, logs) are written to disk and referenced from the database via `JobArtifacts` rather than being embedded in primary rows.

## EvolutionJobStore

- **`EvolutionJobStore`**: database-facing adapter that encapsulates the lifecycle of an evolution job.
  - Constructed with `Settings` to attach worker/application metadata when persisting results.
  - Uses `session_scope()` and the ORM models from `loreley.db.models` (`EvolutionJob`, `CommitCard`, `JobArtifacts`, `Metric`, `JobStatus`) to modify rows transactionally.

### Job lifecycle methods

- **`start_job(job_id)`**:
  - Acquires a row-level lock on the `EvolutionJob` using `SELECT ... FOR UPDATE NOWAIT`.
  - Validates that the job exists, that `base_commit_hash` is present, and that the current `status` is in `{PENDING, QUEUED}`.
  - Marks the job as `RUNNING`, records `started_at`, clears any `last_error`, and returns a `LockedJob` snapshot.
  - Wraps SQL errors into `JobLockConflict` when they indicate a lock-not-available condition, or `EvolutionWorkerError` otherwise.

- **`persist_success(job_ctx, plan, coding, evaluation, commit_hash, commit_message)`**:
  - Updates the `EvolutionJob` row to `SUCCEEDED`, sets `completed_at`, stores `plan_summary`, sets `result_commit_hash`, clears `last_error`, and resets ingestion tracking fields.
  - Inserts a new `CommitCard` row representing the produced commit, with bounded `subject`, `change_summary`, `key_files`, `highlights`, and optional `evaluation_summary`.
  - Inserts one `Metric` row per evaluation metric for the new commit, copying numeric `value`, `unit`, `higher_is_better`, and any structured `details`.
  - Writes planning/coding/evaluation artifacts to disk and upserts a `JobArtifacts` row containing the corresponding filesystem paths.
  - Wraps SQLAlchemy errors into `EvolutionWorkerError` so the caller can surface persistence failures cleanly.

- **`mark_job_failed(job_id, message)`**:
  - Best-effort helper that records a failure reason on an `EvolutionJob` row.
  - If the job no longer exists or has already reached `SUCCEEDED` or `CANCELLED`, the call becomes a no-op.
  - Otherwise sets `status` to `FAILED`, stamps `completed_at`, and stores the latest `last_error` message.
  - Swallows and logs any SQL errors rather than propagating them, to avoid masking the original worker exception.

## Lock conflict detection

- **`_is_lock_conflict(exc)`**: inspects the original DB error to determine whether it represents a NOWAIT lock conflict.
  - For PostgreSQL, checks for error code `"55P03"` (lock_not_available).
  - Falls back to substring checks on the exception message for phrases like `"could not obtain lock"` or `"database is locked"`, covering other backends.

## Time helpers

- **`_utc_now()`**: returns the current UTC `datetime` and is used consistently when stamping `started_at`, `completed_at`, and worker metadata timestamps.


