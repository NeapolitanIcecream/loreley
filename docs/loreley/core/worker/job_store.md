# loreley.core.worker.job_store

Persistence adapter for the evolution worker, responsible for locking jobs, storing results, and recording job failures in the database.

## Domain types and errors

- **`EvolutionWorkerError`**: base runtime error used when the worker cannot complete or persist a job due to configuration, database, or repository issues.
- **`JobLockConflict`**: raised when `start_job()` fails to obtain a NOWAIT lock on a job row, indicating that another worker is already processing the same job.
- **`JobPreconditionError`**: raised when a job cannot start because preconditions are not satisfied (for example, a missing row or an unsupported status).
- **`LockedJob`**: dataclass snapshot of the locked `EvolutionJob` row containing the `job_id`, `base_commit_hash`, optional `island_id`, the bounded job spec fields, and the tuple of `inspiration_commit_hashes`. This is used by `EvolutionWorker` to build its `JobContext`.

## Artifacts

Large, audit/debug oriented payloads (prompts, raw outputs, logs) are written to disk and referenced from the database via `JobArtifacts` rather than being embedded in primary rows.

## EvolutionJobStore

- **`EvolutionJobStore`**: database-facing adapter that encapsulates the lifecycle of an evolution job.
  - Constructed with `Settings` to attach worker/application metadata when persisting results.
  - Uses `session_scope()` and the ORM models from `loreley.db.models` (`EvolutionJob`, `CommitCard`, `JobArtifacts`, `EvaluationArtifactRecord`, `Metric`, `CandidateCommit`, `EvaluationAttempt`, `DiagnosticCapsule`, `JobStatus`) to modify rows transactionally.

### Job lifecycle methods

- **`start_job(job_id)`**:
  - Acquires a row-level lock on the `EvolutionJob` using `SELECT ... FOR UPDATE NOWAIT`.
  - Validates that the job exists, that `base_commit_hash` is present, and that the current `status` is in `{PENDING, QUEUED}`.
  - Marks the job as `RUNNING`, records `started_at`, initial `heartbeat_at`, `lease_expires_at`, generates a fresh `run_token`, records a bounded `worker_id`, clears any `last_error`, resets stale candidate metadata from prior attempts, and returns a `LockedJob` snapshot.
  - Raises `EvolutionWorkerError` when `base_commit_hash` is missing, and wraps SQL errors into `JobLockConflict` when they indicate a lock-not-available condition, or `EvolutionWorkerError` otherwise.

- **`record_candidate_commit(CandidateCommitRecord(...))`**:
  - Stores the latest candidate commit hash and branch name on the `EvolutionJob` row before or after remote publication.
  - Accepts an optional `run_token`; when provided, the write is fenced to the active worker attempt and fails with `JobLeaseLost` if another process already reclaimed the job.
  - When `published=False`, records the candidate pointer while leaving `candidate_published_at` unset.
  - When `published=True`, stamps `candidate_published_at` so post-push failures still leave a durable recovery pointer.
  - Stores the exact Git `source_tree_hash` used for evaluation-reuse lookup.
  - For a supplied manual seed, accepts the detached checkout with no worker
    branch, records publication state as `available`, and rejects a request to
    fabricate a worker publication.
  - Raises `EvolutionWorkerError` if the job disappears, is already terminal, or a database error prevents recording the candidate metadata.

- **`find_reusable_evaluation(...)`**:
  - Finds a passed candidate with the same exact source tree, evaluator name/version, and campaign program.
  - Copies its metric values and candidate identity into a new passed outcome while recording the source evaluation attempt. It does not copy path-backed evaluator artifacts.
  - This shortcut applies to legacy one-shot evaluators. Phased evaluators run
    `prepare` first and use the accepted-measurement cache instead.

- **`record_evaluation_observation(...)`**:
  - Creates a durable `EvaluationAttempt` for every evaluator observation,
    including intermediate rework, exact-tree reuse, and measurement reuse.
  - Records `measurement_executed` separately from `measurement_reused`, links
    reuse to its source attempt and accepted measurement, and persists evaluator
    capacity wait/slot telemetry. Each attempt also receives a stable per-job
    ordinal, run token, and its own fixed-artifact links.

- **`renew_job_lease(job_id, run_token)`**:
  - Extends `heartbeat_at` and `lease_expires_at` for the active `RUNNING` row owned by `run_token`.
  - Raises `JobLeaseLost` when the row is no longer `RUNNING` or the `run_token` no longer matches, which fences stale workers after scheduler reclaim.

- **`persist_success(job_ctx, plan, coding, evaluation, commit_hash, commit_message)`**:
  - Locks the active job row with the same `run_token` used at start-up so stale attempts cannot persist over a newer run.
  - Updates the `EvolutionJob` row to `SUCCEEDED`, sets `completed_at`, stores `plan_summary`, sets `result_commit_hash`, clears `last_error`, clears active lease fields, and resets ingestion tracking fields.
  - Preserves the already-recorded candidate metadata so successful jobs still retain an auditable publication pointer.
  - Inserts a new `CommitCard` row representing the produced commit, with bounded `subject`, `change_summary`, `key_files`, `highlights`, and optional `evaluation_summary`.
  - Inserts one `Metric` row per evaluation metric for the new commit, copying numeric `value`, `unit`, `higher_is_better`, and any structured `details`.
  - Writes planning/coding/evaluation artifacts to disk under a per-`job_id` and per-`run_token` directory, then inserts a `JobArtifacts` row containing the corresponding filesystem paths when artifact writing succeeds.
  - Inserts `EvaluationArtifactRecord` rows for accepted evaluator-declared
    artifacts and links them to the evaluation attempt. Attempt-linked rows are
    append-only, so a retry cannot erase an earlier attempt's evidence.
  - For a fresh cacheable phased pass, inserts the immutable accepted
    measurement, payload hash, evidence manifest, and source-attempt link in the
    same terminal transaction. The worker releases the per-measurement advisory
    lock only after this transaction commits. Evidence keys, sizes, digests,
    artifact records, and stored bytes are checked before acceptance and reuse.
  - Wraps SQLAlchemy errors into `EvolutionWorkerError` so the caller can surface persistence failures cleanly.

- **`mark_job_failed(job_id, message)`**:
  - Best-effort helper that records a failure reason on an `EvolutionJob` row.
  - Accepts an optional `run_token`; when provided, failure persistence becomes a no-op if the lease was already lost to a newer attempt.
  - If the job no longer exists or has already reached `SUCCEEDED` or `CANCELLED`, the call becomes a no-op.
  - Otherwise sets `status` to `FAILED`, stamps `completed_at`, clears active lease fields, stores the latest `last_error` message, and intentionally leaves any previously recorded candidate metadata intact for debugging/recovery.
  - Swallows and logs any SQL errors rather than propagating them, to avoid masking the original worker exception.

## Lock conflict detection

- **`_is_lock_conflict(exc)`**: inspects the original DB error to determine whether it represents a NOWAIT lock conflict.
  - For PostgreSQL, checks for error code `"55P03"` (lock_not_available).
  - Falls back to substring checks on the exception message for phrases like `"could not obtain lock"` or `"database is locked"`, covering other backends.

## Time helpers

- **`_utc_now()`**: returns the current UTC `datetime` and is used consistently when stamping `started_at`, `completed_at`, and worker metadata timestamps.
- **`_lease_ttl()`**: derives the lease duration from `WORKER_JOB_LEASE_TTL_SECONDS`.
