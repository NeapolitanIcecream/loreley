# loreley.core.worker.evolution

Autonomous evolution worker that orchestrates planning, coding, evaluation, repository management, and persistence for a single evolution job.

## Domain types

- **`JobContext`**: in-memory representation of a locked evolution job containing:
  - `job_id`, `run_token`, `base_commit_hash`, optional `island_id`.
  - `inspiration_commit_hashes` (bounded list) used to load lightweight commit context.
  - size-bounded job spec fields: `goal`, `constraints`, `acceptance_criteria`, optional `iteration_hint`, free-form `notes`, and `tags`.
  - seed-job and sampling provenance fields: `is_seed_job`, `sampling_strategy`, optional radius metadata, sampling ordinal/recipe identity, the explicit recipe-reuse flag, and optional fallback-inspiration counts.
- **`_JobLeaseHeartbeat`**: background helper that renews the active job lease while long-running worker stages execute. It periodically calls `EvolutionJobStore.renew_job_lease(...)` and records `JobLeaseLost` when the worker no longer owns the active attempt.
- **`EvolutionWorkerResult`**: structured success payload returned from `EvolutionWorker.run()`, combining the `job_id`, `base_commit_hash`, resulting `candidate_commit_hash`, the full `PlanningAgentResponse`, `CodingAgentResponse`, `EvaluationResult`, `CheckoutContext`, and the final `commit_message` used for the worker commit.

## Public worker API

- **`EvolutionWorker`**: service-layer entry point for running an evolution job synchronously end-to-end.
  - Constructor wires together dependencies, all of which may be overridden for tests or custom orchestration:
    - `WorkerRepository` for git operations.
    - `PlanningAgent` / `CodingAgent` for backend-driven planning and coding (Kilocode by default; pluggable backends).
    - `Evaluator` for running evaluation plugins.
    - `EvolutionJobStore` for DB persistence of job status and results.
  - **`run(job_id)`**:
    - Coerces the `job_id` into a `UUID`.
    - Calls `_start_job()` to lock and validate the job row, building a `JobContext`.
    - Starts `_JobLeaseHeartbeat` so planning, coding, evaluation, and persistence renew the lease in the background.
    - Creates an isolated per-job git worktree via `WorkerRepository.checkout_lease_for_job()`.
    - Runs planning (`_run_planning()`), coding (`_run_coding()`), and evaluation (`_run_evaluation()`) in sequence.
    - Calls `heartbeat.raise_if_lease_lost()` between long-running stages so a reclaimed attempt stops before it can overwrite a newer worker run.
    - Prepares a commit message via `_prepare_commit_message()`, creates a local commit via `_create_commit()`, records candidate metadata through `EvolutionJobStore.record_candidate_commit(...)`, then publishes the branch.
    - Persists success artifacts and metrics through `EvolutionJobStore.persist_success()` and prunes stale job branches.
    - Returns an `EvolutionWorkerResult` when everything succeeds.
    - On failure, records the error via `_mark_job_failed()` and re-raises, or directly propagates job lock, precondition, and lease-lost errors.

## Orchestration helpers

- **`_start_job(job_id)`**: uses `EvolutionJobStore.start_job()` to lock the job row, validates its status, and constructs a `JobContext` by:
  - Reading the size-bounded job spec fields directly from the `EvolutionJob` row (no catch-all payload parsing).
  - Carrying forward the active `run_token` so later writes can be fenced to the same worker attempt.
  - Falling back to `Settings.worker_evolution_global_goal` only when the per-job `goal` is missing.
- **`_run_planning(job_ctx, checkout)`**: batch-loads planning context for base + inspirations, builds a `PlanningAgentRequest` from those commit snapshots plus the global goal and iteration context, invokes `PlanningAgent.plan()`, and wraps `PlanningError` into `EvolutionWorkerError`. Seed-job handling happens earlier in `_build_prompt_context()`: it strips historical metrics and evaluation details from the base context, drops inspirations entirely, and carries the seed-job state through `IterationContext.seed_job` and related facts.
- **`_run_coding(job_ctx, plan, checkout)`**: builds a `CodingAgentRequest` from the plan, base commit, prompt context (`base`, `inspirations`, `iteration_context`), and job notes, runs `CodingAgent.implement()`, and wraps `CodingError` into `EvolutionWorkerError`.
- **`_prepare_commit_message(job_ctx, plan, coding)`**: deterministically reuses the coding report `summary`, then the plan `summary`, then `"Evolution job <id>"`. It does not make another model request or truncate the git message.
- **`_create_commit(checkout, commit_message)`**: ensures the checkout is on a branch and that the job worktree contains changes, stages everything, and creates the local commit hash that will later be published.
- **`_evaluate_or_reuse(...)`**: computes the candidate's exact Git tree identity and reuses an earlier passed result only when the tree, evaluator name/version, and campaign program all match. Reused metrics retain provenance; evaluator artifacts are not copied to a different commit.
- **`_publish_candidate_commit(checkout)`**: pushes the per-job branch using `force-with-lease` after candidate metadata has already been written to the job row.
- **`_run_evaluation(job_ctx, checkout, plan, candidate_commit)`**: constructs an `EvaluationContext` payload that includes only bounded job and plan fields (no raw prompts/JSON dumps), then calls `Evaluator.evaluate()` and wraps `EvaluationError` into `EvolutionWorkerError`.
- **`_prune_job_branches()`**: calls `WorkerRepository.prune_stale_job_branches()` and logs the number of branches removed, swallowing repository errors into warnings.
- **`_mark_job_failed(job_id, exc)`**: logs a red failure message and forwards the concise error text to `EvolutionJobStore.mark_job_failed()`, ensuring job rows still capture failures even when other parts of the worker raise.

## Data extraction and normalisation

- **`_load_commit_planning_contexts(commit_hashes)`**: fetches `CommitCard`,
  `Metric`, and agent-visible evaluation-artifact rows in batch for the base and
  all inspirations, then rebuilds bounded `CommitPlanningContext` instances in
  input order. Planning does not query or expose archive-internal scalar/cell
  fields.
