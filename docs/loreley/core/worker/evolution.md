# loreley.core.worker.evolution

Autonomous evolution worker that orchestrates planning, coding, evaluation, repository management, and persistence for a single evolution job.

## Domain types

- **`JobContext`**: in-memory representation of a locked evolution job containing:
  - `job_id`, `base_commit_hash`, optional `island_id`.
  - `inspiration_commit_hashes` (bounded list) used to load lightweight commit context.
  - size-bounded job spec fields: `goal`, `constraints`, `acceptance_criteria`, optional `iteration_hint`, free-form `notes`, and `tags`.
  - seed-job and sampling provenance fields: `is_seed_job`, `sampling_strategy`, optional radius metadata, and optional fallback-inspiration counts.
- **`EvolutionWorkerResult`**: structured success payload returned from `EvolutionWorker.run()`, combining the `job_id`, `base_commit_hash`, resulting `candidate_commit_hash`, the full `PlanningAgentResponse`, `CodingAgentResponse`, `EvaluationResult`, `CheckoutContext`, and the final `commit_message` used for the worker commit.

## Public worker API

- **`EvolutionWorker`**: service-layer entry point for running an evolution job synchronously end-to-end.
  - Constructor wires together dependencies, all of which may be overridden for tests or custom orchestration:
    - `WorkerRepository` for git operations.
    - `PlanningAgent` / `CodingAgent` for backend-driven planning and coding (Kilocode by default; pluggable backends).
    - `Evaluator` for running evaluation plugins.
    - `CommitSummarizer` for generating concise commit messages.
    - `EvolutionJobStore` for DB persistence of job status and results.
  - **`run(job_id)`**:
    - Coerces the `job_id` into a `UUID`.
    - Calls `_start_job()` to lock and validate the job row, building a `JobContext`.
    - Creates an isolated per-job git worktree via `WorkerRepository.checkout_lease_for_job()`.
    - Runs planning (`_run_planning()`), coding (`_run_coding()`), and evaluation (`_run_evaluation()`) in sequence.
    - Prepares a commit message via `_prepare_commit_message()`, creates a local commit via `_create_commit()`, records candidate metadata through `EvolutionJobStore.record_candidate_commit(...)`, then publishes the branch.
    - Persists success artifacts and metrics through `EvolutionJobStore.persist_success()` and prunes stale job branches.
    - Returns an `EvolutionWorkerResult` when everything succeeds.
    - On failure, records the error via `_mark_job_failed()` and re-raises, or directly propagates job lock/precondition errors.

## Orchestration helpers

- **`_start_job(job_id)`**: uses `EvolutionJobStore.start_job()` to lock the job row, validates its status, and constructs a `JobContext` by:
  - Reading the size-bounded job spec fields directly from the `EvolutionJob` row (no catch-all payload parsing).
  - Falling back to `Settings.worker_evolution_global_goal` only when the per-job `goal` is missing.
- **`_run_planning(job_ctx, checkout)`**: batch-loads planning context for base + inspirations, builds a `PlanningAgentRequest` from those commit snapshots plus the global goal and iteration context, invokes `PlanningAgent.plan()`, and wraps `PlanningError` into `EvolutionWorkerError`. Seed-job handling happens earlier in `_build_prompt_context()`: it strips historical metrics and evaluation details from the base context, drops inspirations entirely, and carries the seed-job state through `IterationContext.seed_job` and related facts.
- **`_run_coding(job_ctx, plan, checkout)`**: builds a `CodingAgentRequest` from the plan, base commit, prompt context (`base`, `inspirations`, `iteration_context`), and job notes, runs `CodingAgent.implement()`, and wraps `CodingError` into `EvolutionWorkerError`.
- **`_prepare_commit_message(job_ctx, plan, coding)`**: delegates to `CommitSummarizer.generate()` to generate an LLM-backed git subject line; if summarisation fails, falls back to the coding report `summary`, plan `summary`, or a generic `"Evolution job <id>"` string.
- **`_create_commit(checkout, commit_message)`**: ensures the checkout is on a branch and that the job worktree contains changes, stages everything, and creates the local commit hash that will later be published.
- **`_publish_candidate_commit(checkout)`**: pushes the per-job branch using `force-with-lease` after candidate metadata has already been written to the job row.
- **`_run_evaluation(job_ctx, checkout, plan, candidate_commit)`**: constructs an `EvaluationContext` payload that includes only bounded job and plan fields (no raw prompts/JSON dumps), then calls `Evaluator.evaluate()` and wraps `EvaluationError` into `EvolutionWorkerError`.
- **`_prune_job_branches()`**: calls `WorkerRepository.prune_stale_job_branches()` and logs the number of branches removed, swallowing repository errors into warnings.
- **`_mark_job_failed(job_id, exc)`**: logs a red failure message and forwards the concise error text to `EvolutionJobStore.mark_job_failed()`, ensuring job rows still capture failures even when other parts of the worker raise.

## Data extraction and normalisation

- **`_load_commit_planning_contexts(commit_hashes, ...)`**: fetches `CommitCard`, `Metric`, and optional `MapElitesArchiveCell` rows in batch for all requested commit hashes, then rebuilds bounded `CommitPlanningContext` instances in input order.
- **`_load_commit_planning_context(commit_hash, ...)`**: compatibility wrapper for single-commit call sites, delegating to the batch loader.
