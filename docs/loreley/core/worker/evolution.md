# loreley.core.worker.evolution

Autonomous evolution worker that orchestrates planning, coding, evaluation, repository management, and persistence for a single evolution job.

## Domain types

- **`JobContext`**: in-memory representation of a locked evolution job containing:
  - `job_id`, `base_commit_hash`, optional `island_id`, optional `experiment_id` and `repository_id`.
  - `inspiration_commit_hashes` (bounded list) used to load lightweight commit context.
  - size-bounded job spec fields: `goal`, `constraints`, `acceptance_criteria`, optional `iteration_hint`, free-form `notes`, and `tags`.
  - a boolean `is_seed_job` flag indicating whether the job is a cold-start seed job.
- **`EvolutionWorkerResult`**: structured success payload returned from `EvolutionWorker.run()`, combining the `job_id`, `base_commit_hash`, resulting `candidate_commit_hash`, the full `PlanningAgentResponse`, `CodingAgentResponse`, `EvaluationResult`, `CheckoutContext`, and the final `commit_message` used for the worker commit.

## Public worker API

- **`EvolutionWorker`**: service-layer entry point for running an evolution job synchronously end-to-end.
  - Constructor wires together dependencies, all of which may be overridden for tests or custom orchestration:
    - `WorkerRepository` for git operations.
    - `PlanningAgent` / `CodingAgent` for Codex-powered planning and coding.
    - `Evaluator` for running evaluation plugins.
    - `CommitSummarizer` for generating concise commit messages.
    - `EvolutionJobStore` for DB persistence of job status and results.
  - **`run(job_id)`**:
    - Coerces the `job_id` into a `UUID`.
    - Calls `_start_job()` to lock and validate the job row, building a `JobContext`.
    - Checks out the base commit via `WorkerRepository.checkout_for_job()`.
    - Runs planning (`_run_planning()`), coding (`_run_coding()`), and evaluation (`_run_evaluation()`) in sequence.
    - Prepares a commit message via `_prepare_commit_message()`, then creates and pushes a new commit via `_create_commit()`.
    - Persists success artifacts and metrics through `EvolutionJobStore.persist_success()` and prunes stale job branches.
    - Returns an `EvolutionWorkerResult` when everything succeeds.
    - On failure, records the error via `_mark_job_failed()` and re-raises, or directly propagates job lock/precondition errors.

## Orchestration helpers

- **`_start_job(job_id)`**: uses `EvolutionJobStore.start_job()` to lock the job row, validates its status, and constructs a `JobContext` by:
  - Reading the size-bounded job spec fields directly from the `EvolutionJob` row (no catch-all payload parsing).
  - Falling back to `Settings.worker_evolution_global_goal` only when the per-job `goal` is missing.
- **`_run_planning(job_ctx, checkout)`**: builds a `PlanningAgentRequest` from commit snapshots and job fields, invokes `PlanningAgent.plan()`, and wraps `PlanningError` into `EvolutionWorkerError`. For seed jobs, `_run_planning` clears metrics, highlights, and evaluation details from the base planning context, drops all inspirations, and passes `cold_start=True` so that the planning agent treats the request as a cold-start seed population design run.
- **`_run_coding(job_ctx, plan, checkout)`**: builds a `CodingAgentRequest` from the plan and job context, runs `CodingAgent.implement()`, and wraps `CodingError` into `EvolutionWorkerError`.
- **`_prepare_commit_message(job_ctx, plan, coding)`**: delegates to `CommitSummarizer.generate()` to generate an LLM-backed git subject line; if summarisation fails, falls back to the coding agent's suggested `commit_message`, plan `summary`, or a generic `"Evolution job <id>"` string.
- **`_create_commit(checkout, commit_message)`**: ensures the checkout is on a branch and that the repository contains changes, stages everything, creates a commit, and pushes the per-job branch using `force-with-lease`.
- **`_run_evaluation(job_ctx, checkout, plan, candidate_commit)`**: constructs an `EvaluationContext` payload that includes only bounded job and plan fields (no raw prompts/JSON dumps), then calls `Evaluator.evaluate()` and wraps `EvaluationError` into `EvolutionWorkerError`.
- **`_prune_job_branches()`**: calls `WorkerRepository.prune_stale_job_branches()` and logs the number of branches removed, swallowing repository errors into warnings.
- **`_mark_job_failed(job_id, exc)`**: logs a red failure message and forwards the concise error text to `EvolutionJobStore.mark_job_failed()`, ensuring job rows still capture failures even when other parts of the worker raise.

## Data extraction and normalisation

- **`_load_commit_planning_context(commit_hash, ...)`**: pulls `CommitCard` and `Metric` rows for a given commit hash via `session_scope()` and returns a bounded `CommitPlanningContext` with `subject`, `change_summary`, `key_files`, `highlights`, `evaluation_summary`, and a small set of metrics.


