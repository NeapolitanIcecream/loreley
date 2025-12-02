# loreley.core.worker.evolution

Autonomous evolution worker that orchestrates planning, coding, evaluation, repository management, and persistence for a single evolution job.

## Domain types

- **`CommitSnapshot`**: immutable snapshot of commit-related data used to build planning context (`commit_hash`, derived `summary`, optional `evaluation_summary`, a tuple of text `highlights`, a tuple of `CommitMetric` instances, and an `extra_context` dict that may include both DB and MAP-Elites metadata). Exposes `to_planning_context()` to convert into a `CommitPlanningContext`.
- **`JobContext`**: in-memory representation of a locked evolution job containing:
  - `job_id`, `base_commit_hash`, optional `island_id`, optional `experiment_id` and `repository_id`, and the raw job `payload`.
  - `base_snapshot` and `inspiration_snapshots` that wrap DB records and/or MAP-Elites payloads.
  - user-facing `goal`, `constraints`, `acceptance_criteria`, optional `iteration_hint`, free-form `notes`, and `tags`, all normalised to tuples of strings.
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
  - Loading commit metadata and metrics from the DB via `_load_commit_snapshot()`.
  - Merging optional MAP-Elites record payloads from the job `payload` into `extra_context`.
  - Deriving the job `goal`, `constraints`, `acceptance_criteria`, `iteration_hint`, `notes`, and `tags` from the payload and extra context using a set of coercion helpers; when no explicit goal is provided in the payload or `extra_context`, the worker falls back to the configured `Settings.worker_evolution_global_goal`.
- **`_run_planning(job_ctx, checkout)`**: builds a `PlanningAgentRequest` from commit snapshots and job fields, invokes `PlanningAgent.plan()`, and wraps `PlanningError` into `EvolutionWorkerError`.
- **`_run_coding(job_ctx, plan, checkout)`**: builds a `CodingAgentRequest` from the plan and job context, runs `CodingAgent.implement()`, and wraps `CodingError` into `EvolutionWorkerError`.
- **`_prepare_commit_message(job_ctx, plan, coding)`**: delegates to `CommitSummarizer.generate()` to generate an LLM-backed git subject line; if summarisation fails, falls back to the coding agent's suggested `commit_message`, plan `summary`, or a generic `"Evolution job <id>"` string.
- **`_create_commit(checkout, commit_message)`**: ensures the checkout is on a branch and that the repository contains changes, stages everything, creates a commit, and pushes the per-job branch using `force-with-lease`.
- **`_run_evaluation(job_ctx, checkout, plan, candidate_commit)`**: constructs an `EvaluationContext` payload that includes job metadata and a normalised plan payload (via `build_plan_payload()`), then calls `Evaluator.evaluate()` and wraps `EvaluationError` into `EvolutionWorkerError`.
- **`_prune_job_branches()`**: calls `WorkerRepository.prune_stale_job_branches()` and logs the number of branches removed, swallowing repository errors into warnings.
- **`_mark_job_failed(job_id, exc)`**: logs a red failure message and forwards the concise error text to `EvolutionJobStore.mark_job_failed()`, ensuring job rows still capture failures even when other parts of the worker raise.

## Data extraction and normalisation

- **`_load_commit_snapshot(commit_hash, fallback)`**: pulls `CommitMetadata` and `Metric` rows for a given commit hash via `session_scope()`, merges DB and fallback MAP-Elites data into a `CommitSnapshot`, and derives:
  - A human-readable `summary` built from commit message, fallback metadata, or a plain `"Commit <hash>"` string.
  - A set of `highlights` assembled from various `highlights`/`snippets`/`notes` fields in DB and payload metadata.
  - A list of `CommitMetric` values taken either from DB rows or fallback payload metrics.
- Additional helpers such as `_extract_goal()`, `_extract_iteration_hint()`, `_map_inspiration_payloads()`, `_extract_mapping()`, `_extract_highlights()`, `_first_non_empty()`, `_coerce_str_sequence()`, and `_coerce_uuid()` encapsulate common logic for turning loosely-structured job payloads into the strongly-typed structures that the planning and coding agents expect. In particular, `_extract_goal()` derives the goal from explicit job fields and the global configuration and does not use commit messages as a fallback.


