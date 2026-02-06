# loreley.core.worker.coding

Execution engine for Loreley's autonomous worker, responsible for driving the Codex-based coding agent that applies a planning agent's plan to a real git worktree.

## Domain types

- **`CodingError`**: custom runtime error raised when the coding agent cannot successfully execute the plan (backend failures, bad working directory, timeouts, etc.).
- **`ExecutionReport`**: Markdown execution report emitted by the backend (`summary`, full `markdown`, optional `commit_message`).
- **`CodingAgentRequest`**: input payload given to the coding agent (`goal`, `plan` from `PlanDocument`, `base_commit`, optional `constraints`, `acceptance_criteria`, `iteration_hint`, and `additional_notes`); the `goal` is the same global evolution objective that the planning agent sees, resolved by the evolution worker from either explicit job row fields (for example `EvolutionJob.goal`) or `Settings.worker_evolution_global_goal`. All sequence fields are normalised to tuples in `__post_init__`.
- **`CodingAgentResponse`**: envelope returned from the agent combining the `report`, raw backend `raw_output`, rendered `prompt`, executed backend `command`, captured `stderr`, number of `attempts`, and total `duration_seconds`.

## Markdown contract

- Coding relies on `loreley.core.worker.agent` for shared backend abstractions (`AgentBackend`, `AgentTask`, `AgentInvocation`) and a shared retry loop (`run_agent_task()`).
- Backends return plain-text output (typically Markdown). The worker requests a simple Markdown structure (Summary / Changes / Tests) and extracts a short summary and an optional commit message on a best-effort basis.

## Coding agent

- **`CodingAgent`**: high-level orchestrator that turns a `CodingAgentRequest` and a `PlanDocument` into a sequence of edits via a configurable backend.
  - Instantiated with a `Settings` object and an optional `AgentBackend` implementation. When no backend is provided, it uses `CodexCliBackend` configured via `WORKER_CODING_CODEX_BIN`, `WORKER_CODING_CODEX_PROFILE`, `WORKER_CODING_MAX_ATTEMPTS`, `WORKER_CODING_TIMEOUT_SECONDS`, and `WORKER_CODING_EXTRA_ENV`. You can override the default by setting `WORKER_CODING_BACKEND` to a dotted Python path (`module:attr` or `module.attr`) that resolves to either an `AgentBackend` instance, a class implementing the `AgentBackend` protocol (constructed with no arguments), or a factory callable that returns such an instance.
  - **`implement(request, *, working_dir)`**: resolves the git worktree path, renders a concise prompt describing the goal, base commit, and plan Markdown, asks the backend to apply the changes, and wraps the output as an `ExecutionReport`.
  - Retries the backend invocation up to `max_attempts` times when process-level `CodingError` issues occur, logging warnings via `loguru` and showing concise progress output with `rich`.
  - Detects and treats a run that leaves the worktree unchanged as a failure, persisting debug artifacts and retrying until attempts are exhausted.
  - Merges any configured extra environment variables into the backend subprocess environment and enforces bounded prompt and log sizes via `_truncate`.

## Exceptions and helpers

- **`_extract_summary()`**, **`_extract_commit_message()`**, and **`_truncate()`**: utilities that keep prompts bounded and extract best-effort structured fields from Markdown output.
  Debug artifacts are written under `logs/<experiment_namespace>/worker/coding` (or `logs/worker/coding` when no experiment namespace is available).

