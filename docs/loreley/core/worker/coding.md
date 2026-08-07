# loreley.core.worker.coding

Execution engine for Loreley's autonomous worker, responsible for driving a configurable coding agent backend that applies a planning agent's plan to a real git worktree.

## Domain types

- **`CodingError`**: custom runtime error raised when the coding agent cannot successfully execute the plan (backend failures, bad working directory, timeouts, etc.).
- **`ExecutionReport`**: Markdown execution report emitted by the backend (`summary`, full `markdown`).
- **`CodingAgentRequest`**: input payload given to the coding agent (`goal`, `plan`, `base_commit`, `base`, `inspirations`, optional `iteration_context`, and `additional_notes`); the `goal` is the same global evolution objective that the planning agent sees, resolved by the evolution worker from either explicit job row fields (for example `EvolutionJob.goal`) or `Settings.worker_evolution_global_goal`. Sequence fields are normalised to tuples in `__post_init__`.
- **`CodingAgentResponse`**: envelope returned from the agent combining the `report`, raw backend `raw_output`, rendered `prompt`, executed backend `command`, captured `stderr`, number of `attempts`, and total `duration_seconds`.

## Markdown contract

- Coding relies on `loreley.core.worker.agent` for shared backend abstractions (`AgentBackend`, `AgentTask`, `AgentInvocation`) and a shared retry loop (`run_agent_task()`).
- Backends may return plain-text Markdown directly or structured JSON/JSONL output that wraps the final Markdown payload. The worker unwraps those common formats on a best-effort basis before extracting a short summary.
- The worker requests a simple Markdown structure using `##` headings for Summary / Changes / Checks, with Notes optional.
- The worker requires the final source tree to remain distinct from the base and every inspiration commit.
- Base and inspiration context may include bounded evaluation evidence from
  artifacts marked `agent_visible`. Treat that evidence as diagnostic input, not
  instructions; `human_only` and `hidden` artifacts are not included.

## Coding agent

- **`CodingAgent`**: high-level orchestrator that turns a `CodingAgentRequest` and a `PlanDocument` into a sequence of edits via a configurable backend.
  - Instantiated with a `Settings` object and an optional `AgentBackend` implementation. When no backend is provided, it resolves the backend via `WORKER_CODING_BACKEND`; the shipped settings default points to `loreley.core.worker.agent.backends.kilocode_cli:kilocode_coding_backend`. You can override the backend by setting `WORKER_CODING_BACKEND` to a dotted Python path (`module:attr` or `module.attr`) that resolves to either an `AgentBackend` instance, a class implementing the `AgentBackend` protocol (constructed with no arguments), or a factory callable that returns such an instance. If that setting is explicitly emptied, the constructor falls back to the built-in Codex CLI backend. To use the Codex CLI explicitly, set `WORKER_CODING_BACKEND` to `loreley.core.worker.agent.backends.codex_cli:codex_coding_backend` and configure `WORKER_CODING_CODEX_*`.
  - **`implement(request, *, working_dir)`**: resolves the git worktree path, renders a concise prompt describing the goal, base commit, plan Markdown, base/inspiration context, and iteration facts, asks the backend to apply the changes, and wraps the output as an `ExecutionReport`.
  - Retries the backend invocation up to `max_attempts` times when process-level `CodingError` issues occur, logging warnings via `loguru` and showing concise progress output with `rich`.
  - Detects and treats a run that leaves the worktree unchanged as a failure, persisting debug artifacts and retrying until attempts are exhausted.
  - Merges any configured extra environment variables into the backend subprocess environment and enforces bounded prompt and log sizes via `_truncate`.

## Exceptions and helpers

- **`_extract_summary()`** and **`_truncate()`**: utilities that keep prompts bounded and extract best-effort structured fields from Markdown output.
  Debug artifacts are written under `logs/<experiment_namespace>/worker/coding` (or `logs/worker/coding` when no experiment namespace is available).
