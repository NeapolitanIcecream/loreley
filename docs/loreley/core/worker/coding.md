# loreley.core.worker.coding

Execution engine for Loreley's autonomous worker, responsible for driving the Codex-based coding agent that applies a planning agent's plan to a real git worktree.

## Domain types

- **`CodingError`**: custom runtime error raised when the coding agent cannot successfully execute the plan (invalid schema, Codex failures, bad working directory, timeouts, etc.).
- **`StepExecutionStatus`**: string `Enum` describing how each plan step was handled (`"completed"`, `"partial"`, or `"skipped"`).
- **`CodingStepReport`**: dataclass capturing the outcome of a single step (`step_id`, `status`, human-readable `summary`, and optional `files` / `commands` touched by that step).
- **`CodingPlanExecution`**: aggregate result for the whole run, including the overall `implementation_summary`, optional `commit_message`, tuple of `step_results`, `tests_executed`, `tests_recommended`, `follow_up_items`, and `notes`.
- **`CodingAgentRequest`**: input payload given to the coding agent (`goal`, `plan` from `PlanningPlan`, `base_commit`, optional `constraints`, `acceptance_criteria`, `iteration_hint`, and `additional_notes`); the `goal` is the same global evolution objective that the planning agent sees, resolved by the evolution worker from either explicit job row fields (for example `EvolutionJob.goal`) or `Settings.worker_evolution_global_goal`. All sequence fields are normalised to tuples in `__post_init__`.
- **`CodingAgentResponse`**: envelope returned from the agent combining the structured `execution`, raw backend `raw_output`, rendered `prompt`, executed backend `command`, captured `stderr`, number of `attempts`, and total `duration_seconds`.

## JSON schema and validation

- **`CODING_OUTPUT_SCHEMA`**: JSON schema describing the expected coding agent output (top-level `implementation_summary`, optional `commit_message`, array of `step_results`, plus optional `tests_executed`, `tests_recommended`, `follow_up_items`, and `notes`), used to validate the backend response.
- **`_StepResultModel`** / **`_CodingOutputModel`**: internal frozen `pydantic` models that validate the JSON payload against `CODING_OUTPUT_SCHEMA` and provide a typed bridge into the domain dataclasses.
- **Agent backend**: coding relies on `loreley.core.worker.agent_backend` for shared backend abstractions (`AgentBackend`, `StructuredAgentTask`, `AgentInvocation`) and the default `CodexCliBackend` implementation used by the coding agent; see that module's documentation for backend configuration details.

## Coding agent

- **`CodingAgent`**: high-level orchestrator that turns a `CodingAgentRequest` and `PlanningPlan` into a sequence of edits via a configurable backend.
  - Instantiated with a `Settings` object and an optional `AgentBackend` implementation. When no backend is provided, it uses `CodexCliBackend` configured via `WORKER_CODING_CODEX_BIN`, `WORKER_CODING_CODEX_PROFILE`, `WORKER_CODING_MAX_ATTEMPTS`, `WORKER_CODING_TIMEOUT_SECONDS`, `WORKER_CODING_EXTRA_ENV`, `WORKER_CODING_SCHEMA_PATH`, and `WORKER_CODING_VALIDATION_MODE`. You can override the default by setting `WORKER_CODING_BACKEND` to a dotted Python path (`module:attr` or `module.attr`) that resolves to either an `AgentBackend` instance, a class implementing the `AgentBackend` protocol (constructed with no arguments), or a factory callable that returns such an instance.
  - **`implement(request, *, working_dir)`**: resolves the git worktree path, renders a detailed natural-language prompt describing the goal, constraints, acceptance criteria, plan steps, focus metrics, guardrails, validation bullets, risks, additional notes, handoff notes, and any fallback plan, builds a `StructuredAgentTask` whose schema usage depends on `WORKER_CODING_VALIDATION_MODE`, and asks the backend to execute it (for `CodexCliBackend` this means `codex exec --full-auto`):
    - In `"strict"` mode, the coding JSON schema is passed to the backend (using Codex's native schema support when available), and the response must parse cleanly as `_CodingOutputModel` or the agent will retry and eventually fail.
    - In `"lenient"` mode, the same schema is still provided to the backend, but JSON decoding / validation failures are treated as non-fatal: the agent first attempts to parse the response into `_CodingOutputModel`, and on failure synthesises a minimal `CodingPlanExecution` from the free-form output.
    - In `"none"` mode, no schema is enforced at the backend level and the agent skips JSON parsing entirely, always building a minimal `CodingPlanExecution` directly from the raw free-form output and job context.
  - Retries the backend invocation up to `max_attempts` times when the process fails or, in strict/lenient modes, when JSON decoding / schema validation errors occur, logging warnings via `loguru` and showing concise progress output with `rich`.
  - On success, either parses JSON into `_CodingOutputModel` and converts it into a `CodingPlanExecution`, or (for lenient/none modes when parsing is not possible) returns a best-effort synthetic `CodingPlanExecution`; on repeated failure or timeout, raises `CodingError` with a descriptive message.
  - Detects and treats a run that leaves the worktree unchanged as a failure, persisting debug artifacts and retrying until attempts are exhausted.
  - Merges any configured extra environment variables into the backend subprocess environment and enforces bounded prompt and log sizes via `_truncate`.

## Exceptions and helpers

- **`_parse_output()`**, **`_log_invalid_output()`**, **`_to_domain()`**, **`_format_plan_step()`**, **`_format_bullets()`**, and **`_truncate()`**: utilities that format human-readable prompt sections, enforce length limits, convert the raw JSON model into domain types, and provide rich logging when backend output cannot be validated.
  Debug artifacts are written under `logs/worker/coding`, including the effective `schema_mode` and `validation_mode` used for each run.

