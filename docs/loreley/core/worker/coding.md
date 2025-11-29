# loreley.core.worker.coding

Execution engine for Loreley's autonomous worker, responsible for driving the Codex-based coding agent that applies a planning agent's plan to a real git worktree.

## Domain types

- **`CodingError`**: custom runtime error raised when the coding agent cannot successfully execute the plan (invalid schema, Codex failures, bad working directory, timeouts, etc.).
- **`StepExecutionStatus`**: string `Enum` describing how each plan step was handled (`COMPLETED`, `PARTIAL`, or `SKIPPED`).
- **`CodingStepReport`**: dataclass capturing the outcome of a single step (`step_id`, `status`, human-readable `summary`, and optional `files` / `commands` touched by that step).
- **`CodingPlanExecution`**: aggregate result for the whole run, including the overall `implementation_summary`, optional `commit_message`, tuple of `step_results`, `tests_executed`, `tests_recommended`, `follow_up_items`, and `notes`.
- **`CodingAgentRequest`**: input payload given to the coding agent (`goal`, `plan` from `PlanningPlan`, `base_commit`, optional `constraints`, `acceptance_criteria`, `iteration_hint`, and `additional_notes`); normalises all sequence fields to tuples in `__post_init__`.
- **`CodingAgentResponse`**: envelope returned from the agent combining the structured `execution`, raw Codex `raw_output`, rendered `prompt`, executed CLI `command`, captured `stderr`, number of `attempts`, and total `duration_seconds`.

## JSON schema and validation

- **`CODING_OUTPUT_SCHEMA`**: JSON schema describing the expected Codex output (top-level `implementation_summary`, optional `commit_message`, array of `step_results`, plus optional `tests_executed`, `tests_recommended`, `follow_up_items`, and `notes`), used to validate the CLI response.
- **`_StepResultModel`** / **`_CodingOutputModel`**: internal frozen `pydantic` models that validate the JSON payload against `CODING_OUTPUT_SCHEMA` and provide a typed bridge into the domain dataclasses.
- **`_CodexInvocation`**: frozen dataclass summarising a single Codex run (`command`, `stdout`, `stderr`, `duration_seconds`) which is logged for debugging and observability.

## Coding agent

- **`CodingAgent`**: high-level orchestrator that turns a `CodingAgentRequest` and `PlanningPlan` into a sequence of edits by calling the external Codex CLI.
  - Configured via `loreley.config.Settings` worker coding options (`WORKER_CODING_CODEX_BIN`, `WORKER_CODING_CODEX_PROFILE`, `WORKER_CODING_MAX_ATTEMPTS`, `WORKER_CODING_TIMEOUT_SECONDS`, `WORKER_CODING_EXTRA_ENV`, `WORKER_CODING_SCHEMA_PATH`) and an internal `_truncate_limit` for long text.
  - **`implement(request, *, working_dir)`**: validates that `working_dir` is a git repository, renders a detailed natural-language prompt describing the goal, constraints, acceptance criteria, plan steps, focus metrics, guardrails, validation bullets, risks, additional notes, handoff notes, and any fallback plan, materialises the JSON schema (either from `WORKER_CODING_SCHEMA_PATH` or a temporary file based on `CODING_OUTPUT_SCHEMA`), and then calls `codex exec` with the prompt and schema.
  - Retries the Codex invocation up to `max_attempts` times when the process fails, the JSON is invalid, or schema validation errors occur, logging warnings via `loguru` and showing concise progress output with `rich`.
  - On success, parses the JSON into `_CodingOutputModel`, converts it into a `CodingPlanExecution`, and returns a `CodingAgentResponse`; on repeated failure or timeout, raises `CodingError` with a descriptive message.
  - Merges any `WORKER_CODING_EXTRA_ENV` values into the subprocess environment and enforces bounded prompt and log sizes via `_truncate`.

## Exceptions and helpers

- **`_validate_workdir()`**: internal helper that ensures `working_dir` exists, is a directory, and contains a `.git` folder before any external commands are run.
- **`_materialise_schema()`**: writes `CODING_OUTPUT_SCHEMA` to a temporary JSON file when no schema override is configured, returning the path and an optional handle for later cleanup.
- **`_invoke_codex()`**: constructs the `codex exec` command, runs it with appropriate timeout and environment, logs the exit status and duration, and wraps stdout/stderr in a `_CodexInvocation`, raising `CodingError` on failures or empty output.
- **`_parse_output()`**, **`_log_invalid_output()`**, **`_to_domain()`**, **`_format_plan_step()`**, **`_format_bullets()`**, and **`_truncate()`**: utilities that format human-readable prompt sections, enforce length limits, convert the raw JSON model into domain types, and provide rich logging when Codex output cannot be validated.

