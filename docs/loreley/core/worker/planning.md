# loreley.core.worker.planning

Planning utilities for Loreley's autonomous worker, responsible for turning commit history and evaluation results into a structured, multi-step plan that a coding agent can execute.

## Domain types

- **`CommitMetric`**: lightweight value object describing a single evaluation metric (`name`, numeric `value`, optional `unit`, `higher_is_better` flag, and human-readable `summary`).
- **`CommitPlanningContext`**: shared context for one commit, including the `commit_hash`, high-level `summary`, optional textual `highlights`, an optional `evaluation_summary`, a sequence of `CommitMetric` instances, and an `extra_context` dict for arbitrary structured details; normalises all collections to tuples/dicts on initialisation.
- **`PlanningAgentRequest`**: input payload for the planning agent containing the `base` commit context, a sequence of `inspirations`, the plain-language global evolution `goal` (resolved by the evolution worker from either per-job payload fields or the `Settings.worker_evolution_global_goal` configuration), optional `constraints` and `acceptance_criteria` bullet lists, an optional `iteration_hint`, and a boolean `cold_start` flag; when `cold_start=True`, the planning agent treats the request as a cold-start seed population design run and adjusts the prompt accordingly. All list-like fields are normalised to tuples.
- **`PlanStep`**: single actionable step in the generated plan (`step_id`, `title`, `intent`, `actions`, `files`, `dependencies`, `validation`, `risks`, `references`) with an `as_dict()` helper that converts all tuples back to plain lists for serialisation.
- **`PlanningPlan`**: structured planning output that aggregates the global `summary`, `rationale`, `focus_metrics`, `guardrails`, `risks`, overall `validation` bullets, the ordered `steps`, optional `handoff_notes`, and an optional free-form `fallback_plan`, again with `as_dict()` for JSON-friendly output.
- **`PlanningAgentResponse`**: envelope returned from the planner containing the domain `plan`, raw Codex JSON `raw_output`, the rendered `prompt`, executed CLI `command`, captured `stderr`, number of `attempts`, and total `duration_seconds`.

## JSON schema and validation

- **`PLANNING_OUTPUT_SCHEMA`**: JSON schema describing the expected shape of the planning output (top-level fields like `plan_summary`, `rationale`, `focus_metrics`, `guardrails`, `risks`, `validation`, `steps`, `handoff_notes`, and `fallback_plan`, plus constraints on each step's fields), used when invoking the external Codex CLI.
- **`_PlanStepModel`** / **`_PlanModel`**: internal `pydantic` models that validate the Codex JSON payload against the schema and provide a typed bridge from raw JSON into the `PlanStep` / `PlanningPlan` domain objects.
- **`_CodexInvocation`**: frozen dataclass summarising a single Codex call (`command`, `stdout`, `stderr`, `duration_seconds`) for logging and debugging.

## Planning agent

- **`PlanningAgent`**: high-level orchestration layer that talks to the external Codex CLI.
  - Configured via `loreley.config.Settings` worker planning options (`WORKER_PLANNING_CODEX_BIN`, `WORKER_PLANNING_CODEX_PROFILE`, `WORKER_PLANNING_MAX_ATTEMPTS`, `WORKER_PLANNING_TIMEOUT_SECONDS`, `WORKER_PLANNING_EXTRA_ENV`, `WORKER_PLANNING_SCHEMA_PATH`), plus internal limits on prompt truncation, maximum highlights, and maximum metrics.
  - **`plan(request, *, working_dir)`**: validates that `working_dir` is a git repository, renders a rich natural-language prompt from the request (including base commit, inspiration commits, constraints, and acceptance criteria), materialises the JSON schema (either from `WORKER_PLANNING_SCHEMA_PATH` or a temporary file based on `PLANNING_OUTPUT_SCHEMA`), and then calls `codex exec` with the schema and prompt.
  - Retries the Codex invocation up to `max_attempts` times when JSON decoding, schema validation, or other `PlanningError` / `ValidationError` issues occur, logging warnings via `loguru` and printing concise progress messages with `rich`.
  - On success, parses the JSON into the `_PlanModel`, converts it into a `PlanningPlan`, and returns a `PlanningAgentResponse`; on repeated failure or timeout, raises `PlanningError` with a descriptive message.
  - Uses `WORKER_PLANNING_EXTRA_ENV` to layer additional environment variables onto the Codex subprocess and performs basic truncation of long text fields to keep prompts and summaries bounded.

## Exceptions and helpers

- **`PlanningError`**: custom runtime error raised when validation fails, Codex returns an error or empty response, the planning schema path is invalid, or the working directory is not a git repository.
- **`_validate_workdir()`**: internal helper that ensures the provided `working_dir` exists, is a directory, and contains a `.git` folder before any external commands are run.
- **`_truncate()`**, **`_format_commit_block()`**, and **`_format_metrics()`**: internal utilities that format commit context and metrics into human-readable sections for the prompt while enforcing length limits and providing clear fallbacks when no metrics or highlights are available.

