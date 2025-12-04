# loreley.core.worker.planning

Planning utilities for Loreley's autonomous worker, responsible for turning commit history and evaluation results into a structured, multi-step plan that a coding agent can execute.

## Domain types

- **`CommitMetric`**: lightweight value object describing a single evaluation metric (`name`, numeric `value`, optional `unit`, `higher_is_better` flag, and human-readable `summary`).
- **`CommitPlanningContext`**: shared context for one commit, including the `commit_hash`, high-level `summary`, optional textual `highlights`, an optional `evaluation_summary`, a sequence of `CommitMetric` instances, and an `extra_context` dict for arbitrary structured details; normalises all collections to tuples/dicts on initialisation.
- **`PlanningAgentRequest`**: input payload for the planning agent containing the `base` commit context, a sequence of `inspirations`, the plain-language global evolution `goal` (resolved by the evolution worker from either per-job payload fields or the `Settings.worker_evolution_global_goal` configuration), optional `constraints` and `acceptance_criteria` bullet lists, an optional `iteration_hint`, and a boolean `cold_start` flag; when `cold_start=True`, the planning agent treats the request as a cold-start seed population design run and adjusts the prompt accordingly. All list-like fields are normalised to tuples.
- **`PlanStep`**: single actionable step in the generated plan (`step_id`, `title`, `intent`, `actions`, `files`, `dependencies`, `validation`, `risks`, `references`) with an `as_dict()` helper that converts all tuples back to plain lists for serialisation.
- **`PlanningPlan`**: structured planning output that aggregates the global `summary`, `rationale`, `focus_metrics`, `guardrails`, `risks`, overall `validation` bullets, the ordered `steps`, optional `handoff_notes`, and an optional free-form `fallback_plan`, again with `as_dict()` for JSON-friendly output.
- **`PlanningAgentResponse`**: envelope returned from the planner containing the domain `plan`, raw backend JSON `raw_output`, the rendered `prompt`, executed backend `command`, captured `stderr`, number of `attempts`, and total `duration_seconds`.

## JSON schema and validation

- **`PLANNING_OUTPUT_SCHEMA`**: JSON schema describing the expected shape of the planning output (top-level fields like `plan_summary`, `rationale`, `focus_metrics`, `guardrails`, `risks`, `validation`, `steps`, `handoff_notes`, and `fallback_plan`, plus constraints on each step's fields), used when invoking the external Codex CLI.
- **`_PlanStepModel`** / **`_PlanModel`**: internal `pydantic` models that validate the Codex JSON payload against the schema and provide a typed bridge from raw JSON into the `PlanStep` / `PlanningPlan` domain objects.
- **Agent backend**: planning relies on `loreley.core.worker.agent_backend` for shared backend abstractions (`AgentBackend`, `StructuredAgentTask`, `AgentInvocation`) and the default `CodexCliBackend` implementation that talks to the `codex` CLI; see that module's documentation for backend configuration details.

## Planning agent

- **`PlanningAgent`**: high-level orchestration layer that prepares a structured planning request and delegates execution to a configurable backend.
  - Instantiated with a `Settings` object and an optional `AgentBackend` implementation. When no backend is provided, it uses `CodexCliBackend` configured via `WORKER_PLANNING_CODEX_BIN`, `WORKER_PLANNING_CODEX_PROFILE`, `WORKER_PLANNING_MAX_ATTEMPTS`, `WORKER_PLANNING_TIMEOUT_SECONDS`, `WORKER_PLANNING_EXTRA_ENV`, and `WORKER_PLANNING_SCHEMA_PATH`. You can override the default by setting `WORKER_PLANNING_BACKEND` to a dotted Python path (`module:attr` or `module.attr`) that resolves to either an `AgentBackend` instance, a class implementing the `AgentBackend` protocol (constructed with no arguments), or a factory callable that returns such an instance.
  - **`plan(request, *, working_dir)`**: resolves the git worktree path, renders a rich natural-language prompt from the request (including base commit, inspiration commits, constraints, and acceptance criteria), builds a `StructuredAgentTask` that references `PLANNING_OUTPUT_SCHEMA`, and asks the backend to execute it.
  - Retries the backend invocation up to `max_attempts` times when JSON decoding, schema validation, or other `PlanningError` / `ValidationError` issues occur, logging warnings via `loguru` and printing concise progress messages with `rich`.
  - On success, parses the JSON into the `_PlanModel`, converts it into a `PlanningPlan`, and returns a `PlanningAgentResponse`; on repeated failure or timeout, raises `PlanningError` with a descriptive message.
  - Performs basic truncation of long text fields to keep prompts and summaries bounded and writes detailed debug artifacts under `logs/worker/planning`.

## Exceptions and helpers

- **`PlanningError`**: custom runtime error raised when validation fails, the backend returns an error or empty response, the planning schema path is invalid, or the working directory is not a git repository.
- **`_truncate()`**, **`_format_commit_block()`**, and **`_format_metrics()`**: internal utilities that format commit context and metrics into human-readable sections for the prompt while enforcing length limits and providing clear fallbacks when no metrics or highlights are available.

