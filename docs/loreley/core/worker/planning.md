# loreley.core.worker.planning

Planning utilities for Loreley's autonomous worker, responsible for turning commit history and evaluation results into a structured, multi-step plan that a coding agent can execute.

## Domain types

- **`CommitMetric`**: lightweight value object describing a single evaluation metric (`name`, numeric `value`, optional `unit`, `higher_is_better` flag, and human-readable `summary`).
- **`CommitPlanningContext`**: shared context for one commit, including the `commit_hash`, bounded `subject` and `change_summary`, optional `trajectory` rollup lines (baseline-aligned unique-path summary), optional `trajectory_meta` counters, optional `key_files`, bounded `highlights`, an optional `evaluation_summary`, a sequence of `CommitMetric` instances, and optional MAP-Elites context (`cell_index`, `objective`, `measures`); normalises all collections to tuples on initialisation.
- **`PlanningAgentRequest`**: input payload for the planning agent containing the `base` commit context, a sequence of `inspirations`, the plain-language global evolution `goal`, optional `constraints` and `acceptance_criteria` bullet lists, an optional `iteration_hint`, and a boolean `cold_start` flag; when `cold_start=True`, the planning agent treats the request as a cold-start seed population design run and adjusts the prompt accordingly. All list-like fields are normalised to tuples.
- **`PlanStep`**: single actionable step in the generated plan (`step_id`, `title`, `intent`, `actions`, `files`, `dependencies`, `validation`, `risks`, `references`) with an `as_dict()` helper that converts all tuples back to plain lists for serialisation.
- **`PlanningPlan`**: structured planning output that aggregates the global `summary`, `rationale`, `focus_metrics`, `guardrails`, `risks`, overall `validation` bullets, the ordered `steps`, optional `handoff_notes`, and an optional free-form `fallback_plan`, again with `as_dict()` for JSON-friendly output.
- **`PlanningAgentResponse`**: envelope returned from the planner containing the domain `plan`, raw backend JSON `raw_output`, the rendered `prompt`, executed backend `command`, captured `stderr`, number of `attempts`, and total `duration_seconds`.

## Inspiration trajectory rollups

When building the planning prompt, inspirations are enriched with a baseline-aligned
trajectory section labelled **"Trajectory (unique vs base)"**. This section is derived
from the CommitCard evolution chain (`CommitCard.parent_commit_hash`) by extracting the
unique path `LCA(base,inspiration) -> inspiration` and rendering a bounded summary:

- Always includes `unique_steps_count` and the `lca` short hash.
- Includes a bounded number of raw step summaries near the LCA ("Earliest unique steps")
  and near the inspiration tip ("Recent unique steps").
- The "Recent unique steps" section is aligned to the nearest chunk boundary to avoid
  gaps when `WORKER_PLANNING_TRAJECTORY_MAX_RAW_STEPS < WORKER_PLANNING_TRAJECTORY_BLOCK_SIZE`.
- Optionally includes cached LLM summaries for older full chunks (root-aligned fixed-size
  blocks anchored at the experiment root commit, falling back to the earliest known
  CommitCard ancestor when the configured root is not reachable) to keep prompts short
  while preserving long-horizon context.
- Reports a single "Omitted N older unique step(s)" line when the unique path is longer
  than the configured budgets.

## JSON schema and validation

- **`PLANNING_OUTPUT_SCHEMA`**: JSON schema describing the expected shape of the planning output (top-level fields like `plan_summary`, `rationale`, `focus_metrics`, `guardrails`, `risks`, `validation`, `steps`, `handoff_notes`, and `fallback_plan`, plus constraints on each step's fields), used when invoking the external Codex CLI.
- **`_PlanStepModel`** / **`_PlanModel`**: internal `pydantic` models that validate the Codex JSON payload against the schema and provide a typed bridge from raw JSON into the `PlanStep` / `PlanningPlan` domain objects.
- **Agent backend**: planning relies on `loreley.core.worker.agent_backend` for shared backend abstractions (`AgentBackend`, `StructuredAgentTask`, `AgentInvocation`) and the default `CodexCliBackend` implementation that talks to the `codex` CLI; see that module's documentation for backend configuration details.

## Planning agent

- **`PlanningAgent`**: high-level orchestration layer that prepares a structured planning request and delegates execution to a configurable backend.
  - Instantiated with a `Settings` object and an optional `AgentBackend` implementation. When no backend is provided, it uses `CodexCliBackend` configured via `WORKER_PLANNING_CODEX_BIN`, `WORKER_PLANNING_CODEX_PROFILE`, `WORKER_PLANNING_MAX_ATTEMPTS`, `WORKER_PLANNING_TIMEOUT_SECONDS`, `WORKER_PLANNING_EXTRA_ENV`, `WORKER_PLANNING_SCHEMA_PATH`, and `WORKER_PLANNING_VALIDATION_MODE`. You can override the default by setting `WORKER_PLANNING_BACKEND` to a dotted Python path (`module:attr` or `module.attr`) that resolves to either an `AgentBackend` instance, a class implementing the `AgentBackend` protocol (constructed with no arguments), or a factory callable that returns such an instance.
  - **`plan(request, *, working_dir)`**: resolves the git worktree path, renders a rich natural-language prompt from the request (including base commit, inspiration commits, constraints, and acceptance criteria), builds a `StructuredAgentTask` whose schema usage depends on `WORKER_PLANNING_VALIDATION_MODE`, and asks the backend to execute it:
    - In `"strict"` mode, the planning JSON schema is passed to the backend (using Codex's native schema support when available), and the response must parse cleanly as `_PlanModel` or the agent will retry and eventually fail.
    - In `"lenient"` mode, the same schema is still provided to the backend, but JSON decoding / validation failures are treated as non-fatal: the agent first attempts to parse the response into `_PlanModel`, and on failure synthesises a minimal `PlanningPlan` from the free-form output while keeping as much structure as possible.
    - In `"none"` mode, no schema is enforced at the backend level and the agent skips JSON parsing entirely, always building a minimal `PlanningPlan` directly from the raw free-form output and job context.
  - Retries the backend invocation up to `max_attempts` times when process-level `PlanningError` / `ValidationError` / JSON decoding issues occur, logging warnings via `loguru` and printing concise progress messages with `rich`.
  - On success, either parses JSON into `_PlanModel` and converts it into a `PlanningPlan`, or (for lenient/none modes when parsing is not possible) returns a best-effort synthetic `PlanningPlan`; on repeated failure or timeout, raises `PlanningError` with a descriptive message.
  - Performs basic truncation of long text fields to keep prompts and summaries bounded and writes detailed debug artifacts under `logs/worker/planning`, including the effective `schema_mode` and `validation_mode`.

## Exceptions and helpers

- **`PlanningError`**: custom runtime error raised when validation fails, the backend returns an error, the planning schema path is invalid, or the working directory is not a git repository.
- **`_truncate()`**, **`_format_commit_block()`**, and **`_format_metrics()`**: internal utilities that format commit context and metrics into human-readable sections for the prompt while enforcing length limits and providing clear fallbacks when no metrics or highlights are available.

