# loreley.core.worker.planning

Planning utilities for Loreley's autonomous worker, responsible for turning commit history and evaluation results into a Markdown plan document that a coding agent can execute.

## Domain types

- **`CommitMetric`**: lightweight value object describing a single evaluation metric (`name`, numeric `value`, optional `unit`, `higher_is_better` flag, and human-readable `summary`).
- **`CommitPlanningContext`**: shared context for one commit, including the `commit_hash`, bounded `subject` and `change_summary`, optional `trajectory` rollup lines (baseline-aligned unique-path summary), optional `trajectory_meta` counters, optional `key_files`, bounded `highlights`, an optional `evaluation_summary`, a sequence of `CommitMetric` instances, and optional MAP-Elites context (`cell_index`, `objective`, `measures`); normalises all collections to tuples on initialisation.
- **`PlanningAgentRequest`**: input payload for the planning agent containing the `base` commit context, a sequence of `inspirations`, the plain-language global evolution `goal`, optional `constraints` and `acceptance_criteria` bullet lists, an optional `iteration_hint`, and a boolean `cold_start` flag; when `cold_start=True`, the planning agent treats the request as a cold-start seed population design run and adjusts the prompt accordingly. All list-like fields are normalised to tuples.
- **`PlanDocument`**: Markdown plan document emitted by the backend (`summary`, full `markdown`, plus best-effort `focus_metrics` and `guardrails` derived from request context).
- **`PlanningAgentResponse`**: envelope returned from the planner containing the domain `plan`, raw backend `raw_output`, the rendered `prompt`, executed backend `command`, captured `stderr`, number of `attempts`, and total `duration_seconds`.

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

## Markdown contract

- Planning relies on `loreley.core.worker.agent` for shared backend abstractions (`AgentBackend`, `AgentTask`, `AgentInvocation`) and a shared retry loop (`run_agent_task()`).
- Backends may return plain-text Markdown directly or structured JSON/JSONL output that wraps the final Markdown payload. The worker unwraps those common formats on a best-effort basis before extracting a short summary.
- The worker requests a simple Markdown structure using `##` headings for Summary / Steps / Validation, with Notes optional.

## Planning agent

- **`PlanningAgent`**: high-level orchestration layer that prepares a structured planning request and delegates execution to a configurable backend.
  - Instantiated with a `Settings` object and an optional `AgentBackend` implementation. When no backend is provided, it resolves the backend via `WORKER_PLANNING_BACKEND` (default: `loreley.core.worker.agent.backends.kilocode_cli:kilocode_planning_backend`). You can override the backend by setting `WORKER_PLANNING_BACKEND` to a dotted Python path (`module:attr` or `module.attr`) that resolves to either an `AgentBackend` instance, a class implementing the `AgentBackend` protocol (constructed with no arguments), or a factory callable that returns such an instance. To use the Codex CLI explicitly, set `WORKER_PLANNING_BACKEND` to `loreley.core.worker.agent.backends.codex_cli:codex_planning_backend` and configure `WORKER_PLANNING_CODEX_*`.
  - **`plan(request, *, working_dir)`**: resolves the git worktree path, renders a concise prompt from the request (goal, constraints, acceptance criteria, and base/inspiration context), asks the backend to return a Markdown plan document, and wraps it as a `PlanDocument`.
  - Retries the backend invocation up to `max_attempts` times when process-level `PlanningError` issues occur, logging warnings via `loguru` and printing concise progress messages with `rich`.
  - Performs basic truncation of long text fields to keep prompts and summaries bounded and writes detailed debug artifacts under `logs/<experiment_namespace>/worker/planning` (or `logs/worker/planning` when no experiment namespace is available).

## Exceptions and helpers

- **`PlanningError`**: custom runtime error raised when the backend returns an error, the working directory is not a git repository, or retries are exhausted.
- **`_truncate()`**, **`_format_commit_block()`**, and **`_format_metrics()`**: internal utilities that format commit context and metrics into human-readable sections for the prompt while enforcing length limits and providing clear fallbacks when no metrics or highlights are available.
