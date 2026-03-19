# loreley.core.worker.agent

Shared contracts and orchestration utilities for planning/coding agents.

Built-in CLI backends live under `loreley.core.worker.agent.backends` and depend only on this core package (no re-exports from the core module), keeping imports acyclic.

## Core types

- **`AgentTask`**: backend-agnostic task payload (`name`, `prompt`).
- **`AgentInvocation`**: immutable record of a single backend call (`command`, `stdout`, `stderr`, `duration_seconds`).
- **`AgentBackend`**: protocol defining `run(task, *, working_dir) -> AgentInvocation`.

## Backend resolution

- **`load_agent_backend(ref, *, label)`**: resolves an `AgentBackend` from a dotted reference (`"module:attr"` or `"module.attr"`). The target may be an instance, a class (no-arg constructor), or a no-arg factory callable.

## Execution utilities

- **`run_agent_task()`**: shared retry loop for agent tasks with optional debug hooks, progress callbacks, and post-checks.

## Worker utilities

- **`resolve_worker_debug_dir(logs_base_dir, kind, experiment_id=None)`**: ensures `logs/<experiment_namespace>/worker/{kind}` exists and returns the path (falls back to `logs/worker/{kind}` when no experiment namespace is available). The experiment namespace is derived from `EXPERIMENT_ID` when configured.
- **`TruncationMixin`** / **`truncate_text()`**: consistent truncation helpers for prompts and logs.
- **`coerce_agent_stdout_text()`**: best-effort unwrapping for backend stdout that may arrive as plain Markdown, a single JSON object/array, or line-delimited JSON events containing the final Markdown payload.
- **`validate_workdir()`**: helper used by CLI backends to validate the working directory is a git repository.

## Built-in CLI backends

- **`CodexCliBackend`**: `loreley.core.worker.agent.backends.codex_cli.CodexCliBackend`
  - **Factory helpers**: `codex_planning_backend()`, `codex_coding_backend()`
- **`CursorCliBackend`**: `loreley.core.worker.agent.backends.cursor_cli.CursorCliBackend`
  - **Factory helpers**: `cursor_backend()`, `cursor_planning_backend()`, `cursor_coding_backend()`
- **`KilocodeCliBackend`**: `loreley.core.worker.agent.backends.kilocode_cli.KilocodeCliBackend`
  - **Factory helpers**: `kilocode_backend()`, `kilocode_planning_backend()`, `kilocode_coding_backend()`
