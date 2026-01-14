# loreley.core.worker.agent

Shared contracts and orchestration utilities for structured planning/coding agents.

Built-in CLI backends live under `loreley.core.worker.agent.backends` and depend only on this core package (no re-exports from the core module), keeping imports acyclic.

## Core types

- **`SchemaMode`**: schema handling mode (`"native"`, `"prompt"`, `"none"`).
- **`ValidationMode`**: validation strictness (`"strict"`, `"lenient"`, `"none"`).
- **`StructuredAgentTask`**: backend-agnostic task payload (`name`, `prompt`, optional `schema`, and `schema_mode`).
- **`AgentInvocation`**: immutable record of a single backend call (`command`, `stdout`, `stderr`, `duration_seconds`).
- **`AgentBackend`**: protocol defining `run(task, *, working_dir) -> AgentInvocation`.

## Backend resolution

- **`load_agent_backend(ref, *, label)`**: resolves an `AgentBackend` from a dotted reference (`"module:attr"` or `"module.attr"`). The target may be an instance, a class (instantiated with no args), or a factory callable.

## Structured execution utilities

- **`build_structured_agent_task()`**: builds a `StructuredAgentTask` that disables schemas when `validation_mode="none"`.
- **`coerce_structured_output()`**: parses stdout under strict/lenient/none semantics, with optional error hooks and free-form fallbacks.
- **`run_structured_agent_task()`**: shared retry loop for structured tasks with optional debug hooks, progress callbacks, and post-checks.

## Worker utilities

- **`resolve_schema_mode(configured_mode, api_spec)`**: resolves `"auto"` to `"prompt"` for Chat Completions and `"native"` for Responses.
- **`resolve_worker_debug_dir(logs_base_dir, kind)`**: ensures `logs/worker/{kind}` exists and returns the path.
- **`TruncationMixin`** / **`truncate_text()`**: consistent truncation helpers for prompts and logs.
- **`validate_workdir()`** / **`materialise_schema_to_temp()`**: small helpers used by CLI backends for repo validation and schema materialization.

## Built-in CLI backends

- **`CodexCliBackend`**: `loreley.core.worker.agent.backends.codex_cli.CodexCliBackend`
- **`CursorCliBackend`**: `loreley.core.worker.agent.backends.cursor_cli.CursorCliBackend`
  - **Factory helper**: `cursor_backend_from_settings()`

