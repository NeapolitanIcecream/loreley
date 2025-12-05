# loreley.core.worker.agent_backend

Shared abstractions and helpers for structured planning/coding agents plus the default Codex CLI backend.

## Core types

- **`SchemaMode`**: type alias restricting schema handling modes to `"native"`, `"prompt"`, or `"none"`.  
  Used by structured agents to decide whether JSON schema is enforced by the backend API itself (`"native"`), injected into the prompt (`"prompt"`), or completely disabled (`"none"`).

- **`AgentInvocation`**: immutable result of a single backend invocation.  
  - Captures the executed `command` (argv tuple), captured `stdout` and `stderr`, and total `duration_seconds`.  
  - Serves as the low‑level record returned by all `AgentBackend` implementations.

- **`StructuredAgentTask`**: backend‑agnostic description of a structured agent call.  
  - Fields:
    - `name`: human‑readable label used in logs and error messages.  
    - `prompt`: full natural‑language prompt passed to the backend.  
    - `schema`: optional JSON schema dict describing the expected output structure.  
    - `schema_mode`: one of the `SchemaMode` values controlling how `schema` should be applied.  
  - Planning and coding agents construct a `StructuredAgentTask` and delegate execution to a concrete backend.

- **`AgentBackend`**: `Protocol` defining the contract for all agent backends.  
  - Requires a single method `run(task, *, working_dir)` that:
    - validates and normalises the `working_dir`,  
    - executes the task using a concrete backend mechanism (CLI, HTTP API, etc.), and  
    - returns an `AgentInvocation` with the raw output and timing information.

## Backend resolution helpers

- **`resolve_schema_mode(configured_mode, api_spec)`**: determines the effective schema mode based on configuration and the selected API surface.  
  - If `configured_mode` is anything other than `"auto"`, it is returned unchanged.  
  - When `configured_mode="auto"` and `api_spec="chat_completions"`, it returns `"prompt"` so that schemas are enforced via prompt engineering.  
  - When `configured_mode="auto"` and `api_spec="responses"`, it returns `"native"` so that native tool/JSON capabilities are used.

- **`load_agent_backend(ref, *, label)`**: resolves and instantiates an `AgentBackend` from a dotted reference string.  
  - Accepts both `"module:attr"` and `"module.attr"` forms.  
  - Internally splits the reference, imports the target module, and walks the attribute path.  
  - Supports three patterns:
    - an already‑instantiated backend object exposing a callable `run` method,  
    - a class implementing the `AgentBackend` protocol (constructed with no arguments), or  
    - a factory callable that returns a backend instance when called with no arguments.  
  - Validates that the resulting object has a callable `run` attribute and raises a descriptive `RuntimeError` otherwise.

## CLI backends

- **`CodexCliBackend`**: concrete `AgentBackend` implementation that delegates to the external Codex CLI.  
  - Configuration fields:
    - `bin`: CLI executable to invoke (for example, `"codex"`).  
    - `profile`: optional profile name passed as `--profile` to select a Codex configuration.  
    - `timeout_seconds`: hard timeout for the subprocess invocation.  
    - `extra_env`: dict of additional environment variables merged into the subprocess environment.  
    - `schema_override`: optional path to a JSON schema file to use instead of the task‑provided schema.  
    - `error_cls`: concrete `RuntimeError` subtype used for all user‑facing errors (planning/coding use module‑specific error types).  
    - `full_auto`: when `True`, appends `--full-auto` so the CLI can drive multi‑step interactions autonomously.
  - **`run(task, *, working_dir)`**:
    - Validates that `working_dir` exists, is a directory, and contains a `.git` folder via `_validate_workdir()`.  
    - Builds the CLI `command` list starting from `[bin, "exec"]`, optionally adding `--full-auto`.  
    - When `task.schema_mode=="native"`:
      - uses `schema_override` if provided and points to an existing file, or  
      - materialises `task.schema` into a temporary JSON file via `_materialise_schema_to_temp()` and passes it as `--output-schema`.  
    - Adds `--profile` when a profile is configured, merges `extra_env` into a copy of `os.environ`, and feeds `task.prompt` on stdin.  
    - Runs the process with `subprocess.run(...)`, capturing stdout/stderr, enforcing the timeout, and deleting any temporary schema file afterwards.  
    - Raises `error_cls` when the process exits non‑zero or times out; even when stdout is empty it still returns an `AgentInvocation`, leaving it to higher‑level agents (and their validation modes) to decide whether an empty payload is acceptable.

- **`CursorCliBackend`**: concrete `AgentBackend` implementation that delegates to the Cursor Agent CLI (`cursor-agent`).  
  - Configuration fields:
    - `bin`: CLI executable to invoke (default `"cursor-agent"`).  
    - `model`: model identifier passed as `--model`; defaults to `"gpt-5.1-codex-max-high"` and can be overridden (for example, `"gpt-5"`), including via the `cursor_backend_from_settings()` helper that reads `WORKER_CURSOR_MODEL`.  
    - `timeout_seconds`: hard timeout for the subprocess invocation.  
    - `extra_env`: dict of additional environment variables merged into the subprocess environment.  
    - `output_format`: value passed as `--output-format` (default `"text"`), typically left as `"text"` so the agent can emit a single JSON object as plain text.  
    - `force`: when `True` (default), appends `--force` so the Cursor agent allows commands unless explicitly denied; set to `False` to omit the flag.  
    - `error_cls`: concrete `RuntimeError` subtype used for all user‑facing errors.
  - **`run(task, *, working_dir)`**:
    - Validates that `working_dir` exists, is a directory, and contains a `.git` folder via `_validate_workdir()`.  
    - Builds the CLI `command` list starting from `[bin]`, adding:
      - `-p <prompt>` to forward `task.prompt` to the Cursor agent,  
      - `--model` when `model` is configured, and  
      - `--output-format` when `output_format` is set, and  
      - `--force` when `force=True`.  
    - Merges `extra_env` into a copy of `os.environ` and runs `cursor-agent` in the provided working directory.  
    - Captures stdout/stderr and enforces the timeout via `subprocess.run(...)`.  
    - Raises `error_cls` when the process exits non‑zero or times out; even when stdout is empty it still returns an `AgentInvocation`, leaving it to higher‑level agents and validation logic to decide how to handle the result.  
    - Does not pass JSON Schema to the CLI directly; structured agents are expected to enforce schemas via prompt engineering (for example by embedding the schema in the prompt when using `"prompt"` schema mode).
  - **Factory helper**: `cursor_backend_from_settings()` builds a `CursorCliBackend` using application settings (notably `WORKER_CURSOR_MODEL`) so deployments can pick a Cursor model without writing custom wiring.

## Internal utilities

- **`_validate_workdir()`**: expands and checks the working directory path, ensuring it exists, is a directory, and contains a `.git` subdirectory; raises the configured error type with clear messages when any condition fails.  
- **`_materialise_schema_to_temp()`**: writes a JSON schema dict to a temporary file with a deterministic prefix/suffix and returns its path, used only when running in `"native"` schema mode without a `schema_override`.  
- **`_split_backend_reference()`** and **`_import_backend_target()`**: low‑level helpers used by `load_agent_backend()` to parse dotted references and resolve the final object, with detailed error messages for invalid module names or attribute paths.  
- These helpers are internal but form part of the stable backend loading behaviour relied on by the planning and coding agents.


