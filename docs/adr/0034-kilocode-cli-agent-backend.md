# ADR 0034: Kilocode CLI agent backend

## Status

Accepted

## Context

Loreley needs support for multiple agent backends to drive planning and coding
tasks. The Kilocode CLI provides a non-interactive flow (`kilo run --auto`) suitable for
headless orchestration, CI/CD pipelines, and parallel task execution.

## Decision

Add `KilocodeCliBackend` following the same `AgentBackend` protocol used by
`CodexCliBackend` and `CursorCliBackend`. The backend invokes
`kilo run --auto [--agent AGENT] [--model MODEL] [--variant VARIANT] "PROMPT"`
as a subprocess, captures stdout/stderr, and returns an `AgentInvocation`.
Configuration is managed via `WORKER_KILOCODE_*` environment variables through
pydantic-settings.

Loreley defaults the Kilocode backend to plain-text output. The CLI's
`--format json` mode emits raw JSON events rather than a single final Markdown
document, so structured output remains opt-in (`WORKER_KILOCODE_JSON_OUTPUT`)
and Loreley only applies best-effort JSON/JSONL unwrapping when it is enabled.

When wiring Kilocode to an OpenAI-compatible provider, worker-specific
`WORKER_KILOCODE_OPENAI_*` values take precedence; otherwise Loreley falls back
to the global `OPENAI_*` / `LORELEY_LLM_*` settings for API key, base URL, and
API spec.

Provide a generic `kilocode_backend()` factory plus worker-aware factories that
raise the planning/coding agent error types for retries and debug artifacts.

## Consequences

- Workers can use Kilocode by setting `WORKER_CODING_BACKEND` or
  `WORKER_PLANNING_BACKEND` to:
  - `loreley.core.worker.agent.backends.kilocode_cli:kilocode_coding_backend`
  - `loreley.core.worker.agent.backends.kilocode_cli:kilocode_planning_backend`
- No new runtime dependencies; the Kilocode CLI (`kilo`) must be installed on the host.
- `WORKER_KILOCODE_MODE` remains a backward-compatible alias for selecting the
  Kilocode agent, but new configuration should prefer `WORKER_KILOCODE_AGENT`.
- Structured JSON output is no longer the default; enable it explicitly only
  when you want raw Kilocode event streams.
