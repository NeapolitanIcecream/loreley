# ADR 0034 — Kilocode CLI Agent Backend

## Status

Accepted

## Context

Loreley needs support for multiple agent backends to drive planning and coding
tasks. Kilocode provides a non-interactive CLI (`kilocode --auto`) suitable for
headless orchestration, CI/CD pipelines, and parallel task execution.

## Decision

Add `KilocodeCliBackend` following the same `AgentBackend` protocol used by
`CodexCliBackend` and `CursorCliBackend`. The backend invokes
`kilocode --auto [--json] [--mode MODE] "PROMPT"` as a subprocess, captures
stdout/stderr, and returns an `AgentInvocation`. Configuration is managed via
`WORKER_KILOCODE_*` environment variables through pydantic-settings. Provide a
generic `kilocode_backend()` factory plus worker-aware factories that raise the
planning/coding agent error types for retries and debug artifacts.

## Consequences

- Workers can use Kilocode by setting `WORKER_CODING_BACKEND` or
  `WORKER_PLANNING_BACKEND` to:
  - `loreley.core.worker.agent.backends.kilocode_cli:kilocode_coding_backend`
  - `loreley.core.worker.agent.backends.kilocode_cli:kilocode_planning_backend`
- No new runtime dependencies; Kilocode CLI must be installed on the host.
- Structured JSON output is enabled by default for downstream parsing.
