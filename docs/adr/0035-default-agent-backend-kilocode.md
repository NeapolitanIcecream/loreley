# ADR 0035 — Default planning/coding backend is Kilocode

## Status

Accepted

## Context

Loreley supports multiple CLI agent backends for planning and coding. We want a single default that works well in headless automation without interactive prompts and can provide optional structured stdout for downstream tools.

## Decision

Default `WORKER_PLANNING_BACKEND` and `WORKER_CODING_BACKEND` to the worker-aware Kilocode factories (`loreley.core.worker.agent.backends.kilocode_cli:kilocode_planning_backend` and `loreley.core.worker.agent.backends.kilocode_cli:kilocode_coding_backend`). When a backend emits JSON-wrapped stdout, Loreley extracts the best-effort Markdown payload before deriving summaries and commit messages.

## Consequences

- `kilocode` is required by default; other backends remain opt-in via `WORKER_*_BACKEND`.
- Debug artifacts preserve raw stdout/stderr while the persisted plan/report uses the extracted Markdown payload.
