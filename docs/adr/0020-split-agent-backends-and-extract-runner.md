# ADR 0020: Split built-in agent backends and extract a shared structured-agent runner

Date: 2026-01-14

Context: `planning` and `coding` duplicated the same retry loop, task construction, truncation, and debug-dir logic, while `agent_backend` mixed core abstractions with concrete CLI backend implementations.
Decision: Move `CodexCliBackend` and `CursorCliBackend` into `loreley.core.worker.agent_backends.*` modules and extract shared orchestration utilities into `loreley.core.worker.agent_backend` (`build_structured_agent_task`, `coerce_structured_output`, `run_structured_agent_task`, `resolve_worker_debug_dir`, `TruncationMixin`).
Constraints: Treat `loreley.core.worker.agent_backends` as the canonical import path for built-in backends and keep `agent_backend` backend-agnostic to avoid import-order circular dependencies.
Consequences: Worker agents become thinner and easier to evolve; adding new agents/backends requires less boilerplate, while callers should reference built-in backends via `loreley.core.worker.agent_backends`.

