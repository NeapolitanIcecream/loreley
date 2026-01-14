# ADR 0020: Split built-in agent backends and extract a shared structured-agent runner

Date: 2026-01-14

Context: `planning` and `coding` duplicated the same retry loop, task construction, truncation, and debug-dir logic, while concrete CLI backends and core contracts/utilities were not clearly separated.
Decision: Create a dedicated core package `loreley.core.worker.agent` for shared contracts and orchestration utilities (`build_structured_agent_task`, `coerce_structured_output`, `run_structured_agent_task`, `resolve_worker_debug_dir`, `TruncationMixin`, plus loader/utils) and implement built-in CLI backends under `loreley.core.worker.agent.backends.*` (`CodexCliBackend`, `CursorCliBackend`).
Constraints: Keep `loreley.core.worker.agent` backend-agnostic (no imports/re-exports of built-in backends) so imports remain acyclic; built-in backends may depend only on `loreley.core.worker.agent` APIs.
Consequences: Worker agents become thinner and easier to evolve; adding new backends requires less boilerplate, and configuration strings should reference `loreley.core.worker.agent.backends`.

