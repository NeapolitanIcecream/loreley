# ADR 0018: Remove legacy `script/` wrappers

Date: 2026-01-14

Context: Loreley previously shipped standalone `script/*.py` wrappers for common operations (doctor, scheduler, worker, UI API, UI, DB reset). With the unified Typer-based CLI and the shared runtime helpers in `loreley.entrypoints`, these wrappers duplicated behavior and documentation.

Decision: Remove the `script/*.py` wrappers and document a single operational entrypoint via the package CLI (`uv run loreley ...`). Expose `loreley reset-db --yes` for destructive schema resets.

Consequences: There is one canonical invocation path with consistent help output, options, and exit codes. Operational documentation remains, but references the unified CLI only.

