# ADR 0011: Remove repo-state max-files cap; require interactive startup approval

Date: 2026-01-08

Context: A repo-state max-files cap can permanently disable incremental-only ingestion when the eligible file count hovers around the cap. Large repositories should be an explicit operator choice, not an implicit runtime heuristic.
Decision: `MAPELITES_REPO_STATE_MAX_FILES` is removed. The scheduler performs a startup scan of the experiment root commit (`MAPELITES_EXPERIMENT_ROOT_COMMIT`) and reports the initial eligible file count plus key filter knobs; the operator must explicitly approve before the scheduler enters the main loop.
Constraints: Approval is interactive by default (y/n prompt). In non-interactive environments, approval must be provided explicitly via `--yes` or `SCHEDULER_STARTUP_APPROVE=true`; otherwise the scheduler exits (fail fast).
Consequences: Experiments fail fast on unexpectedly large repositories while retaining incremental-only correctness; operators opt in explicitly at startup with clear on-screen context.


