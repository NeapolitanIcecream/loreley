# ADR 0011: Remove repo-state max-files cap; require interactive startup approval

Date: 2026-01-08

Context: A repo-state max-files cap can permanently disable incremental-only ingestion when the eligible file count hovers around the cap. Large repositories should be an explicit operator choice, not an implicit runtime heuristic.
Decision: `MAPELITES_REPO_STATE_MAX_FILES` is removed. The scheduler performs a startup scan of the experiment root commit (`MAPELITES_EXPERIMENT_ROOT_COMMIT`) and reports the initial eligible file count plus key filter knobs; the operator must explicitly approve via an interactive y/n prompt before the scheduler enters the main loop.
Constraints: Approval requires stdin to be a TTY; in non-interactive environments the scheduler exits (fail fast) instead of proceeding with an unapproved repository scale.
Consequences: Experiments fail fast on unexpectedly large repositories while retaining incremental-only correctness; operators opt in explicitly at startup with clear on-screen context.


