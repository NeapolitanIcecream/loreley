# ADR 0011: Remove repo-state max-files cap; require startup approval by file count

Date: 2026-01-08

Context: A repo-state max-files cap can permanently disable incremental-only ingestion when the eligible file count hovers around the cap. Large repositories should be an explicit operator choice, not an implicit runtime heuristic.
Decision: `MAPELITES_REPO_STATE_MAX_FILES` is removed. The scheduler performs a startup scan of the experiment root commit (`MAPELITES_EXPERIMENT_ROOT_COMMIT`) and reports the initial eligible file count; the operator must explicitly approve this count before the scheduler enters the main loop.
Constraints: Approval is provided non-interactively (config/CLI) and must match the observed count to prevent stale acknowledgements.
Consequences: Experiments fail fast on unexpectedly large repositories while retaining incremental-only correctness; operators opt in with a clear, auditable acknowledgement.


