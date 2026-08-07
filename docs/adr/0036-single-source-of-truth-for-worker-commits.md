# ADR 0036: Single source of truth for worker commits

Date: 2026-02-09

Context: Evolution jobs require a single, reliable place to stage, commit, and generate the final git subject used for persistence and branch publishing.

Decision: The evolution worker (`loreley.core.worker.evolution.EvolutionWorker`) owns commit creation and commit subject generation. The coding agent only applies repository changes and returns a Markdown execution report (summary + markdown). Coding prompts explicitly forbid creating git commits or pushing branches.

Consequences: Commit messages deterministically reuse the coding report summary, then the plan summary, then the job id. This avoids a separate model call while keeping commit creation in one place. The coding backend contract remains limited to applying changes and returning its report.
