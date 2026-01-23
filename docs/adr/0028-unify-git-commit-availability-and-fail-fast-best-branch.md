# ADR 0028: Unify git commit availability and fail-fast best-fitness branch

Date: 2026-01-23

Status: Accepted

Decision

- Centralize “ensure commit available locally” in `loreley.core.git.require_commit(...)`.
- Use a single fetch strategy: `git fetch --prune --tags origin [--depth=N]`, and `git fetch --unshallow origin` when the repo is shallow and the commit is still missing.
- Treat missing commits and git fetch/unshallow failures as fatal errors (fail fast) to avoid silent drift across components.
- Make the best-fitness branch a first-class deliverable: scheduler must fail fast if it cannot resolve the best commit or update the branch.
- Use a stable branch name `evolution/best/<experiment>` and force-update it to the best commit to keep the deliverable deterministic across restarts.

