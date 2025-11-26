# app.core.worker.repository

Git worktree management for Loreley worker processes, responsible for cloning, syncing, and cleaning the upstream repository used for evolutionary jobs.

## Types

- **`RepositoryError`**: custom runtime error raised when a git operation fails, capturing the command, return code, stdout, and stderr for easier debugging.
- **`CheckoutContext`**: frozen dataclass describing the result of preparing a job checkout (`job_id`, derived `branch_name`, selected `base_commit`, and local `worktree` path).

## Repository

- **`WorkerRepository`**: high-level manager for the worker git worktree.
  - Configured via `app.config.Settings` worker repository options (`WORKER_REPO_REMOTE_URL`, `WORKER_REPO_BRANCH`, `WORKER_REPO_WORKTREE`, `WORKER_REPO_GIT_BIN`, `WORKER_REPO_FETCH_DEPTH`, `WORKER_REPO_CLEAN_EXCLUDES`, `WORKER_REPO_JOB_BRANCH_PREFIX`, `WORKER_REPO_ENABLE_LFS`).
  - `prepare()` ensures the worktree directory exists, clones the remote if necessary, aligns the local tracking branch with the configured upstream, and refreshes tags/LFS where enabled, logging progress via `rich` and `loguru`.
  - `checkout_for_job(job_id, base_commit, create_branch=True)` cleans the worktree, ensures the `base_commit` is available locally, then either checks out that commit in detached mode or creates a per-job branch under the configured job-branch prefix, returning a `CheckoutContext`.
  - `clean_worktree()` hard-resets tracked files and runs `git clean -xdf`, preserving any paths configured in `WORKER_REPO_CLEAN_EXCLUDES`.
  - `current_commit()` returns the current HEAD commit hash for observability and scheduling.
