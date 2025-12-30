# loreley.core.worker.repository

Git repository management for Loreley worker processes. The worker maintains a
single **base clone** (configured via `WORKER_REPO_WORKTREE`) and creates
short-lived **per-job worktrees** via `git worktree` so concurrent jobs do not
share a mutable working directory.

## Types

- **`RepositoryError`**: custom runtime error raised when a git operation fails, capturing the command, return code, stdout, and stderr for easier debugging.
- **`CheckoutContext`**: frozen dataclass describing the result of preparing a job checkout (`job_id`, derived `branch_name`, selected `base_commit`, and local `worktree` path).

## Repository

- **`WorkerRepository`**: high-level manager for the worker git repository built on top of `git.Repo`.
  - Configured via `loreley.config.Settings` worker repository options (`WORKER_REPO_REMOTE_URL`, `WORKER_REPO_BRANCH`, `WORKER_REPO_WORKTREE`, `WORKER_REPO_WORKTREE_RANDOMIZE`, `WORKER_REPO_WORKTREE_RANDOM_SUFFIX_LEN`, `WORKER_REPO_GIT_BIN`, `WORKER_REPO_FETCH_DEPTH`, `WORKER_REPO_CLEAN_EXCLUDES`, `WORKER_REPO_JOB_BRANCH_PREFIX`, `WORKER_REPO_ENABLE_LFS`, `WORKER_REPO_JOB_BRANCH_TTL_HOURS`) and honours commit author settings for worker-produced commits. `WORKER_REPO_REMOTE_URL` is mandatory; when it is absent the repository raises `RepositoryError` during construction. When `WORKER_REPO_WORKTREE_RANDOMIZE` is true, the final path segment of `WORKER_REPO_WORKTREE` gains a random hexadecimal suffix (length controlled by `WORKER_REPO_WORKTREE_RANDOM_SUFFIX_LEN`, default 8) so multiple workers on the same host can use isolated base clones.
  - **Concurrency model**:
    - The base clone is protected by a cross-process file lock (`.<WORKER_REPO_WORKTREE basename>.lock`) to avoid races during clone/fetch and worktree bookkeeping.
    - Each job runs inside its own git worktree under `<WORKER_REPO_WORKTREE>-worktrees/`, so multiple worker processes can execute jobs concurrently without sharing a mutable working directory.
  - `prepare()` ensures the base clone exists, clones the remote with the configured depth/branch if necessary, aligns the local tracking branch with the configured upstream, and refreshes tags/LFS where enabled, logging progress via `rich` and `loguru`.
  - `checkout_lease_for_job(job_id, base_commit, create_branch=True)` is the recommended API: it creates an isolated per-job worktree, checks out the requested commit (optionally creating a job branch), yields a `CheckoutContext`, and removes the worktree when the context exits.
  - `checkout_for_job(job_id, base_commit, create_branch=True)` is a low-level helper that returns a `CheckoutContext` for a newly created per-job worktree but does not automatically clean up the worktree; prefer `checkout_lease_for_job()` in long-running workers.
  - `clean_worktree(worktree=...)` hard-resets tracked files and runs `git clean -xdf`, preserving any paths configured in `WORKER_REPO_CLEAN_EXCLUDES`.
  - `current_commit(worktree=...)` returns the current HEAD commit hash for observability and debugging.
  - `has_changes(worktree=...)` reports whether a worktree is dirty (including untracked files).
  - `stage_all(worktree=...)` stages all tracked and untracked changes, and `commit(message, worktree=...)` creates a commit and returns its hash.
  - `push_branch(branch_name, worktree=..., remote=\"origin\", force_with_lease=False)` pushes the named branch to the configured remote (optionally with `--force-with-lease`), and `delete_remote_branch(branch_name, remote=\"origin\")` removes remote job branches without affecting local history.
  - `prune_stale_job_branches()` enumerates remote job branches under the configured job-branch prefix and deletes those whose last commit is older than `WORKER_REPO_JOB_BRANCH_TTL_HOURS`, logging a concise summary of how many branches were pruned.
  - Internal helpers such as `_ensure_worktree_ready()`, `_sync_upstream()`, `_ensure_remote_origin()`, `_fetch()`, `_sync_lfs()`, `_ensure_commit_available()`, and `_wrap_git_error()` encapsulate the GitPython integration, remote configuration, LFS sync, shallow/unshallow behaviour, and consistent error wrapping with sanitised git commands.
