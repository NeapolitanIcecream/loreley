# loreley.core.worker.repository

Git worktree management for Loreley worker processes, responsible for cloning, syncing, cleaning, and publishing the upstream repository used for evolutionary jobs.

## Types

- **`RepositoryError`**: custom runtime error raised when a git operation fails, capturing the command, return code, stdout, and stderr for easier debugging.
- **`CheckoutContext`**: frozen dataclass describing the result of preparing a job checkout (`job_id`, derived `branch_name`, selected `base_commit`, and local `worktree` path).

## Repository

- **`WorkerRepository`**: high-level manager for the worker git worktree built on top of `git.Repo`.
  - Configured via `loreley.config.Settings` worker repository options (`WORKER_REPO_REMOTE_URL`, `WORKER_REPO_BRANCH`, `WORKER_REPO_WORKTREE`, `WORKER_REPO_WORKTREE_RANDOMIZE`, `WORKER_REPO_WORKTREE_RANDOM_SUFFIX_LEN`, `WORKER_REPO_GIT_BIN`, `WORKER_REPO_FETCH_DEPTH`, `WORKER_REPO_CLEAN_EXCLUDES`, `WORKER_REPO_JOB_BRANCH_PREFIX`, `WORKER_REPO_ENABLE_LFS`, `WORKER_REPO_JOB_BRANCH_TTL_HOURS`) and honours commit author settings for worker-produced commits. `WORKER_REPO_REMOTE_URL` is mandatory; when it is absent the repository raises `RepositoryError` during construction. When `WORKER_REPO_WORKTREE_RANDOMIZE` is true, the final path segment of `WORKER_REPO_WORKTREE` gains a random hexadecimal suffix (length controlled by `WORKER_REPO_WORKTREE_RANDOM_SUFFIX_LEN`, default 8) so multiple workers on the same host can use isolated clones.
  - `prepare()` ensures the worktree directory exists, clones the remote with the configured depth/branch if necessary, aligns the local tracking branch with the configured upstream, and refreshes tags/LFS where enabled, logging progress via `rich` and `loguru`.
  - `checkout_for_job(job_id, base_commit, create_branch=True)` cleans the worktree, ensures the `base_commit` object is available locally (unshallows the repository when needed), then either checks out that commit in detached mode or creates a per-job branch under the configured job-branch prefix, returning a `CheckoutContext`.
  - `clean_worktree()` hard-resets tracked files and runs `git clean -xdf`, preserving any paths configured in `WORKER_REPO_CLEAN_EXCLUDES`.
  - `current_commit()` returns the current HEAD commit hash for observability and scheduling.
  - `has_changes()` reports whether the worktree is dirty (including untracked files), which the evolution worker uses to decide if there is anything to commit after coding.
  - `stage_all()` stages all tracked and untracked changes, and `commit(message)` creates a commit and returns its hash, using GitPython under the hood.
  - `push_branch(branch_name, remote=\"origin\", force_with_lease=False)` pushes the current branch to the configured remote (optionally with `--force-with-lease`), and `delete_remote_branch(branch_name, remote=\"origin\")` removes remote job branches without affecting local history.
  - `prune_stale_job_branches()` enumerates remote job branches under the configured job-branch prefix and deletes those whose last commit is older than `WORKER_REPO_JOB_BRANCH_TTL_HOURS`, logging a concise summary of how many branches were pruned.
  - Internal helpers such as `_ensure_worktree_ready()`, `_sync_upstream()`, `_ensure_remote_origin()`, `_fetch()`, `_sync_lfs()`, `_ensure_commit_available()`, and `_wrap_git_error()` encapsulate the GitPython integration, remote configuration, LFS sync, shallow/unshallow behaviour, and consistent error wrapping with sanitised git commands.
