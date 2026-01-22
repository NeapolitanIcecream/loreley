# loreley.core.experiments

Helpers for resolving repository identity and validating instance metadata in single-tenant databases.

## Errors

- **`ExperimentError`**: runtime error raised when the repository or instance context cannot be resolved.
  Used for git discovery failures (non‑existent or non‑repository paths) and instance metadata mismatches.

## Repository identity

- **`RepositoryIdentity`**: frozen dataclass containing `slug`, `canonical_origin`, and `root_path`.
- **`_normalise_remote_url()`**: strips credentials and normalises HTTPS/SSH remotes into a stable URL string.
- **`_build_slug_from_source()`**: builds a stable slug from canonical remote URLs or local paths.
- **`_resolve_repository_identity()`**: extracts `origin` metadata from a `git.Repo` and returns a `RepositoryIdentity`.

## Instance bootstrap

- **`bootstrap_instance(*, settings=None, repo_root=None)`**:
  - Resolves settings via `get_settings()` when not provided explicitly.
  - Chooses the repository root in this order: explicit `repo_root`, `Settings.scheduler_repo_root`, then `Settings.worker_repo_worktree`.
  - Validates that the chosen root is a git repository, logging and raising `ExperimentError` when it is not.
  - Resolves the canonical root commit from `MAPELITES_EXPERIMENT_ROOT_COMMIT` and pins repository‑root ignore rules by reading `.gitignore` + `.loreleyignore` from that commit.
  - Stores the combined ignore text + SHA256 in `Settings` for the scheduler process lifetime (env-only settings model).
  - Validates the single-row `InstanceMetadata` marker (schema version, experiment id, root commit) and updates
    repository fields when available.
  - Returns `(RepositoryIdentity, Settings)` so callers can pass settings downstream consistently.

## Logging and error handling

- All operations are logged through a `loguru` logger bound with `module="core.experiments"` plus a `rich` console.
- Git and metadata failures are wrapped into `ExperimentError` with concise, user‑oriented messages.

