# loreley.core.experiments

Helpers for deriving canonical repository and experiment context from the current git worktree and `Settings`.

## Errors

- **`ExperimentError`**: runtime error raised when the repository or experiment context cannot be resolved.  
  Used for git discovery failures (non‑existent or non‑repository paths) and database errors when reading or writing `Repository` / `Experiment` rows.

## Repository normalisation

- **`canonicalise_repository(*, settings=None, repo_root=None, repo=None)`**: resolves or creates a `Repository` row for a given git worktree.  
  - Expands and normalises the target path, defaulting to `Settings.worker_repo_worktree` when `repo_root` is not provided.  
  - Validates that the path is a git repository and extracts the `origin` remote URL when available.  
  - Uses `_normalise_remote_url()` to strip credentials, support both HTTPS and SSH scp‑style URLs, and produce a canonical `remote_url` for hashing and storage.  
  - Builds a stable `slug` from either the canonical remote URL or the local path via `_build_slug_from_source()`.  
  - Populates an `extra` JSON payload with the canonical origin, root path, and all remotes (with URLs normalised for safe storage).  
  - Within a DB `session_scope()`, either:
    - returns an existing `Repository` with the same `slug` after best‑effort metadata refresh (remote URL, root path, extra), or  
    - creates and persists a new `Repository` row with the derived slug, remote URL, root path, and extra metadata.  
  - Logs concise status messages via `rich` (for human‑friendly console output) and `loguru` (for structured logs).

## Experiment identity (env-only settings)

Loreley assumes **runtime behaviour settings are provided via environment variables** and remain stable for the lifetime of a database. The database does not persist a settings snapshot.

- **Identity anchor**: `MAPELITES_EXPERIMENT_ROOT_COMMIT` (resolved to a canonical full hash).
- **`Experiment.config_hash`**: derived from the canonical root commit only (stable across unrelated env tweaks).

## Experiment derivation

- **`derive_experiment(settings, repository, *, repo)`**: returns or creates an `Experiment` row for a given repository and settings.  
  - Resolves the configured root commit to a canonical full hash and computes `config_hash` from that value.  
  - When found, returns the existing row unchanged.  
  - Otherwise creates a new `Experiment` with:
    - `name` derived from `repository.slug` plus the first 8 characters of the config hash,  
    - `status="active"`.  
  - Logs both to the console and to the structured logger when creating a new experiment.

- **`get_or_create_experiment(*, settings=None, repo_root=None)`**: convenience helper that resolves the `Repository` / `Experiment` pair and returns settings for the scheduler process.  
  - Resolves settings via `get_settings()` when not provided explicitly.  
  - Chooses the repository root in this order: explicit `repo_root`, `Settings.scheduler_repo_root`, then `Settings.worker_repo_worktree`.  
  - Validates that the chosen root is a git repository, logging and raising `ExperimentError` when it is not.  
  - Reuses the discovered `git.Repo` instance when calling `canonicalise_repository()` to avoid redundant discovery work.  
  - Pins repository-root ignore rules for repo-state embeddings by reading `.gitignore` + `.loreleyignore` from the root commit and storing the combined ignore text + hash in `Settings` for the scheduler process lifetime.  
  - Calls `derive_experiment()` to obtain the current experiment and logs the selected `(repository.slug, experiment.id, experiment.config_hash)` pair.  
  - Returns `(Repository, Experiment, Settings)` so callers can pass the settings downstream consistently.

## Logging and error handling

- All operations are logged through a `loguru` logger bound with `module="core.experiments"` plus a `rich` console for user‑facing status messages.  
- Git and database failures are wrapped into `ExperimentError` with concise, user‑oriented messages while preserving the original exception as the cause.  
- Experiment identity is intentionally minimal so that operational tweaks do not fragment experiments in the database.


