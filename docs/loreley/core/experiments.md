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

## Experiment configuration snapshots

- **`build_experiment_config_snapshot(settings)`**: extracts just the configuration fields that define an experiment.  
  - Starts from `settings.model_dump()`.  
  - Keeps only keys with prefixes `mapelites_` and `worker_evaluator_`, so that experiments stay stable across unrelated configuration changes (logging, Redis URLs, etc.).  
  - Recursively applies `_coerce_json_compatible()` so that non‑finite floats (NaN/±inf) and nested containers are converted into JSON‑serialisable equivalents, making the snapshot safe for PostgreSQL JSONB.

- **`hash_experiment_config(snapshot)`**: computes a stable SHA‑256 hash for a configuration snapshot.  
  - Serialises the snapshot with `json.dumps(..., sort_keys=True, separators=(",", ":"), default=str)` so that key ordering does not affect the result.  
  - Returns a hex digest used as `Experiment.config_hash` and as part of the default experiment name.

## Experiment derivation

- **`derive_experiment(settings, repository)`**: returns or creates an `Experiment` row for a given repository and settings.  
  - Builds a snapshot via `build_experiment_config_snapshot()`, hashes it with `hash_experiment_config()`, and looks for an existing `Experiment` with the same `(repository_id, config_hash)`.  
  - When found, returns the existing row unchanged.  
  - Otherwise creates a new `Experiment` with:
    - `name` derived from `repository.slug` plus the first 8 characters of the config hash,  
    - `config_snapshot` set to the JSON‑compatible snapshot, and  
    - `status="active"`.  
  - Logs both to the console and to the structured logger when creating a new experiment.

- **`get_or_create_experiment(*, settings=None, repo_root=None)`**: convenience helper that resolves both `Repository` and `Experiment` for the current process.  
  - Resolves settings via `get_settings()` when not provided explicitly.  
  - Chooses the repository root in this order: explicit `repo_root`, `Settings.scheduler_repo_root`, then `Settings.worker_repo_worktree`.  
  - Validates that the chosen root is a git repository, logging and raising `ExperimentError` when it is not.  
  - Reuses the discovered `git.Repo` instance when calling `canonicalise_repository()` to avoid redundant discovery work.  
  - Calls `derive_experiment()` to obtain the current experiment and logs the selected `(repository.slug, experiment.id, experiment.config_hash)` pair.  
  - Returns a `(Repository, Experiment)` tuple that callers (such as `loreley.scheduler.main.EvolutionScheduler`) pass downstream to the MAP‑Elites manager and sampler so that all jobs and archive state share the same experiment identifier.

## Logging and error handling

- All operations are logged through a `loguru` logger bound with `module="core.experiments"` plus a `rich` console for user‑facing status messages.  
- Git and database failures are wrapped into `ExperimentError` with concise, user‑oriented messages while preserving the original exception as the cause.  
- Configuration snapshots intentionally focus on MAP‑Elites and evaluator knobs so that operational tweaks (logging verbosity, queue names, etc.) do not fragment experiments in the database.


