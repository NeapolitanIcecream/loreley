# app.config

Centralised configuration for the Loreley application, backed by `pydantic-settings` and environment variables.

## Settings

- **`Settings`**: `BaseSettings` subclass that loads core application configuration.
  - **Environment**: `app_name`, `environment`, `log_level`.
  - **Database**: either a raw `DATABASE_URL` or individual `DB_*` fields (scheme, host, port, username, password, database name, pool options, echo flag).
  - **Metrics**: `metrics_retention_days` controls how long metrics are retained.
  - **Worker repository**: `WORKER_REPO_REMOTE_URL`, `WORKER_REPO_BRANCH`, `WORKER_REPO_WORKTREE`, `WORKER_REPO_GIT_BIN`, `WORKER_REPO_FETCH_DEPTH`, `WORKER_REPO_CLEAN_EXCLUDES`, `WORKER_REPO_JOB_BRANCH_PREFIX`, and `WORKER_REPO_ENABLE_LFS` configure the git worktree used by worker processes (upstream remote and branch, local checkout path, git binary, shallow clone depth, clean exclusions, job branch naming, and optional Git LFS support), used by `app.core.worker.repository.WorkerRepository`.
  - **Worker planning**: `WORKER_PLANNING_*` options configuring how the external Codex CLI planner is invoked (binary path, optional profile, maximum attempts, timeout, extra environment variables, and an optional JSON schema override), used by `app.core.worker.planning.PlanningAgent`.
  - **Map-Elites preprocessing**: `MAPELITES_PREPROCESS_*` options controlling which changed code files are considered for feature extraction (limits on file count/size, allowed extensions/filenames, excluded path globs, whitespace handling, and comment stripping), used by `app.core.map-elites.preprocess.CodePreprocessor`.
  - **Map-Elites chunking**: `MAPELITES_CHUNK_*` options controlling how preprocessed files are split into chunks (target and minimum lines per chunk, overlap, maximum chunks per file, and boundary keywords used by the chunker), used by `app.core.map-elites.chunk.CodeChunker`.
  - **Map-Elites code embedding**: `MAPELITES_CODE_EMBEDDING_*` options configuring the embedding model, optional output dimensions, batch size, per-commit chunk budget, retry count, and exponential backoff for embedding requests, used by `app.core.map-elites.code_embedding.CodeEmbedder`.
  - **Map-Elites summary embedding**: `MAPELITES_SUMMARY_*` and `MAPELITES_SUMMARY_EMBEDDING_*` options configuring the LLM summary model (name, temperature, max output tokens, source excerpt character limit, retries, and backoff) and the summary embedding model (name, dimensions, and batch size), used by `app.core.map-elites.summarization_embedding.SummaryEmbedder`.
  - **Map-Elites dimensionality reduction**: `MAPELITES_DIMENSION_REDUCTION_*` options controlling how penultimate code/summary embeddings are normalised, the target feature dimensions, minimum sample count for fitting PCA, rolling history size, and refit cadence, used by `app.core.map-elites.dimension_reduction.DimensionReducer`.
  - **Map-Elites feature bounds and archive**: `MAPELITES_FEATURE_*` and `MAPELITES_ARCHIVE_*` options defining the search space for behaviour features (lower/upper bounds) and the grid resolution and learning parameters for the underlying MAP-Elites archive, used by `app.core.map-elites.map-elites.MapElitesManager`.
  - **Map-Elites fitness and sampling**: `MAPELITES_FITNESS_*` and `MAPELITES_SAMPLER_*` options that configure which metric to optimise, how to treat fitness direction/floor, and how new jobs are drawn from the archive (inspiration count, neighbour radius, fallback sampling, default priority, and whether to include metadata), used by `app.core.map-elites.map-elites.MapElitesManager` and `app.core.map-elites.sampler.MapElitesSampler`.
- **`database_dsn`**: computed property that returns a SQLAlchemy-compatible DSN, preferring `DATABASE_URL` when set and otherwise building one from the individual DB fields (with credentials URL-encoded).
- **`export_safe()`**: helper that returns a dict of non-sensitive configuration values suitable for logging.

## Access helpers

- **`get_settings()`**: cached factory that instantiates `Settings`, logs a concise summary of the environment and DB host using `rich`/`loguru`, and returns a singleton instance for reuse across the app.
