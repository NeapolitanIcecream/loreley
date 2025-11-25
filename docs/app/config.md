# app.config

Centralised configuration for the Loreley application, backed by `pydantic-settings` and environment variables.

## Settings

- **`Settings`**: `BaseSettings` subclass that loads core application configuration.
  - **Environment**: `app_name`, `environment`, `log_level`.
  - **Database**: either a raw `DATABASE_URL` or individual `DB_*` fields (scheme, host, port, username, password, database name, pool options, echo flag).
  - **Metrics**: `metrics_retention_days` controls how long metrics are retained.
  - **Map-Elites preprocessing**: `MAPELITES_PREPROCESS_*` options controlling which changed code files are considered for feature extraction (limits on file count/size, allowed extensions/filenames, excluded path globs, whitespace handling, and comment stripping), used by `app.core.map-elites.preprocess.CodePreprocessor`.
  - **Map-Elites chunking**: `MAPELITES_CHUNK_*` options controlling how preprocessed files are split into chunks (target and minimum lines per chunk, overlap, maximum chunks per file, and boundary keywords used by the chunker), used by `app.core.map-elites.chunk.CodeChunker`.
  - **Map-Elites code embedding**: `MAPELITES_CODE_EMBEDDING_*` options configuring the embedding model, optional output dimensions, batch size, per-commit chunk budget, retry count, and exponential backoff for embedding requests, used by `app.core.map-elites.code_embedding.CodeEmbedder`.
- **`database_dsn`**: computed property that returns a SQLAlchemy-compatible DSN, preferring `DATABASE_URL` when set and otherwise building one from the individual DB fields (with credentials URL-encoded).
- **`export_safe()`**: helper that returns a dict of non-sensitive configuration values suitable for logging.

## Access helpers

- **`get_settings()`**: cached factory that instantiates `Settings`, logs a concise summary of the environment and DB host using `rich`/`loguru`, and returns a singleton instance for reuse across the app.
