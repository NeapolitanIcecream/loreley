# app.config

Centralised configuration for the Loreley application, backed by `pydantic-settings` and environment variables.

## Settings

- **`Settings`**: `BaseSettings` subclass that loads core application configuration.
  - **Environment**: `app_name`, `environment`, `log_level`.
  - **Database**: either a raw `DATABASE_URL` or individual `DB_*` fields (scheme, host, port, username, password, database name, pool options, echo flag).
  - **Metrics**: `metrics_retention_days` controls how long metrics are retained.
- **`database_dsn`**: computed property that returns a SQLAlchemy-compatible DSN, preferring `DATABASE_URL` when set and otherwise building one from the individual DB fields (with credentials URL-encoded).
- **`export_safe()`**: helper that returns a dict of non-sensitive configuration values suitable for logging.

## Access helpers

- **`get_settings()`**: cached factory that instantiates `Settings`, logs a concise summary of the environment and DB host using `rich`/`loguru`, and returns a singleton instance for reuse across the app.
