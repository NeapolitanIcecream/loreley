# loreley.db.base

Database engine and session management for Loreley.

## Engine and session factory

- **`_sanitize_dsn(raw_dsn)`**: masks the password portion of a database DSN so it can be safely logged.
- **`get_engine()`**: cached factory that returns the global SQLAlchemy engine created from `Settings.database_dsn`, configured with `pool_pre_ping`, connection pool sizing, timeouts, and optional SQL echoing. The engine is initialised lazily on first use.

## Declarative base and context manager

- **`Base`**: shared declarative base class used by all ORM models in `loreley.db.models`.
- **`session_scope()`**: context manager that yields a `Session`, commits on success, rolls back on exception, logs failures with `loguru`, and always closes the session.

## Schema helpers

- **`ensure_database_schema()`**: imports `loreley.db.models`, calls `Base.metadata.create_all(bind=get_engine())` to create any missing tables, then validates the single-row `InstanceMetadata` marker against the current environment. This is safe to call multiple times and is used by the UI API at startup.
- **`reset_database_schema()`**: drops and recreates all ORM tables, then seeds `InstanceMetadata` from the current environment (`EXPERIMENT_ID` and `MAPELITES_EXPERIMENT_ROOT_COMMIT`). This is the supported upgrade path for destructive schema changes.
