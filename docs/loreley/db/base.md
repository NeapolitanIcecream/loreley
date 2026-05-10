# loreley.db.base

Database engine and session management for Loreley.

## Engine and session factory

- **`_sanitize_dsn(raw_dsn)`**: masks the password portion of a database DSN so it can be safely logged.
- **`get_engine()`**: cached factory that returns the global SQLAlchemy engine created from `Settings.database_dsn`, configured with `pool_pre_ping`, connection pool sizing, timeouts, and optional SQL echoing. The engine is initialised lazily on first use.

## Declarative base and context manager

- **`Base`**: shared declarative base class used by all ORM models in `loreley.db.models`.
- **`session_scope()`**: context manager that yields a `Session`, commits on success, rolls back on exception, logs failures with `loguru`, and always closes the session.

## Schema helpers

- **`ensure_database_schema()`**: imports `loreley.db.models`, creates a fresh current schema when the database is empty and `DB_AUTO_MIGRATE=true`, or runs native migrations for supported older `InstanceMetadata.schema_version` values before marker validation. With `DB_AUTO_MIGRATE=false`, fresh initialization and upgrades must be run explicitly with `uv run loreley db migrate`. This is safe to call multiple times and is used by API/scheduler/worker startup.
- **`reset_database_schema()`**: drops and recreates all ORM tables through the native current-schema initializer, including `InstanceMetadata`, migration audit state, and managed indexes. This is a destructive local fallback; normal upgrades should use `uv run loreley db migrate`.
