"""Native Loreley database migrations."""

from loreley.db.migrations.registry import MIGRATIONS, SchemaMigration, migration_path
from loreley.db.migrations.runner import (
    MigrationError,
    MigrationRequiredError,
    MigrationResult,
    SchemaStatus,
    describe_schema,
    ensure_schema_current,
    validate_database_identity,
    validate_database_schema,
)

__all__ = [
    "MIGRATIONS",
    "MigrationError",
    "MigrationRequiredError",
    "MigrationResult",
    "SchemaMigration",
    "SchemaStatus",
    "describe_schema",
    "ensure_schema_current",
    "migration_path",
    "validate_database_identity",
    "validate_database_schema",
]
