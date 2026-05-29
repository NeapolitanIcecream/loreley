from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal

from loguru import logger
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from loreley.config import Settings
from loreley.db.instance import (
    InstanceMetadataError,
    MIGRATE_DB_HINT,
    root_commit_matches,
    resolve_instance_identity,
)
from loreley.db.migrations.registry import RegistryError, SchemaMigration, migration_path

SCHEMA_MIGRATION_LOCK_KEY = 77221120260509

_AUDIT_TABLE = "loreley_schema_migrations"
_LORELEY_TABLES = (
    "instance_metadata",
    "commit_cards",
    "commit_chunk_summaries",
    "metrics",
    "evolution_jobs",
    "job_artifacts",
    "map_elites_states",
    "map_elites_archive_cells",
    "map_elites_pca_history",
    "map_elites_file_embedding_cache",
    "map_elites_repo_state_aggregates",
    "evaluation_artifacts",
    "candidate_commits",
    "diagnostic_capsules",
    "evaluation_attempts",
    "campaign_programs",
    "campaign_baselines",
    "operator_tasks",
    "agent_actions",
    "llm_usage_events",
)
_CURRENT_SCHEMA_TABLES = (*_LORELEY_TABLES, _AUDIT_TABLE)
_CURRENT_SCHEMA_INDEXES = (
    "ix_evolution_jobs_ingestion_sort_expr",
    "ix_evolution_jobs_ui_sort_expr",
    "ix_map_elites_archive_cells_island_commit",
    "ix_evolution_jobs_campaign_program_hash",
    "ix_candidate_commits_campaign_program_hash",
    "ix_evaluation_attempts_campaign_program_hash",
    "uq_operator_tasks_active_baseline_ensure",
    "uq_agent_actions_action_idempotency",
    "ix_llm_usage_events_job_created",
    "ix_llm_usage_events_run_token",
    "ix_llm_usage_events_source_created",
    "ix_llm_usage_events_phase_created",
    "ix_llm_usage_events_model_created",
    "uq_llm_usage_events_external_usage_id",
)
_CURRENT_SCHEMA_CONSTRAINTS = (
    ("evaluation_artifacts", "uq_evaluation_artifacts_job_key"),
    ("candidate_commits", "uq_candidate_commits_commit_hash"),
    ("campaign_baselines", "uq_campaign_baselines_key_hash"),
)

log = logger.bind(module="db.migrations")

SchemaState = Literal["fresh", "current", "migratable", "future", "unsupported", "damaged"]


class MigrationError(InstanceMetadataError):
    """Raised when native schema migration cannot complete."""


class MigrationRequiredError(MigrationError):
    """Raised when a database is old and automatic migration is disabled."""


class FutureSchemaError(MigrationError):
    """Raised when the database marker is newer than this binary."""


class UnsupportedMigrationError(MigrationError):
    """Raised when no native migration chain supports the database version."""


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Result of ensuring the database schema."""

    from_version: int | None
    to_version: int
    applied_versions: tuple[int, ...]
    fresh_database: bool = False


@dataclass(frozen=True, slots=True)
class SchemaStatus:
    """Raw schema marker status for CLI/preflight use."""

    schema_version: int | None
    target_version: int
    state: SchemaState
    needs_migration: bool
    detail: str = ""


def describe_schema(*, engine: Engine, target_version: int) -> SchemaStatus:
    """Inspect schema marker state without creating or migrating anything."""

    with engine.connect() as conn:
        return _describe_schema(conn, target_version=target_version)


def ensure_schema_current(
    *,
    engine: Engine,
    settings: Settings,
    target_version: int,
    auto_migrate: bool,
) -> MigrationResult:
    """Create or migrate the Loreley schema to ``target_version``."""

    with engine.begin() as conn:
        _acquire_migration_lock(conn, settings=settings)
        if not _table_exists(conn, "instance_metadata"):
            return _create_fresh_schema(
                conn,
                settings=settings,
                target_version=int(target_version),
                auto_migrate=auto_migrate,
            )
        return _ensure_existing_schema_current(
            conn,
            settings=settings,
            target_version=int(target_version),
            auto_migrate=auto_migrate,
        )


def _create_fresh_schema(
    conn: Connection,
    *,
    settings: Settings,
    target_version: int,
    auto_migrate: bool,
) -> MigrationResult:
    if _has_loreley_tables_without_marker(conn):
        raise MigrationError(
            "instance_metadata table is missing but Loreley tables exist; "
            "refusing to guess whether this database is damaged.",
        )
    if not auto_migrate:
        raise MigrationRequiredError(
            "Loreley schema is not initialized. Run `uv run loreley db migrate` "
            "before API/scheduler/worker startup.",
        )

    import loreley.db.models  # noqa: F401  # pylint: disable=unused-import
    from loreley.db.base import Base

    _create_audit_table(conn)
    Base.metadata.create_all(bind=conn)
    _seed_marker(conn, settings=settings, schema_version=target_version)
    _record_audit(
        conn,
        version=target_version,
        name="fresh_create_all",
        checksum="",
        duration_ms=0,
    )
    log.info("Created fresh Loreley schema at version {}", target_version)
    return MigrationResult(
        from_version=None,
        to_version=target_version,
        applied_versions=(),
        fresh_database=True,
    )


def _ensure_existing_schema_current(
    conn: Connection,
    *,
    settings: Settings,
    target_version: int,
    auto_migrate: bool,
) -> MigrationResult:
    marker = _require_instance_marker(conn)
    current_version = int(marker["schema_version"] or 0)
    if current_version == target_version:
        _create_audit_table(conn)
        _validate_current_schema(conn, target_version=target_version)
        return MigrationResult(
            from_version=current_version,
            to_version=target_version,
            applied_versions=(),
        )
    if current_version > target_version:
        raise FutureSchemaError(
            f"Database schema_version={current_version} is newer than "
            f"this Loreley binary target={target_version}.",
        )

    path = _migration_path_or_raise(current_version, target_version)
    if not auto_migrate:
        raise MigrationRequiredError(
            f"Database schema_version={current_version} requires migration "
            f"to {target_version}. {MIGRATE_DB_HINT}",
        )

    _create_audit_table(conn)
    _validate_marker_identity(marker, settings=settings)
    applied = _run_migrations(
        conn,
        path=path,
        settings=settings,
        target_version=target_version,
    )
    _validate_current_schema(conn, target_version=target_version)
    return MigrationResult(
        from_version=current_version,
        to_version=target_version,
        applied_versions=tuple(applied),
    )


def _require_instance_marker(conn: Connection) -> dict[str, Any]:
    marker = _load_marker(conn)
    if marker is None:
        raise MigrationError(
            "instance_metadata table exists but marker row id=1 is missing; "
            "refusing automatic migration.",
        )
    return marker


def _migration_path_or_raise(current_version: int, target_version: int) -> tuple[SchemaMigration, ...]:
    try:
        return migration_path(current_version, target_version)
    except RegistryError as exc:
        raise UnsupportedMigrationError(str(exc)) from exc


def validate_database_schema(
    *,
    engine: Engine,
    settings: Settings,
    target_version: int,
    validate_identity: bool = True,
) -> SchemaStatus:
    """Validate that the database is current and structurally usable."""

    with engine.connect() as conn:
        status = _describe_schema(conn, target_version=target_version)
        _raise_if_schema_not_current(status)
        if validate_identity:
            _validate_current_schema_identity(conn, settings=settings)
        _validate_current_schema(conn, target_version=int(target_version))
        return status


def _raise_if_schema_not_current(status: SchemaStatus) -> None:
    if status.state == "fresh":
        raise MigrationError(
            "Loreley schema is not initialized. Run `uv run loreley db migrate` "
            "or start API/scheduler/worker with DB_AUTO_MIGRATE=true.",
        )
    if status.state == "damaged":
        raise MigrationError(status.detail)
    if status.state == "future":
        raise FutureSchemaError(status.detail)
    if status.state == "unsupported":
        raise UnsupportedMigrationError(
            status.detail
            or (
                f"No Loreley native migration path from schema_version={status.schema_version} "
                f"to {status.target_version}."
            ),
        )
    if status.state == "migratable":
        raise MigrationRequiredError(
            f"Database schema_version={status.schema_version} requires migration "
            f"to {status.target_version}. {MIGRATE_DB_HINT}",
        )


def _validate_current_schema_identity(conn: Connection, *, settings: Settings) -> None:
    marker = _load_marker(conn)
    if marker is None:
        raise MigrationError("instance_metadata marker row id=1 is missing.")
    _validate_marker_identity(marker, settings=settings)


def validate_database_identity(*, engine: Engine, settings: Settings) -> None:
    """Validate the raw instance marker identity without requiring current schema."""

    with engine.connect() as conn:
        marker = _load_marker(conn)
        if marker is None:
            raise MigrationError("instance_metadata marker row id=1 is missing.")
        _validate_marker_identity(marker, settings=settings)


def _run_migrations(
    conn: Connection,
    *,
    path: tuple[SchemaMigration, ...],
    settings: Settings,
    target_version: int,
) -> list[int]:
    applied: list[int] = []
    for migration in path:
        started = time.monotonic()
        log.info(
            "Applying Loreley schema migration {} from {} to {}",
            migration.name,
            migration.from_version,
            migration.to_version,
        )
        migration.upgrade(conn, settings)
        _stamp_marker(conn, schema_version=migration.to_version)
        duration_ms = int((time.monotonic() - started) * 1000)
        _record_audit(
            conn,
            version=migration.to_version,
            name=migration.name,
            checksum="",
            duration_ms=duration_ms,
        )
        applied.append(migration.to_version)
        log.info(
            "Applied Loreley schema migration {} duration_ms={}",
            migration.name,
            duration_ms,
        )

    final = _load_marker(conn)
    final_version = int(final["schema_version"] or 0) if final is not None else None
    if final_version != int(target_version):
        raise MigrationError(
            f"Migration chain ended at schema_version={final_version}, expected {target_version}.",
        )
    return applied


def _describe_schema(conn: Connection, *, target_version: int) -> SchemaStatus:
    try:
        has_marker_table = _table_exists(conn, "instance_metadata")
        if not has_marker_table:
            if _has_loreley_tables_without_marker(conn):
                return SchemaStatus(
                    schema_version=None,
                    target_version=int(target_version),
                    state="damaged",
                    needs_migration=False,
                    detail="instance_metadata table is missing but Loreley tables exist.",
                )
            return SchemaStatus(
                schema_version=None,
                target_version=int(target_version),
                state="fresh",
                needs_migration=False,
                detail="database has no Loreley instance marker",
            )

        marker = _load_marker(conn)
        if marker is None:
            return SchemaStatus(
                schema_version=None,
                target_version=int(target_version),
                state="damaged",
                needs_migration=False,
                detail="instance_metadata marker row id=1 is missing.",
            )

        current = int(marker["schema_version"] or 0)
        if current == int(target_version):
            return SchemaStatus(
                schema_version=current,
                target_version=int(target_version),
                state="current",
                needs_migration=False,
                detail="schema is current",
            )
        if current > int(target_version):
            return SchemaStatus(
                schema_version=current,
                target_version=int(target_version),
                state="future",
                needs_migration=False,
                detail=(
                    f"Database schema_version={current} is newer than "
                    f"this Loreley binary target={target_version}."
                ),
            )
        try:
            migration_path(current, int(target_version))
        except RegistryError as exc:
            return SchemaStatus(
                schema_version=current,
                target_version=int(target_version),
                state="unsupported",
                needs_migration=True,
                detail=str(exc),
            )
        return SchemaStatus(
            schema_version=current,
            target_version=int(target_version),
            state="migratable",
            needs_migration=True,
            detail=f"schema_version={current} can migrate to {target_version}",
        )
    except SQLAlchemyError as exc:
        return SchemaStatus(
            schema_version=None,
            target_version=int(target_version),
            state="damaged",
            needs_migration=False,
            detail=f"failed to inspect Loreley schema ({exc})",
        )


def _acquire_migration_lock(conn: Connection, *, settings: Settings) -> None:
    if getattr(conn.dialect, "name", "") != "postgresql":
        return
    timeout_seconds = max(1, int(getattr(settings, "db_migration_lock_timeout_seconds", 30)))
    conn.execute(text(f"SET LOCAL lock_timeout = '{timeout_seconds * 1000}ms'"))
    conn.execute(
        text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": SCHEMA_MIGRATION_LOCK_KEY},
    )


def _create_audit_table(conn: Connection) -> None:
    conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {_AUDIT_TABLE} (
                version INTEGER PRIMARY KEY,
                name VARCHAR(128) NOT NULL,
                applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                checksum VARCHAR(64) NOT NULL DEFAULT '',
                duration_ms INTEGER,
                success BOOLEAN NOT NULL DEFAULT TRUE
            )
            """,
        )
    )


def _record_audit(
    conn: Connection,
    *,
    version: int,
    name: str,
    checksum: str,
    duration_ms: int,
) -> None:
    conn.execute(
        text(
            f"""
            INSERT INTO {_AUDIT_TABLE} (version, name, checksum, duration_ms, success)
            VALUES (:version, :name, :checksum, :duration_ms, TRUE)
            ON CONFLICT (version) DO UPDATE
            SET name = EXCLUDED.name,
                applied_at = now(),
                checksum = EXCLUDED.checksum,
                duration_ms = EXCLUDED.duration_ms,
                success = TRUE
            """,
        ),
        {
            "version": int(version),
            "name": name,
            "checksum": checksum,
            "duration_ms": int(duration_ms),
        },
    )


def _table_exists(conn: Connection, table_name: str) -> bool:
    return bool(inspect(conn).has_table(table_name))


def _has_loreley_tables_without_marker(conn: Connection) -> bool:
    for table_name in _LORELEY_TABLES:
        if table_name == "instance_metadata":
            continue
        if _table_exists(conn, table_name):
            return True
    return False


def _load_marker(conn: Connection) -> dict[str, Any] | None:
    row = (
        conn.execute(
            text(
                """
                SELECT
                    schema_version,
                    experiment_id_raw,
                    experiment_uuid::text AS experiment_uuid,
                    root_commit_hash
                FROM instance_metadata
                WHERE id = 1
                """,
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row is not None else None


def _seed_marker(conn: Connection, *, settings: Settings, schema_version: int) -> None:
    identity = resolve_instance_identity(settings)
    conn.execute(
        text(
            """
            INSERT INTO instance_metadata (
                id,
                schema_version,
                experiment_id_raw,
                experiment_uuid,
                root_commit_hash
            )
            VALUES (
                1,
                :schema_version,
                :experiment_id_raw,
                :experiment_uuid,
                :root_commit_hash
            )
            ON CONFLICT (id) DO UPDATE
            SET schema_version = EXCLUDED.schema_version,
                experiment_id_raw = EXCLUDED.experiment_id_raw,
                experiment_uuid = EXCLUDED.experiment_uuid,
                root_commit_hash = EXCLUDED.root_commit_hash,
                updated_at = now()
            """,
        ),
        {
            "schema_version": int(schema_version),
            "experiment_id_raw": identity.experiment_raw,
            "experiment_uuid": identity.experiment_uuid,
            "root_commit_hash": identity.root_commit,
        },
    )


def _stamp_marker(conn: Connection, *, schema_version: int) -> None:
    conn.execute(
        text(
            """
            UPDATE instance_metadata
            SET schema_version = :schema_version,
                updated_at = now()
            WHERE id = 1
            """,
        ),
        {"schema_version": int(schema_version)},
    )


def _validate_marker_identity(marker: dict[str, Any], *, settings: Settings) -> None:
    identity = resolve_instance_identity(settings)
    if str(marker.get("experiment_id_raw") or "").strip() != identity.experiment_raw:
        raise MigrationError("EXPERIMENT_ID does not match the database marker.")
    if str(marker.get("experiment_uuid") or "") != str(identity.experiment_uuid):
        raise MigrationError("EXPERIMENT_ID UUID mapping does not match the database marker.")
    marker_root = str(marker.get("root_commit_hash") or "").strip()
    if not root_commit_matches(marker_root, identity.root_commit):
        raise MigrationError(
            "MAPELITES_EXPERIMENT_ROOT_COMMIT does not match the database marker.",
        )


def _validate_current_schema(conn: Connection, *, target_version: int) -> None:
    marker = _load_marker(conn)
    if marker is None:
        raise MigrationError("instance_metadata marker row id=1 is missing.")
    current = int(marker["schema_version"] or 0)
    if current != int(target_version):
        raise MigrationError(
            f"instance_metadata schema_version={current}, expected {target_version}.",
        )

    missing_tables = _missing_current_schema_tables(conn)
    if missing_tables:
        raise MigrationError(f"Missing current schema tables: {', '.join(missing_tables)}.")

    if _job_kind_null_count(conn) != 0:
        raise MigrationError("evolution_jobs.job_kind contains NULL values after migration.")

    missing_indexes = _missing_current_schema_indexes(conn)
    if missing_indexes:
        raise MigrationError(f"Missing current schema indexes: {', '.join(missing_indexes)}.")

    missing_constraints = _missing_current_schema_constraints(conn)
    if missing_constraints:
        raise MigrationError(
            f"Missing current schema constraints: {', '.join(missing_constraints)}.",
        )


def _missing_current_schema_tables(conn: Connection) -> list[str]:
    return [table for table in _CURRENT_SCHEMA_TABLES if not _table_exists(conn, table)]


def _job_kind_null_count(conn: Connection) -> int:
    count = conn.execute(text("SELECT count(*) FROM evolution_jobs WHERE job_kind IS NULL")).scalar_one()
    return int(count or 0)


def _missing_current_schema_indexes(conn: Connection) -> list[str]:
    return [index_name for index_name in _CURRENT_SCHEMA_INDEXES if not _index_exists(conn, index_name)]


def _missing_current_schema_constraints(conn: Connection) -> list[str]:
    return [
        constraint_name
        for table_name, constraint_name in _CURRENT_SCHEMA_CONSTRAINTS
        if not _constraint_exists(conn, table_name=table_name, constraint_name=constraint_name)
    ]


def _index_exists(conn: Connection, index_name: str) -> bool:
    return bool(
        conn.execute(
            text("SELECT to_regclass(:index_name) IS NOT NULL"),
            {"index_name": index_name},
        ).scalar_one()
    )


def _constraint_exists(conn: Connection, *, table_name: str, constraint_name: str) -> bool:
    return bool(
        conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = :constraint_name
                      AND conrelid = to_regclass(:table_name)
                )
                """,
            ),
            {"table_name": table_name, "constraint_name": constraint_name},
        ).scalar_one()
    )
