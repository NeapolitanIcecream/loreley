from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sqlalchemy.engine import Connection

from loreley.config import Settings


class RegistryError(RuntimeError):
    """Raised when no native migration path exists for a schema version."""


@dataclass(frozen=True, slots=True)
class SchemaMigration:
    """One integer schema-version migration."""

    from_version: int
    to_version: int
    name: str
    upgrade: Callable[[Connection, Settings], None]


def _load_migrations() -> tuple[SchemaMigration, ...]:
    from loreley.db.migrations.versions import (
        v0006_evaluation_artifacts,
        v0007_failed_candidate_repair_pool,
        v0008_campaign_programs,
        v0009_campaign_baselines,
        v0010_operator_tasks,
        v0011_operator_active_baseline_guard,
        v0012_agent_actions_and_cleanup,
        v0013_llm_usage_events,
        v0014_embedding_cache_manifests,
        v0015_multiobjective_islands,
    )

    return (
        v0006_evaluation_artifacts.MIGRATION,
        v0007_failed_candidate_repair_pool.MIGRATION,
        v0008_campaign_programs.MIGRATION,
        v0009_campaign_baselines.MIGRATION,
        v0010_operator_tasks.MIGRATION,
        v0011_operator_active_baseline_guard.MIGRATION,
        v0012_agent_actions_and_cleanup.MIGRATION,
        v0013_llm_usage_events.MIGRATION,
        v0014_embedding_cache_manifests.MIGRATION,
        v0015_multiobjective_islands.MIGRATION,
    )


MIGRATIONS: tuple[SchemaMigration, ...] = _load_migrations()


def migration_path(
    from_version: int,
    to_version: int,
    *,
    migrations: Sequence[SchemaMigration] = MIGRATIONS,
) -> tuple[SchemaMigration, ...]:
    """Return the ordered migration chain from ``from_version`` to ``to_version``."""

    current = int(from_version)
    target = int(to_version)
    if current == target:
        return ()
    if current > target:
        raise RegistryError(
            f"Database schema_version={current} is newer than target={target}.",
        )

    by_from = {migration.from_version: migration for migration in migrations}
    path: list[SchemaMigration] = []
    while current < target:
        migration = by_from.get(current)
        if migration is None:
            raise RegistryError(
                f"No Loreley native migration path from schema_version={current} to {target}.",
            )
        if migration.to_version <= migration.from_version:
            raise RegistryError(
                f"Invalid migration {migration.name}: "
                f"{migration.from_version} -> {migration.to_version}.",
            )
        path.append(migration)
        current = migration.to_version

    if current != target:
        raise RegistryError(
            f"No exact Loreley native migration path from schema_version={from_version} to {target}.",
        )
    return tuple(path)
