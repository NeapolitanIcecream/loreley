from __future__ import annotations

import pytest

from loreley.db.migrations.registry import RegistryError, migration_path


def test_migration_registry_orders_v5_to_current_chain() -> None:
    path = migration_path(5, 13)

    assert [(migration.from_version, migration.to_version) for migration in path] == [
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
    ]


def test_migration_registry_rejects_unsupported_old_version() -> None:
    with pytest.raises(RegistryError, match="No Loreley native migration path"):
        migration_path(4, 13)


def test_migration_registry_rejects_future_version() -> None:
    with pytest.raises(RegistryError, match="newer than target"):
        migration_path(14, 13)
