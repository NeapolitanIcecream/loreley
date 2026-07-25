from __future__ import annotations

from pathlib import Path

import pytest

from tools.migrate_v15_config import main, migrate_env_text


def test_migrate_v15_config_converts_scalar_archive_settings_once() -> None:
    source = """\
# experiment
MAPELITES_DEFAULT_ISLAND_ID=circle_packing
MAPELITES_FITNESS_METRIC=sum_radii
MAPELITES_FITNESS_HIGHER_IS_BETTER=false
MAPELITES_FITNESS_FLOOR=-100
MAPELITES_ARCHIVE_EPSILON=0.001
MAPELITES_ARCHIVE_LEARNING_RATE=1
MAPELITES_ARCHIVE_THRESHOLD_MIN=-inf
MAPELITES_ARCHIVE_QD_SCORE_OFFSET=0
OTHER_SETTING=keep
"""

    migrated = migrate_env_text(source)

    assert 'MAPELITES_ISLANDS=["circle_packing"]' in migrated
    assert (
        'MAPELITES_OBJECTIVES=[{"name":"sum_radii","direction":"min"}]'
        in migrated
    )
    assert "MAPELITES_PARETO_EPSILON=0.001" in migrated
    assert "MAPELITES_DEFAULT_ISLAND_ID" not in migrated
    assert "MAPELITES_FITNESS_" not in migrated
    assert "MAPELITES_ARCHIVE_" not in migrated
    assert "OTHER_SETTING=keep" in migrated
    assert migrate_env_text(migrated) == migrated


def test_migrate_v15_config_does_not_overwrite_explicit_new_contracts() -> None:
    source = """\
MAPELITES_ISLANDS=["a","b"]
MAPELITES_DEFAULT_ISLAND_ID=legacy
MAPELITES_OBJECTIVES=[{"name":"quality","direction":"max"}]
MAPELITES_FITNESS_METRIC=legacy_score
MAPELITES_FITNESS_HIGHER_IS_BETTER=true
MAPELITES_PARETO_EPSILON=0.5
MAPELITES_ARCHIVE_EPSILON=0.1
"""

    migrated = migrate_env_text(source)

    assert migrated.count("MAPELITES_ISLANDS=") == 1
    assert migrated.count("MAPELITES_OBJECTIVES=") == 1
    assert migrated.count("MAPELITES_PARETO_EPSILON=") == 1
    assert "legacy_score" not in migrated


@pytest.mark.parametrize(
    ("legacy_value", "expected_island"),
    [
        ("", "main"),
        ("   ", "main"),
        ('"   "', "main"),
        ("'   '", "main"),
        (" # empty legacy override", "main"),
        ('"  explore  "', "explore"),
    ],
)
def test_migration_normalizes_the_legacy_default_island(
    legacy_value: str,
    expected_island: str,
) -> None:
    migrated = migrate_env_text(
        f"MAPELITES_DEFAULT_ISLAND_ID={legacy_value}\n"
        "OTHER_SETTING=keep\n"
    )

    assert migrated == (
        f'MAPELITES_ISLANDS=["{expected_island}"]\n'
        "OTHER_SETTING=keep\n"
    )


def test_migration_preserves_a_direction_override_of_the_legacy_default_metric() -> None:
    migrated = migrate_env_text(
        "MAPELITES_FITNESS_HIGHER_IS_BETTER=false\n"
    )

    assert migrated == (
        'MAPELITES_OBJECTIVES=[{"name":"composite_score","direction":"min"}]\n'
    )


def test_migration_ignores_unquoted_dotenv_inline_comments() -> None:
    migrated = migrate_env_text(
        "MAPELITES_FITNESS_METRIC=score # primary metric\n"
        "MAPELITES_FITNESS_HIGHER_IS_BETTER=false # minimize\n"
    )

    assert migrated == (
        'MAPELITES_OBJECTIVES=[{"name":"score","direction":"min"}]\n'
    )


def test_migration_preserves_hashes_inside_quoted_dotenv_values() -> None:
    migrated = migrate_env_text(
        'MAPELITES_FITNESS_METRIC="score # primary"\n'
    )

    assert migrated == (
        'MAPELITES_OBJECTIVES=[{"name":"score # primary","direction":"max"}]\n'
    )


def test_migration_command_writes_a_separate_file_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.env"
    target = tmp_path / "v15.env"
    legacy_text = "MAPELITES_FITNESS_METRIC=quality\n"
    source.write_text(legacy_text, encoding="utf-8")

    assert main([str(source), "--output", str(target)]) == 0

    assert source.read_text(encoding="utf-8") == legacy_text
    assert target.read_text(encoding="utf-8") == (
        'MAPELITES_OBJECTIVES=[{"name":"quality","direction":"max"}]\n'
    )


def test_migration_command_can_replace_the_source_in_place(tmp_path: Path) -> None:
    source = tmp_path / "legacy.env"
    source.write_text("MAPELITES_DEFAULT_ISLAND_ID=explore\n", encoding="utf-8")

    assert main([str(source), "--in-place"]) == 0

    assert source.read_text(encoding="utf-8") == (
        'MAPELITES_ISLANDS=["explore"]\n'
    )
