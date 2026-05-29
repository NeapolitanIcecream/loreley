from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from loreley.cli import main
from loreley.db.base import INSTANCE_SCHEMA_VERSION
from loreley.db.migrations.runner import MigrationResult, SchemaStatus
from tests.support import TestSettings


def _patch_cli_db_common(monkeypatch: pytest.MonkeyPatch) -> TestSettings:
    settings = TestSettings(
        EXPERIMENT_ID="db-cli",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="deadbeef",
    )
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    monkeypatch.setattr("loreley.db.base.get_engine", lambda: SimpleNamespace(name="engine"))
    return settings


def test_db_current_prints_script_friendly_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_cli_db_common(monkeypatch)
    monkeypatch.setattr(
        "loreley.db.migrations.runner.describe_schema",
        lambda **_kwargs: SchemaStatus(
            schema_version=5,
            target_version=INSTANCE_SCHEMA_VERSION,
            state="migratable",
            needs_migration=True,
            detail="schema_version=5 can migrate",
        ),
    )

    code = main(["db", "current"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.strip() == (
        f"schema_version=5 target={INSTANCE_SCHEMA_VERSION} state=migratable "
        "needs_migration=true"
    )


def test_db_current_json_includes_schema_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_cli_db_common(monkeypatch)
    monkeypatch.setattr(
        "loreley.db.migrations.runner.describe_schema",
        lambda **_kwargs: SchemaStatus(
            schema_version=INSTANCE_SCHEMA_VERSION,
            target_version=INSTANCE_SCHEMA_VERSION,
            state="current",
            needs_migration=False,
            detail="schema is current",
        ),
    )

    code = main(["db", "current", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["schema_version"] == INSTANCE_SCHEMA_VERSION
    assert payload["needs_migration"] is False


def test_db_current_json_stdout_stays_json_with_real_setup_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Regression: settings and engine console logs used to prefix the JSON payload."""

    from loreley.config import get_settings as cached_get_settings
    from loreley.db.base import get_engine as cached_get_engine

    cached_get_settings.cache_clear()
    cached_get_engine.cache_clear()
    request.addfinalizer(cached_get_settings.cache_clear)
    request.addfinalizer(cached_get_engine.cache_clear)
    monkeypatch.setenv("EXPERIMENT_ID", "db-cli")
    monkeypatch.setenv("MAPELITES_EXPERIMENT_ROOT_COMMIT", "deadbeef")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://loreley:secret@localhost:5432/loreley_cli")
    monkeypatch.setenv("LOGS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setattr("loreley.db.base.create_engine", lambda *_args, **_kwargs: SimpleNamespace(name="engine"))
    monkeypatch.setattr(
        "loreley.db.migrations.runner.describe_schema",
        lambda **_kwargs: SchemaStatus(
            schema_version=INSTANCE_SCHEMA_VERSION,
            target_version=INSTANCE_SCHEMA_VERSION,
            state="current",
            needs_migration=False,
            detail="schema is current",
        ),
    )

    code = main(["db", "current", "--json"])
    captured = capsys.readouterr()

    expected = {
        "detail": "schema is current",
        "needs_migration": False,
        "schema_version": INSTANCE_SCHEMA_VERSION,
        "state": "current",
        "target": INSTANCE_SCHEMA_VERSION,
    }
    assert code == 0
    assert captured.out == f"{json.dumps(expected, ensure_ascii=False, sort_keys=True)}\n"
    assert "Loaded settings" in captured.err
    assert "SQLAlchemy engine ready" in captured.err


def test_db_migrate_runs_explicit_migration_even_when_auto_disabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = TestSettings(
        EXPERIMENT_ID="db-cli",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="deadbeef",
        DB_AUTO_MIGRATE=False,
    )
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)
    monkeypatch.setattr("loreley.cli._configure_logging_or_exit", lambda **_kwargs: None)
    monkeypatch.setattr("loreley.db.base.get_engine", lambda: SimpleNamespace(name="engine"))
    calls: list[dict[str, object]] = []

    def fake_ensure(**kwargs):
        calls.append(kwargs)
        return MigrationResult(
            from_version=5,
            to_version=INSTANCE_SCHEMA_VERSION,
            applied_versions=(6, 7, 8, 9, 10, 11, 12, 13),
        )

    monkeypatch.setattr("loreley.db.migrations.runner.ensure_schema_current", fake_ensure)
    monkeypatch.setattr(
        "loreley.db.migrations.runner.validate_database_schema",
        lambda **_kwargs: SchemaStatus(
            schema_version=INSTANCE_SCHEMA_VERSION,
            target_version=INSTANCE_SCHEMA_VERSION,
            state="current",
            needs_migration=False,
        ),
    )

    code = main(["db", "migrate"])
    captured = capsys.readouterr()

    assert code == 0
    assert calls[0]["auto_migrate"] is True
    assert (
        f"from=5 to={INSTANCE_SCHEMA_VERSION} applied=6,7,8,9,10,11,12,13 fresh=false"
        in captured.out
    )


def test_db_validate_reports_current_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_cli_db_common(monkeypatch)
    monkeypatch.setattr(
        "loreley.db.migrations.runner.validate_database_schema",
        lambda **_kwargs: SchemaStatus(
            schema_version=INSTANCE_SCHEMA_VERSION,
            target_version=INSTANCE_SCHEMA_VERSION,
            state="current",
            needs_migration=False,
        ),
    )

    code = main(["db", "validate"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.strip() == (
        f"valid schema_version={INSTANCE_SCHEMA_VERSION} target={INSTANCE_SCHEMA_VERSION}"
    )
