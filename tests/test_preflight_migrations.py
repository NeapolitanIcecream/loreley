from __future__ import annotations

from types import SimpleNamespace

import pytest

from loreley.db.base import INSTANCE_SCHEMA_VERSION
from loreley.db.migrations.runner import SchemaStatus
from loreley.preflight import CheckResult, check_instance_marker, preflight_scheduler, preflight_worker
from tests.support import TestSettings


def test_preflight_accepts_migratable_schema_without_reset_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = TestSettings(DB_AUTO_MIGRATE=True)
    monkeypatch.setattr("loreley.db.base.get_engine", lambda: SimpleNamespace(name="engine"))
    monkeypatch.setattr(
        "loreley.db.migrations.runner.describe_schema",
        lambda **_kwargs: SchemaStatus(
            schema_version=5,
            target_version=INSTANCE_SCHEMA_VERSION,
            state="migratable",
            needs_migration=True,
        ),
    )
    monkeypatch.setattr("loreley.db.migrations.runner.validate_database_identity", lambda **_kwargs: None)

    result = check_instance_marker(schema_version=INSTANCE_SCHEMA_VERSION, settings=settings)

    assert result.status == "ok"
    assert "migratable" in result.details
    assert "reset-db" not in result.details


def test_preflight_fails_with_migrate_hint_when_auto_migrate_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = TestSettings(DB_AUTO_MIGRATE=False)
    monkeypatch.setattr("loreley.db.base.get_engine", lambda: SimpleNamespace(name="engine"))
    monkeypatch.setattr(
        "loreley.db.migrations.runner.describe_schema",
        lambda **_kwargs: SchemaStatus(
            schema_version=5,
            target_version=INSTANCE_SCHEMA_VERSION,
            state="migratable",
            needs_migration=True,
        ),
    )
    monkeypatch.setattr("loreley.db.migrations.runner.validate_database_identity", lambda **_kwargs: None)

    result = check_instance_marker(schema_version=INSTANCE_SCHEMA_VERSION, settings=settings)

    assert result.status == "fail"
    assert "uv run loreley db migrate" in result.details


def test_preflight_accepts_empty_database_when_auto_migrate_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = TestSettings(
        DB_AUTO_MIGRATE=True,
        EXPERIMENT_ID="preflight-fresh",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="deadbeef",
    )
    monkeypatch.setattr("loreley.db.base.get_engine", lambda: SimpleNamespace(name="engine"))
    monkeypatch.setattr(
        "loreley.db.migrations.runner.describe_schema",
        lambda **_kwargs: SchemaStatus(
            schema_version=None,
            target_version=INSTANCE_SCHEMA_VERSION,
            state="fresh",
            needs_migration=False,
        ),
    )

    result = check_instance_marker(schema_version=INSTANCE_SCHEMA_VERSION, settings=settings)

    assert result.status == "ok"
    assert "initialize" in result.details


@pytest.mark.parametrize(
    ("settings", "expected_detail"),
    (
        (
            TestSettings(
                DB_AUTO_MIGRATE=True,
                EXPERIMENT_ID=None,
                MAPELITES_EXPERIMENT_ROOT_COMMIT="deadbeef",
            ),
            "EXPERIMENT_ID is required",
        ),
        (
            TestSettings(
                DB_AUTO_MIGRATE=True,
                EXPERIMENT_ID="preflight-fresh",
                MAPELITES_EXPERIMENT_ROOT_COMMIT=None,
            ),
            "MAPELITES_EXPERIMENT_ROOT_COMMIT is required",
        ),
    ),
)
def test_preflight_fails_empty_database_when_marker_identity_cannot_be_seeded(
    monkeypatch: pytest.MonkeyPatch,
    settings: TestSettings,
    expected_detail: str,
) -> None:
    """Regression: fresh auto-migration must fail fast if startup cannot seed the marker."""

    monkeypatch.setattr("loreley.db.base.get_engine", lambda: SimpleNamespace(name="engine"))
    monkeypatch.setattr(
        "loreley.db.migrations.runner.describe_schema",
        lambda **_kwargs: SchemaStatus(
            schema_version=None,
            target_version=INSTANCE_SCHEMA_VERSION,
            state="fresh",
            needs_migration=False,
        ),
    )

    result = check_instance_marker(schema_version=INSTANCE_SCHEMA_VERSION, settings=settings)

    assert result.status == "fail"
    assert expected_detail in result.details


@pytest.mark.parametrize("preflight_func", (preflight_scheduler, preflight_worker))
def test_scheduler_and_worker_preflight_include_instance_marker_check(
    monkeypatch: pytest.MonkeyPatch,
    preflight_func,
) -> None:
    """Regression: non-API startup preflight must honor DB_AUTO_MIGRATE=false."""

    settings = TestSettings(
        DB_AUTO_MIGRATE=False,
        EXPERIMENT_ID="preflight-fresh",
        MAPELITES_EXPERIMENT_ROOT_COMMIT="deadbeef",
        MAPELITES_CODE_EMBEDDING_DIMENSIONS=8,
        SCHEDULER_MAX_TOTAL_JOBS=1,
        WORKER_REPO_REMOTE_URL="https://example.com/repo.git",
        WORKER_EVALUATOR_PLUGIN="tests.support_dynamic_provider:token_provider",
    )
    marker_result = CheckResult(
        "instance_metadata",
        "fail",
        "instance metadata is missing; run `uv run loreley db migrate`.",
    )

    monkeypatch.setattr(
        "loreley.preflight.check_database",
        lambda **_kwargs: CheckResult("database", "ok", "reachable"),
    )
    monkeypatch.setattr(
        "loreley.preflight.check_instance_marker",
        lambda **_kwargs: marker_result,
    )
    monkeypatch.setattr(
        "loreley.preflight.check_redis",
        lambda **_kwargs: CheckResult("redis", "ok", "reachable"),
    )
    monkeypatch.setattr(
        "loreley.preflight.check_git_repo",
        lambda *_args, **_kwargs: CheckResult("repo", "ok", "valid"),
    )
    monkeypatch.setattr(
        "loreley.preflight.check_binary",
        lambda *_args, **_kwargs: CheckResult("git", "ok", "available"),
    )
    monkeypatch.setattr(
        "loreley.preflight._check_openai_api_key_for_scheduler",
        lambda _settings: CheckResult("openai_api_key", "ok", "configured"),
    )
    monkeypatch.setattr(
        "loreley.preflight._check_openai_api_key_for_worker",
        lambda _settings: CheckResult("openai_api_key", "ok", "configured"),
    )
    monkeypatch.setattr(
        "loreley.preflight.check_embedding_dimensions",
        lambda _settings: CheckResult("embedding_dimensions", "ok", "configured"),
    )
    monkeypatch.setattr(
        "loreley.preflight.check_scheduler_max_total_jobs",
        lambda _settings: CheckResult("scheduler_max_total_jobs", "ok", "configured"),
    )
    monkeypatch.setattr(
        "loreley.preflight.check_campaign_program",
        lambda _settings: CheckResult("campaign_program", "ok", "configured"),
    )
    monkeypatch.setattr(
        "loreley.preflight.check_evaluator_plugin",
        lambda **_kwargs: CheckResult("worker_evaluator_plugin", "ok", "configured"),
    )
    monkeypatch.setattr("loreley.preflight._check_agent_backend", lambda **_kwargs: [])
    monkeypatch.setattr("loreley.preflight._check_dynamic_openai_agent_ttl", lambda _settings: [])

    results = preflight_func(settings, timeout_seconds=0.2)

    assert marker_result in results
