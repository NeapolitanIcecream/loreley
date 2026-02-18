from __future__ import annotations

import json

import pytest

from loreley.cli import main
from tests.support import TestSettings


def _make_settings() -> TestSettings:
    return TestSettings(
        OPENAI_API_KEY="sk-test-secret",
        OPENAI_BASE_URL="https://gateway.example.com/v1",
        DB_PASSWORD="db-secret-password",
        DB_HOST="db.internal",
        DB_PORT=5433,
        DB_NAME="loreley_dev",
        TASKS_REDIS_PASSWORD="redis-secret",
        TASKS_REDIS_URL="redis://:redis-secret@redis.internal:6380/2",
        WORKER_KILOCODE_OPENAI_API_KEY="kilo-secret",
        WORKER_REPO_REMOTE_URL="https://token@example.com/repo.git",
    )


def test_config_dump_json_masks_secrets_by_default(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)

    code = main(["config", "dump", "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["openai_api_key"] == "***"
    assert payload["tasks_redis_password"] == "***"
    assert payload["database_dsn"].startswith("postgresql+psycopg://loreley:***@")
    assert payload["worker_repo_remote_url"] == "https://example.com/repo.git"


def test_config_dump_json_can_disable_secret_masking(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)

    code = main(["config", "dump", "--json", "--no-mask-secrets"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["openai_api_key"] == "sk-test-secret"
    assert payload["tasks_redis_password"] == "redis-secret"
    assert "db-secret-password" in payload["database_dsn"]


def test_config_dump_yaml_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = pytest.importorskip("yaml")

    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)

    code = main(["config", "dump", "--yaml"])
    captured = capsys.readouterr()

    assert code == 0
    payload = yaml.safe_load(captured.out)
    assert payload["openai_base_url"] == "https://gateway.example.com/v1"
    assert payload["tasks_redis_url"] == "redis://redis.internal:6380/2"
    assert "scheduler_poll_interval_seconds" in payload


def test_config_dump_rejects_multiple_output_formats(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _make_settings()
    monkeypatch.setattr("loreley.cli.get_settings", lambda: settings)

    code = main(["config", "dump", "--json", "--yaml"])
    captured = capsys.readouterr()

    assert code == 1
    assert "choose exactly one output format" in captured.out.lower()
