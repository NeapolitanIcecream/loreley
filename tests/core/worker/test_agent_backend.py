from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from loreley.config import Settings
from loreley.core.worker import agent_backend
from loreley.core.worker.agent_backend import (
    CodexCliBackend,
    CursorCliBackend,
    StructuredAgentTask,
    cursor_backend_from_settings,
    load_agent_backend,
    resolve_schema_mode,
)


def test_resolve_schema_mode_honours_config_and_api_spec() -> None:
    assert resolve_schema_mode("native", "chat_completions") == "native"
    assert resolve_schema_mode("auto", "chat_completions") == "prompt"
    assert resolve_schema_mode("auto", "responses") == "native"


def test_validate_workdir_requires_git_repo(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    with pytest.raises(RuntimeError):
        agent_backend._validate_workdir(  # type: ignore[attr-defined]
            repo_dir,
            error_cls=RuntimeError,
            agent_name="test",
        )

    git_dir = repo_dir / ".git"
    git_dir.mkdir()
    resolved = agent_backend._validate_workdir(  # type: ignore[attr-defined]
        repo_dir,
        error_cls=RuntimeError,
        agent_name="test",
    )
    assert resolved == repo_dir.resolve()


def test_materialise_schema_writes_json(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    path = agent_backend._materialise_schema_to_temp(  # type: ignore[attr-defined]
        schema,
        error_cls=RuntimeError,
    )
    try:
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == schema
    finally:
        path.unlink(missing_ok=True)


def test_load_agent_backend_supports_instance_and_factory(monkeypatch) -> None:
    module = types.ModuleType("dummy_backend_mod")

    class DummyBackend:
        def run(self, task, working_dir):  # pragma: no cover - trivial
            return (task, working_dir)

    module.backend_instance = DummyBackend()

    def backend_factory():
        return DummyBackend()

    module.backend_factory = backend_factory
    sys.modules[module.__name__] = module

    instance = load_agent_backend("dummy_backend_mod.backend_instance", label="test")
    assert instance is module.backend_instance

    factory_instance = load_agent_backend("dummy_backend_mod:backend_factory", label="test")
    assert isinstance(factory_instance, DummyBackend)

    with pytest.raises(RuntimeError):
        load_agent_backend("dummy_backend_mod.missing", label="test")


def test_codex_cli_backend_runs_and_cleans_schema(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    schema_path = tmp_path / "schema.json"

    def fake_materialise(schema, *, error_cls):  # noqa: ANN001
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        return schema_path

    monkeypatch.setattr(agent_backend, "_materialise_schema_to_temp", fake_materialise)

    captured: dict[str, object] = {}

    def fake_run(command, cwd, env, input, text, capture_output, timeout, check):  # noqa: ANN001
        captured.update(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "input": input,
                "timeout": timeout,
            }
        )
        return types.SimpleNamespace(stdout="{}", stderr="", returncode=0)

    monkeypatch.setattr(agent_backend.subprocess, "run", fake_run)

    backend = CodexCliBackend(
        bin="codex",
        profile=None,
        timeout_seconds=5,
        extra_env={"A": "1"},
        schema_override=None,
        error_cls=RuntimeError,
        full_auto=True,
    )

    task = StructuredAgentTask(
        name="code",
        prompt="do things",
        schema={"foo": "bar"},
        schema_mode="native",
    )

    invocation = backend.run(task, working_dir=repo_dir)

    assert "--output-schema" in invocation.command
    assert "--full-auto" in invocation.command
    assert str(schema_path) in invocation.command
    assert captured["cwd"] == str(repo_dir.resolve())
    assert captured["input"] == "do things"
    assert captured["env"] and captured["env"]["A"] == "1"
    assert not schema_path.exists()


def test_codex_cli_backend_raises_on_failure(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    schema_path = tmp_path / "schema.json"

    def fake_materialise(schema, *, error_cls):  # noqa: ANN001
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        return schema_path

    monkeypatch.setattr(agent_backend, "_materialise_schema_to_temp", fake_materialise)

    def fake_run(*args, **kwargs):  # noqa: ANN001, ANN002
        return types.SimpleNamespace(stdout="", stderr="boom", returncode=1)

    monkeypatch.setattr(agent_backend.subprocess, "run", fake_run)

    backend = CodexCliBackend(
        bin="codex",
        profile=None,
        timeout_seconds=5,
        extra_env={},
        schema_override=None,
        error_cls=RuntimeError,
        full_auto=False,
    )

    task = StructuredAgentTask(
        name="code",
        prompt="run",
        schema={"foo": "bar"},
        schema_mode="native",
    )

    with pytest.raises(RuntimeError):
        backend.run(task, working_dir=repo_dir)
    assert not schema_path.exists()


def test_cursor_cli_backend_builds_command(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        captured.update({"command": command, "cwd": cwd, "env": env, "timeout": timeout})
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(agent_backend.subprocess, "run", fake_run)

    backend = CursorCliBackend(
        bin="cursor-agent",
        model="cursor-model",
        timeout_seconds=10,
        extra_env={"X": "1"},
        output_format="json",
        force=False,
        error_cls=RuntimeError,
    )

    task = StructuredAgentTask(
        name="cursor",
        prompt="do it",
        schema=None,
        schema_mode="none",
    )

    invocation = backend.run(task, working_dir=repo_dir)

    command_list = list(invocation.command)
    assert "-p" in command_list and "do it" in command_list
    assert "--model" in command_list and "cursor-model" in command_list
    assert "--output-format" in command_list and "json" in command_list
    assert "--force" not in command_list
    assert captured["env"] and captured["env"]["X"] == "1"
    assert captured["cwd"] == str(repo_dir.resolve())
    assert invocation.stdout == "ok"


def test_cursor_backend_from_settings_uses_defaults(settings: Settings) -> None:
    settings.worker_cursor_model = "custom-model"
    settings.worker_cursor_force = False

    backend = cursor_backend_from_settings(settings=settings, error_cls=RuntimeError)

    assert isinstance(backend, CursorCliBackend)
    assert backend.model == "custom-model"
    assert backend.force is False
