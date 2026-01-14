from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from loreley.config import Settings
from loreley.core.worker.agent import (
    AgentInvocation,
    StructuredAgentTask,
    coerce_structured_output,
    load_agent_backend,
    materialise_schema_to_temp,
    resolve_schema_mode,
    run_structured_agent_task,
    validate_workdir,
)
from loreley.core.worker.agent.backends import (
    CodexCliBackend,
    CursorCliBackend,
    DEFAULT_CURSOR_MODEL,
    codex_cli,
    cursor_backend_from_settings,
    cursor_cli,
)


def test_resolve_schema_mode_honours_config_and_api_spec() -> None:
    assert resolve_schema_mode("native", "chat_completions") == "native"
    assert resolve_schema_mode("auto", "chat_completions") == "prompt"
    assert resolve_schema_mode("auto", "responses") == "native"


def test_validate_workdir_requires_git_repo(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    with pytest.raises(RuntimeError):
        validate_workdir(
            repo_dir,
            error_cls=RuntimeError,
            agent_name="test",
        )

    git_dir = repo_dir / ".git"
    git_dir.mkdir()
    resolved = validate_workdir(
        repo_dir,
        error_cls=RuntimeError,
        agent_name="test",
    )
    assert resolved == repo_dir.resolve()


def test_materialise_schema_writes_json(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    path = materialise_schema_to_temp(
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
    module: Any = types.ModuleType("dummy_backend_mod")

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

    monkeypatch.setattr(codex_cli, "materialise_schema_to_temp", fake_materialise)

    captured: dict[str, Any] = {}

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

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

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

    monkeypatch.setattr(codex_cli, "materialise_schema_to_temp", fake_materialise)

    def fake_run(*args, **kwargs):  # noqa: ANN001, ANN002
        return types.SimpleNamespace(stdout="", stderr="boom", returncode=1)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

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

    captured: dict[str, Any] = {}

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        captured.update({"command": command, "cwd": cwd, "env": env, "timeout": timeout})
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(cursor_cli.subprocess, "run", fake_run)

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


def test_import_order_is_safe_for_agent_backends_without_reexports() -> None:
    code = "\n".join(
        [
            "import loreley.core.worker.agent.backends.codex_cli",
            "import loreley.core.worker.agent.backends.cursor_cli",
            "import loreley.core.worker.agent.backends as backends",
            "from loreley.core.worker.agent.backends import (",
            "    CodexCliBackend,",
            "    CursorCliBackend,",
            "    DEFAULT_CURSOR_MODEL,",
            "    cursor_backend_from_settings,",
            ")",
            "import loreley.core.worker.agent as agent",
            "assert CodexCliBackend is backends.CodexCliBackend",
            "assert CursorCliBackend is backends.CursorCliBackend",
            "assert isinstance(DEFAULT_CURSOR_MODEL, str) and DEFAULT_CURSOR_MODEL",
            "assert callable(cursor_backend_from_settings)",
            "assert hasattr(agent, 'load_agent_backend')",
            "assert not hasattr(agent, 'CodexCliBackend')",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_coerce_structured_output_honours_validation_mode() -> None:
    called = {"parse": 0, "build": 0, "on_error": 0}

    def parse(stdout: str) -> dict[str, object]:
        called["parse"] += 1
        return json.loads(stdout)

    def build_from_freeform(stdout: str) -> dict[str, object]:
        called["build"] += 1
        return {"raw": stdout}

    def on_parse_error(exc: Exception) -> None:  # noqa: ARG001 - behaviour test
        called["on_error"] += 1

    with pytest.raises(json.JSONDecodeError):
        coerce_structured_output(
            validation_mode="strict",
            stdout="not-json",
            parse=parse,
            build_from_freeform=build_from_freeform,
            parse_exceptions=(json.JSONDecodeError,),
        )

    value = coerce_structured_output(
        validation_mode="lenient",
        stdout="not-json",
        parse=parse,
        build_from_freeform=build_from_freeform,
        on_parse_error=on_parse_error,
        parse_exceptions=(json.JSONDecodeError,),
    )
    assert value == {"raw": "not-json"}
    assert called["on_error"] == 1

    called2 = {"parse": 0, "build": 0}

    def parse2(stdout: str) -> str:  # pragma: no cover - should not run
        called2["parse"] += 1
        return stdout

    def build2(stdout: str) -> str:
        called2["build"] += 1
        return stdout

    value2 = coerce_structured_output(
        validation_mode="none",
        stdout="freeform",
        parse=parse2,
        build_from_freeform=build2,
        parse_exceptions=(Exception,),
    )
    assert value2 == "freeform"
    assert called2["parse"] == 0
    assert called2["build"] == 1


def test_run_structured_agent_task_retries_on_post_check(tmp_path: Path) -> None:
    class DummyBackend:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: StructuredAgentTask, *, working_dir: Path) -> AgentInvocation:  # noqa: ARG002
            self.calls += 1
            return AgentInvocation(
                command=("dummy", str(self.calls)),
                stdout=str(self.calls),
                stderr="",
                duration_seconds=0.0,
            )

    backend = DummyBackend()
    task = StructuredAgentTask(name="test", prompt="hi", schema=None, schema_mode="none")

    debug_events: list[tuple[int, str | None, int | None, str | None]] = []

    def debug_hook(
        attempt: int,
        invocation: AgentInvocation | None,
        result: int | None,
        error: Exception | None,
    ) -> None:
        debug_events.append(
            (
                attempt,
                invocation.stdout if invocation else None,
                result,
                type(error).__name__ if error else None,
            )
        )

    def post_check(_invocation: AgentInvocation, result: int) -> Exception | None:
        if result < 2:
            return RuntimeError("too-small")
        return None

    value, invocation, attempts = run_structured_agent_task(
        backend=backend,
        task=task,
        working_dir=tmp_path,
        max_attempts=3,
        coerce_result=lambda inv: int(inv.stdout),
        retryable_exceptions=(ValueError,),
        error_cls=RuntimeError,
        error_message="should-not-fail",
        debug_hook=debug_hook,
        post_check=post_check,
    )

    assert value == 2
    assert invocation.stdout == "2"
    assert attempts == 2
    assert backend.calls == 2
    assert debug_events[0][3] == "RuntimeError"
    assert debug_events[1][3] is None
