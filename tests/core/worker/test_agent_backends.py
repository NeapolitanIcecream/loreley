from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from loreley.config import Settings
from loreley.core.usage import LLMUsageEventPayload
from loreley.core.worker.agent import (
    AgentInvocation,
    AgentTask,
    load_agent_backend,
    run_agent_task,
    validate_workdir,
)
from loreley.core.worker.agent.backends import (
    CodexCliBackend,
    CursorCliBackend,
    DEFAULT_CURSOR_MODEL,
    KilocodeCliBackend,
    codex_coding_backend,
    codex_planning_backend,
    cursor_backend,
    cursor_coding_backend,
    cursor_planning_backend,
    kilocode_backend,
    kilocode_coding_backend,
    kilocode_planning_backend,
)
from loreley.core.worker.agent.backends import codex_cli, cursor_cli, kilocode_cli


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


def test_load_agent_backend_requires_no_arg_factory() -> None:
    module: Any = types.ModuleType("dummy_backend_mod_settings")

    class DummyBackend:
        def run(self, task, working_dir):  # pragma: no cover - trivial
            return (task, working_dir)

    def backend_factory(*, settings: Settings) -> DummyBackend:  # noqa: ARG001 - behaviour test
        return DummyBackend()

    module.backend_factory = backend_factory
    sys.modules[module.__name__] = module

    with pytest.raises(TypeError):
        load_agent_backend("dummy_backend_mod_settings:backend_factory", label="test")


def test_codex_cli_backend_builds_noninteractive_command_and_reads_last_message(
    tmp_path: Path, monkeypatch
) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    captured: dict[str, Any] = {}

    def fake_run(command, cwd, env, input, text, capture_output, timeout, check):  # noqa: ANN001
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("## Summary\n- Completed safely.\n", encoding="utf-8")
        captured.update(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "input": input,
                "timeout": timeout,
            }
        )
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    backend = CodexCliBackend(
        bin="codex",
        model="gpt-5.4",
        profile="prof",
        timeout_seconds=5,
        extra_env={"A": "1"},
        error_cls=RuntimeError,
        full_auto=True,
    )

    task = AgentTask(
        name="code",
        prompt="do things",
    )

    invocation = backend.run(task, working_dir=repo_dir)

    command_list = list(invocation.command)
    exec_index = command_list.index("exec")
    assert command_list[:3] == ["codex", "--model", "gpt-5.4"]
    assert command_list[exec_index - 6 : exec_index] == [
        "--profile",
        "prof",
        "-a",
        "never",
        "--sandbox",
        "workspace-write",
    ]
    assert "--full-auto" not in command_list
    assert "--ephemeral" in command_list
    assert "-a" in command_list and "never" in command_list
    assert "--sandbox" in command_list and "workspace-write" in command_list
    assert "--color" in command_list and "never" in command_list
    assert "--json" in command_list
    assert "--output-last-message" in command_list
    assert "--profile" in command_list and "prof" in command_list
    assert "--model" in command_list and "gpt-5.4" in command_list
    assert "--output-schema" not in command_list
    assert captured["cwd"] == str(repo_dir.resolve())
    assert captured["input"] == "do things"
    assert captured["env"] and captured["env"]["A"] == "1"
    assert captured["env"]["CODEX_QUIET_MODE"] == "1"
    assert invocation.stdout == "## Summary\n- Completed safely."


def test_codex_cli_backend_allows_omitting_model_and_skips_model_flag() -> None:
    backend = CodexCliBackend(
        bin="codex",
        timeout_seconds=5,
        extra_env={},
        error_cls=RuntimeError,
        profile="prof",
    )

    command = backend._build_command(  # noqa: SLF001 - spec-level assertion
        output_last_message_path=None,
    )

    assert command[:1] == ["codex"]
    assert "--model" not in command
    assert command[1:7] == [
        "--profile",
        "prof",
        "-a",
        "never",
        "--sandbox",
        "read-only",
    ]
    assert command[7:] == ["exec", "--ephemeral", "--color", "never", "--json"]


def test_codex_cli_backend_isolates_codex_home_for_read_only_without_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = CodexCliBackend(
        bin="codex",
        model="gpt-5.4",
        profile=None,
        timeout_seconds=5,
        extra_env={},
        error_cls=RuntimeError,
    )
    source_home = tmp_path / "source-home"
    source_home.mkdir()
    auth_path = source_home / "auth.json"
    auth_path.write_text('{"token":"secret"}', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(source_home))

    env = {"CODEX_QUIET_MODE": "1"}
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    worktree = tmp_path / "repo"
    worktree.mkdir()

    backend._prepare_isolated_home(  # noqa: SLF001 - spec-level assertion
        env=env,
        temp_dir=temp_dir,
        worktree=worktree,
    )

    isolated_home = Path(env["CODEX_HOME"])
    assert isolated_home != source_home
    assert isolated_home.parent == temp_dir
    assert (isolated_home / "auth.json").read_text(encoding="utf-8") == '{"token":"secret"}'
    assert not (isolated_home / "config.toml").exists()


def test_codex_cli_backend_marks_workspace_write_worktree_trusted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = CodexCliBackend(
        bin="codex",
        model="gpt-5.4",
        profile=None,
        timeout_seconds=5,
        extra_env={},
        error_cls=RuntimeError,
        sandbox="workspace-write",
    )
    source_home = tmp_path / "source-home"
    source_home.mkdir()
    auth_path = source_home / "auth.json"
    auth_path.write_text('{"token":"secret"}', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(source_home))

    env = {"CODEX_QUIET_MODE": "1"}
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    worktree = tmp_path / "repo"
    worktree.mkdir()

    backend._prepare_isolated_home(  # noqa: SLF001 - spec-level assertion
        env=env,
        temp_dir=temp_dir,
        worktree=worktree,
    )

    isolated_home = Path(env["CODEX_HOME"])
    config_text = (isolated_home / "config.toml").read_text(encoding="utf-8")
    assert "[projects." in config_text
    assert str(worktree.resolve()) in config_text
    assert "trust_level = \"trusted\"" in config_text


def test_codex_cli_backend_raises_on_failure(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN002
        return types.SimpleNamespace(stdout="", stderr="boom", returncode=1)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    backend = CodexCliBackend(
        bin="codex",
        model=None,
        profile=None,
        timeout_seconds=5,
        extra_env={},
        error_cls=RuntimeError,
        full_auto=False,
    )

    task = AgentTask(
        name="code",
        prompt="run",
    )

    with pytest.raises(RuntimeError):
        backend.run(task, working_dir=repo_dir)


def test_codex_cli_backend_defaults_to_read_only_for_planning(
    tmp_path: Path, monkeypatch
) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    captured: dict[str, Any] = {}

    def fake_run(command, cwd, env, input, text, capture_output, timeout, check):  # noqa: ANN001
        captured["command"] = command
        return types.SimpleNamespace(stdout="plan", stderr="", returncode=0)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    backend = CodexCliBackend(
        bin="codex",
        model=None,
        profile=None,
        timeout_seconds=5,
        extra_env={},
        error_cls=RuntimeError,
        full_auto=False,
    )

    task = AgentTask(name="planning", prompt="plan")
    backend.run(task, working_dir=repo_dir)

    command_list = list(captured["command"])
    assert "--sandbox" in command_list and "read-only" in command_list
    assert "-a" in command_list and "never" in command_list


def test_codex_cli_backend_parses_json_token_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(command, **_kwargs):  # noqa: ANN001
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("## Summary\n- Done.\n", encoding="utf-8")
        payload = {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 25,
                        "output_tokens": 40,
                        "reasoning_output_tokens": 10,
                        "total_tokens": 140,
                    }
                },
            },
        }
        return types.SimpleNamespace(stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    job_id = uuid4()
    run_token = uuid4()
    backend = CodexCliBackend(
        bin="codex",
        model="gpt-codex",
        timeout_seconds=5,
        extra_env={},
        error_cls=RuntimeError,
        full_auto=False,
    )

    invocation = backend.run(
        AgentTask(
            name="planning",
            prompt="plan",
            job_id=job_id,
            run_token=run_token,
            phase="planning",
            attempt=3,
        ),
        working_dir=repo_dir,
    )

    assert invocation.stdout == "## Summary\n- Done."
    assert len(invocation.usage_events) == 1
    event = invocation.usage_events[0]
    assert event.source == "codex_cli"
    assert event.phase == "planning"
    assert event.job_id == job_id
    assert event.run_token == run_token
    assert event.input_tokens == 100
    assert event.cached_input_tokens == 25
    assert event.output_tokens == 40
    assert event.reasoning_output_tokens == 10
    assert event.external_usage_id == f"codex:{job_id}:{run_token}:planning:attempt:3"


def test_codex_backend_uses_env_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from loreley.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("WORKER_PLANNING_CODEX_MODEL", "gpt-5.4")
    monkeypatch.setenv("WORKER_CODING_CODEX_MODEL", "gpt-5.4")
    get_settings.cache_clear()

    planning = codex_planning_backend()
    coding = codex_coding_backend()

    assert planning.model == "gpt-5.4"
    assert planning.sandbox == "read-only"
    assert coding.model == "gpt-5.4"
    assert coding.sandbox == "workspace-write"

    get_settings.cache_clear()


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

    task = AgentTask(
        name="cursor",
        prompt="do it",
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


def test_cursor_backend_uses_env_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from loreley.config import get_settings

    # Ensure we do not reuse cached Settings across tests.
    get_settings.cache_clear()
    monkeypatch.setenv("WORKER_CURSOR_MODEL", "custom-model")
    monkeypatch.setenv("WORKER_CURSOR_FORCE", "false")
    get_settings.cache_clear()

    backend = cursor_backend()

    assert isinstance(backend, CursorCliBackend)
    assert backend.model == "custom-model"
    assert backend.force is False

    get_settings.cache_clear()


def test_cursor_coding_backend_is_retryable_in_worker_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker retry loop should treat Cursor CLI failures as CodingError."""
    from loreley.core.worker.coding import CodingError

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    calls = {"count": 0}

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        calls["count"] += 1
        return types.SimpleNamespace(stdout="", stderr="boom", returncode=1)

    monkeypatch.setattr(cursor_cli.subprocess, "run", fake_run)

    backend = cursor_cli.cursor_coding_backend()
    task = AgentTask(name="coding", prompt="do it")

    debug_events: list[tuple[int, str]] = []

    def debug_hook(
        attempt: int,
        _invocation: AgentInvocation | None,
        _result: Any | None,
        error: Exception | None,
    ) -> None:
        debug_events.append((attempt, type(error).__name__ if error else "None"))

    with pytest.raises(CodingError) as exc_info:
        run_agent_task(
            backend=backend,
            task=task,
            working_dir=repo_dir,
            max_attempts=2,
            coerce_result=lambda inv: inv.stdout,
            retryable_exceptions=(CodingError,),
            error_cls=CodingError,
            error_message="cursor backend should be retryable for coding",
            debug_hook=debug_hook,
        )

    assert calls["count"] == 2
    assert debug_events == [(1, "CodingError"), (2, "CodingError")]
    assert isinstance(exc_info.value.__cause__, CodingError)


def test_kilocode_planning_backend_is_retryable_in_worker_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker retry loop should treat Kilocode CLI failures as PlanningError."""
    from loreley.core.worker.planning import PlanningError

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    calls = {"count": 0}

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        calls["count"] += 1
        return types.SimpleNamespace(stdout="", stderr="connection failed", returncode=1)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = kilocode_cli.kilocode_planning_backend()
    task = AgentTask(name="planning", prompt="plan")

    debug_events: list[tuple[int, str]] = []

    def debug_hook(
        attempt: int,
        _invocation: AgentInvocation | None,
        _result: Any | None,
        error: Exception | None,
    ) -> None:
        debug_events.append((attempt, type(error).__name__ if error else "None"))

    with pytest.raises(PlanningError) as exc_info:
        run_agent_task(
            backend=backend,
            task=task,
            working_dir=repo_dir,
            max_attempts=2,
            coerce_result=lambda inv: inv.stdout,
            retryable_exceptions=(PlanningError,),
            error_cls=PlanningError,
            error_message="kilocode backend should be retryable for planning",
            debug_hook=debug_hook,
        )

    assert calls["count"] == 2
    assert debug_events == [(1, "PlanningError"), (2, "PlanningError")]
    assert isinstance(exc_info.value.__cause__, PlanningError)


def test_import_order_is_safe_for_agent_backends_without_reexports() -> None:
    code = "\n".join(
        [
            "import loreley.core.worker.agent.backends.codex_cli",
            "import loreley.core.worker.agent.backends.cursor_cli",
            "import loreley.core.worker.agent.backends.kilocode_cli",
            "import loreley.core.worker.agent.backends as backends",
            "from loreley.core.worker.agent.backends import (",
            "    CodexCliBackend,",
            "    CursorCliBackend,",
            "    DEFAULT_CURSOR_MODEL,",
            "    KilocodeCliBackend,",
            "    codex_coding_backend,",
            "    codex_planning_backend,",
            "    cursor_backend,",
            "    cursor_coding_backend,",
            "    cursor_planning_backend,",
            "    kilocode_backend,",
            "    kilocode_coding_backend,",
            "    kilocode_planning_backend,",
            ")",
            "import loreley.core.worker.agent as agent",
            "assert CodexCliBackend is backends.CodexCliBackend",
            "assert CursorCliBackend is backends.CursorCliBackend",
            "assert KilocodeCliBackend is backends.KilocodeCliBackend",
            "assert isinstance(DEFAULT_CURSOR_MODEL, str) and DEFAULT_CURSOR_MODEL",
            "assert callable(codex_coding_backend)",
            "assert callable(codex_planning_backend)",
            "assert callable(cursor_backend)",
            "assert callable(cursor_coding_backend)",
            "assert callable(cursor_planning_backend)",
            "assert callable(kilocode_backend)",
            "assert callable(kilocode_coding_backend)",
            "assert callable(kilocode_planning_backend)",
            "assert hasattr(agent, 'load_agent_backend')",
            "assert hasattr(agent, 'run_agent_task')",
            "assert hasattr(agent, 'AgentTask')",
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


def test_kilocode_cli_backend_builds_run_command_with_passthrough_flags(
    tmp_path: Path, monkeypatch
) -> None:
    """KilocodeCliBackend constructs ``kilo run --auto`` with prompt as positional arg."""
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    captured: dict[str, Any] = {}

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        captured.update({"command": command, "cwd": cwd, "env": env, "timeout": timeout})
        return types.SimpleNamespace(stdout="done", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilo",
        model="openai/gpt-5.4",
        agent="architect",
        variant="high",
        timeout_seconds=60,
        extra_env={"KEY": "val"},
        json_output=True,
        error_cls=RuntimeError,
    )

    task = AgentTask(name="coding", prompt="implement feature X")
    invocation = backend.run(task, working_dir=repo_dir)

    command_list = list(invocation.command)
    assert command_list[:3] == ["kilo", "run", "--auto"]
    assert "--auto" in command_list
    assert "--format" in command_list and "json" in command_list
    assert "--agent" in command_list and "architect" in command_list
    assert "--model" in command_list and "openai/gpt-5.4" in command_list
    assert "--variant" in command_list and "high" in command_list
    assert "implement feature X" in command_list
    assert captured["cwd"] == str(repo_dir.resolve())
    assert captured["env"] and captured["env"]["KEY"] == "val"
    assert invocation.stdout == "done"


def test_kilocode_cli_backend_omits_optional_flags_when_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    """Omits optional passthrough flags when not configured."""
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=60,
        extra_env={},
        json_output=False,
        error_cls=RuntimeError,
    )

    task = AgentTask(name="planning", prompt="plan something")
    invocation = backend.run(task, working_dir=repo_dir)

    command_list = list(invocation.command)
    assert command_list[:3] == ["kilo", "run", "--auto"]
    assert "--format" not in command_list
    assert "--agent" not in command_list
    assert "--model" not in command_list
    assert "--variant" not in command_list
    assert "--auto" in command_list
    assert "plan something" in command_list


def test_kilocode_cli_backend_titles_session_and_reads_usage_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)
    usage_db = tmp_path / "kilo.db"
    with sqlite3.connect(usage_db) as conn:
        conn.execute(
            """
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                title TEXT,
                directory TEXT,
                time_created INTEGER,
                time_updated INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                time_created INTEGER,
                data TEXT
            )
            """
        )

    def fake_run(command, **kwargs):  # noqa: ANN001
        cwd = kwargs["cwd"]
        title = command[command.index("--title") + 1]
        with sqlite3.connect(usage_db) as conn:
            conn.execute(
                "INSERT INTO session (id, title, directory, time_created, time_updated) VALUES (?, ?, ?, ?, ?)",
                ("sess-usage", title, cwd, 1, 2),
            )
            conn.execute(
                "INSERT INTO message (id, session_id, time_created, data) VALUES (?, ?, ?, ?)",
                (
                    "msg-1",
                    "sess-usage",
                    3,
                    json.dumps(
                        {
                            "role": "assistant",
                            "providerID": "openrouter",
                            "modelID": "openai/gpt-5.2",
                            "cost": 0.025,
                            "tokens": {
                                "input": 10,
                                "output": 20,
                                "reasoning": 3,
                                "cache": {"read": 100, "write": 5},
                            },
                        }
                    ),
                ),
            )
        return types.SimpleNamespace(stdout="done", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)
    job_id = uuid4()
    run_token = uuid4()
    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=30,
        extra_env={},
        error_cls=RuntimeError,
        usage_db_path=str(usage_db),
    )

    invocation = backend.run(
        AgentTask(
            name="coding",
            prompt="do it",
            job_id=job_id,
            run_token=run_token,
            phase="coding",
            attempt=4,
        ),
        working_dir=repo_dir,
    )

    command_list = list(invocation.command)
    assert "--title" in command_list
    assert (
        command_list[command_list.index("--title") + 1]
        == f"loreley:{job_id}:{run_token}:coding:attempt:4"
    )
    assert len(invocation.usage_events) == 1
    event = invocation.usage_events[0]
    assert event.source == "kilo_cli"
    assert event.provider == "openrouter"
    assert event.model == "openai/gpt-5.2"
    assert event.cached_input_tokens == 100
    assert event.cache_write_tokens == 5
    assert event.cost_source == "provider_reported"
    assert str(event.cost_usd) == "0.025"
    assert event.external_usage_id == f"kilo:{job_id}:{run_token}:coding:attempt:4"


def test_kilocode_cli_backend_raises_on_nonzero_exit_with_stdout_and_stderr_context(
    tmp_path: Path, monkeypatch
) -> None:
    """Non-zero exit code from kilo includes both stderr and stdout snippets."""
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN002
        return types.SimpleNamespace(
            stdout='{"event":"error","message":"permission denied"}',
            stderr="connection failed",
            returncode=1,
        )

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=10,
        extra_env={},
        error_cls=RuntimeError,
    )

    task = AgentTask(name="coding", prompt="run")

    with pytest.raises(RuntimeError, match="exit code 1") as exc_info:
        backend.run(task, working_dir=repo_dir)

    message = str(exc_info.value)
    assert "stderr: connection failed" in message
    assert 'stdout: {"event":"error","message":"permission denied"}' in message


def test_kilocode_cli_backend_raises_on_timeout(tmp_path: Path, monkeypatch) -> None:
    """Subprocess timeout is surfaced via the configured error class."""
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN002
        raise subprocess.TimeoutExpired(cmd="kilocode", timeout=5)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=5,
        extra_env={},
        error_cls=RuntimeError,
    )

    task = AgentTask(name="coding", prompt="slow task")

    with pytest.raises(RuntimeError, match="timed out"):
        backend.run(task, working_dir=repo_dir)


def test_kilocode_cli_backend_warns_on_empty_stdout(
    tmp_path: Path, monkeypatch, captured_logs
) -> None:
    """Empty stdout emits a warning log with module binding."""
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN002
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=10,
        extra_env={},
        error_cls=RuntimeError,
    )

    task = AgentTask(name="test", prompt="empty output")
    backend.run(task, working_dir=repo_dir)

    warning_msgs = [r for r in captured_logs if r["level"] == "WARNING"]
    assert any("empty stdout" in r["message"].lower() for r in warning_msgs)


def test_kilocode_backend_factory_uses_env_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory reads WORKER_KILOCODE_* env vars via Settings."""
    from loreley.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("WORKER_KILOCODE_BIN", "/usr/local/bin/kilo")
    monkeypatch.setenv("WORKER_KILOCODE_AGENT", "architect")
    monkeypatch.setenv("WORKER_KILOCODE_MODEL", "openai/gpt-5.4")
    monkeypatch.setenv("WORKER_KILOCODE_VARIANT", "high")
    monkeypatch.setenv("WORKER_KILOCODE_JSON_OUTPUT", "true")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_MODEL", "gpt-4o-mini")
    get_settings.cache_clear()

    backend = kilocode_backend()

    assert isinstance(backend, KilocodeCliBackend)
    assert backend.bin == "/usr/local/bin/kilo"
    assert backend.agent == "architect"
    assert backend.model == "openai/gpt-5.4"
    assert backend.variant == "high"
    assert backend.json_output is True
    assert backend.extra_env["KILO_PROVIDER_TYPE"] == "openai-responses"
    assert backend.extra_env["KILO_OPENAI_BASE_URL"] == "https://example.invalid/v1"
    assert backend.extra_env["KILO_OPENAI_MODEL_ID"] == "gpt-4o-mini"

    get_settings.cache_clear()


def test_kilocode_backend_factory_falls_back_to_global_openai_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global OPENAI/LORELEY aliases feed the Kilo provider env when worker-specific values are absent."""
    from loreley.config import get_settings

    get_settings.cache_clear()
    monkeypatch.delenv("WORKER_KILOCODE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("WORKER_KILOCODE_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("WORKER_KILOCODE_OPENAI_API_SPEC", raising=False)
    monkeypatch.setenv("LORELEY_LLM_API_KEY", "sk-alias")
    monkeypatch.setenv("LORELEY_LLM_BASE_URL", "https://alias.example.com/v1")
    monkeypatch.setenv("OPENAI_API_SPEC", "chat_completions")
    get_settings.cache_clear()

    backend = kilocode_backend()

    assert backend.extra_env["KILO_PROVIDER_TYPE"] == "openai"
    assert backend.extra_env["KILO_OPENAI_BASE_URL"] == "https://alias.example.com/v1"

    get_settings.cache_clear()


def test_kilocode_backend_factory_prefers_worker_specific_openai_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker-specific Kilo config wins over the global OpenAI-compatible aliases."""
    from loreley.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("LORELEY_LLM_API_KEY", "sk-global")
    monkeypatch.setenv("LORELEY_LLM_BASE_URL", "https://global.example.com/v1")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_API_KEY", "sk-worker")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_BASE_URL", "https://worker.example.com/v1")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_API_SPEC", "responses")
    get_settings.cache_clear()

    backend = kilocode_backend()

    assert backend.extra_env["KILO_PROVIDER_TYPE"] == "openai-responses"
    assert backend.extra_env["KILO_OPENAI_BASE_URL"] == "https://worker.example.com/v1"

    get_settings.cache_clear()


def test_kilocode_backend_factory_keeps_global_api_spec_under_partial_worker_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial worker-specific provider overrides still inherit the global API spec."""
    from loreley.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_SPEC", "responses")
    monkeypatch.setenv("LORELEY_LLM_API_KEY", "sk-global")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_BASE_URL", "https://worker.example.com/v1")
    monkeypatch.delenv("WORKER_KILOCODE_OPENAI_API_SPEC", raising=False)
    get_settings.cache_clear()

    backend = kilocode_backend()

    assert backend.extra_env["KILO_PROVIDER_TYPE"] == "openai-responses"
    assert backend.extra_env["KILO_OPENAI_BASE_URL"] == "https://worker.example.com/v1"

    get_settings.cache_clear()


def test_kilocode_backend_factory_maps_legacy_mode_to_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WORKER_KILOCODE_MODE remains a backward-compatible alias for the agent selector."""
    from loreley.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("WORKER_KILOCODE_MODE", "debug")
    get_settings.cache_clear()

    backend = kilocode_backend()

    assert backend.agent == "debug"

    get_settings.cache_clear()


def test_codex_backend_factories_use_worker_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from loreley.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("WORKER_PLANNING_CODEX_BIN", "/usr/local/bin/codex-plan")
    monkeypatch.setenv("WORKER_CODING_CODEX_BIN", "/usr/local/bin/codex-code")
    monkeypatch.setenv("WORKER_PLANNING_CODEX_PROFILE", "planner")
    monkeypatch.setenv("WORKER_CODING_CODEX_PROFILE", "coder")
    get_settings.cache_clear()

    planning_backend = codex_planning_backend()
    coding_backend = codex_coding_backend()

    assert isinstance(planning_backend, CodexCliBackend)
    assert planning_backend.bin == "/usr/local/bin/codex-plan"
    assert planning_backend.profile == "planner"
    assert planning_backend.full_auto is False

    assert isinstance(coding_backend, CodexCliBackend)
    assert coding_backend.bin == "/usr/local/bin/codex-code"
    assert coding_backend.profile == "coder"
    assert coding_backend.full_auto is True

    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("api_spec", "expected_provider_type"),
    [
        ("chat_completions", "openai"),
        ("responses", "openai-responses"),
    ],
)
def test_kilocode_backend_factory_maps_openai_api_spec_to_provider_type(
    monkeypatch: pytest.MonkeyPatch,
    api_spec: str,
    expected_provider_type: str,
) -> None:
    """Kilo provider type follows WORKER_KILOCODE_OPENAI_API_SPEC."""
    from loreley.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_API_SPEC", api_spec)
    get_settings.cache_clear()

    backend = kilocode_backend()

    assert backend.extra_env["KILO_PROVIDER_TYPE"] == expected_provider_type

    get_settings.cache_clear()


def test_kilocode_backend_resolves_api_key_at_run_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    captured_keys: list[str] = []

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        captured_keys.append(env["KILO_OPENAI_API_KEY"])
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)
    api_keys = iter(["dyn-1", "dyn-2"])
    monkeypatch.setattr(
        kilocode_cli,
        "get_agent_openai_api_key",
        lambda _settings: next(api_keys),
    )

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=30,
        extra_env={"KILO_PROVIDER_TYPE": "openai-responses"},
        settings=Settings.model_validate({}),
        error_cls=RuntimeError,
    )

    backend.run(AgentTask(name="planning", prompt="plan"), working_dir=repo_dir)
    backend.run(AgentTask(name="planning", prompt="plan again"), working_dir=repo_dir)

    assert captured_keys == ["dyn-1", "dyn-2"]


def test_kilocode_backend_runtime_api_key_does_not_overwrite_shared_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    from loreley.core.openai_auth import DynamicOpenAIKeyManager

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    values = iter(["shared-token", "agent-token"])
    manager = DynamicOpenAIKeyManager(
        provider=lambda: next(values),
        provider_ref="tests.provider:token",
        ttl_seconds=600,
        refresh_skew_seconds=60,
        start_refresh_thread=False,
    )
    settings = Settings.model_validate({})
    monkeypatch.setattr(
        kilocode_cli,
        "get_agent_openai_api_key",
        lambda _settings: manager.get_agent_token(),
    )

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=30,
        extra_env={"KILO_PROVIDER_TYPE": "openai-responses"},
        settings=settings,
        error_cls=RuntimeError,
    )

    assert manager.get_shared_token() == "shared-token"
    backend.run(AgentTask(name="planning", prompt="plan"), working_dir=repo_dir)
    assert manager.get_shared_token() == "shared-token"


def test_kilocode_backend_preserves_explicit_extra_env_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    captured_keys: list[str] = []

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        captured_keys.append(env["KILO_OPENAI_API_KEY"])
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        kilocode_cli,
        "get_agent_openai_api_key",
        lambda _settings: "runtime-key",
    )

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=30,
        extra_env={
            "KILO_PROVIDER_TYPE": "openai-responses",
            "KILO_OPENAI_API_KEY": "explicit-extra-env-key",
        },
        settings=Settings.model_validate({}),
        error_cls=RuntimeError,
    )

    backend.run(AgentTask(name="planning", prompt="plan"), working_dir=repo_dir)

    assert captured_keys == ["explicit-extra-env-key"]


def test_kilocode_backend_skips_runtime_api_key_lookup_when_extra_env_sets_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    captured_keys: list[str] = []

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        captured_keys.append(env["KILO_OPENAI_API_KEY"])
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    def fail_lookup(_settings):  # noqa: ANN001
        raise RuntimeError("dynamic provider unavailable")

    monkeypatch.setattr(kilocode_cli, "get_agent_openai_api_key", fail_lookup)

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=30,
        extra_env={
            "KILO_PROVIDER_TYPE": "openai-responses",
            "KILO_OPENAI_API_KEY": "explicit-extra-env-key",
        },
        settings=Settings.model_validate({}),
        error_cls=RuntimeError,
    )

    backend.run(AgentTask(name="planning", prompt="plan"), working_dir=repo_dir)

    assert captured_keys == ["explicit-extra-env-key"]


def test_kilocode_backend_runtime_api_key_overrides_inherited_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    captured_keys: list[str] = []

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        captured_keys.append(env["KILO_OPENAI_API_KEY"])
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        kilocode_cli,
        "get_agent_openai_api_key",
        lambda _settings: "runtime-key",
    )
    monkeypatch.setenv("KILO_OPENAI_API_KEY", "stale-process-key")

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=30,
        extra_env={"KILO_PROVIDER_TYPE": "openai-responses"},
        settings=Settings.model_validate({}),
        error_cls=RuntimeError,
    )

    backend.run(AgentTask(name="planning", prompt="plan"), working_dir=repo_dir)

    assert captured_keys == ["runtime-key"]


def test_run_agent_task_retries_when_kilocode_runtime_api_key_lookup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from loreley.core.openai_auth import DynamicOpenAIKeyUnavailableError
    from loreley.core.worker.planning import PlanningError

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    lookup_calls = {"count": 0}
    subprocess_calls = {"count": 0}

    def flaky_lookup(_settings):  # noqa: ANN001
        lookup_calls["count"] += 1
        if lookup_calls["count"] == 1:
            raise DynamicOpenAIKeyUnavailableError("provider unavailable")
        return "runtime-key"

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        subprocess_calls["count"] += 1
        assert env["KILO_OPENAI_API_KEY"] == "runtime-key"
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli, "get_agent_openai_api_key", flaky_lookup)
    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilo",
        timeout_seconds=30,
        extra_env={"KILO_PROVIDER_TYPE": "openai-responses"},
        settings=Settings.model_validate({}),
        error_cls=PlanningError,
    )

    value, invocation, attempts = run_agent_task(
        backend=backend,
        task=AgentTask(name="planning", prompt="plan"),
        working_dir=repo_dir,
        max_attempts=2,
        coerce_result=lambda inv: inv.stdout,
        retryable_exceptions=(PlanningError,),
        error_cls=PlanningError,
        error_message="should-not-fail",
    )

    assert value == "ok"
    assert invocation.stdout == "ok"
    assert attempts == 2
    assert lookup_calls["count"] == 2
    assert subprocess_calls["count"] == 1


def test_run_agent_task_retries_on_post_check(tmp_path: Path) -> None:
    class DummyBackend:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: AgentTask, *, working_dir: Path) -> AgentInvocation:  # noqa: ARG002
            self.calls += 1
            return AgentInvocation(
                command=("dummy", str(self.calls)),
                stdout=str(self.calls),
                stderr="",
                duration_seconds=0.0,
                usage_events=(
                    _usage_event(
                        external_usage_id=f"dummy:{task.attempt}",
                        input_tokens=10 * self.calls,
                    ),
                ),
            )

    backend = DummyBackend()
    task = AgentTask(name="test", prompt="hi")

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

    value, invocation, attempts = run_agent_task(
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
    assert [event.external_usage_id for event in invocation.usage_events] == [
        "dummy:1",
        "dummy:2",
    ]
    assert [event.input_tokens for event in invocation.usage_events] == [10, 20]


def test_run_agent_task_preserves_coercion_failure_usage(tmp_path: Path) -> None:
    class DummyBackend:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: AgentTask, *, working_dir: Path) -> AgentInvocation:  # noqa: ARG002
            self.calls += 1
            stdout = "bad" if self.calls == 1 else "2"
            return AgentInvocation(
                command=("dummy", str(self.calls)),
                stdout=stdout,
                stderr="",
                duration_seconds=0.0,
                usage_events=(
                    _usage_event(
                        external_usage_id=f"dummy:{task.attempt}",
                        output_tokens=self.calls,
                    ),
                ),
            )

    backend = DummyBackend()

    value, invocation, attempts = run_agent_task(
        backend=backend,
        task=AgentTask(name="test", prompt="hi"),
        working_dir=tmp_path,
        max_attempts=2,
        coerce_result=lambda inv: int(inv.stdout),
        retryable_exceptions=(ValueError,),
        error_cls=RuntimeError,
        error_message="should-not-fail",
    )

    assert value == 2
    assert attempts == 2
    assert [event.external_usage_id for event in invocation.usage_events] == [
        "dummy:1",
        "dummy:2",
    ]
    assert [event.output_tokens for event in invocation.usage_events] == [1, 2]


def test_run_agent_task_preserves_retryable_exception_usage(tmp_path: Path) -> None:
    class DummyBackend:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: AgentTask, *, working_dir: Path) -> AgentInvocation:  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                error = RuntimeError("transient")
                setattr(
                    error,
                    "usage_events",
                    (
                        _usage_event(
                            external_usage_id=f"dummy:{task.attempt}",
                            input_tokens=5,
                        ),
                    ),
                )
                raise error
            return AgentInvocation(
                command=("dummy", "ok"),
                stdout="ok",
                stderr="",
                duration_seconds=0.0,
                usage_events=(
                    _usage_event(
                        external_usage_id=f"dummy:{task.attempt}",
                        input_tokens=7,
                    ),
                ),
            )

    value, invocation, attempts = run_agent_task(
        backend=DummyBackend(),
        task=AgentTask(name="test", prompt="hi"),
        working_dir=tmp_path,
        max_attempts=2,
        coerce_result=lambda inv: inv.stdout,
        retryable_exceptions=(RuntimeError,),
        error_cls=RuntimeError,
        error_message="should-not-fail",
    )

    assert value == "ok"
    assert attempts == 2
    assert [event.external_usage_id for event in invocation.usage_events] == [
        "dummy:1",
        "dummy:2",
    ]
    assert [event.input_tokens for event in invocation.usage_events] == [5, 7]


def test_run_agent_task_exhausted_error_carries_attempt_usage(tmp_path: Path) -> None:
    class DummyBackend:
        def run(self, task: AgentTask, *, working_dir: Path) -> AgentInvocation:  # noqa: ARG002
            return AgentInvocation(
                command=("dummy", str(task.attempt)),
                stdout="not-an-int",
                stderr="",
                duration_seconds=0.0,
                usage_events=(
                    _usage_event(
                        external_usage_id=f"dummy:{task.attempt}",
                        input_tokens=int(task.attempt or 0),
                    ),
                ),
            )

    with pytest.raises(RuntimeError) as exc_info:
        run_agent_task(
            backend=DummyBackend(),
            task=AgentTask(name="test", prompt="hi"),
            working_dir=tmp_path,
            max_attempts=2,
            coerce_result=lambda inv: int(inv.stdout),
            retryable_exceptions=(ValueError,),
            error_cls=RuntimeError,
            error_message="all attempts failed",
        )

    usage_events = exc_info.value.usage_events
    assert [event.external_usage_id for event in usage_events] == ["dummy:1", "dummy:2"]
    assert [event.input_tokens for event in usage_events] == [1, 2]


def _usage_event(
    *,
    external_usage_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> LLMUsageEventPayload:
    return LLMUsageEventPayload(
        source="dummy",
        phase="test",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        external_usage_id=external_usage_id,
    )
