from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from loreley.config import Settings
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
    cursor_backend,
    kilocode_backend,
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


def test_codex_cli_backend_builds_command_and_passes_stdin(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

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
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    backend = CodexCliBackend(
        bin="codex",
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
    assert command_list[:2] == ["codex", "exec"]
    assert "--full-auto" in command_list
    assert "--profile" in command_list and "prof" in command_list
    assert "--output-schema" not in command_list
    assert captured["cwd"] == str(repo_dir.resolve())
    assert captured["input"] == "do things"
    assert captured["env"] and captured["env"]["A"] == "1"


def test_codex_cli_backend_raises_on_failure(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN002
        return types.SimpleNamespace(stdout="", stderr="boom", returncode=1)

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    backend = CodexCliBackend(
        bin="codex",
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
            "    cursor_backend,",
            "    kilocode_backend,",
            ")",
            "import loreley.core.worker.agent as agent",
            "assert CodexCliBackend is backends.CodexCliBackend",
            "assert CursorCliBackend is backends.CursorCliBackend",
            "assert KilocodeCliBackend is backends.KilocodeCliBackend",
            "assert isinstance(DEFAULT_CURSOR_MODEL, str) and DEFAULT_CURSOR_MODEL",
            "assert callable(cursor_backend)",
            "assert callable(kilocode_backend)",
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


def test_kilocode_cli_backend_builds_auto_command(tmp_path: Path, monkeypatch) -> None:
    """KilocodeCliBackend constructs ``kilocode --auto`` with prompt as positional arg."""
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    captured: dict[str, Any] = {}

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        captured.update({"command": command, "cwd": cwd, "env": env, "timeout": timeout})
        return types.SimpleNamespace(stdout="done", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilocode",
        mode="code",
        timeout_seconds=60,
        extra_env={"KEY": "val"},
        json_output=True,
        error_cls=RuntimeError,
    )

    task = AgentTask(name="coding", prompt="implement feature X")
    invocation = backend.run(task, working_dir=repo_dir)

    command_list = list(invocation.command)
    assert command_list[0] == "kilocode"
    assert "--auto" in command_list
    assert "--json" in command_list
    assert "--mode" in command_list and "code" in command_list
    assert "implement feature X" in command_list
    assert captured["cwd"] == str(repo_dir.resolve())
    assert captured["env"] and captured["env"]["KEY"] == "val"
    assert invocation.stdout == "done"


def test_kilocode_cli_backend_omits_optional_flags_when_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    """Omits --json and --mode when not configured."""
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(command, cwd, env, text, capture_output, timeout, check):  # noqa: ANN001
        return types.SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilocode",
        mode=None,
        timeout_seconds=60,
        extra_env={},
        json_output=False,
        error_cls=RuntimeError,
    )

    task = AgentTask(name="planning", prompt="plan something")
    invocation = backend.run(task, working_dir=repo_dir)

    command_list = list(invocation.command)
    assert "--json" not in command_list
    assert "--mode" not in command_list
    assert "--auto" in command_list
    assert "plan something" in command_list


def test_kilocode_cli_backend_raises_on_nonzero_exit(tmp_path: Path, monkeypatch) -> None:
    """Non-zero exit code from kilocode produces an error with stderr context."""
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN002
        return types.SimpleNamespace(stdout="", stderr="connection failed", returncode=1)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilocode",
        timeout_seconds=10,
        extra_env={},
        error_cls=RuntimeError,
    )

    task = AgentTask(name="coding", prompt="run")

    with pytest.raises(RuntimeError, match="exit code 1"):
        backend.run(task, working_dir=repo_dir)


def test_kilocode_cli_backend_raises_on_timeout(tmp_path: Path, monkeypatch) -> None:
    """Subprocess timeout is surfaced via the configured error class."""
    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_run(*_args, **_kwargs):  # noqa: ANN001, ANN002
        raise subprocess.TimeoutExpired(cmd="kilocode", timeout=5)

    monkeypatch.setattr(kilocode_cli.subprocess, "run", fake_run)

    backend = KilocodeCliBackend(
        bin="kilocode",
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
        bin="kilocode",
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
    monkeypatch.setenv("WORKER_KILOCODE_MODE", "architect")
    monkeypatch.setenv("WORKER_KILOCODE_JSON_OUTPUT", "false")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("WORKER_KILOCODE_OPENAI_MODEL", "gpt-4o-mini")
    get_settings.cache_clear()

    backend = kilocode_backend()

    assert isinstance(backend, KilocodeCliBackend)
    assert backend.bin == "/usr/local/bin/kilo"
    assert backend.mode == "architect"
    assert backend.json_output is False
    assert backend.extra_env["KILO_PROVIDER_TYPE"] == "openai"
    assert backend.extra_env["KILO_OPENAI_API_KEY"] == "sk-test"
    assert backend.extra_env["KILO_OPENAI_BASE_URL"] == "https://example.invalid/v1"
    assert backend.extra_env["KILO_OPENAI_MODEL_ID"] == "gpt-4o-mini"

    get_settings.cache_clear()


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

