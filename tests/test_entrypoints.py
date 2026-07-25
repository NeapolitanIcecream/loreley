from __future__ import annotations

import signal
import sys
from types import SimpleNamespace

from rich.console import Console

import loreley.entrypoints as entrypoints
from loreley.preflight import CheckResult


class _FakeProcess:
    def __init__(self, *, pid: int, polls: list[int | None]) -> None:
        self.pid = pid
        self._polls = polls
        self.returncode = polls[-1] if polls else 0

    def poll(self) -> int | None:
        if self._polls:
            value = self._polls.pop(0)
            if value is not None:
                self.returncode = value
            return value
        return self.returncode


def test_apply_dramatiq_prefetch_settings_uses_loreley_configuration(
    monkeypatch,
    settings,
) -> None:
    import dramatiq.worker as dramatiq_worker

    monkeypatch.setattr(dramatiq_worker, "QUEUE_PREFETCH", 99)
    monkeypatch.setattr(dramatiq_worker, "DELAY_QUEUE_PREFETCH", 999)
    settings.tasks_queue_prefetch = 1
    settings.tasks_delay_queue_prefetch = 3

    entrypoints._apply_dramatiq_prefetch_settings(  # noqa: SLF001 - spec-level assertion
        settings=settings,
        console=Console(record=True),
    )

    assert dramatiq_worker.QUEUE_PREFETCH == 1
    assert dramatiq_worker.DELAY_QUEUE_PREFETCH == 3


def test_apply_dramatiq_prefetch_settings_preserves_explicit_zero(
    monkeypatch,
    settings,
) -> None:
    import dramatiq.worker as dramatiq_worker

    monkeypatch.setattr(dramatiq_worker, "QUEUE_PREFETCH", 99)
    monkeypatch.setattr(dramatiq_worker, "DELAY_QUEUE_PREFETCH", 999)
    settings.tasks_queue_prefetch = 0
    settings.tasks_delay_queue_prefetch = 0

    entrypoints._apply_dramatiq_prefetch_settings(  # noqa: SLF001 - spec-level assertion
        settings=settings,
        console=Console(record=True),
    )

    assert dramatiq_worker.QUEUE_PREFETCH == 0
    assert dramatiq_worker.DELAY_QUEUE_PREFETCH == 0


def test_worker_pool_uses_spawned_single_threaded_dramatiq_processes(
    monkeypatch,
) -> None:
    captured: list[object] = []
    child_environment: list[str | None] = []

    def fake_main(args: object) -> int:
        captured.append(args)
        child_environment.append(
            entrypoints.os.environ.get("WORKER_REPO_WORKTREE_RANDOMIZE")
        )
        return 7

    monkeypatch.setattr("dramatiq.cli.main", fake_main)
    monkeypatch.setenv("WORKER_REPO_WORKTREE_RANDOMIZE", "false")

    rc = entrypoints._run_dramatiq_worker_pool(  # noqa: SLF001 - process contract
        processes=3,
        console=Console(record=True),
    )

    assert rc == 7
    assert len(captured) == 1
    args = captured[0]
    assert getattr(args, "processes") == 3
    assert getattr(args, "threads") == 1
    assert getattr(args, "use_spawn") is True
    assert getattr(args, "broker") == "loreley.tasks.worker_runtime:broker"
    assert child_environment == ["true"]
    assert entrypoints.os.environ["WORKER_REPO_WORKTREE_RANDOMIZE"] == "false"


def test_run_worker_delegates_multi_process_lifecycle_to_dramatiq(
    monkeypatch,
    settings,
) -> None:
    schema_calls: list[object] = []
    pool_calls: list[int] = []
    monkeypatch.setattr(
        "loreley.db.base.ensure_database_schema",
        lambda *, settings: schema_calls.append(settings),
    )
    monkeypatch.setattr(
        entrypoints,
        "_run_dramatiq_worker_pool",
        lambda *, processes, console: pool_calls.append(processes) or 0,
    )

    rc = entrypoints.run_worker(
        settings=settings,
        console=Console(record=True),
        processes=4,
        preflight=False,
    )

    assert rc == 0
    assert schema_calls == [settings]
    assert pool_calls == [4]


def test_process_log_paths_are_unique_within_the_same_second(
    monkeypatch,
    tmp_path,
    settings,
) -> None:
    settings.logs_base_dir = str(tmp_path)
    monkeypatch.setattr(entrypoints.os, "getpid", lambda: 101)
    first = entrypoints.configure_process_logging(
        settings=settings,
        console=Console(record=True),
        role="worker",
    )
    monkeypatch.setattr(entrypoints.os, "getpid", lambda: 202)
    second = entrypoints.configure_process_logging(
        settings=settings,
        console=Console(record=True),
        role="worker",
    )

    assert first != second
    assert "pid-101" in first.name
    assert "pid-202" in second.name


def test_run_ui_starts_streamlit_with_api_environment_when_api_is_reachable(
    monkeypatch,
    settings,
) -> None:
    popen_calls: list[tuple[list[str], dict[str, object]]] = []
    streamlit_proc = _FakeProcess(pid=1234, polls=[0])

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakeProcess:
        popen_calls.append((cmd, kwargs))
        return streamlit_proc

    monkeypatch.setattr(entrypoints, "_is_ui_api_reachable", lambda *args, **kwargs: True)
    monkeypatch.setattr(entrypoints.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(entrypoints.signal, "signal", lambda *args, **kwargs: None)

    rc = entrypoints.run_ui(
        settings=settings,
        console=Console(record=True),
        api_base_url=" http://127.0.0.1:8123 ",
        host="0.0.0.0",
        port=8502,
        headless=True,
        preflight=False,
        preflight_timeout_seconds=0.5,
    )

    assert rc == 0
    assert len(popen_calls) == 1
    cmd, kwargs = popen_calls[0]
    expected_ui_script = str(
        (entrypoints.Path(entrypoints.__file__).resolve().parent / "ui" / "app.py").resolve()
    )
    assert cmd == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        expected_ui_script,
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8502",
        "--server.headless",
        "true",
    ]
    env = kwargs["env"]
    assert isinstance(env, dict)
    assert env["LORELEY_UI_API_BASE_URL"] == "http://127.0.0.1:8123"
    if entrypoints.os.name == "posix":
        assert kwargs["start_new_session"] is True


def test_run_ui_preflight_warning_blocks_startup_before_spawning_processes(
    monkeypatch,
    settings,
) -> None:
    timeouts: list[float] = []

    def fake_preflight_ui(_settings: object, *, timeout_seconds: float) -> list[CheckResult]:
        timeouts.append(timeout_seconds)
        return [CheckResult("streamlit_deps", "warn", "missing optional dependency")]

    def fail_popen(*args: object, **kwargs: object) -> None:
        raise AssertionError("run_ui should not spawn processes when preflight blocks startup")

    monkeypatch.setattr(entrypoints, "preflight_ui", fake_preflight_ui)
    monkeypatch.setattr(entrypoints.subprocess, "Popen", fail_popen)

    console = Console(record=True)
    rc = entrypoints.run_ui(
        settings=settings,
        console=console,
        api_base_url="http://127.0.0.1:8123",
        host="localhost",
        port=8501,
        headless=False,
        preflight=True,
        preflight_timeout_seconds=1.25,
    )

    assert rc == 1
    assert timeouts == [1.25]
    assert "Preflight failed" in console.export_text()


def test_run_api_preflight_token_warnings_do_not_block_read_only_startup(
    monkeypatch,
    settings,
) -> None:
    """Regression: missing control tokens should warn without disabling read-only routes."""

    run_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        entrypoints,
        "preflight_api",
        lambda *_args, **_kwargs: [
            CheckResult(
                "api_write_token",
                "warn",
                "LORELEY_API_WRITE_TOKEN is not set; POST routes are disabled.",
            ),
            CheckResult(
                "agent_api_token",
                "warn",
                "LORELEY_AGENT_API_TOKEN is not set; agent routes are disabled.",
            ),
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: run_calls.append((args, kwargs))),
    )

    console = Console(record=True)
    rc = entrypoints.run_api(
        settings=settings,
        console=console,
        host="127.0.0.1",
        port=8123,
        reload=False,
        preflight=True,
        preflight_timeout_seconds=0.25,
    )

    assert rc == 0
    assert run_calls
    output = console.export_text()
    assert "LORELEY_API_WRITE_TOKEN is not set" in output
    assert "LORELEY_AGENT_API_TOKEN is not set" in output
    assert "Preflight failed" not in output


def test_run_api_preflight_dependency_warnings_block_startup(
    monkeypatch,
    settings,
) -> None:
    """Regression: API dependency/setup warnings must not be treated like token warnings."""

    run_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        entrypoints,
        "preflight_api",
        lambda *_args, **_kwargs: [
            CheckResult("ui_api_deps", "warn", "missing modules: ['fastapi']."),
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: run_calls.append((args, kwargs))),
    )

    console = Console(record=True)
    rc = entrypoints.run_api(
        settings=settings,
        console=console,
        host="127.0.0.1",
        port=8123,
        reload=False,
        preflight=True,
        preflight_timeout_seconds=0.25,
    )

    assert rc == 1
    assert not run_calls
    assert "Preflight failed" in console.export_text()


def test_run_ui_autostarts_local_api_before_streamlit_when_api_is_unreachable(
    monkeypatch,
    settings,
) -> None:
    reachability_results = iter([False, False, True])
    popen_calls: list[tuple[list[str], dict[str, object], _FakeProcess]] = []
    api_proc = _FakeProcess(pid=2001, polls=[None, None])
    streamlit_proc = _FakeProcess(pid=2002, polls=[0])
    processes = iter([api_proc, streamlit_proc])
    stop_calls: list[tuple[_FakeProcess, int]] = []

    def fake_is_reachable(*args: object, **kwargs: object) -> bool:
        return next(reachability_results)

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakeProcess:
        proc = next(processes)
        popen_calls.append((cmd, kwargs, proc))
        return proc

    def fake_stop_proc(proc: _FakeProcess, *, console: Console, first_signal: int) -> None:
        stop_calls.append((proc, first_signal))

    monkeypatch.setattr(entrypoints, "_is_ui_api_reachable", fake_is_reachable)
    monkeypatch.setattr(entrypoints.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(entrypoints.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(entrypoints, "_stop_proc", fake_stop_proc)

    rc = entrypoints.run_ui(
        settings=settings,
        console=Console(record=True),
        api_base_url=" http://127.0.0.1:9123/ ",
        host="localhost",
        port=8501,
        headless=False,
        preflight=False,
        preflight_timeout_seconds=0.25,
    )

    assert rc == 0
    assert len(popen_calls) == 2
    api_cmd, api_kwargs, _ = popen_calls[0]
    assert api_cmd == [
        sys.executable,
        "-m",
        "loreley",
        "--log-level",
        str(settings.log_level or "INFO"),
        "api",
        "--host",
        "127.0.0.1",
        "--port",
        "9123",
        "--preflight-timeout-seconds",
        "0.25",
        "--no-preflight",
    ]
    api_env = api_kwargs["env"]
    assert isinstance(api_env, dict)
    assert api_env["LORELEY_UI_API_BASE_URL"] == "http://127.0.0.1:9123/"

    streamlit_cmd, streamlit_kwargs, _ = popen_calls[1]
    assert streamlit_cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert "--server.headless" not in streamlit_cmd
    streamlit_env = streamlit_kwargs["env"]
    assert isinstance(streamlit_env, dict)
    assert streamlit_env["LORELEY_UI_API_BASE_URL"] == "http://127.0.0.1:9123/"
    if entrypoints.os.name == "posix":
        assert api_kwargs["start_new_session"] is True
        assert streamlit_kwargs["start_new_session"] is True
    assert stop_calls == [(api_proc, signal.SIGTERM)]
