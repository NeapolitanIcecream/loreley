from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Literal

from loguru import logger

from loreley.config import Settings, get_settings
from loreley.core.openai_auth import get_agent_openai_api_key
from loreley.core.usage import kilo_usage_event_from_messages, unavailable_usage_event
from loreley.core.worker.agent.contracts import AgentInvocation, AgentTask
from loreley.core.worker.agent.utils import truncate_text, validate_workdir

log = logger.bind(module="worker.agent.backends.kilocode_cli")

KILO_CONFIG_CONTENT_ENV = "KILO_CONFIG_CONTENT"
LORELEY_KILO_OPENAI_API_KEY_ENV = "LORELEY_KILO_OPENAI_API_KEY"
LORELEY_KILO_OPENAI_BASE_URL_ENV = "LORELEY_KILO_OPENAI_BASE_URL"
KILO_CONFIG_SCHEMA_URL = "https://app.kilo.ai/config.json"
KILO_CHAT_COMPLETIONS_PROVIDER_ID = "loreley-openai-compatible"
KILO_RESPONSES_PROVIDER_ID = "loreley-openai-responses"
KILO_HEADLESS_AGENT = "loreley-headless"
KILO_PROVIDER_CONFIG_MODES = ("auto", "config", "legacy_env", "none")
DEFAULT_KILOCODE_TIMEOUT_SECONDS = 1800
KILO_TERMINATION_GRACE_SECONDS = 5.0
KiloProviderConfigMode = Literal["auto", "config", "legacy_env", "none"]

_KILO_RUN_FLAG_PATTERN = re.compile(r"(?<![\w-])--[A-Za-z0-9][A-Za-z0-9-]*")
_KILO_VERSION_PATTERN = re.compile(r"\b(\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.]+)?)\b")
_KILO_FAILURE_REASON_CODES = frozenset(
    {
        "job_token_budget_exhausted",
        "request_limit_reached",
    }
)
_KILO_USAGE_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "session": frozenset({"id", "title", "directory", "time_created", "time_updated"}),
}


class KiloUsageUnavailableError(RuntimeError):
    """Usage data is unavailable after a successful Kilo invocation."""


class KiloWorkspaceIsolationError(RuntimeError):
    """Kilo bound a session to a directory outside the requested job worktree."""


@dataclass(frozen=True, slots=True)
class KiloCliCapabilities:
    """Best-effort view of the installed Kilo CLI command surface."""

    version: str | None
    run_flags: frozenset[str]
    supports_db_path: bool
    provider_config_mode: str = "unknown"

    @property
    def supports_auto(self) -> bool:
        return "--auto" in self.run_flags

    @property
    def supports_agent(self) -> bool:
        return "--agent" in self.run_flags

    @property
    def supports_model(self) -> bool:
        return "--model" in self.run_flags

    @property
    def supports_variant(self) -> bool:
        return "--variant" in self.run_flags

    @property
    def supports_title(self) -> bool:
        return "--title" in self.run_flags

    @property
    def supports_dir(self) -> bool:
        return "--dir" in self.run_flags

    @property
    def supports_pure(self) -> bool:
        return "--pure" in self.run_flags

    @property
    def supports_format_json(self) -> bool:
        return "--format" in self.run_flags


def parse_kilo_version(output: str) -> str | None:
    """Extract the first semantic-version-looking token from Kilo output."""

    match = _KILO_VERSION_PATTERN.search(str(output or ""))
    return match.group(1) if match else None


def parse_kilo_run_flags(help_output: str) -> frozenset[str]:
    """Extract long flags from ``kilo run --help`` output."""

    return frozenset(_KILO_RUN_FLAG_PATTERN.findall(str(help_output or "")))


def discover_kilo_cli_capabilities(
    kilo_bin: str,
    *,
    timeout_seconds: float = 2.0,
) -> KiloCliCapabilities:
    """Probe the installed Kilo CLI without requiring provider credentials."""

    version_result = _run_kilo_probe(
        [kilo_bin, "--version"], timeout_seconds=timeout_seconds
    )
    help_result = _run_kilo_probe(
        [kilo_bin, "run", "--help"], timeout_seconds=timeout_seconds
    )
    db_path_result = _run_kilo_probe(
        [kilo_bin, "db", "path"], timeout_seconds=timeout_seconds
    )
    version_text = "\n".join(
        filter(None, [version_result.stdout, version_result.stderr])
    )
    help_text = "\n".join(filter(None, [help_result.stdout, help_result.stderr]))
    return KiloCliCapabilities(
        version=parse_kilo_version(version_text),
        run_flags=parse_kilo_run_flags(help_text),
        supports_db_path=db_path_result.returncode == 0
        and bool((db_path_result.stdout or "").strip()),
    )


def probe_kilo_config_content_support(
    kilo_bin: str,
    *,
    provider_id: str = "openai",
    timeout_seconds: float = 2.0,
) -> tuple[bool, str]:
    """Return whether ``KILO_CONFIG_CONTENT`` is loaded by ``kilo debug config``."""

    env = os.environ.copy()
    env[LORELEY_KILO_OPENAI_API_KEY_ENV] = "loreley-probe-key"
    env[LORELEY_KILO_OPENAI_BASE_URL_ENV] = "https://example.invalid/v1"
    env[KILO_CONFIG_CONTENT_ENV] = _build_kilo_config_content(
        provider_id=provider_id,
        base_url=env[LORELEY_KILO_OPENAI_BASE_URL_ENV],
        model="loreley-probe-model",
        include_api_key=True,
    )
    result = _run_kilo_probe(
        [kilo_bin, "debug", "config"],
        timeout_seconds=timeout_seconds,
        env=env,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, detail or "kilo debug config failed"
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return False, f"kilo debug config did not emit JSON ({exc})"

    provider_payload = payload.get("provider", {}).get(provider_id, {})
    options = (
        provider_payload.get("options", {})
        if isinstance(provider_payload, dict)
        else {}
    )
    if options.get("apiKey") != env[LORELEY_KILO_OPENAI_API_KEY_ENV]:
        return False, f"{KILO_CONFIG_CONTENT_ENV} did not resolve apiKey env references"
    if options.get("baseURL") != env[LORELEY_KILO_OPENAI_BASE_URL_ENV]:
        return (
            False,
            f"{KILO_CONFIG_CONTENT_ENV} did not resolve baseURL env references",
        )
    return True, KILO_CONFIG_CONTENT_ENV


def _run_kilo_probe(
    command: list[str],
    *,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            command, returncode=127, stdout="", stderr=str(exc)
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"timed out after {timeout_seconds}s",
        )


def _signal_kilo_process_tree(
    process: subprocess.Popen[str],
    signum: int,
) -> None:
    """Signal Kilo and its descendants when the platform supports groups."""

    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signum)
        elif signum == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _collect_kilo_timeout_output(
    process: subprocess.Popen[str],
) -> tuple[str | None, str | None]:
    """Stop a timed-out Kilo process group and collect its final output."""

    _signal_kilo_process_tree(process, signal.SIGTERM)
    try:
        return process.communicate(timeout=KILO_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_kilo_process_tree(
            process,
            getattr(signal, "SIGKILL", signal.SIGTERM),
        )
        return process.communicate()


def _run_kilo_process(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run Kilo in a process group so timeout cleanup reaches descendants."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _collect_kilo_timeout_output(process)
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=timeout,
            output=stdout,
            stderr=stderr,
        ) from exc

    return subprocess.CompletedProcess(
        command,
        returncode=process.wait(),
        stdout=stdout,
        stderr=stderr,
    )


@dataclass(slots=True)
class KilocodeCliBackend:
    """AgentBackend implementation that delegates to the Kilocode CLI.

    Uses Kilo's non-interactive ``run --auto`` flow for headless worker
    orchestration. Loreley defaults to plain-text output because the CLI's
    structured ``--format json`` mode emits raw events that are not guaranteed
    to be a single final Markdown document.
    """

    bin: str = "kilo"
    mode: str | None = None
    agent: str | None = None
    model: str | None = None
    variant: str | None = None
    pure: bool = False
    timeout_seconds: int = DEFAULT_KILOCODE_TIMEOUT_SECONDS
    extra_env: dict[str, str] = field(default_factory=dict)
    json_output: bool = False
    error_cls: type[RuntimeError] = RuntimeError
    settings: Settings | None = None
    usage_tracking_enabled: bool = True
    usage_db_path: str | None = None
    state_root: str | None = None

    def run(
        self,
        task: AgentTask,
        *,
        working_dir: Path,
    ) -> AgentInvocation:
        worktree = validate_workdir(
            working_dir,
            error_cls=self.error_cls,
            agent_name=task.name or "Agent",
        )

        usage_title = self._usage_title(task)
        command = self._build_command(task.prompt, title=usage_title, worktree=worktree)
        command_for_log = command[:-1] + [f"<prompt:{len(task.prompt)} chars>"]

        env = self._build_env(task=task)
        try:
            result, duration = self._run_cli(
                command=command,
                command_for_log=command_for_log,
                env=env,
                task=task,
                worktree=worktree,
            )
        except RuntimeError as exc:
            usage_events = self._usage_events_from_kilo_db(
                task=task,
                title=usage_title,
                worktree=worktree,
            )
            if usage_events:
                setattr(exc, "usage_events", usage_events)
            raise
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        usage_events = self._usage_events_from_kilo_db(
            task=task,
            title=usage_title,
            worktree=worktree,
        )

        log.debug(
            "Kilocode CLI finished (exit_code={}, duration={:.2f}s) for task={}",
            result.returncode,
            duration,
            task.name,
        )

        self._raise_for_failure(
            result=result,
            stdout=stdout,
            stderr=stderr,
            usage_events=usage_events,
        )

        if not stdout:
            log.warning(
                "Kilocode CLI produced an empty stdout payload for task={} (command={})",
                task.name,
                command_for_log,
            )

        return AgentInvocation(
            command=tuple(command),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            usage_events=usage_events,
            working_directory=str(worktree),
        )

    def _build_env(self, *, task: AgentTask | None = None) -> dict[str, str]:
        explicit_extra_env = self.extra_env or {}
        env = os.environ.copy()
        env.update(explicit_extra_env)
        env.update(self._isolated_state_env(task))
        if _has_explicit_kilocode_api_key(explicit_extra_env):
            return env
        runtime_api_key_env = _runtime_api_key_env_name(explicit_extra_env)
        if runtime_api_key_env is None:
            return env
        runtime_api_key = self._resolved_runtime_api_key()
        if runtime_api_key:
            env[runtime_api_key_env] = runtime_api_key
        return env

    def _isolated_state_env(self, task: AgentTask | None) -> dict[str, str]:
        configured = str(self.state_root or "").strip()
        if not configured:
            return {}
        if task is None or task.job_id is None or task.run_token is None:
            raise self.error_cls(
                "Per-job Kilo state isolation requires job_id and run_token.",
            )
        state_dir = (
            Path(configured).expanduser().resolve()
            / f"{task.job_id}-{task.run_token}"
        )
        paths = {
            "HOME": state_dir / "home",
            "XDG_DATA_HOME": state_dir / "data",
            "XDG_CONFIG_HOME": state_dir / "config",
            "XDG_CACHE_HOME": state_dir / "cache",
            "TMPDIR": state_dir / "tmp",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.chmod(0o700)
        return {name: str(path) for name, path in paths.items()}

    def _resolved_runtime_api_key(self) -> str | None:
        try:
            return self._resolve_api_key()
        except Exception as exc:
            raise self.error_cls(
                "failed to resolve Kilocode OpenAI API key before launch.",
            ) from exc

    def _run_cli(
        self,
        *,
        command: list[str],
        command_for_log: list[str],
        env: dict[str, str],
        task: AgentTask,
        worktree: Path,
    ):
        start = monotonic()
        log.debug(
            "Running Kilocode CLI command: {} (cwd={}) for task={}",
            command_for_log,
            worktree,
            task.name,
        )
        try:
            result = _run_kilo_process(
                command,
                cwd=str(worktree),
                env=env,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise self.error_cls(
                f"kilo run timed out after {self.timeout_seconds}s.",
            ) from exc
        return result, monotonic() - start

    def _raise_for_failure(
        self,
        *,
        result,
        stdout: str,
        stderr: str,
        usage_events: tuple,
    ) -> None:
        failure_reason_code = _kilo_failure_reason_code(
            stdout=stdout,
            stderr=stderr,
        )
        if result.returncode == 0:
            error_detail = _kilo_json_error_detail(stdout)
            if error_detail is None:
                return
            error = self.error_cls(
                "kilo run reported an error event despite exit code 0. "
                f"stdout: {error_detail}",
            )
            if failure_reason_code:
                setattr(error, "failure_reason_code", failure_reason_code)
            if usage_events:
                setattr(error, "usage_events", usage_events)
            raise error
        detail_suffix = self._failure_detail_suffix(stdout=stdout, stderr=stderr)
        reason_suffix = (
            f" reason_code={failure_reason_code}." if failure_reason_code else ""
        )
        error = self.error_cls(
            f"kilo run failed with exit code {result.returncode}."
            f"{reason_suffix}{detail_suffix}",
        )
        if failure_reason_code:
            setattr(error, "failure_reason_code", failure_reason_code)
        if usage_events:
            setattr(error, "usage_events", usage_events)
        raise error

    @staticmethod
    def _failure_detail_suffix(*, stdout: str, stderr: str) -> str:
        details: list[str] = []
        for label, payload in (("stderr", stderr), ("stdout", stdout)):
            snippet = truncate_text(payload, limit=400)
            if snippet:
                details.append(f"{label}: {snippet}")
        return f" {' '.join(details)}" if details else ""

    def _resolve_api_key(self) -> str | None:
        settings = self.settings or get_settings()
        return get_agent_openai_api_key(settings)

    def _build_command(
        self,
        prompt: str,
        *,
        title: str | None = None,
        worktree: Path | None = None,
    ) -> list[str]:
        command: list[str] = [self.bin, "run", "--auto"]

        if self.pure:
            command.append("--pure")

        if worktree is not None:
            command.extend(["--dir", str(worktree)])

        if self.json_output:
            command.extend(["--format", "json"])

        selected_agent = (self.agent or self.mode or "").strip()
        if selected_agent:
            command.extend(["--agent", selected_agent])

        selected_model = (self.model or "").strip()
        if selected_model:
            command.extend(["--model", selected_model])

        selected_variant = (self.variant or "").strip()
        if selected_variant:
            command.extend(["--variant", selected_variant])

        if title:
            command.extend(["--title", title])

        command.append(prompt)
        return command

    def _usage_title(self, task: AgentTask) -> str | None:
        if not self.usage_tracking_enabled:
            return None
        phase = task.phase or task.name or ""
        if task.job_id is None or task.run_token is None or not phase:
            return None
        title = f"loreley:{task.job_id}:{task.run_token}:{phase}"
        if task.invocation is not None:
            title = f"{title}:invocation:{max(1, int(task.invocation))}"
        if task.attempt is not None:
            title = f"{title}:attempt:{max(1, int(task.attempt))}"
        return title

    def _usage_events_from_kilo_db(
        self,
        *,
        task: AgentTask,
        title: str | None,
        worktree: Path,
    ) -> tuple:
        if not self.usage_tracking_enabled or not title:
            return ()
        phase = task.phase or task.name or ""
        external_id = self._usage_external_id(task, phase=phase)
        try:
            event = self._read_usage_event(
                title=title,
                worktree=worktree,
                task=task,
                external_usage_id=external_id,
            )
        except KiloWorkspaceIsolationError as exc:
            raise self.error_cls(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - filesystem/SQLite dependent
            log.warning("Failed to read Kilocode usage for title={}: {}", title, exc)
            event = unavailable_usage_event(
                source="kilo_cli",
                phase=phase,
                api_surface="kilo_run",
                job_id=task.job_id,
                run_token=task.run_token,
                reason=f"kilo usage DB read failed: {exc}",
                external_usage_id=external_id,
            )
        if event is None:
            event = unavailable_usage_event(
                source="kilo_cli",
                phase=phase,
                api_surface="kilo_run",
                job_id=task.job_id,
                run_token=task.run_token,
                reason="kilo usage session was not found or had no token records",
                external_usage_id=external_id,
            )
        return (event,)

    def _read_usage_event(
        self,
        *,
        title: str,
        worktree: Path,
        task: AgentTask,
        external_usage_id: str,
    ):
        db_path = self._resolved_usage_db_path(task)
        if not db_path.is_file():
            raise KiloUsageUnavailableError(f"kilo usage DB not found: {db_path}")
        uri = f"file:{db_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            missing_schema = kilo_usage_schema_missing_reason(conn)
            if missing_schema:
                raise KiloUsageUnavailableError(missing_schema)
            session_row = _latest_kilo_session(conn, title)
            if session_row is None:
                return None
            session_id = str(session_row["id"])
            settings = self.settings or get_settings()
            session_rows = _kilo_session_tree(conn, session_row)
            _assert_kilo_session_directories(
                session_rows=session_rows,
                worktree=worktree,
                settings=settings,
            )
        event = kilo_usage_event_from_messages(
            (),
            session_rows=tuple(dict(row) for row in session_rows),
            phase=task.phase or task.name or "",
            job_id=task.job_id,
            run_token=task.run_token,
            title=title,
            session_id=session_id,
            settings=self.settings or get_settings(),
            external_usage_id=external_usage_id,
        )
        if event is None:
            return None
        return event.with_context(
            job_id=task.job_id, run_token=task.run_token, phase=task.phase
        )

    @staticmethod
    def _usage_external_id(task: AgentTask, *, phase: str) -> str:
        if task.job_id is None or task.run_token is None or not phase:
            return ""
        external_id = f"kilo:{task.job_id}:{task.run_token}:{phase}"
        if task.invocation is not None:
            external_id = f"{external_id}:invocation:{max(1, int(task.invocation))}"
        if task.attempt is not None:
            external_id = f"{external_id}:attempt:{max(1, int(task.attempt))}"
        return external_id

    def _resolved_usage_db_path(self, task: AgentTask | None = None) -> Path:
        if self.state_root:
            if task is None or task.job_id is None or task.run_token is None:
                raise KiloUsageUnavailableError(
                    "Per-job Kilo usage lookup requires job_id and run_token."
                )
            return (
                Path(self.state_root).expanduser().resolve()
                / f"{task.job_id}-{task.run_token}"
                / "data"
                / "kilo"
                / "kilo.db"
            )
        settings = self.settings or get_settings()
        return resolved_kilo_usage_db_path(
            kilo_bin=self.bin,
            configured_path=self.usage_db_path
            or getattr(settings, "worker_kilocode_usage_db_path", None),
        )


def _latest_kilo_session(conn: sqlite3.Connection, title: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT *
        FROM session
        WHERE title = ?
        ORDER BY time_updated DESC, time_created DESC
        LIMIT 1
        """,
        (title,),
    ).fetchone()


def _kilo_session_tree(
    conn: sqlite3.Connection,
    session_row: sqlite3.Row,
) -> list[sqlite3.Row]:
    session_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(session)").fetchall()
    }
    if "parent_id" not in session_columns:
        return [session_row]
    return conn.execute(
        """
        WITH RECURSIVE session_tree(id) AS (
            SELECT id
            FROM session
            WHERE id = ?
            UNION
            SELECT child.id
            FROM session AS child
            JOIN session_tree AS parent
              ON child.parent_id = parent.id
        )
        SELECT session.*
        FROM session
        JOIN session_tree ON session.id = session_tree.id
        """,
        (str(session_row["id"]),),
    ).fetchall()


def _assert_kilo_session_directories(
    *,
    session_rows: list[sqlite3.Row],
    worktree: Path,
    settings: Settings,
) -> None:
    for row in session_rows:
        assert_kilo_session_directory(
            expected_worktree=worktree,
            actual_directory=str(row["directory"] or ""),
            settings=settings,
        )


def _build_kilocode_openai_env(
    settings, *, api_key: str | None = None
) -> dict[str, str]:
    """Translate Loreley OpenAI-compatible settings into Kilo provider config."""

    mode = _kilocode_provider_config_mode(settings)
    if mode == "none":
        return {KILO_CONFIG_CONTENT_ENV: build_kilo_headless_config_content()}

    provider_input = _kilocode_provider_input(settings, api_key=api_key)
    if not provider_input["has_provider_config"]:
        return {KILO_CONFIG_CONTENT_ENV: build_kilo_headless_config_content()}
    if mode == "legacy_env":
        return {
            KILO_CONFIG_CONTENT_ENV: build_kilo_headless_config_content(),
            **_build_kilocode_legacy_openai_env(provider_input, api_key=api_key),
        }
    return _build_kilocode_config_openai_env(provider_input, api_key=api_key)


def _kilocode_provider_config_mode(settings) -> KiloProviderConfigMode:
    raw = str(
        getattr(settings, "worker_kilocode_provider_config_mode", "auto") or "auto"
    ).strip()
    normalized = raw.lower()
    if normalized not in KILO_PROVIDER_CONFIG_MODES:
        return "auto"
    return normalized  # type: ignore[return-value]


def _stripped_setting(settings, name: str) -> str:
    return str(getattr(settings, name, "") or "").strip()


def _has_kilocode_api_key_source(settings, *, api_key: str | None = None) -> bool:
    values = (
        api_key,
        getattr(settings, "worker_kilocode_openai_api_key", ""),
        getattr(settings, "openai_api_key", ""),
        getattr(settings, "openai_dynamic_api_key_provider", ""),
    )
    return any(str(value or "").strip() for value in values)


def _kilocode_provider_input(
    settings, *, api_key: str | None = None
) -> dict[str, object]:
    worker_base_url = _stripped_setting(settings, "worker_kilocode_openai_base_url")
    worker_model = _kilocode_gateway_model(settings)
    worker_api_spec = getattr(settings, "worker_kilocode_openai_api_spec", None)

    base_url = worker_base_url or _stripped_setting(settings, "openai_base_url")
    api_spec = worker_api_spec or getattr(settings, "openai_api_spec", None)
    has_api_key_source = _has_kilocode_api_key_source(settings, api_key=api_key)
    has_provider_config = any(
        (has_api_key_source, base_url, worker_model, worker_api_spec)
    )
    provider_id = _kilocode_provider_id(
        api_spec=api_spec, has_provider_config=has_provider_config
    )
    return {
        "api_spec": api_spec,
        "base_url": base_url,
        "model": worker_model,
        "provider_id": provider_id,
        "has_api_key_source": has_api_key_source,
        "has_provider_config": has_provider_config,
    }


def _kilocode_provider_id(*, api_spec: object, has_provider_config: bool) -> str | None:
    if api_spec == "responses":
        return KILO_RESPONSES_PROVIDER_ID
    if api_spec == "chat_completions":
        return KILO_CHAT_COMPLETIONS_PROVIDER_ID
    if has_provider_config:
        return KILO_CHAT_COMPLETIONS_PROVIDER_ID
    return None


def _kilocode_gateway_model(settings) -> str:
    explicit = _stripped_setting(settings, "worker_kilocode_openai_model")
    if explicit:
        return explicit
    selected = _stripped_setting(settings, "worker_kilocode_model")
    provider_id, separator, model_id = selected.partition("/")
    if not separator:
        return selected
    if provider_id in {
        "openai",
        "openai-responses",
        KILO_CHAT_COMPLETIONS_PROVIDER_ID,
        KILO_RESPONSES_PROVIDER_ID,
    }:
        return model_id
    return ""


def _build_kilocode_legacy_openai_env(
    provider_input: dict[str, object],
    *,
    api_key: str | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    provider_id = _legacy_kilocode_provider_id(provider_input.get("api_spec"))
    base_url = str(provider_input.get("base_url") or "")
    model = str(provider_input.get("model") or "")
    if provider_id:
        env["KILO_PROVIDER_TYPE"] = provider_id

    resolved_api_key = str(api_key or "").strip()
    if resolved_api_key:
        env["KILO_OPENAI_API_KEY"] = resolved_api_key
    if base_url:
        env["KILO_OPENAI_BASE_URL"] = base_url
    if model:
        env["KILO_OPENAI_MODEL_ID"] = model
    return env


def _legacy_kilocode_provider_id(api_spec: object) -> str:
    return "openai-responses" if api_spec == "responses" else "openai"


def _build_kilocode_config_openai_env(
    provider_input: dict[str, object],
    *,
    api_key: str | None = None,
) -> dict[str, str]:
    provider_id = str(provider_input.get("provider_id") or "")
    if not provider_id:
        return {}
    base_url = str(provider_input.get("base_url") or "")
    model = str(provider_input.get("model") or "")
    include_api_key = bool(provider_input.get("has_api_key_source"))
    env: dict[str, str] = {
        KILO_CONFIG_CONTENT_ENV: _build_kilo_config_content(
            provider_id=provider_id,
            base_url=base_url,
            model=model,
            include_api_key=include_api_key,
        )
    }
    if base_url:
        env[LORELEY_KILO_OPENAI_BASE_URL_ENV] = base_url
    resolved_api_key = str(api_key or "").strip()
    if resolved_api_key:
        env[LORELEY_KILO_OPENAI_API_KEY_ENV] = resolved_api_key
    return env


def _build_kilo_config_content(
    *,
    provider_id: str,
    base_url: str,
    model: str,
    include_api_key: bool,
) -> str:
    options: dict[str, str] = {}
    if include_api_key:
        options["apiKey"] = f"{{env:{LORELEY_KILO_OPENAI_API_KEY_ENV}}}"
    if base_url:
        options["baseURL"] = f"{{env:{LORELEY_KILO_OPENAI_BASE_URL_ENV}}}"
    provider: dict[str, object] = {
        "name": "Loreley OpenAI-compatible gateway",
        "npm": _kilo_provider_npm(provider_id),
        "options": options,
    }
    provider_model_id = _kilo_provider_model_id(
        provider_id=provider_id,
        model=model,
    )
    if provider_model_id:
        # Kilo validates --model against the provider catalog before making an
        # API request. Register custom OpenAI-compatible model IDs explicitly.
        provider["models"] = {
            provider_model_id: {
                "name": provider_model_id,
            }
        }
    payload = _kilo_headless_config_payload()
    payload.update(
        {
            "$schema": KILO_CONFIG_SCHEMA_URL,
            "provider": {
                provider_id: provider,
            },
        }
    )
    normalized_model = _kilo_config_model(provider_id=provider_id, model=model)
    if normalized_model:
        payload["model"] = normalized_model
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def build_kilo_headless_config_content() -> str:
    return json.dumps(
        _kilo_headless_config_payload(),
        separators=(",", ":"),
        sort_keys=True,
    )


def _kilo_headless_config_payload() -> dict[str, object]:
    permissions = {
        "interactive_terminal": "deny",
        "plan_enter": "deny",
        "plan_exit": "deny",
        "question": "deny",
        "suggest": "deny",
    }
    disabled_tools = {"suggest": False}
    return {
        "$schema": KILO_CONFIG_SCHEMA_URL,
        "agent": {
            KILO_HEADLESS_AGENT: {
                "description": "Non-interactive Loreley worker agent",
                "mode": "primary",
                "permission": dict(permissions),
                "tools": dict(disabled_tools),
                "prompt": (
                    "You are running headlessly inside Loreley. Never call "
                    "interactive_terminal, question, suggest, plan_enter, or "
                    "plan_exit. Complete the task with non-interactive tools, "
                    "then return the requested result immediately."
                ),
            }
        },
        "default_agent": KILO_HEADLESS_AGENT,
        "permission": permissions,
        "tools": disabled_tools,
    }


def _kilo_config_model(*, provider_id: str, model: str) -> str:
    value = str(model or "").strip()
    if not value:
        return ""
    if "/" in value:
        return value
    return f"{provider_id}/{value}"


def _kilo_provider_npm(provider_id: str) -> str:
    if provider_id == KILO_RESPONSES_PROVIDER_ID:
        return "@ai-sdk/openai"
    return "@ai-sdk/openai-compatible"


def _kilo_provider_model_id(*, provider_id: str, model: str) -> str:
    value = str(model or "").strip()
    if not value:
        return ""
    configured_provider, separator, model_id = value.partition("/")
    if not separator:
        return value
    if configured_provider != provider_id:
        return ""
    return model_id


def _kilo_json_error_detail(stdout: str) -> str | None:
    """Return the first structured Kilo error event from mixed stdout."""

    for line in str(stdout or "").splitlines():
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        payload = _kilo_json_object(candidate)
        if payload is None or payload.get("type") != "error":
            continue
        return _kilo_error_event_detail(payload, fallback=candidate)
    return None


def _kilo_failure_reason_code(*, stdout: str, stderr: str) -> str | None:
    """Return an allowlisted, non-sensitive reason code from Kilo output."""

    combined = f"{stdout}\n{stderr}"
    return next(
        (
            reason
            for reason in sorted(_KILO_FAILURE_REASON_CODES)
            if reason in combined
        ),
        None,
    )


def _kilo_json_object(candidate: str) -> dict[str, object] | None:
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _kilo_error_event_detail(
    payload: dict[str, object],
    *,
    fallback: str,
) -> str:
    error = payload.get("error")
    if not isinstance(error, dict):
        return truncate_text(fallback, limit=400)
    data = error.get("data")
    if isinstance(data, dict) and data.get("message"):
        return truncate_text(str(data["message"]), limit=400)
    return truncate_text(str(error.get("name") or fallback), limit=400)


def _kilocode_backend_model(settings, extra_env: dict[str, str]) -> str | None:
    config_content = extra_env.get(KILO_CONFIG_CONTENT_ENV)
    if config_content:
        try:
            configured_model = json.loads(config_content).get("model")
        except (json.JSONDecodeError, AttributeError):
            configured_model = None
        if str(configured_model or "").strip():
            return str(configured_model)
    selected_model = getattr(settings, "worker_kilocode_model", None)
    return str(selected_model) if selected_model else None


def _has_explicit_kilocode_api_key(env: dict[str, str]) -> bool:
    return bool(
        str(env.get(LORELEY_KILO_OPENAI_API_KEY_ENV, "") or "").strip()
        or str(env.get("KILO_OPENAI_API_KEY", "") or "").strip()
    )


def _runtime_api_key_env_name(env: dict[str, str]) -> str | None:
    if LORELEY_KILO_OPENAI_API_KEY_ENV in str(env.get(KILO_CONFIG_CONTENT_ENV, "")):
        return LORELEY_KILO_OPENAI_API_KEY_ENV
    if any(
        key in env
        for key in (
            "KILO_PROVIDER_TYPE",
            "KILO_OPENAI_BASE_URL",
            "KILO_OPENAI_MODEL_ID",
        )
    ):
        return "KILO_OPENAI_API_KEY"
    return None


def resolved_kilo_usage_db_path(
    *,
    kilo_bin: str,
    configured_path: str | None = None,
    timeout_seconds: float = 2.0,
) -> Path:
    raw_configured = str(configured_path or "").strip()
    if raw_configured:
        return Path(raw_configured).expanduser()
    result = _run_kilo_probe([kilo_bin, "db", "path"], timeout_seconds=timeout_seconds)
    raw = (result.stdout or "").strip()
    if result.returncode == 0 and raw:
        return Path(raw).expanduser()
    return Path("~/.local/share/kilo/kilo.db").expanduser()


def kilo_usage_schema_missing_reason(conn: sqlite3.Connection) -> str | None:
    for table, required_columns in _KILO_USAGE_REQUIRED_COLUMNS.items():
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if not rows:
            return f"kilo usage DB schema incompatible: missing table {table}"
        present = {str(row[1]) for row in rows}
        missing = sorted(required_columns - present)
        if missing:
            return (
                f"kilo usage DB schema incompatible: missing column "
                f"{table}.{missing[0]}"
            )
    return None


def assert_kilo_session_directory(
    *,
    expected_worktree: Path,
    actual_directory: str,
    settings: Settings,
) -> None:
    """Fail closed if Kilo recorded a session outside the job worktree."""

    expected = expected_worktree.expanduser().resolve()
    raw_actual = str(actual_directory or "").strip()
    if not raw_actual:
        raise KiloWorkspaceIsolationError(
            "Kilo workspace isolation failed: usage session did not record a "
            f"workspace directory (expected={expected})."
        )

    actual = Path(raw_actual).expanduser().resolve()
    if actual == expected:
        return

    danger = _kilo_session_directory_danger_reason(
        actual=actual,
        expected=expected,
        settings=settings,
    )
    danger_suffix = f" ({danger})" if danger else ""
    raise KiloWorkspaceIsolationError(
        "Kilo workspace isolation failed: usage session directory did not match "
        f"the requested job worktree{danger_suffix}; expected={expected}; "
        f"actual={actual}. Refusing to continue after kilo run."
    )


def _kilo_session_directory_danger_reason(
    *,
    actual: Path,
    expected: Path,
    settings: Settings,
) -> str:
    if actual in expected.parents:
        return "actual directory is a parent of the job worktree"

    dangerous_roots: list[tuple[str, Path]] = []
    scheduler_root = str(getattr(settings, "scheduler_repo_root", "") or "").strip()
    if scheduler_root:
        dangerous_roots.append(("scheduler repo root", Path(scheduler_root)))

    worker_base = str(getattr(settings, "worker_repo_worktree", "") or "").strip()
    if worker_base:
        dangerous_roots.append(("worker base worktree", Path(worker_base)))

    dangerous_roots.extend(
        [
            ("Loreley source checkout", Path(__file__).resolve().parents[5]),
            ("process current working directory", Path.cwd()),
        ]
    )
    for label, raw_root in dangerous_roots:
        root = raw_root.expanduser().resolve()
        if actual == root:
            return f"actual directory is the {label}"
    return ""


def kilocode_backend() -> KilocodeCliBackend:
    """Factory to build a Kilocode backend using env-only settings."""

    settings = get_settings()
    bin_value = getattr(settings, "worker_kilocode_bin", "kilo")
    mode_value = getattr(settings, "worker_kilocode_mode", None)
    agent_value = getattr(settings, "worker_kilocode_agent", None) or mode_value
    variant_value = getattr(settings, "worker_kilocode_variant", None)
    pure_value = getattr(settings, "worker_kilocode_pure", False)
    json_output_value = getattr(settings, "worker_kilocode_json_output", False)
    extra_env = _build_kilocode_openai_env(settings)
    model_value = _kilocode_backend_model(settings, extra_env)
    return KilocodeCliBackend(
        bin=str(bin_value),
        mode=str(mode_value) if mode_value else None,
        agent=str(agent_value) if agent_value else None,
        model=str(model_value) if model_value else None,
        variant=str(variant_value) if variant_value else None,
        pure=bool(pure_value),
        timeout_seconds=DEFAULT_KILOCODE_TIMEOUT_SECONDS,
        json_output=bool(json_output_value),
        extra_env=extra_env,
        settings=settings,
        usage_tracking_enabled=bool(settings.llm_usage_tracking_enabled),
        usage_db_path=settings.worker_kilocode_usage_db_path,
        state_root=settings.worker_kilocode_state_root,
    )


def kilocode_planning_backend() -> KilocodeCliBackend:
    """Factory to build a Kilocode backend for the planning agent.

    Uses the planning agent's error type so the shared retry loop can capture
    failures, emit debug artifacts, and retry when appropriate.
    """

    from loreley.core.worker.planning import PlanningError

    settings = get_settings()
    bin_value = getattr(settings, "worker_kilocode_bin", "kilo")
    mode_value = getattr(settings, "worker_kilocode_mode", None)
    agent_value = getattr(settings, "worker_kilocode_agent", None) or mode_value
    variant_value = getattr(settings, "worker_kilocode_variant", None)
    pure_value = getattr(settings, "worker_kilocode_pure", False)
    json_output_value = getattr(settings, "worker_kilocode_json_output", False)
    extra_env = _build_kilocode_openai_env(settings)
    model_value = _kilocode_backend_model(settings, extra_env)
    return KilocodeCliBackend(
        bin=str(bin_value),
        mode=str(mode_value) if mode_value else None,
        agent=str(agent_value) if agent_value else None,
        model=str(model_value) if model_value else None,
        variant=str(variant_value) if variant_value else None,
        pure=bool(pure_value),
        timeout_seconds=settings.worker_planning_timeout_seconds,
        json_output=bool(json_output_value),
        extra_env=extra_env,
        error_cls=PlanningError,
        settings=settings,
        usage_tracking_enabled=bool(settings.llm_usage_tracking_enabled),
        usage_db_path=settings.worker_kilocode_usage_db_path,
        state_root=settings.worker_kilocode_state_root,
    )


def kilocode_coding_backend() -> KilocodeCliBackend:
    """Factory to build a Kilocode backend for the coding agent.

    Uses the coding agent's error type so the shared retry loop can capture
    failures, emit debug artifacts, and retry when appropriate.
    """

    from loreley.core.worker.coding import CodingError

    settings = get_settings()
    bin_value = getattr(settings, "worker_kilocode_bin", "kilo")
    mode_value = getattr(settings, "worker_kilocode_mode", None)
    agent_value = getattr(settings, "worker_kilocode_agent", None) or mode_value
    variant_value = getattr(settings, "worker_kilocode_variant", None)
    pure_value = getattr(settings, "worker_kilocode_pure", False)
    json_output_value = getattr(settings, "worker_kilocode_json_output", False)
    extra_env = _build_kilocode_openai_env(settings)
    model_value = _kilocode_backend_model(settings, extra_env)
    return KilocodeCliBackend(
        bin=str(bin_value),
        mode=str(mode_value) if mode_value else None,
        agent=str(agent_value) if agent_value else None,
        model=str(model_value) if model_value else None,
        variant=str(variant_value) if variant_value else None,
        pure=bool(pure_value),
        timeout_seconds=settings.worker_coding_timeout_seconds,
        json_output=bool(json_output_value),
        extra_env=extra_env,
        error_cls=CodingError,
        settings=settings,
        usage_tracking_enabled=bool(settings.llm_usage_tracking_enabled),
        usage_db_path=settings.worker_kilocode_usage_db_path,
        state_root=settings.worker_kilocode_state_root,
    )


__all__ = [
    "KilocodeCliBackend",
    "KiloCliCapabilities",
    "KiloWorkspaceIsolationError",
    "assert_kilo_session_directory",
    "build_kilo_headless_config_content",
    "discover_kilo_cli_capabilities",
    "kilocode_backend",
    "kilocode_coding_backend",
    "kilocode_planning_backend",
    "parse_kilo_run_flags",
    "parse_kilo_version",
]
