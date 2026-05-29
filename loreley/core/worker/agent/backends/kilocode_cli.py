from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from loguru import logger

from loreley.config import Settings, get_settings
from loreley.core.openai_auth import get_agent_openai_api_key
from loreley.core.usage import kilo_usage_event_from_messages, unavailable_usage_event
from loreley.core.worker.agent.contracts import AgentInvocation, AgentTask
from loreley.core.worker.agent.utils import truncate_text, validate_workdir

log = logger.bind(module="worker.agent.backends.kilocode_cli")


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
    timeout_seconds: int = 1800
    extra_env: dict[str, str] = field(default_factory=dict)
    json_output: bool = False
    error_cls: type[RuntimeError] = RuntimeError
    settings: Settings | None = None
    usage_tracking_enabled: bool = True
    usage_db_path: str | None = None

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
        command = self._build_command(task.prompt, title=usage_title)
        command_for_log = command[:-1] + [f"<prompt:{len(task.prompt)} chars>"]

        env = self._build_env()
        result, duration = self._run_cli(
            command=command,
            command_for_log=command_for_log,
            env=env,
            task=task,
            worktree=worktree,
        )
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
        )

    def _build_env(self) -> dict[str, str]:
        explicit_extra_env = self.extra_env or {}
        env = os.environ.copy()
        env.update(explicit_extra_env)
        if "KILO_OPENAI_API_KEY" in explicit_extra_env:
            return env
        runtime_api_key = self._resolved_runtime_api_key()
        if runtime_api_key:
            env["KILO_OPENAI_API_KEY"] = runtime_api_key
        return env

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
            result = subprocess.run(
                command,
                cwd=str(worktree),
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
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
        if result.returncode == 0:
            return
        detail_suffix = self._failure_detail_suffix(stdout=stdout, stderr=stderr)
        error = self.error_cls(
            f"kilo run failed with exit code {result.returncode}.{detail_suffix}",
        )
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

    def _build_command(self, prompt: str, *, title: str | None = None) -> list[str]:
        command: list[str] = [self.bin, "run", "--auto"]

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
        db_path = self._resolved_usage_db_path()
        if not db_path.is_file():
            return None
        uri = f"file:{db_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            session_row = conn.execute(
                """
                SELECT id
                FROM session
                WHERE title = ?
                  AND (directory = ? OR ? = '')
                ORDER BY time_updated DESC, time_created DESC
                LIMIT 1
                """,
                (title, str(worktree), str(worktree)),
            ).fetchone()
            if session_row is None:
                session_row = conn.execute(
                    """
                    SELECT id
                    FROM session
                    WHERE title = ?
                    ORDER BY time_updated DESC, time_created DESC
                    LIMIT 1
                    """,
                    (title,),
                ).fetchone()
            if session_row is None:
                return None
            session_id = str(session_row["id"])
            rows = conn.execute(
                """
                SELECT data
                FROM message
                WHERE session_id = ?
                ORDER BY time_created ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        messages: list[dict[str, object]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["data"] or "{}"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                messages.append(payload)
        event = kilo_usage_event_from_messages(
            messages,
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
        return event.with_context(job_id=task.job_id, run_token=task.run_token, phase=task.phase)

    @staticmethod
    def _usage_external_id(task: AgentTask, *, phase: str) -> str:
        if task.job_id is None or task.run_token is None or not phase:
            return ""
        external_id = f"kilo:{task.job_id}:{task.run_token}:{phase}"
        if task.attempt is not None:
            external_id = f"{external_id}:attempt:{max(1, int(task.attempt))}"
        return external_id

    def _resolved_usage_db_path(self) -> Path:
        settings = self.settings or get_settings()
        raw = (
            self.usage_db_path
            or getattr(settings, "worker_kilocode_usage_db_path", None)
            or "~/.local/share/kilo/kilo.db"
        )
        return Path(str(raw)).expanduser()


def _build_kilocode_openai_env(settings, *, api_key: str | None = None) -> dict[str, str]:
    """Translate Loreley settings into Kilo Code CLI provider env config.

    Worker-specific ``WORKER_KILOCODE_OPENAI_*`` values take precedence. When
    absent, Loreley falls back to the global OpenAI-compatible settings so the
    same gateway credentials can drive both internal SDK calls and the spawned
    Kilo subprocess.

    Kilo Code CLI supports provider configuration through environment variables.
    For OpenAI-compatible
    endpoints, use:
    - ``KILO_PROVIDER_TYPE=openai`` (Chat Completions)
    - ``KILO_PROVIDER_TYPE=openai-responses`` (Responses)
    - ``KILO_OPENAI_API_KEY``
    - ``KILO_OPENAI_BASE_URL`` (optional; required for OpenAI-compatible gateways)
    - ``KILO_OPENAI_MODEL_ID``

    Loreley maps ``WORKER_KILOCODE_OPENAI_API_SPEC`` to the provider type:
    - ``chat_completions`` -> ``openai``
    - ``responses`` -> ``openai-responses``

    Reference: ``cli/docs/ENVIRONMENT_VARIABLES.md`` in the upstream Kilocode repo.
    """

    worker_base_url = (getattr(settings, "worker_kilocode_openai_base_url", None) or "").strip()
    worker_model = (getattr(settings, "worker_kilocode_openai_model", None) or "").strip()
    worker_api_spec = getattr(settings, "worker_kilocode_openai_api_spec", None)

    base_url = worker_base_url or (getattr(settings, "openai_base_url", None) or "").strip()
    model = worker_model
    api_spec = worker_api_spec or getattr(settings, "openai_api_spec", None)

    env: dict[str, str] = {}
    provider_type: str | None = None
    if api_spec == "responses":
        provider_type = "openai-responses"
    elif api_spec == "chat_completions":
        provider_type = "openai"
    elif api_key or base_url or model:
        provider_type = "openai"

    if provider_type:
        env["KILO_PROVIDER_TYPE"] = provider_type

    resolved_api_key = str(api_key or "").strip()
    if resolved_api_key:
        env["KILO_OPENAI_API_KEY"] = resolved_api_key
    if base_url:
        env["KILO_OPENAI_BASE_URL"] = base_url
    if model:
        env["KILO_OPENAI_MODEL_ID"] = model
    return env


def kilocode_backend() -> KilocodeCliBackend:
    """Factory to build a Kilocode backend using env-only settings."""

    settings = get_settings()
    bin_value = getattr(settings, "worker_kilocode_bin", "kilo")
    mode_value = getattr(settings, "worker_kilocode_mode", None)
    agent_value = getattr(settings, "worker_kilocode_agent", None) or mode_value
    model_value = getattr(settings, "worker_kilocode_model", None)
    variant_value = getattr(settings, "worker_kilocode_variant", None)
    json_output_value = getattr(settings, "worker_kilocode_json_output", False)
    extra_env = _build_kilocode_openai_env(settings)
    return KilocodeCliBackend(
        bin=str(bin_value),
        mode=str(mode_value) if mode_value else None,
        agent=str(agent_value) if agent_value else None,
        model=str(model_value) if model_value else None,
        variant=str(variant_value) if variant_value else None,
        json_output=bool(json_output_value),
        extra_env=extra_env,
        settings=settings,
        usage_tracking_enabled=bool(settings.llm_usage_tracking_enabled),
        usage_db_path=settings.worker_kilocode_usage_db_path,
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
    model_value = getattr(settings, "worker_kilocode_model", None)
    variant_value = getattr(settings, "worker_kilocode_variant", None)
    json_output_value = getattr(settings, "worker_kilocode_json_output", False)
    extra_env = _build_kilocode_openai_env(settings)
    return KilocodeCliBackend(
        bin=str(bin_value),
        mode=str(mode_value) if mode_value else None,
        agent=str(agent_value) if agent_value else None,
        model=str(model_value) if model_value else None,
        variant=str(variant_value) if variant_value else None,
        json_output=bool(json_output_value),
        extra_env=extra_env,
        error_cls=PlanningError,
        settings=settings,
        usage_tracking_enabled=bool(settings.llm_usage_tracking_enabled),
        usage_db_path=settings.worker_kilocode_usage_db_path,
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
    model_value = getattr(settings, "worker_kilocode_model", None)
    variant_value = getattr(settings, "worker_kilocode_variant", None)
    json_output_value = getattr(settings, "worker_kilocode_json_output", False)
    extra_env = _build_kilocode_openai_env(settings)
    return KilocodeCliBackend(
        bin=str(bin_value),
        mode=str(mode_value) if mode_value else None,
        agent=str(agent_value) if agent_value else None,
        model=str(model_value) if model_value else None,
        variant=str(variant_value) if variant_value else None,
        json_output=bool(json_output_value),
        extra_env=extra_env,
        error_cls=CodingError,
        settings=settings,
        usage_tracking_enabled=bool(settings.llm_usage_tracking_enabled),
        usage_db_path=settings.worker_kilocode_usage_db_path,
    )


__all__ = [
    "KilocodeCliBackend",
    "kilocode_backend",
    "kilocode_coding_backend",
    "kilocode_planning_backend",
]
