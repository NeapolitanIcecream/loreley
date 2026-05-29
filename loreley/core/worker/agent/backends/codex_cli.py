from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import tempfile
from time import monotonic

from loguru import logger

from loreley.config import get_settings
from loreley.core.usage import codex_usage_event_from_jsonl, unavailable_usage_event
from loreley.core.worker.agent.contracts import AgentInvocation, AgentTask
from loreley.core.worker.agent.utils import truncate_text, validate_workdir

log = logger.bind(module="worker.agent.backends.codex_cli")


@dataclass(slots=True)
class CodexCliBackend:
    """AgentBackend implementation that delegates to the Codex CLI.

    Loreley runs Codex in a fully non-interactive worker context, so it avoids
    ``--full-auto`` and instead passes explicit approval/sandbox flags. This is
    more predictable because Codex documents ``--full-auto`` as
    ``-a on-request --sandbox workspace-write``, while non-interactive runs
    should prefer ``--ask-for-approval never``.
    """

    bin: str
    timeout_seconds: int
    extra_env: dict[str, str]
    error_cls: type[RuntimeError]
    model: str | None = None
    profile: str | None = None
    full_auto: bool = False
    approval_policy: str = "never"
    sandbox: str | None = None
    color: str = "never"
    ephemeral: bool = True
    capture_last_message: bool = True
    isolate_home: bool = True
    usage_tracking_enabled: bool = True

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

        env = os.environ.copy()
        env.setdefault("CODEX_QUIET_MODE", "1")
        env.update(self.extra_env or {})

        start = monotonic()
        with tempfile.TemporaryDirectory(prefix="loreley-codex-") as temp_dir:
            temp_path = Path(temp_dir)
            output_last_message_path = (
                temp_path / "last-message.md" if self.capture_last_message else None
            )
            self._prepare_isolated_home(
                env=env,
                temp_dir=temp_path,
                worktree=worktree,
            )
            command = self._build_command(output_last_message_path=output_last_message_path)
            log.debug(
                "Running Codex CLI command: {} (cwd={}) for task={}",
                command,
                worktree,
                task.name,
            )
            try:
                result = subprocess.run(
                    command,
                    cwd=str(worktree),
                    env=env,
                    input=task.prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise self.error_cls(
                    f"codex exec timed out after {self.timeout_seconds}s.",
                ) from exc

            captured_last_message = self._read_last_message(output_last_message_path)

        duration = monotonic() - start
        raw_stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        stdout = captured_last_message or raw_stdout
        usage_events = self._usage_events_from_stdout(raw_stdout=raw_stdout, task=task)

        log.debug(
            "Codex CLI finished (exit_code={}, duration={:.2f}s) for task={}",
            result.returncode,
            duration,
            task.name,
        )

        if result.returncode != 0:
            details: list[str] = []
            for label, payload in (
                ("stderr", stderr),
                ("stdout", raw_stdout),
                ("last_message", captured_last_message),
            ):
                snippet = truncate_text(payload, limit=400)
                if snippet and (label != "last_message" or snippet != truncate_text(raw_stdout, limit=400)):
                    details.append(f"{label}: {snippet}")
            detail_suffix = f" {' '.join(details)}" if details else ""
            error = self.error_cls(
                f"codex exec failed with exit code {result.returncode}.{detail_suffix}",
            )
            if usage_events:
                setattr(error, "usage_events", usage_events)
            raise error

        if captured_last_message and raw_stdout and captured_last_message != raw_stdout:
            log.debug(
                "Codex CLI emitted auxiliary stdout ({} chars); using last-message payload ({} chars) for task={}",
                len(raw_stdout),
                len(captured_last_message),
                task.name,
            )

        if not stdout:
            log.warning(
                "Codex CLI produced an empty stdout payload for task={} (command={})",
                task.name,
                command,
            )

        return AgentInvocation(
            command=tuple(command),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            usage_events=usage_events,
        )

    def _build_command(
        self,
        *,
        output_last_message_path: Path | None,
    ) -> list[str]:
        command: list[str] = [self.bin]

        if self.model:
            command.extend(["--model", self.model])

        if self.profile:
            command.extend(["--profile", self.profile])

        if self.approval_policy:
            command.extend(["-a", self.approval_policy])

        sandbox = self._requested_sandbox()
        if sandbox:
            command.extend(["--sandbox", sandbox])

        command.append("exec")

        if self.ephemeral:
            command.append("--ephemeral")

        if self.color:
            command.extend(["--color", self.color])

        if self.usage_tracking_enabled:
            command.append("--json")

        if output_last_message_path is not None:
            command.extend(["--output-last-message", str(output_last_message_path)])

        return command

    def _usage_events_from_stdout(self, *, raw_stdout: str, task: AgentTask) -> tuple:
        if not self.usage_tracking_enabled:
            return ()
        phase = task.phase or task.name or ""
        external_id = self._usage_external_id(task, phase=phase)
        event = codex_usage_event_from_jsonl(
            raw_stdout,
            phase=phase,
            job_id=task.job_id,
            run_token=task.run_token,
            model=self.model,
            settings=get_settings(),
            external_usage_id=external_id,
        )
        if event is not None:
            return (event,)
        if task.job_id is None or task.run_token is None:
            return ()
        return (
            unavailable_usage_event(
                source="codex_cli",
                phase=phase,
                provider="openai",
                model=self.model or "",
                api_surface="codex_exec",
                job_id=task.job_id,
                run_token=task.run_token,
                reason="codex exec --json did not emit a token_count event",
                external_usage_id=external_id,
            ),
        )

    @staticmethod
    def _usage_external_id(task: AgentTask, *, phase: str) -> str:
        if task.job_id is None or task.run_token is None or not phase:
            return ""
        external_id = f"codex:{task.job_id}:{task.run_token}:{phase}"
        if task.attempt is not None:
            external_id = f"{external_id}:attempt:{max(1, int(task.attempt))}"
        return external_id

    def _read_last_message(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            log.debug("Failed to read Codex last-message output from {}: {}", path, exc)
            return ""

    def _prepare_isolated_home(
        self,
        *,
        env: dict[str, str],
        temp_dir: Path,
        worktree: Path,
    ) -> None:
        if not self.isolate_home:
            return
        if "CODEX_HOME" in (self.extra_env or {}):
            return

        source_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex")).expanduser()
        source_auth = source_home / "auth.json"
        isolated_home = temp_dir / "codex-home"
        isolated_home.mkdir(parents=True, exist_ok=True)
        if source_auth.is_file():
            shutil.copy2(source_auth, isolated_home / "auth.json")
        trusted_worktree = None
        if self._requested_sandbox() != "read-only":
            trusted_worktree = json.dumps(str(worktree.resolve()))
            (isolated_home / "config.toml").write_text(
                f"[projects.{trusted_worktree}]\ntrust_level = \"trusted\"\n",
                encoding="utf-8",
            )
        env["CODEX_HOME"] = str(isolated_home)
        log.debug(
            "Prepared isolated CODEX_HOME={} for task auth_copied={} trusted_worktree={}",
            isolated_home,
            source_auth.is_file(),
            trusted_worktree or "<none>",
        )

    def _requested_sandbox(self) -> str | None:
        return self.sandbox or ("workspace-write" if self.full_auto else "read-only")


def codex_planning_backend() -> CodexCliBackend:
    """Factory to build a Codex backend for the planning agent."""

    from loreley.core.worker.planning import PlanningError

    settings = get_settings()
    return CodexCliBackend(
        bin=settings.worker_planning_codex_bin,
        model=settings.worker_planning_codex_model,
        profile=settings.worker_planning_codex_profile,
        timeout_seconds=settings.worker_planning_timeout_seconds,
        extra_env=dict(settings.worker_planning_extra_env or {}),
        error_cls=PlanningError,
        full_auto=False,
        sandbox="read-only",
        usage_tracking_enabled=settings.llm_usage_tracking_enabled,
    )


def codex_coding_backend() -> CodexCliBackend:
    """Factory to build a Codex backend for the coding agent."""

    from loreley.core.worker.coding import CodingError

    settings = get_settings()
    return CodexCliBackend(
        bin=settings.worker_coding_codex_bin,
        model=settings.worker_coding_codex_model,
        profile=settings.worker_coding_codex_profile,
        timeout_seconds=settings.worker_coding_timeout_seconds,
        extra_env=dict(settings.worker_coding_extra_env or {}),
        error_cls=CodingError,
        full_auto=True,
        sandbox="workspace-write",
        usage_tracking_enabled=settings.llm_usage_tracking_enabled,
    )


__all__ = [
    "CodexCliBackend",
    "codex_coding_backend",
    "codex_planning_backend",
]
