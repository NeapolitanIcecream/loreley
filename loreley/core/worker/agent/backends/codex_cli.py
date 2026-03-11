from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
import tempfile
from time import monotonic

from loguru import logger

from loreley.config import get_settings
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
    profile: str | None
    timeout_seconds: int
    extra_env: dict[str, str]
    error_cls: type[RuntimeError]
    full_auto: bool = False
    approval_policy: str = "never"
    sandbox: str | None = None
    color: str = "never"
    ephemeral: bool = True
    capture_last_message: bool = True

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
            output_last_message_path = (
                Path(temp_dir) / "last-message.md" if self.capture_last_message else None
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
            raise self.error_cls(
                f"codex exec failed with exit code {result.returncode}.{detail_suffix}",
            )

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
        )

    def _build_command(
        self,
        *,
        output_last_message_path: Path | None,
    ) -> list[str]:
        command: list[str] = [self.bin, "exec"]

        if self.ephemeral:
            command.append("--ephemeral")

        if self.profile:
            command.extend(["--profile", self.profile])

        if self.approval_policy:
            command.extend(["--ask-for-approval", self.approval_policy])

        sandbox = self.sandbox or ("workspace-write" if self.full_auto else "read-only")
        if sandbox:
            command.extend(["--sandbox", sandbox])

        if self.color:
            command.extend(["--color", self.color])

        if output_last_message_path is not None:
            command.extend(["--output-last-message", str(output_last_message_path)])

        return command

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


def codex_planning_backend() -> CodexCliBackend:
    """Factory to build a Codex backend for the planning agent."""

    from loreley.core.worker.planning import PlanningError

    settings = get_settings()
    return CodexCliBackend(
        bin=settings.worker_planning_codex_bin,
        profile=settings.worker_planning_codex_profile,
        timeout_seconds=settings.worker_planning_timeout_seconds,
        extra_env=dict(settings.worker_planning_extra_env or {}),
        error_cls=PlanningError,
        full_auto=False,
    )


def codex_coding_backend() -> CodexCliBackend:
    """Factory to build a Codex backend for the coding agent."""

    from loreley.core.worker.coding import CodingError

    settings = get_settings()
    return CodexCliBackend(
        bin=settings.worker_coding_codex_bin,
        profile=settings.worker_coding_codex_profile,
        timeout_seconds=settings.worker_coding_timeout_seconds,
        extra_env=dict(settings.worker_coding_extra_env or {}),
        error_cls=CodingError,
        full_auto=True,
    )


__all__ = [
    "CodexCliBackend",
    "codex_coding_backend",
    "codex_planning_backend",
]
