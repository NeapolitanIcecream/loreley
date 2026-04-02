from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from loguru import logger

from loreley.config import Settings, get_settings
from loreley.core.openai_auth import get_agent_openai_api_key
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

        command = self._build_command(task.prompt)
        command_for_log = command[:-1] + [f"<prompt:{len(task.prompt)} chars>"]

        explicit_extra_env = self.extra_env or {}
        preserve_extra_env_api_key = "KILO_OPENAI_API_KEY" in explicit_extra_env
        env = os.environ.copy()
        env.update(explicit_extra_env)
        runtime_api_key = self._resolve_api_key()
        if runtime_api_key and not preserve_extra_env_api_key:
            env["KILO_OPENAI_API_KEY"] = runtime_api_key

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

        duration = monotonic() - start
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        log.debug(
            "Kilocode CLI finished (exit_code={}, duration={:.2f}s) for task={}",
            result.returncode,
            duration,
            task.name,
        )

        if result.returncode != 0:
            details: list[str] = []
            for label, payload in (("stderr", stderr), ("stdout", stdout)):
                snippet = truncate_text(payload, limit=400)
                if snippet:
                    details.append(f"{label}: {snippet}")
            detail_suffix = f" {' '.join(details)}" if details else ""
            raise self.error_cls(
                f"kilo run failed with exit code {result.returncode}.{detail_suffix}",
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
        )

    def _resolve_api_key(self) -> str | None:
        settings = self.settings or get_settings()
        return get_agent_openai_api_key(settings)

    def _build_command(self, prompt: str) -> list[str]:
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

        command.append(prompt)
        return command


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
    )


__all__ = [
    "KilocodeCliBackend",
    "kilocode_backend",
    "kilocode_coding_backend",
    "kilocode_planning_backend",
]
