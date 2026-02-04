from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from git import Repo
from git.exc import InvalidGitRepositoryError, NoSuchPathError

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.core.worker.agent import (
    AgentBackend,
    AgentInvocation,
    SchemaMode,
    TruncationMixin,
    ValidationMode,
    build_structured_agent_task,
    coerce_structured_output,
    load_agent_backend,
    resolve_schema_mode,
    resolve_worker_debug_dir,
    run_structured_agent_task,
)
from loreley.core.worker.agent.backends import CodexCliBackend
from loreley.core.worker.planning import PlanStep, PlanningPlan
from loreley.core.worker.output_sanitizer import sanitize_json_payload

console = Console()
log = logger.bind(module="worker.coding")

__all__ = [
    "CodingAgent",
    "CodingAgentRequest",
    "CodingAgentResponse",
    "CodingError",
    "CodingPlanExecution",
    "CodingStepReport",
    "StepExecutionStatus",
]


class CodingError(RuntimeError):
    """Raised when the coding agent cannot implement a plan."""


class StepExecutionStatus(str, Enum):
    """Enum describing how a plan step was handled."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass(slots=True)
class CodingStepReport:
    """Structured summary of a single plan step execution."""

    step_id: str
    status: StepExecutionStatus
    summary: str
    files: tuple[str, ...] = field(default_factory=tuple)
    commands: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class CodingPlanExecution:
    """Aggregate execution metadata emitted by the coding agent."""

    implementation_summary: str
    commit_message: str | None
    step_results: tuple[CodingStepReport, ...]
    tests_executed: tuple[str, ...]
    tests_recommended: tuple[str, ...]
    follow_up_items: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(slots=True)
class CodingAgentRequest:
    """Input payload for the coding agent."""

    goal: str
    plan: PlanningPlan
    base_commit: str
    constraints: Sequence[str] = field(default_factory=tuple)
    acceptance_criteria: Sequence[str] = field(default_factory=tuple)
    iteration_hint: str | None = None
    additional_notes: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.constraints = tuple(self.constraints or ())
        self.acceptance_criteria = tuple(self.acceptance_criteria or ())
        self.additional_notes = tuple(self.additional_notes or ())


@dataclass(slots=True)
class CodingAgentResponse:
    """Envelope containing coding agent output."""

    execution: CodingPlanExecution
    raw_output: str
    prompt: str
    command: tuple[str, ...]
    stderr: str
    attempts: int
    duration_seconds: float


class _StepResultModel(BaseModel):
    """Pydantic schema for plan step execution results."""

    model_config = ConfigDict(frozen=True)

    step_id: str
    status: StepExecutionStatus
    summary: str
    files: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)


class _CodingOutputModel(BaseModel):
    """Top-level schema representing coding agent output."""

    model_config = ConfigDict(frozen=True)

    implementation_summary: str
    commit_message: str | None = None
    step_results: list[_StepResultModel]
    tests_executed: list[str] = Field(default_factory=list)
    tests_recommended: list[str] = Field(default_factory=list)
    follow_up_items: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


CODING_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "implementation_summary": {"type": "string", "minLength": 1, "maxLength": 2000},
        "commit_message": {"type": ["string", "null"], "maxLength": 200},
        "step_results": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "status": {
                        "type": "string",
                        "enum": [status.value for status in StepExecutionStatus],
                    },
                    "summary": {"type": "string", "minLength": 1, "maxLength": 800},
                    "files": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 50},
                    "commands": {"type": "array", "items": {"type": "string", "maxLength": 512}, "maxItems": 50},
                },
                "required": ["step_id", "status", "summary"],
                "additionalProperties": False,
            },
        },
        "tests_executed": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 50},
        "tests_recommended": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 50},
        "follow_up_items": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 50},
        "notes": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 50},
    },
    "required": ["implementation_summary", "step_results"],
    "additionalProperties": False,
}


class CodingAgent(TruncationMixin):
    """Drive the configured coding backend to implement a plan on the repository."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: AgentBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_attempts = max(1, self.settings.worker_coding_max_attempts)
        self.validation_mode: ValidationMode = self.settings.worker_coding_validation_mode
        self.schema_mode: SchemaMode = resolve_schema_mode(
            configured_mode=self.settings.worker_coding_codex_schema_mode,
            api_spec=self.settings.openai_api_spec,
        )
        self._truncate_limit = 2000
        self._debug_dir = resolve_worker_debug_dir(
            logs_base_dir=self.settings.logs_base_dir,
            kind="coding",
            experiment_id=self.settings.experiment_id,
        )
        if backend is not None:
            self.backend: AgentBackend = backend
        elif self.settings.worker_coding_backend:
            self.backend = load_agent_backend(
                self.settings.worker_coding_backend,
                label="coding backend",
            )
        else:
            self.backend = CodexCliBackend(
                bin=self.settings.worker_coding_codex_bin,
                profile=self.settings.worker_coding_codex_profile,
                timeout_seconds=self.settings.worker_coding_timeout_seconds,
                extra_env=dict(self.settings.worker_coding_extra_env or {}),
                schema_override=self.settings.worker_coding_schema_path,
                error_cls=CodingError,
                full_auto=True,
            )

    def implement(
        self,
        request: CodingAgentRequest,
        *,
        working_dir: Path,
    ) -> CodingAgentResponse:
        """Execute the provided plan and return structured results."""
        worktree = Path(working_dir).expanduser().resolve()
        prompt = self._render_prompt(request, worktree=worktree)
        baseline_status = self._snapshot_worktree_state(worktree)

        task = build_structured_agent_task(
            name="coding",
            prompt=prompt,
            schema=CODING_OUTPUT_SCHEMA,
            schema_mode=self.schema_mode,
            validation_mode=self.validation_mode,
        )

        class _NoRepoChangeError(CodingError):
            """Raised when a coding attempt does not produce repository changes."""

        def _debug_hook(
            attempt: int,
            invocation: AgentInvocation | None,
            execution: CodingPlanExecution | None,
            error: Exception | None,
        ) -> None:
            self._dump_debug_artifact(
                request=request,
                worktree=worktree,
                invocation=invocation,
                prompt=prompt,
                attempt=attempt,
                execution=execution,
                error=error,
            )

        def _post_check(invocation: AgentInvocation, execution: CodingPlanExecution) -> Exception | None:
            current_status = self._snapshot_worktree_state(worktree)
            if current_status == baseline_status:
                return _NoRepoChangeError(
                    "Coding agent finished without producing repository changes.",
                )
            return None

        def _on_attempt_start(attempt: int, total: int) -> None:
            console.log(
                "[cyan]Coding agent[/] requesting execution "
                f"(attempt {attempt}/{total})",
            )

        def _on_attempt_success(
            attempt: int,
            total: int,
            invocation: AgentInvocation,
            _execution: CodingPlanExecution,
        ) -> None:
            console.log(
                "[bold green]Coding agent[/] finished in "
                f"{invocation.duration_seconds:.1f}s "
                f"(attempt {attempt}/{total})",
            )

        def _on_attempt_retry(attempt: int, _total: int, exc: Exception) -> None:
            if isinstance(exc, _NoRepoChangeError):
                console.log(
                    "[yellow]Coding agent[/] produced no repository changes; retrying…",
                )
                log.warning("Coding attempt {} produced no repository changes", attempt)
                return
            log.warning("Coding attempt {} failed: {}", attempt, exc)

        execution, invocation, attempts = run_structured_agent_task(
            backend=self.backend,
            task=task,
            working_dir=worktree,
            max_attempts=self.max_attempts,
            coerce_result=lambda inv: self._coerce_execution_from_invocation(
                request=request,
                invocation=inv,
            ),
            retryable_exceptions=(CodingError, ValidationError, json.JSONDecodeError),
            error_cls=CodingError,
            error_message=(
                "Coding agent could not produce a valid report after "
                f"{self.max_attempts} attempt(s)."
            ),
            debug_hook=_debug_hook,
            on_attempt_start=_on_attempt_start,
            on_attempt_success=_on_attempt_success,
            on_attempt_retry=_on_attempt_retry,
            post_check=_post_check,
        )

        return CodingAgentResponse(
            execution=execution,
            raw_output=invocation.stdout,
            prompt=prompt,
            command=invocation.command,
            stderr=invocation.stderr,
            attempts=attempts,
            duration_seconds=invocation.duration_seconds,
        )

    # Internal helpers --------------------------------------------------

    def _render_prompt(
        self,
        request: CodingAgentRequest,
        *,
        worktree: Path,
    ) -> str:
        plan = request.plan
        steps_block = "\n\n".join(
            self._format_plan_step(idx + 1, step) for idx, step in enumerate(plan.steps)
        )
        is_freeform_plan = bool(plan.steps) and all(
            step.step_id.startswith("freeform-") for step in plan.steps
        )
        if is_freeform_plan and plan.fallback_plan:
            plan_block = self._truncate(plan.fallback_plan.strip(), limit=2000)
        else:
            plan_block = steps_block
        plan_block = plan_block.strip() or plan.summary.strip() or "N/A"

        prompt = f"""
You are the coding agent inside Loreley's evolution worker.
Apply the plan to the repository at {worktree}, starting from base commit {request.base_commit}.

Goal:
{request.goal.strip()}

Plan:
{plan_block}

When done:
- Apply the required changes.
- Provide a short free-form summary of what you changed.
"""
        return textwrap.dedent(prompt).strip()

    def _format_plan_step(self, ordinal: int, step: PlanStep) -> str:
        actions = self._format_bullets(step.actions, indent="  ")
        files = self._format_bullets(step.files, indent="  ")
        return (
            f"Step {ordinal} ({step.step_id}) — {step.title}\n"
            f"Intent: {step.intent}\n"
            f"Actions:\n{actions}\n"
            f"Files:\n{files}"
        )

    def _format_bullets(
        self,
        values: Sequence[str] | Sequence[Any],
        *,
        indent: str = "",
    ) -> str:
        items = [
            f"{indent}- {self._truncate(str(value))}"
            for value in values
            if str(value).strip()
        ]
        if not items:
            return f"{indent}- None"
        return "\n".join(items)

    def _coerce_execution_from_invocation(
        self,
        *,
        request: CodingAgentRequest,
        invocation: AgentInvocation,
    ) -> CodingPlanExecution:
        """Turn backend output into a CodingPlanExecution, honouring the validation mode."""

        def parse(stdout: str) -> CodingPlanExecution:
            output_model = self._parse_output(stdout)
            return self._to_domain(output_model)

        return coerce_structured_output(
            validation_mode=self.validation_mode,
            stdout=invocation.stdout,
            parse=parse,
            build_from_freeform=lambda stdout: self._build_execution_from_freeform_output(
                request=request,
                raw_output=stdout,
            ),
            on_parse_error=lambda exc: self._log_invalid_output(invocation, exc),
            parse_exceptions=(ValidationError, json.JSONDecodeError),
        )

    def _parse_output(self, payload: str) -> _CodingOutputModel:
        cleaned = sanitize_json_payload(payload)
        return _CodingOutputModel.model_validate_json(cleaned)

    def _log_invalid_output(
        self,
        invocation: AgentInvocation,
        exc: Exception,
    ) -> None:
        stdout_preview = self._truncate(invocation.stdout, limit=2000) or "<empty>"
        stderr_preview = self._truncate(invocation.stderr, limit=1000) or "<empty>"
        log.warning(
            "Invalid coding agent output: {} | stdout preview: {} | stderr preview: {}",
            exc,
            stdout_preview,
            stderr_preview,
        )

    def _to_domain(self, output: _CodingOutputModel) -> CodingPlanExecution:
        step_results = tuple(
            CodingStepReport(
                step_id=step.step_id,
                status=step.status,
                summary=step.summary,
                files=tuple(step.files),
                commands=tuple(step.commands),
            )
            for step in output.step_results
        )
        return CodingPlanExecution(
            implementation_summary=output.implementation_summary,
            commit_message=output.commit_message,
            step_results=step_results,
            tests_executed=tuple(output.tests_executed),
            tests_recommended=tuple(output.tests_recommended),
            follow_up_items=tuple(output.follow_up_items),
            notes=tuple(output.notes),
        )

    def _build_execution_from_freeform_output(
        self,
        *,
        request: CodingAgentRequest,
        raw_output: str,
    ) -> CodingPlanExecution:
        """Build a best-effort CodingPlanExecution from free-form agent output."""
        raw_text = (raw_output or "").strip()

        summary_line = ""
        for line in raw_text.splitlines():
            cleaned = line.strip()
            if cleaned:
                summary_line = cleaned
                break
        if not summary_line:
            summary_line = f"Free-form coding agent output for goal: {self._truncate(request.goal, limit=200)}"

        implementation_summary = self._truncate(summary_line, limit=800)

        bullet_re = re.compile(r"^\s*(?:[-*]|•)\s+(?P<item>.+\S)\s*$")
        numbered_re = re.compile(r"^\s*(?P<num>\d{1,2})[.)]\s*(?P<rest>.*)$")
        step_re = re.compile(
            r"^\s*Step\s+(?P<num>\d{1,2})\s*(?::|[.)-])\s*(?P<rest>.*)$",
            re.I,
        )

        def extract_files(text: str) -> tuple[str, ...]:
            hits: list[str] = []
            for match in re.finditer(r"`([^`]+)`", text):
                token = match.group(1).strip()
                if not token or len(token) > 256:
                    continue
                if "://" in token:
                    continue
                if "/" in token or any(
                    token.endswith(ext)
                    for ext in (
                        ".py",
                        ".md",
                        ".toml",
                        ".yaml",
                        ".yml",
                        ".json",
                        ".txt",
                        ".ini",
                        ".cfg",
                    )
                ):
                    hits.append(token)
            seen: set[str] = set()
            unique: list[str] = []
            for item in hits:
                if item in seen:
                    continue
                seen.add(item)
                unique.append(item)
            return tuple(unique[:20])

        def parse_numbered_steps(text: str) -> list[tuple[str, list[str]]]:
            steps: list[tuple[str, list[str]]] = []
            current_title = ""
            current_lines: list[str] = []
            started = False
            for line in text.splitlines():
                match = numbered_re.match(line) or step_re.match(line)
                if match:
                    if started:
                        steps.append((current_title.strip(), current_lines))
                    started = True
                    current_title = (match.group("rest") or "").strip()
                    current_lines = []
                    continue
                if started:
                    current_lines.append(line.rstrip())
            if started:
                steps.append((current_title.strip(), current_lines))
            return steps

        def summarise_block(title: str, body_lines: list[str]) -> str:
            bullets = []
            for line in body_lines:
                match = bullet_re.match(line)
                if match:
                    bullets.append(match.group("item").strip())
            if bullets:
                return " / ".join(bullets[:6])
            parts = [title.strip()] if title.strip() else []
            parts.extend(line.strip() for line in body_lines if line.strip())
            return " ".join(parts).strip()

        step_reports: list[CodingStepReport] = []
        parsed = parse_numbered_steps(raw_text) if raw_text else []
        for idx, (title, body_lines) in enumerate(parsed[:12]):
            block_summary = summarise_block(title, body_lines) or title.strip() or summary_line
            block_text = "\n".join([title.strip()] + [line.rstrip() for line in body_lines]).strip()
            step_reports.append(
                CodingStepReport(
                    step_id=f"freeform-{idx + 1}",
                    status=StepExecutionStatus.PARTIAL,
                    summary=self._truncate(block_summary, limit=800),
                    files=extract_files(block_text),
                    commands=tuple(),
                )
            )

        if not step_reports:
            step_reports = [
                CodingStepReport(
                    step_id="freeform-1",
                    status=StepExecutionStatus.PARTIAL,
                    summary=self._truncate(raw_text or summary_line, limit=800),
                    files=extract_files(raw_text),
                    commands=tuple(),
                )
            ]

        notes = (
            f"Coding used free-form synthesis (validation_mode={self.validation_mode!r}). "
            "The full backend output is preserved in raw_output.",
        )

        return CodingPlanExecution(
            implementation_summary=implementation_summary,
            commit_message=None,
            step_results=tuple(step_reports),
            tests_executed=tuple(),
            tests_recommended=tuple(),
            follow_up_items=tuple(),
            notes=notes,
        )

    def _snapshot_worktree_state(self, worktree: Path) -> tuple[str, ...]:
        """Return a stable snapshot of the worktree status for change detection."""
        try:
            repo = Repo(worktree)
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:  # pragma: no cover - defensive
            raise CodingError(f"Invalid git worktree for coding agent: {worktree}") from exc

        try:
            status_output = repo.git.status("--porcelain", "--untracked-files=all")
        except Exception as exc:  # pragma: no cover - defensive
            raise CodingError("Failed to inspect worktree status during coding run.") from exc

        lines = [line.strip() for line in status_output.splitlines() if line.strip()]
        return tuple(sorted(lines))

    def _dump_debug_artifact(
        self,
        *,
        request: CodingAgentRequest,
        worktree: Path,
        invocation: AgentInvocation | None,
        prompt: str,
        attempt: int,
        execution: CodingPlanExecution | None,
        error: Exception | None,
    ) -> None:
        """Persist coding agent prompt and backend interaction for debugging."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            commit_prefix = (request.base_commit or "unknown")[:12]
            filename = f"coding-{commit_prefix}-attempt{attempt}-{timestamp}.json"
            payload: dict[str, Any] = {
                "timestamp": timestamp,
                "status": "error" if error else "ok",
                "error": repr(error) if error else None,
                "attempt": attempt,
                "schema_mode": self.schema_mode,
                "validation_mode": self.validation_mode,
                "working_dir": str(worktree),
                "goal": request.goal,
                "base_commit": request.base_commit,
                "constraints": list(request.constraints),
                "acceptance_criteria": list(request.acceptance_criteria),
                "backend_command": list(invocation.command) if invocation else None,
                "backend_duration_seconds": (
                    invocation.duration_seconds if invocation else None
                ),
                "backend_stdout": invocation.stdout if invocation else None,
                "backend_stderr": invocation.stderr if invocation else None,
                "prompt": prompt,
                "execution": {
                    "implementation_summary": execution.implementation_summary,
                    "commit_message": execution.commit_message,
                    "step_results": [
                        {
                            "step_id": step.step_id,
                            "status": step.status.value,
                            "summary": step.summary,
                            "files": list(step.files),
                            "commands": list(step.commands),
                        }
                        for step in execution.step_results
                    ],
                    "tests_executed": list(execution.tests_executed),
                    "tests_recommended": list(execution.tests_recommended),
                    "follow_up_items": list(execution.follow_up_items),
                    "notes": list(execution.notes),
                }
                if execution
                else None,
            }
            path = self._debug_dir / filename
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - best-effort logging
            log.debug("Failed to write coding debug artifact: {}", exc)


