from __future__ import annotations

import json
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
from loreley.core.worker.agent_backend import (
    AgentBackend,
    AgentInvocation,
    CodexCliBackend,
    SchemaMode,
    StructuredAgentTask,
    load_agent_backend,
    resolve_schema_mode,
)
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


class CodingAgent:
    """Drive the configured coding backend to implement a plan on the repository."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: AgentBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_attempts = max(1, self.settings.worker_coding_max_attempts)
        self.validation_mode: str = self.settings.worker_coding_validation_mode
        self.schema_mode: SchemaMode = resolve_schema_mode(
            configured_mode=self.settings.worker_coding_codex_schema_mode,
            api_spec=self.settings.openai_api_spec,
        )
        self._truncate_limit = 2000
        self._debug_dir = self._resolve_debug_dir()
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

        if self.validation_mode in ("strict", "lenient"):
            task = StructuredAgentTask(
                name="coding",
                prompt=prompt,
                schema=CODING_OUTPUT_SCHEMA,
                schema_mode=self.schema_mode,
            )
        else:
            # In 'none' validation mode we do not enforce a JSON schema at the backend level.
            task = StructuredAgentTask(
                name="coding",
                prompt=prompt,
                schema=None,
                schema_mode="none",
            )

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            invocation: AgentInvocation | None = None
            try:
                console.log(
                    "[cyan]Coding agent[/] requesting execution "
                    f"(attempt {attempt}/{self.max_attempts})",
                )
                invocation = self.backend.run(
                    task,
                    working_dir=worktree,
                )
                execution = self._coerce_execution_from_invocation(
                    request=request,
                    invocation=invocation,
                )
                current_status = self._snapshot_worktree_state(worktree)
                if current_status == baseline_status:
                    no_change_error = CodingError(
                        "Coding agent finished without producing repository changes.",
                    )
                    self._dump_debug_artifact(
                        request=request,
                        worktree=worktree,
                        invocation=invocation,
                        prompt=prompt,
                        attempt=attempt,
                        execution=execution,
                        error=no_change_error,
                    )
                    console.log(
                        "[yellow]Coding agent[/] produced no repository changes; retrying…",
                    )
                    log.warning("Coding attempt {} produced no repository changes", attempt)
                    last_error = no_change_error
                    continue
                self._dump_debug_artifact(
                    request=request,
                    worktree=worktree,
                    invocation=invocation,
                    prompt=prompt,
                    attempt=attempt,
                    execution=execution,
                    error=None,
                )
                console.log(
                    "[bold green]Coding agent[/] finished in "
                    f"{invocation.duration_seconds:.1f}s "
                    f"(attempt {attempt}/{self.max_attempts})",
                )
                return CodingAgentResponse(
                    execution=execution,
                    raw_output=invocation.stdout,
                    prompt=prompt,
                    command=invocation.command,
                    stderr=invocation.stderr,
                    attempts=attempt,
                    duration_seconds=invocation.duration_seconds,
                )
            except (CodingError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                self._dump_debug_artifact(
                    request=request,
                    worktree=worktree,
                    invocation=None,
                    prompt=prompt,
                    attempt=attempt,
                    execution=None,
                    error=exc,
                )
                log.warning("Coding attempt {} failed: {}", attempt, exc)
        raise CodingError(
            "Coding agent could not produce a valid report after "
            f"{self.max_attempts} attempt(s).",
        ) from last_error

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
        constraints = self._format_bullets(request.constraints)
        acceptance = self._format_bullets(request.acceptance_criteria)
        focus_metrics = self._format_bullets(plan.focus_metrics)
        guardrails = self._format_bullets(plan.guardrails)
        validation = self._format_bullets(plan.validation)
        risks = self._format_bullets(plan.risks)
        notes = self._format_bullets(request.additional_notes)
        handoff_notes = self._format_bullets(plan.handoff_notes)
        fallback_plan_text = self._truncate(plan.fallback_plan or "None provided")
        iteration_hint = request.iteration_hint or "None provided"

        schema_contract_block = ""
        json_requirements_block = ""
        if self.validation_mode == "strict":
            if self.schema_mode in ("prompt", "none"):
                schema_json = json.dumps(
                    CODING_OUTPUT_SCHEMA,
                    ensure_ascii=True,
                    indent=2,
                )
                schema_contract_block = (
                    "\n\nOutput JSON schema contract:\n"
                    f"{schema_json}\n"
                )
            json_requirements_block = (
                "- summarise your work using the provided JSON schema\n"
                "- respond ONLY with a single JSON object following that schema; "
                "no prose outside JSON\n"
                "- do NOT wrap the JSON in markdown code fences (no ``` or ```json)"
            )
        elif self.validation_mode == "lenient":
            # In lenient mode, JSON is a best-effort contract rather than a hard requirement.
            if self.schema_mode in ("prompt", "none"):
                schema_json = json.dumps(
                    CODING_OUTPUT_SCHEMA,
                    ensure_ascii=True,
                    indent=2,
                )
                schema_contract_block = (
                    "\n\nSuggested JSON schema for structuring your summary (optional):\n"
                    f"{schema_json}\n"
                )
            json_requirements_block = (
                "- prefer summarising your work using the JSON schema below when convenient "
                "(without markdown code fences such as ```json)\n"
                "- it is acceptable to return free-form text if a structured JSON "
                "object is difficult to produce"
            )
        else:
            # In 'none' validation mode there is no expectation about JSON structure.
            json_requirements_block = (
                "- provide your summary in clear free-form text; JSON formatting is optional."
            )

        prompt = f"""
You are the coding agent running inside Loreley's autonomous worker.
Your mission is to modify the repository located at {worktree} so that it
implements the provided plan starting from base commit {request.base_commit}.
You may inspect files, run tests, and edit code directly.

Global objective:
{request.goal.strip()}

Plan summary:
{plan.summary}

Plan rationale:
{plan.rationale}

Focus metrics:
{focus_metrics}

Guardrails to respect:
{guardrails}

Validation expectations:
{validation}

Known risks:
{risks}

Additional constraints:
{constraints}

Acceptance criteria / definition of done:
{acceptance}

Iteration hint:
{iteration_hint}

Extra worker notes:
{notes}

Handoff notes from planning agent:
{handoff_notes}

Fallback plan if things go wrong:
{fallback_plan_text}

Detailed plan steps:
{steps_block}

When you finish applying the plan:
- ensure repository changes are ready for review (lint/tests as needed)
- keep `implementation_summary` concise (<= 2000 chars), plain text, and free of code fences/JSON dumps
{json_requirements_block}
{schema_contract_block}
"""
        return textwrap.dedent(prompt).strip()

    def _format_plan_step(self, ordinal: int, step: PlanStep) -> str:
        actions = self._format_bullets(step.actions, indent="  ")
        files = self._format_bullets(step.files, indent="  ")
        dependencies = self._format_bullets(step.dependencies, indent="  ")
        validation = self._format_bullets(step.validation, indent="  ")
        risks = self._format_bullets(step.risks, indent="  ")
        references = self._format_bullets(step.references, indent="  ")
        return (
            f"Step {ordinal} ({step.step_id}) — {step.title}\n"
            f"Intent: {step.intent}\n"
            f"Actions:\n{actions}\n"
            f"Files:\n{files}\n"
            f"Dependencies:\n{dependencies}\n"
            f"Validation:\n{validation}\n"
            f"Risks:\n{risks}\n"
            f"References:\n{references}"
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
        if self.validation_mode == "strict":
            output_model = self._parse_output(invocation.stdout)
            return self._to_domain(output_model)

        if self.validation_mode == "lenient":
            try:
                output_model = self._parse_output(invocation.stdout)
                return self._to_domain(output_model)
            except (ValidationError, json.JSONDecodeError) as exc:
                self._log_invalid_output(invocation, exc)
                return self._build_execution_from_freeform_output(
                    request=request,
                    raw_output=invocation.stdout,
                )

        # validation_mode == "none": skip JSON parsing entirely and treat output as free-form.
        return self._build_execution_from_freeform_output(
            request=request,
            raw_output=invocation.stdout,
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
        """Build a minimal CodingPlanExecution from free-form agent output under non-strict validation."""
        summary_source = (raw_output or "").strip()
        if not summary_source:
            summary_source = (
                f"Free-form coding agent output for goal: {self._truncate(request.goal)}"
            )

        implementation_summary = self._truncate(summary_source)

        step = CodingStepReport(
            step_id="lenient-1",
            status=StepExecutionStatus.PARTIAL,
            summary=self._truncate(summary_source),
            files=tuple(),
            commands=tuple(),
        )

        notes = (
            "Coding ran in a non-strict validation mode. The full free-form output is "
            "available via raw_output; this structured execution summary is a best-effort "
            "synthesis.",
        )

        return CodingPlanExecution(
            implementation_summary=implementation_summary,
            commit_message=None,
            step_results=(step,),
            tests_executed=tuple(),
            tests_recommended=tuple(),
            follow_up_items=tuple(),
            notes=notes,
        )

    def _truncate(self, text: str, limit: int | None = None) -> str:
        if not text:
            return ""
        active_limit = limit or self._truncate_limit
        stripped = text.strip()
        if len(stripped) <= active_limit:
            return stripped
        return f"{stripped[:active_limit]}…"

    def _resolve_debug_dir(self) -> Path:
        """Resolve directory for coding debug artifacts."""
        if self.settings.logs_base_dir:
            base_dir = Path(self.settings.logs_base_dir).expanduser()
        else:
            base_dir = Path.cwd()
        logs_root = base_dir / "logs" / "worker" / "coding"
        logs_root.mkdir(parents=True, exist_ok=True)
        return logs_root

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


