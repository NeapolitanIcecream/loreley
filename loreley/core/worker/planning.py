from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

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
from loreley.core.worker.output_sanitizer import sanitize_json_payload

console = Console()
log = logger.bind(module="worker.planning")

__all__ = [
    "CommitMetric",
    "CommitPlanningContext",
    "PlanningAgent",
    "PlanningAgentRequest",
    "PlanningAgentResponse",
    "PlanningError",
    "PlanningPlan",
    "PlanStep",
]


class PlanningError(RuntimeError):
    """Raised when the planning agent cannot produce a plan."""


@dataclass(slots=True)
class CommitMetric:
    """Lightweight representation of an evaluation metric."""

    name: str
    value: float
    unit: str | None = None
    higher_is_better: bool | None = None
    summary: str | None = None


@dataclass(slots=True)
class CommitPlanningContext:
    """Context shared with the planning agent for a single commit."""

    commit_hash: str
    subject: str
    change_summary: str
    trajectory: Sequence[str] = field(default_factory=tuple)
    trajectory_meta: dict[str, Any] | None = None
    key_files: Sequence[str] = field(default_factory=tuple)
    highlights: Sequence[str] = field(default_factory=tuple)
    evaluation_summary: str | None = None
    metrics: Sequence[CommitMetric] = field(default_factory=tuple)
    map_elites_cell_index: int | None = None
    map_elites_objective: float | None = None
    map_elites_measures: Sequence[float] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.subject = " ".join((self.subject or "").split()).strip() or f"Commit {self.commit_hash}"
        self.change_summary = (self.change_summary or "").strip() or "N/A"
        self.trajectory = tuple(self.trajectory or ())
        self.key_files = tuple(self.key_files or ())
        self.highlights = tuple(self.highlights or ())
        self.metrics = tuple(self.metrics or ())
        self.map_elites_measures = tuple(self.map_elites_measures or ())


@dataclass(slots=True)
class PlanningAgentRequest:
    """Input payload for the planning agent."""

    base: CommitPlanningContext
    inspirations: Sequence[CommitPlanningContext]
    goal: str
    constraints: Sequence[str] = field(default_factory=tuple)
    acceptance_criteria: Sequence[str] = field(default_factory=tuple)
    iteration_hint: str | None = None
    cold_start: bool = False

    def __post_init__(self) -> None:
        self.inspirations = tuple(self.inspirations or ())
        self.constraints = tuple(self.constraints or ())
        self.acceptance_criteria = tuple(self.acceptance_criteria or ())


@dataclass(slots=True)
class PlanStep:
    """Single actionable step returned by the planning agent."""

    step_id: str
    title: str
    intent: str
    actions: tuple[str, ...]
    files: tuple[str, ...]
    dependencies: tuple[str, ...]
    validation: tuple[str, ...]
    risks: tuple[str, ...]
    references: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "intent": self.intent,
            "actions": list(self.actions),
            "files": list(self.files),
            "dependencies": list(self.dependencies),
            "validation": list(self.validation),
            "risks": list(self.risks),
            "references": list(self.references),
        }


@dataclass(slots=True)
class PlanningPlan:
    """Structured planning output ready for the coding agent."""

    summary: str
    rationale: str
    focus_metrics: tuple[str, ...]
    guardrails: tuple[str, ...]
    risks: tuple[str, ...]
    validation: tuple[str, ...]
    steps: tuple[PlanStep, ...]
    handoff_notes: tuple[str, ...]
    fallback_plan: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "rationale": self.rationale,
            "focus_metrics": list(self.focus_metrics),
            "guardrails": list(self.guardrails),
            "risks": list(self.risks),
            "validation": list(self.validation),
            "steps": [step.as_dict() for step in self.steps],
            "handoff_notes": list(self.handoff_notes),
            "fallback_plan": self.fallback_plan,
        }


@dataclass(slots=True)
class PlanningAgentResponse:
    """Envelope containing planning results and metadata."""

    plan: PlanningPlan
    raw_output: str
    prompt: str
    command: tuple[str, ...]
    stderr: str
    attempts: int
    duration_seconds: float


class _PlanStepModel(BaseModel):
    """pydantic schema for validating planning backend output."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    step_id: str = Field(..., alias="id")
    title: str
    intent: str
    actions: list[str]
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    validation: list[str]
    risks: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class _PlanModel(BaseModel):
    """Top-level plan schema."""

    model_config = ConfigDict(frozen=True)

    plan_summary: str = Field(min_length=1, max_length=512)
    rationale: str = Field(min_length=1, max_length=2048)
    focus_metrics: list[str]
    guardrails: list[str]
    risks: list[str]
    validation: list[str]
    steps: list[_PlanStepModel]
    handoff_notes: list[str] = Field(default_factory=list)
    fallback_plan: str | None = None


PLANNING_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan_summary": {"type": "string", "minLength": 1, "maxLength": 512},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 2048},
        "focus_metrics": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
        },
        "guardrails": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 20,
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 20,
        },
        "validation": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 20,
        },
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1, "maxLength": 64},
                    "title": {"type": "string", "minLength": 1, "maxLength": 120},
                    "intent": {"type": "string", "minLength": 1, "maxLength": 400},
                    "actions": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 200},
                        "minItems": 1,
                        "maxItems": 12,
                    },
                    "files": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 30},
                    "dependencies": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 12},
                    "validation": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 200},
                        "minItems": 1,
                        "maxItems": 12,
                    },
                    "risks": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 12},
                    "references": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 12},
                },
                "required": ["id", "title", "intent", "actions", "validation"],
                "additionalProperties": False,
            },
        },
        "handoff_notes": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 20},
        "fallback_plan": {"type": ["string", "null"], "maxLength": 1200},
    },
    "required": [
        "plan_summary",
        "rationale",
        "focus_metrics",
        "guardrails",
        "risks",
        "validation",
        "steps",
    ],
    "additionalProperties": False,
}


class PlanningAgent:
    """Bridge between Loreley's worker and the configured planning backend."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: AgentBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_attempts = max(1, self.settings.worker_planning_max_attempts)
        self.validation_mode: str = self.settings.worker_planning_validation_mode
        self.schema_mode: SchemaMode = resolve_schema_mode(
            configured_mode=self.settings.worker_planning_codex_schema_mode,
            api_spec=self.settings.openai_api_spec,
        )
        self._truncate_limit = 2000
        self._max_highlights = 8
        self._max_metrics = 10
        self._debug_dir = self._resolve_debug_dir()
        if backend is not None:
            self.backend: AgentBackend = backend
        elif self.settings.worker_planning_backend:
            self.backend = load_agent_backend(
                self.settings.worker_planning_backend,
                label="planning backend",
            )
        else:
            self.backend = CodexCliBackend(
                bin=self.settings.worker_planning_codex_bin,
                profile=self.settings.worker_planning_codex_profile,
                timeout_seconds=self.settings.worker_planning_timeout_seconds,
                extra_env=dict(self.settings.worker_planning_extra_env or {}),
                schema_override=self.settings.worker_planning_schema_path,
                error_cls=PlanningError,
                full_auto=False,
            )

    def plan(
        self,
        request: PlanningAgentRequest,
        *,
        working_dir: Path,
    ) -> PlanningAgentResponse:
        """Generate a structured plan using the configured backend."""
        worktree = Path(working_dir).expanduser().resolve()
        prompt = self._render_prompt(request)

        if self.validation_mode in ("strict", "lenient"):
            task = StructuredAgentTask(
                name="planning",
                prompt=prompt,
                schema=PLANNING_OUTPUT_SCHEMA,
                schema_mode=self.schema_mode,
            )
        else:
            # In 'none' validation mode we do not enforce a JSON schema at the backend level.
            task = StructuredAgentTask(
                name="planning",
                prompt=prompt,
                schema=None,
                schema_mode="none",
            )

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            invocation: AgentInvocation | None = None
            try:
                console.log(
                    "[cyan]Planning agent[/] requesting plan "
                    f"(attempt {attempt}/{self.max_attempts})",
                )
                invocation = self.backend.run(
                    task,
                    working_dir=worktree,
                )
                plan = self._coerce_plan_from_invocation(
                    request=request,
                    invocation=invocation,
                )
                self._dump_debug_artifact(
                    request=request,
                    worktree=worktree,
                    invocation=invocation,
                    prompt=prompt,
                    attempt=attempt,
                    plan=plan,
                    error=None,
                )
                console.log(
                    "[bold green]Planning agent[/] generated plan "
                    f"in {invocation.duration_seconds:.1f}s "
                    f"(attempt {attempt}/{self.max_attempts})",
                )
                return PlanningAgentResponse(
                    plan=plan,
                    raw_output=invocation.stdout,
                    prompt=prompt,
                    command=invocation.command,
                    stderr=invocation.stderr,
                    attempts=attempt,
                    duration_seconds=invocation.duration_seconds,
                )
            except (PlanningError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                self._dump_debug_artifact(
                    request=request,
                    worktree=worktree,
                    invocation=None,
                    prompt=prompt,
                    attempt=attempt,
                    plan=None,
                    error=exc,
                )
                log.warning(
                    "Planning attempt {} failed: {}",
                    attempt,
                    exc,
                )
        raise PlanningError(
            "Planning agent could not produce a valid plan after "
            f"{self.max_attempts} attempt(s).",
        ) from last_error

    def _render_prompt(self, request: PlanningAgentRequest) -> str:
        """Compose the narrative prompt for the planning backend."""
        base_block = self._format_commit_block("Base commit", request.base)
        insp_blocks = "\n\n".join(
            self._format_commit_block(f"Inspiration #{idx + 1}", ctx)
            for idx, ctx in enumerate(request.inspirations)
        )
        constraints = "\n".join(f"- {item}" for item in request.constraints) or "None"
        acceptance = (
            "\n".join(f"- {item}" for item in request.acceptance_criteria) or "None"
        )
        iteration_hint = request.iteration_hint or "None provided"

        schema_contract_block = ""
        json_requirement_line = ""
        if self.validation_mode == "strict":
            if self.schema_mode in ("prompt", "none"):
                schema_json = json.dumps(
                    PLANNING_OUTPUT_SCHEMA,
                    ensure_ascii=True,
                    indent=2,
                )
                schema_contract_block = (
                    "\n\nOutput JSON schema contract:\n"
                    f"{schema_json}\n"
                )
            json_requirement_line = (
                "- Respond ONLY with a single JSON object that matches the expected schema. "
                "Do NOT wrap the JSON in markdown code fences (no ``` or ```json) and do NOT "
                "include any other text before/after the JSON."
            )
        elif self.validation_mode == "lenient":
            # In lenient mode, JSON is a best-effort contract rather than a hard requirement.
            if self.schema_mode in ("prompt", "none"):
                schema_json = json.dumps(
                    PLANNING_OUTPUT_SCHEMA,
                    ensure_ascii=True,
                    indent=2,
                )
                schema_contract_block = (
                    "\n\nSuggested JSON schema (optional, best-effort):\n"
                    f"{schema_json}\n"
                )
            json_requirement_line = (
                "- Prefer returning a single JSON object that roughly follows the schema "
                "(without markdown code fences such as ```json). Free-form text is also acceptable."
            )
        else:
            # In 'none' validation mode there is no expectation about JSON structure.
            json_requirement_line = (
                "- Provide your answer in clear free-form text; JSON formatting is optional."
            )

        cold_start_block = ""
        if request.cold_start:
            cold_start_block = (
                "This is a cold-start seed population design run. The MAP-Elites archive\n"
                "is currently empty. Propose diverse, high-variance initial directions\n"
                "that all respect the global objective and constraints. Favour\n"
                "exploration and higher-temperature behaviour.\n\n"
            )

        prompt = f"""
You are the planning agent inside Loreley's autonomous evolution worker.
Your job is to convert the available commit knowledge into a concrete, multi-step
implementation plan that a coding agent can execute without further clarification.

{cold_start_block}\

Global objective:
{request.goal.strip()}

Constraints that must be respected:
{constraints}

Acceptance criteria / definition of done:
{acceptance}

Iteration / island hint:
{iteration_hint}

Base commit context:
{base_block}

Inspiration commits:
{insp_blocks or "None"}

Deliverable requirements:
- Produce 3-6 coherent steps with explicit actions and files to touch.
- Reference evaluation metrics to justify why the plan should work.
- Call out any risks, guardrails, and validation activities per step.
{json_requirement_line}
{schema_contract_block}
"""
        return textwrap.dedent(prompt).strip()

    def _format_commit_block(
        self,
        title: str,
        context: CommitPlanningContext,
    ) -> str:
        metrics_block = self._format_metrics(context.metrics)
        highlights = tuple(context.highlights)[: self._max_highlights]
        highlight_block = (
            "\n".join(f"  - {self._truncate(snippet)}" for snippet in highlights)
            if highlights
            else "  - None"
        )
        key_files = tuple(context.key_files or ())[:20]
        key_files_block = (
            "\n".join(f"  - {self._truncate(path)}" for path in key_files) if key_files else "  - None"
        )
        trajectory_block = "\n".join(context.trajectory) if context.trajectory else "  - None"
        evaluation_summary = self._truncate(context.evaluation_summary or "N/A")
        map_elites_block = ""
        if context.map_elites_cell_index is not None:
            measures = (
                ", ".join(self._truncate(str(v), limit=48) for v in context.map_elites_measures[:4])
                if context.map_elites_measures
                else "N/A"
            )
            obj = (
                f"{float(context.map_elites_objective):.4f}"
                if context.map_elites_objective is not None
                else "N/A"
            )
            map_elites_block = (
                "\n- MAP-Elites:\n"
                f"  - cell_index: {int(context.map_elites_cell_index)}\n"
                f"  - objective: {obj}\n"
                f"  - measures_head: {measures}"
            )
        return (
            f"{title}\n"
            f"- Hash: {context.commit_hash}\n"
            f"- Subject: {self._truncate(context.subject)}\n"
            f"- Change summary: {self._truncate(context.change_summary, limit=512)}\n"
            f"- Trajectory (unique vs base):\n{trajectory_block}\n"
            f"- Key files:\n{key_files_block}\n"
            f"- Evaluation summary: {evaluation_summary}\n"
            f"- Highlights:\n{highlight_block}\n"
            f"- Metrics:\n{metrics_block}"
            f"{map_elites_block}"
        )

    def _format_metrics(self, metrics: Sequence[CommitMetric]) -> str:
        sliced = tuple(metrics)[: self._max_metrics]
        if not sliced:
            return "  - None"

        lines: list[str] = []
        for metric in sliced:
            detail = f"{metric.value}"
            if metric.unit:
                detail = f"{detail}{metric.unit}"
            hb = ""
            if metric.higher_is_better is not None:
                hb = " (higher is better)" if metric.higher_is_better else " (lower is better)"
            summary = f" — {self._truncate(metric.summary)}" if metric.summary else ""
            lines.append(f"  - {metric.name}: {detail}{hb}{summary}")
        if len(metrics) > self._max_metrics:
            lines.append("  - ... (truncated)")
        return "\n".join(lines)

    def _coerce_plan_from_invocation(
        self,
        *,
        request: PlanningAgentRequest,
        invocation: AgentInvocation,
    ) -> PlanningPlan:
        """Turn backend output into a PlanningPlan, honouring the validation mode."""
        if self.validation_mode == "strict":
            plan_model = self._parse_plan(invocation.stdout)
            return self._to_domain(plan_model)

        if self.validation_mode == "lenient":
            try:
                plan_model = self._parse_plan(invocation.stdout)
                return self._to_domain(plan_model)
            except (ValidationError, json.JSONDecodeError) as exc:
                self._log_invalid_output(invocation, exc)
                return self._build_plan_from_freeform_output(
                    request=request,
                    raw_output=invocation.stdout,
                )

        # validation_mode == "none": skip JSON parsing entirely and treat output as free-form.
        return self._build_plan_from_freeform_output(
            request=request,
            raw_output=invocation.stdout,
        )

    def _parse_plan(self, payload: str) -> _PlanModel:
        """Validate JSON output from the planning backend against the schema."""
        cleaned = sanitize_json_payload(payload)
        return _PlanModel.model_validate_json(cleaned)

    def _to_domain(self, plan_model: _PlanModel) -> PlanningPlan:
        steps = tuple(
            PlanStep(
                step_id=step.step_id,
                title=step.title,
                intent=step.intent,
                actions=tuple(step.actions),
                files=tuple(step.files),
                dependencies=tuple(step.dependencies),
                validation=tuple(step.validation),
                risks=tuple(step.risks),
                references=tuple(step.references),
            )
            for step in plan_model.steps
        )

        return PlanningPlan(
            summary=plan_model.plan_summary,
            rationale=plan_model.rationale,
            focus_metrics=tuple(plan_model.focus_metrics),
            guardrails=tuple(plan_model.guardrails),
            risks=tuple(plan_model.risks),
            validation=tuple(plan_model.validation),
            steps=steps,
            handoff_notes=tuple(plan_model.handoff_notes),
            fallback_plan=plan_model.fallback_plan,
        )

    def _build_plan_from_freeform_output(
        self,
        *,
        request: PlanningAgentRequest,
        raw_output: str,
    ) -> PlanningPlan:
        """Build a minimal PlanningPlan from free-form agent output under non-strict validation."""
        summary_source = (raw_output or "").strip() or request.goal
        summary = self._truncate(summary_source)
        rationale = (
            "Planning output could not be parsed as structured JSON or validation was disabled; "
            "this plan was synthesised from free-form text in a non-strict validation mode."
        )

        focus_metrics = tuple(metric.name for metric in request.base.metrics)[:3]
        guardrails = tuple(request.constraints)
        risks: tuple[str, ...] = ()
        validation = tuple(request.acceptance_criteria) or (
            "Run the project's tests and ensure there are no regressions.",
        )

        synthetic_step = PlanStep(
            step_id="lenient-1",
            title="Apply the free-form planning output",
            intent="Follow the planning agent's free-form suggestions from the raw output.",
            actions=(self._truncate(summary_source),),
            files=tuple(),
            dependencies=tuple(),
            validation=validation,
            risks=risks,
            references=tuple(),
        )

        handoff_notes = (
            "Planning ran in lenient validation mode. The full free-form output is "
            "available to downstream consumers via raw_output; this structured plan is "
            "a best-effort synthesis.",
        )

        fallback_plan = (raw_output or "").strip() or None

        return PlanningPlan(
            summary=summary,
            rationale=rationale,
            focus_metrics=focus_metrics,
            guardrails=guardrails,
            risks=risks,
            validation=validation,
            steps=(synthetic_step,),
            handoff_notes=handoff_notes,
            fallback_plan=fallback_plan,
        )

    def _log_invalid_output(
        self,
        invocation: AgentInvocation,
        exc: Exception,
    ) -> None:
        stdout_preview = self._truncate(invocation.stdout, limit=2000) or "<empty>"
        stderr_preview = self._truncate(invocation.stderr, limit=1000) or "<empty>"
        log.warning(
            "Invalid planning agent output: {} | stdout preview: {} | stderr preview: {}",
            exc,
            stdout_preview,
            stderr_preview,
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
        """Resolve directory for planning debug artifacts."""
        if self.settings.logs_base_dir:
            base_dir = Path(self.settings.logs_base_dir).expanduser()
        else:
            base_dir = Path.cwd()
        logs_root = base_dir / "logs" / "worker" / "planning"
        logs_root.mkdir(parents=True, exist_ok=True)
        return logs_root

    def _dump_debug_artifact(
        self,
        *,
        request: PlanningAgentRequest,
        worktree: Path,
        invocation: AgentInvocation | None,
        prompt: str,
        attempt: int,
        plan: PlanningPlan | None,
        error: Exception | None,
    ) -> None:
        """Persist planning agent prompt and backend interaction for debugging."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            commit_prefix = (request.base.commit_hash or "unknown")[:12]
            filename = f"planning-{commit_prefix}-attempt{attempt}-{timestamp}.json"
            payload: dict[str, Any] = {
                "timestamp": timestamp,
                "status": "error" if error else "ok",
                "error": repr(error) if error else None,
                "attempt": attempt,
                "schema_mode": self.schema_mode,
                "validation_mode": self.validation_mode,
                "working_dir": str(worktree),
                "goal": request.goal,
                "base_commit": request.base.commit_hash,
                "constraints": list(request.constraints),
                "acceptance_criteria": list(request.acceptance_criteria),
                "backend_command": list(invocation.command) if invocation else None,
                "backend_duration_seconds": (
                    invocation.duration_seconds if invocation else None
                ),
                "backend_stdout": invocation.stdout if invocation else None,
                "backend_stderr": invocation.stderr if invocation else None,
                "prompt": prompt,
                "parsed_plan": plan.as_dict() if plan else None,
            }
            path = self._debug_dir / filename
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - best-effort logging
            log.debug("Failed to write planning debug artifact: {}", exc)



