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
    summary: str
    highlights: Sequence[str] = field(default_factory=tuple)
    evaluation_summary: str | None = None
    metrics: Sequence[CommitMetric] = field(default_factory=tuple)
    extra_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.highlights = tuple(self.highlights or ())
        self.metrics = tuple(self.metrics or ())
        self.extra_context = dict(self.extra_context or {})


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
    """pydantic schema for validating Codex output."""

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

    plan_summary: str
    rationale: str
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
        "plan_summary": {"type": "string", "minLength": 1},
        "rationale": {"type": "string", "minLength": 1},
        "focus_metrics": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "guardrails": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "validation": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "steps": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "intent": {"type": "string"},
                    "actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "files": {"type": "array", "items": {"type": "string"}},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "validation": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "references": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "title", "intent", "actions", "validation"],
                "additionalProperties": False,
            },
        },
        "handoff_notes": {"type": "array", "items": {"type": "string"}},
        "fallback_plan": {"type": ["string", "null"]},
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
    """Bridge between Loreley's worker and the Codex CLI planning workflow."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: AgentBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_attempts = max(1, self.settings.worker_planning_max_attempts)
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

        task = StructuredAgentTask(
            name="planning",
            prompt=prompt,
            schema=PLANNING_OUTPUT_SCHEMA,
            schema_mode=self.schema_mode,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                console.log(
                    "[cyan]Planning agent[/] requesting plan "
                    f"(attempt {attempt}/{self.max_attempts})",
                )
                invocation: AgentInvocation = self.backend.run(
                    task,
                    working_dir=worktree,
                )
                plan_model = self._parse_plan(invocation.stdout)
                plan = self._to_domain(plan_model)
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
        """Compose the narrative prompt for Codex."""
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
        if self.schema_mode in ("prompt", "none"):
            schema_json = json.dumps(PLANNING_OUTPUT_SCHEMA, ensure_ascii=True, indent=2)
            schema_contract_block = (
                "\n\nOutput JSON schema contract:\n"
                f"{schema_json}\n"
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
 - Respond ONLY with a single JSON object that matches the expected schema.
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
        extra = ""
        if context.extra_context:
            serialized = self._truncate(json.dumps(context.extra_context, indent=2))
            extra = f"\nAdditional context:\n{serialized}"

        evaluation_summary = self._truncate(context.evaluation_summary or "N/A")
        return (
            f"{title}\n"
            f"- Hash: {context.commit_hash}\n"
            f"- Summary: {self._truncate(context.summary)}\n"
            f"- Evaluation summary: {evaluation_summary}\n"
            f"- Key snippets:\n{highlight_block}\n"
            f"- Metrics:\n{metrics_block}"
            f"{extra}"
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

    def _parse_plan(self, payload: str) -> _PlanModel:
        """Validate JSON output from Codex against the schema."""
        return _PlanModel.model_validate_json(payload)

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
        """Persist planning agent prompt and Codex output for debugging."""
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
                "working_dir": str(worktree),
                "goal": request.goal,
                "base_commit": request.base.commit_hash,
                "constraints": list(request.constraints),
                "acceptance_criteria": list(request.acceptance_criteria),
                "codex_command": list(invocation.command) if invocation else None,
                "codex_duration_seconds": (
                    invocation.duration_seconds if invocation else None
                ),
                "codex_stdout": invocation.stdout if invocation else None,
                "codex_stderr": invocation.stderr if invocation else None,
                "prompt": prompt,
                "parsed_plan": plan.as_dict() if plan else None,
            }
            path = self._debug_dir / filename
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - best-effort logging
            log.debug("Failed to write planning debug artifact: {}", exc)



