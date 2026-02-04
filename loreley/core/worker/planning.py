from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

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


class PlanningAgent(TruncationMixin):
    """Bridge between Loreley's worker and the configured planning backend."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: AgentBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_attempts = max(1, self.settings.worker_planning_max_attempts)
        self.validation_mode: ValidationMode = self.settings.worker_planning_validation_mode
        self.schema_mode: SchemaMode = resolve_schema_mode(
            configured_mode=self.settings.worker_planning_codex_schema_mode,
            api_spec=self.settings.openai_api_spec,
        )
        self._truncate_limit = 2000
        self._max_highlights = 8
        self._max_metrics = 10
        self._debug_dir = resolve_worker_debug_dir(
            logs_base_dir=self.settings.logs_base_dir,
            kind="planning",
            experiment_id=self.settings.experiment_id,
        )
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

        task = build_structured_agent_task(
            name="planning",
            prompt=prompt,
            schema=PLANNING_OUTPUT_SCHEMA,
            schema_mode=self.schema_mode,
            validation_mode=self.validation_mode,
        )

        def _debug_hook(
            attempt: int,
            invocation: AgentInvocation | None,
            plan: PlanningPlan | None,
            error: Exception | None,
        ) -> None:
            self._dump_debug_artifact(
                request=request,
                worktree=worktree,
                invocation=invocation,
                prompt=prompt,
                attempt=attempt,
                plan=plan,
                error=error,
            )

        def _on_attempt_start(attempt: int, total: int) -> None:
            console.log(
                "[cyan]Planning agent[/] requesting plan "
                f"(attempt {attempt}/{total})",
            )

        def _on_attempt_success(
            attempt: int,
            total: int,
            invocation: AgentInvocation,
            _plan: PlanningPlan,
        ) -> None:
            console.log(
                "[bold green]Planning agent[/] generated plan "
                f"in {invocation.duration_seconds:.1f}s "
                f"(attempt {attempt}/{total})",
            )

        def _on_attempt_retry(attempt: int, _total: int, exc: Exception) -> None:
            log.warning("Planning attempt {} failed: {}", attempt, exc)

        plan, invocation, attempts = run_structured_agent_task(
            backend=self.backend,
            task=task,
            working_dir=worktree,
            max_attempts=self.max_attempts,
            coerce_result=lambda inv: self._coerce_plan_from_invocation(
                request=request,
                invocation=inv,
            ),
            retryable_exceptions=(PlanningError, ValidationError, json.JSONDecodeError),
            error_cls=PlanningError,
            error_message=(
                "Planning agent could not produce a valid plan after "
                f"{self.max_attempts} attempt(s)."
            ),
            debug_hook=_debug_hook,
            on_attempt_start=_on_attempt_start,
            on_attempt_success=_on_attempt_success,
            on_attempt_retry=_on_attempt_retry,
        )

        return PlanningAgentResponse(
            plan=plan,
            raw_output=invocation.stdout,
            prompt=prompt,
            command=invocation.command,
            stderr=invocation.stderr,
            attempts=attempts,
            duration_seconds=invocation.duration_seconds,
        )

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

        cold_start_block = ""
        if request.cold_start:
            cold_start_block = (
                "Cold-start seed job: the MAP-Elites archive is empty, so propose diverse\n"
                "starting directions that respect the goal and constraints.\n\n"
            )

        prompt = f"""
You are the planning agent inside Loreley's evolution worker.
Produce a concise plan that a coding agent can execute.

{cold_start_block}\

Goal:
{request.goal.strip()}

Constraints:
{constraints}

Acceptance criteria:
{acceptance}

Iteration hint:
{iteration_hint}

Base commit context:
{base_block}

Inspiration commits:
{insp_blocks or "None"}

Requested output:
- A short free-form plan.
- Prefer 3-6 numbered steps with clear actions and files when possible.
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

        def parse(stdout: str) -> PlanningPlan:
            plan_model = self._parse_plan(stdout)
            return self._to_domain(plan_model)

        return coerce_structured_output(
            validation_mode=self.validation_mode,
            stdout=invocation.stdout,
            parse=parse,
            build_from_freeform=lambda stdout: self._build_plan_from_freeform_output(
                request=request,
                raw_output=stdout,
            ),
            on_parse_error=lambda exc: self._log_invalid_output(invocation, exc),
            parse_exceptions=(ValidationError, json.JSONDecodeError),
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
        """Build a best-effort PlanningPlan from free-form agent output."""
        raw_text = (raw_output or "").strip()
        fallback_plan = raw_text or None

        summary_line = ""
        for line in raw_text.splitlines():
            cleaned = line.strip()
            if cleaned:
                summary_line = cleaned
                break
        summary_source = summary_line or request.goal.strip()
        summary = self._truncate(summary_source, limit=512)

        rationale = (
            "Planning output was handled as free-form text "
            f"(validation_mode={self.validation_mode!r}); "
            "this plan was synthesised into structured steps for downstream consumers."
        )

        focus_metrics = tuple(metric.name for metric in request.base.metrics)[:3]
        guardrails = tuple(request.constraints)
        risks: tuple[str, ...] = ()
        validation = tuple(request.acceptance_criteria) or (
            "Run the project's tests and ensure there are no regressions.",
        )

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

        def build_actions(title: str, body_lines: list[str]) -> tuple[str, ...]:
            bullets = []
            for line in body_lines:
                match = bullet_re.match(line)
                if match:
                    bullets.append(match.group("item").strip())
            if bullets:
                return tuple(self._truncate(item, limit=200) for item in bullets[:12])
            cleaned_lines = []
            if title.strip():
                cleaned_lines.append(title.strip())
            for line in body_lines:
                text = line.strip()
                if not text:
                    continue
                match = bullet_re.match(line)
                if match:
                    text = match.group("item").strip()
                cleaned_lines.append(text)
            cleaned_lines = [item for item in cleaned_lines if item]
            if not cleaned_lines:
                return (self._truncate(summary_source, limit=200),)
            return tuple(self._truncate(item, limit=200) for item in cleaned_lines[:8])

        parsed = parse_numbered_steps(raw_text) if raw_text else []
        step_objects: list[PlanStep] = []
        for idx, (title, body_lines) in enumerate(parsed[:6]):
            title_candidate = title.strip()
            if not title_candidate:
                for line in body_lines:
                    cleaned = line.strip()
                    if not cleaned:
                        continue
                    match = bullet_re.match(cleaned)
                    if match:
                        cleaned = match.group("item").strip()
                    title_candidate = cleaned
                    break
            title_candidate = title_candidate or f"Step {idx + 1}"
            block_text = "\n".join([title.strip()] + [line.rstrip() for line in body_lines]).strip()
            intent_source = " ".join(
                part.strip()
                for part in [title_candidate, block_text]
                if part and part.strip()
            )
            step_objects.append(
                PlanStep(
                    step_id=f"freeform-{idx + 1}",
                    title=self._truncate(title_candidate, limit=120),
                    intent=self._truncate(intent_source, limit=400),
                    actions=build_actions(title, body_lines),
                    files=extract_files(block_text),
                    dependencies=tuple(),
                    validation=validation,
                    risks=risks,
                    references=tuple(),
                )
            )

        if not step_objects:
            action_text = raw_text or request.goal
            step_objects = [
                PlanStep(
                    step_id="freeform-1",
                    title="Apply the free-form planning output",
                    intent="Follow the planning agent's free-form suggestions from the raw output.",
                    actions=(self._truncate(action_text, limit=200),),
                    files=extract_files(action_text),
                    dependencies=tuple(),
                    validation=validation,
                    risks=risks,
                    references=tuple(),
                )
            ]

        handoff_notes = (
            f"Planning used free-form synthesis (validation_mode={self.validation_mode!r}). "
            "Consult fallback_plan for the full text output when available.",
        )

        return PlanningPlan(
            summary=summary,
            rationale=rationale,
            focus_metrics=focus_metrics,
            guardrails=guardrails,
            risks=risks,
            validation=validation,
            steps=tuple(step_objects),
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



