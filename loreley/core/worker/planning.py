from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rich.console import Console

from loreley.config import Settings, get_settings

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


@dataclass(slots=True, frozen=True)
class _CodexInvocation:
    """Internal helper summarising a Codex CLI call."""

    command: tuple[str, ...]
    stdout: str
    stderr: str
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

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.codex_bin = self.settings.worker_planning_codex_bin
        self.profile = self.settings.worker_planning_codex_profile
        self.max_attempts = max(1, self.settings.worker_planning_max_attempts)
        self.timeout = self.settings.worker_planning_timeout_seconds
        self.extra_env = dict(self.settings.worker_planning_extra_env or {})
        self.schema_override = self.settings.worker_planning_schema_path
        self.schema_mode = self._resolve_schema_mode(
            configured_mode=self.settings.worker_planning_codex_schema_mode,
            api_spec=self.settings.openai_api_spec,
        )
        self._truncate_limit = 2000
        self._max_highlights = 8
        self._max_metrics = 10

    def plan(
        self,
        request: PlanningAgentRequest,
        *,
        working_dir: Path,
    ) -> PlanningAgentResponse:
        """Generate a structured plan using Codex."""
        worktree = self._validate_workdir(working_dir)
        prompt = self._render_prompt(request)

        schema_path: Path | None = None
        cleanup_handle: Path | None = None
        if self.schema_mode == "native":
            schema_path, cleanup_handle = self._materialise_schema()
        last_error: Exception | None = None
        try:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    invocation = self._invoke_codex(prompt, schema_path, worktree)
                    plan_model = self._parse_plan(invocation.stdout)
                    plan = self._to_domain(plan_model)
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
                    log.warning(
                        "Planning attempt {} failed: {}",
                        attempt,
                        exc,
                    )
            raise PlanningError(
                "Planning agent could not produce a valid plan after "
                f"{self.max_attempts} attempt(s).",
            ) from last_error
        finally:
            if cleanup_handle:
                cleanup_handle.unlink(missing_ok=True)

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

    def _materialise_schema(self) -> tuple[Path, Path | None]:
        """Return the schema path and optional temp file to clean up."""
        if self.schema_override:
            path = Path(self.schema_override).expanduser().resolve()
            if not path.exists():
                raise PlanningError(
                    f"Configured planning schema {path} does not exist.",
                )
            return path, None

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="loreley-planning-schema-",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        try:
            json.dump(PLANNING_OUTPUT_SCHEMA, tmp, ensure_ascii=True, indent=2)
        finally:
            tmp.close()
        path = Path(tmp.name)
        return path, path

    def _invoke_codex(
        self,
        prompt: str,
        schema_path: Path | None,
        worktree: Path,
    ) -> _CodexInvocation:
        command: list[str] = [
            self.codex_bin,
            "exec",
        ]
        if self.schema_mode == "native":
            if schema_path is None:
                raise PlanningError("Schema path is required when schema_mode='native'.")
            command.extend(
                [
                    "--output-schema",
                    str(schema_path),
                ],
            )
        if self.profile:
            command.extend(["--profile", self.profile])

        env = os.environ.copy()
        env.update(self.extra_env)
        start = monotonic()
        console.log("[cyan]Planning agent[/] requesting Codex plan...")
        try:
            result = subprocess.run(
                command,
                cwd=str(worktree),
                env=env,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PlanningError(
                f"codex exec timed out after {self.timeout}s.",
            ) from exc
        duration = monotonic() - start
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        log.debug(
            "codex exec finished (exit_code={}, duration={:.2f}s)",
            result.returncode,
            duration,
        )

        if result.returncode != 0:
            raise PlanningError(
                f"codex exec failed with exit code {result.returncode}. "
                f"stderr: {stderr or 'N/A'}",
            )

        if not stdout:
            raise PlanningError("codex exec returned an empty response.")

        return _CodexInvocation(
            command=tuple(command),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
        )

    @staticmethod
    def _resolve_schema_mode(
        configured_mode: str,
        api_spec: str,
    ) -> str:
        """Resolve the effective schema mode from configuration and API spec."""
        if configured_mode != "auto":
            return configured_mode
        if api_spec == "chat_completions":
            return "prompt"
        return "native"

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

    @staticmethod
    def _validate_workdir(working_dir: Path) -> Path:
        path = Path(working_dir).expanduser().resolve()
        if not path.exists():
            raise PlanningError(f"Working directory {path} does not exist.")
        if not path.is_dir():
            raise PlanningError(f"Working directory {path} is not a directory.")
        git_dir = path / ".git"
        if not git_dir.exists():
            raise PlanningError(
                f"Planning agent requires a git repository at {path} (missing .git).",
            )
        return path


