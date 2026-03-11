from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.core.worker.agent import (
    AgentBackend,
    AgentInvocation,
    AgentTask,
    TruncationMixin,
    coerce_agent_stdout_text,
    load_agent_backend,
    resolve_worker_debug_dir,
    run_agent_task,
)
from loreley.core.worker.agent.backends import CodexCliBackend
from loreley.core.worker.markdown import extract_markdown_summary

console = Console()
log = logger.bind(module="worker.planning")

__all__ = [
    "CommitMetric",
    "CommitPlanningContext",
    "PlanningAgent",
    "PlanningAgentRequest",
    "PlanningAgentResponse",
    "PlanningError",
    "PlanDocument",
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
class PlanDocument:
    """Markdown plan document emitted by the planning agent."""

    summary: str
    markdown: str
    focus_metrics: tuple[str, ...] = field(default_factory=tuple)
    guardrails: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "markdown": self.markdown,
            "focus_metrics": list(self.focus_metrics),
            "guardrails": list(self.guardrails),
        }


@dataclass(slots=True)
class PlanningAgentResponse:
    """Envelope containing planning results and metadata."""

    plan: PlanDocument
    raw_output: str
    prompt: str
    command: tuple[str, ...]
    stderr: str
    attempts: int
    duration_seconds: float


class PlanningAgent(TruncationMixin):
    """Bridge between Loreley's worker and the configured planning backend."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: AgentBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_attempts = max(1, self.settings.worker_planning_max_attempts)
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
                error_cls=PlanningError,
                full_auto=False,
            )

    def plan(
        self,
        request: PlanningAgentRequest,
        *,
        working_dir: Path,
    ) -> PlanningAgentResponse:
        """Generate a Markdown plan document using the configured backend."""
        worktree = Path(working_dir).expanduser().resolve()
        prompt = self._render_prompt(request)

        task = AgentTask(name="planning", prompt=prompt)

        def _debug_hook(
            attempt: int,
            invocation: AgentInvocation | None,
            plan: PlanDocument | None,
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
            _plan: PlanDocument,
        ) -> None:
            console.log(
                "[bold green]Planning agent[/] generated plan "
                f"in {invocation.duration_seconds:.1f}s "
                f"(attempt {attempt}/{total})",
            )

        def _on_attempt_retry(attempt: int, _total: int, exc: Exception) -> None:
            log.warning("Planning attempt {} failed: {}", attempt, exc)

        plan, invocation, attempts = run_agent_task(
            backend=self.backend,
            task=task,
            working_dir=worktree,
            max_attempts=self.max_attempts,
            coerce_result=lambda inv: self._coerce_plan_from_invocation(
                request=request,
                invocation=inv,
            ),
            retryable_exceptions=(PlanningError,),
            error_cls=PlanningError,
            error_message=(
                "Planning agent could not produce a plan after "
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
Produce a Markdown plan document that a coding agent can execute.

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

Output requirements:
- Operate non-interactively: do not ask for clarification, approval, or confirmation; make reasonable assumptions and proceed.
- Return a single Markdown document.
- Use '##' headings for these sections: Summary, Steps, Validation, Notes (optional).
- In Steps, prefer 3-6 numbered steps with concrete actions and relevant files.
- Mention file paths in backticks.
- Avoid fenced code blocks.
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
    ) -> PlanDocument:
        """Coerce backend stdout into a PlanDocument (best-effort)."""

        raw_text = (invocation.stdout or "").strip()
        markdown = coerce_agent_stdout_text(raw_text)
        summary = self._extract_summary(markdown) or (request.goal or "").strip() or "N/A"
        summary = self._truncate(summary, limit=512)

        focus_metrics = tuple(metric.name for metric in request.base.metrics)[:3]
        guardrails = tuple(request.constraints)

        if not markdown:
            markdown = f"## Summary\n- {summary}\n"

        return PlanDocument(
            summary=summary,
            markdown=markdown,
            focus_metrics=focus_metrics,
            guardrails=guardrails,
        )

    def _extract_summary(self, markdown: str) -> str:
        """Extract a short summary line from a Markdown document (best-effort)."""

        return extract_markdown_summary(markdown)

    def _dump_debug_artifact(
        self,
        *,
        request: PlanningAgentRequest,
        worktree: Path,
        invocation: AgentInvocation | None,
        prompt: str,
        attempt: int,
        plan: PlanDocument | None,
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
                "plan": plan.as_dict() if plan else None,
            }
            path = self._debug_dir / filename
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - best-effort logging
            log.debug("Failed to write planning debug artifact: {}", exc)
