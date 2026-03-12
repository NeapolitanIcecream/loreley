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
    "IterationContext",
    "PlanningAgent",
    "PlanningAgentRequest",
    "PlanningAgentResponse",
    "PlanningError",
    "PlanDocument",
    "render_shared_prompt_packet",
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
class IterationContext:
    """Small, structured facts about the current sampling stage."""

    seed_job: bool = False
    sampling_strategy: str | None = None
    facts: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.sampling_strategy = (self.sampling_strategy or "").strip() or None
        self.facts = tuple(str(item).strip() for item in (self.facts or ()) if str(item).strip())


@dataclass(slots=True)
class PlanningAgentRequest:
    """Input payload for the planning agent."""

    base: CommitPlanningContext
    inspirations: Sequence[CommitPlanningContext]
    goal: str
    iteration_context: IterationContext | None = None

    def __post_init__(self) -> None:
        self.inspirations = tuple(self.inspirations or ())
        self.goal = (self.goal or "").strip()


WORKER_CONTRACT_LINES: tuple[str, ...] = (
    "Operate non-interactively. Do not ask for clarification, approval, or confirmation.",
    "Modify repository files only. Do not create commits, tags, or branches. Do not push.",
    "Do not run Loreley's evaluator or any framework-managed end-to-end benchmark flow.",
    "Leave the repository in a modified worktree state.",
    "Prefer focused, minimal changes that directly improve the task.",
    "Preserve existing tracked files unless a change is clearly necessary for the task.",
)

WORKER_CONTRACT_GUARDRAILS: tuple[str, ...] = (
    "non_interactive_worker",
    "framework_managed_evaluation",
    "leave_modified_worktree",
    "no_git_commits",
)


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


class _PromptPacketRenderer(TruncationMixin):
    """Render compact, transfer-friendly prompt context blocks."""

    def __init__(self, *, truncate_limit: int = 2000, max_metrics: int = 4) -> None:
        self._truncate_limit = truncate_limit
        self._max_metrics = max(1, int(max_metrics))

    def render(
        self,
        *,
        goal: str,
        iteration_context: IterationContext | None,
        base: CommitPlanningContext,
        inspirations: Sequence[CommitPlanningContext],
    ) -> str:
        inspiration_blocks = "\n\n".join(
            self._format_inspiration_block(index=idx + 1, base=base, context=ctx)
            for idx, ctx in enumerate(inspirations)
        )
        packet = f"""
You are working inside Loreley's evolution worker.

Evolution Goal:
{goal.strip() or "N/A"}

Worker Contract:
{self._format_worker_contract()}

Iteration Context:
{self._format_iteration_context(iteration_context)}

Base Commit Context:
{self._format_base_commit_block(base)}

Inspiration Commits:
{inspiration_blocks or "None"}
"""
        return textwrap.dedent(packet).strip()

    def _format_worker_contract(self) -> str:
        return "\n".join(f"- {line}" for line in WORKER_CONTRACT_LINES)

    def _format_iteration_context(self, context: IterationContext | None) -> str:
        context = context or IterationContext()
        lines = [f"- seed_job: {'true' if context.seed_job else 'false'}"]
        if context.sampling_strategy:
            lines.append(f"- sampling_strategy: {context.sampling_strategy}")
        if context.facts:
            lines.append("- sampler_facts:")
            lines.extend(f"  - {self._truncate(fact, limit=200)}" for fact in context.facts)
        return "\n".join(lines)

    def _format_base_commit_block(self, context: CommitPlanningContext) -> str:
        lines = [
            f"- hash: {context.commit_hash}",
            f"- summary: {self._truncate(context.subject)}",
            f"- change_summary: {self._truncate(context.change_summary, limit=512)}",
        ]
        if context.evaluation_summary:
            lines.append(
                f"- evaluation_summary: {self._truncate(context.evaluation_summary, limit=512)}"
            )
        metrics_block = self._format_metrics_block(context.metrics)
        if metrics_block:
            lines.append("- selected_metrics:")
            lines.extend(metrics_block)
        key_files_block = self._format_key_files_block(context.key_files)
        if key_files_block:
            lines.append("- key_files:")
            lines.extend(key_files_block)
        return "\n".join(lines)

    def _format_inspiration_block(
        self,
        *,
        index: int,
        base: CommitPlanningContext,
        context: CommitPlanningContext,
    ) -> str:
        lines = [
            f"Inspiration #{index}",
            f"- hash: {context.commit_hash}",
            f"- why_it_matters: {self._derive_why_it_matters(base=base, inspiration=context)}",
        ]
        distinctive_changes = self._extract_distinctive_changes(context.trajectory)
        if distinctive_changes:
            lines.append("- distinctive_changes_vs_base:")
            lines.extend(f"  - {self._truncate(change, limit=240)}" for change in distinctive_changes)
        if context.evaluation_summary:
            lines.append(
                f"- evaluation_summary: {self._truncate(context.evaluation_summary, limit=512)}"
            )
        metrics_block = self._format_metrics_block(context.metrics)
        if metrics_block:
            lines.append("- selected_metrics:")
            lines.extend(metrics_block)
        key_files_block = self._format_key_files_block(context.key_files)
        if key_files_block:
            lines.append("- key_files:")
            lines.extend(key_files_block)
        return "\n".join(lines)

    def _format_metrics_block(self, metrics: Sequence[CommitMetric]) -> list[str]:
        sliced = tuple(metrics)[: self._max_metrics]
        lines: list[str] = []
        for metric in sliced:
            detail = f"{metric.value:g}"
            if metric.unit:
                detail = f"{detail}{metric.unit}"
            hb = ""
            if metric.higher_is_better is not None:
                hb = " (higher is better)" if metric.higher_is_better else " (lower is better)"
            summary = f" - {self._truncate(metric.summary, limit=120)}" if metric.summary else ""
            lines.append(f"  - `{metric.name}`: {detail}{hb}{summary}")
        return lines

    def _format_key_files_block(self, key_files: Sequence[str]) -> list[str]:
        return [f"  - `{self._truncate(path, limit=200)}`" for path in tuple(key_files)[:8]]

    def _derive_why_it_matters(
        self,
        *,
        base: CommitPlanningContext,
        inspiration: CommitPlanningContext,
    ) -> str:
        metric_reason = self._derive_metric_reason(base=base, inspiration=inspiration)
        if metric_reason:
            return metric_reason
        for candidate in (inspiration.evaluation_summary, inspiration.change_summary):
            if candidate and candidate != "N/A":
                return self._truncate(candidate, limit=240)
        return "Offers an alternative implementation direction relative to base."

    def _derive_metric_reason(
        self,
        *,
        base: CommitPlanningContext,
        inspiration: CommitPlanningContext,
    ) -> str | None:
        base_by_name = {
            metric.name: metric
            for metric in base.metrics
            if metric.name and metric.higher_is_better is not None
        }
        improvements: list[tuple[float, str]] = []
        for metric in inspiration.metrics:
            base_metric = base_by_name.get(metric.name)
            if (
                base_metric is None
                or metric.higher_is_better is None
                or base_metric.higher_is_better != metric.higher_is_better
            ):
                continue
            delta = float(metric.value) - float(base_metric.value)
            better = delta > 0 if metric.higher_is_better else delta < 0
            if not better:
                continue
            magnitude = abs(delta)
            improvements.append(
                (
                    magnitude,
                    f"`{metric.name}` ({base_metric.value:g} -> {metric.value:g})",
                )
            )
        if not improvements:
            return None
        improvements.sort(key=lambda item: item[0], reverse=True)
        top = [text for _, text in improvements[:2]]
        if len(top) == 1:
            return f"Improves {top[0]} versus base."
        return f"Improves {top[0]} and {top[1]} versus base."

    def _extract_distinctive_changes(self, trajectory: Sequence[str]) -> tuple[str, ...]:
        buckets: dict[str, list[str]] = {
            "earliest": [],
            "older": [],
            "recent": [],
        }
        current: str | None = None
        for raw_line in trajectory:
            stripped = raw_line.strip()
            if stripped.startswith("- Earliest unique steps"):
                current = "earliest"
                continue
            if stripped.startswith("- Older unique steps"):
                current = "older"
                continue
            if stripped.startswith("- Recent unique steps"):
                current = "recent"
                continue
            if not stripped.startswith("- "):
                continue
            item = stripped[2:].strip()
            if current is None or not item:
                continue
            if item.startswith("unique_steps_count:") or item.startswith("Omitted "):
                continue
            if item.startswith("Trajectory unavailable:"):
                continue
            buckets[current].append(item)

        selected = [*buckets["earliest"][:2], *buckets["older"][:1], *buckets["recent"][:2]]
        deduped: list[str] = []
        seen: set[str] = set()
        for item in selected:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) >= 4:
                break
        return tuple(deduped)


def render_shared_prompt_packet(
    *,
    goal: str,
    iteration_context: IterationContext | None,
    base: CommitPlanningContext,
    inspirations: Sequence[CommitPlanningContext],
    truncate_limit: int = 2000,
    max_metrics: int = 4,
) -> str:
    """Render the thin Loreley task packet shared by planning and coding."""

    renderer = _PromptPacketRenderer(
        truncate_limit=truncate_limit,
        max_metrics=max_metrics,
    )
    return renderer.render(
        goal=goal,
        iteration_context=iteration_context,
        base=base,
        inspirations=inspirations,
    )


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
        """Compose the thin planning prompt for the configured backend."""
        shared_packet = render_shared_prompt_packet(
            goal=request.goal,
            iteration_context=request.iteration_context,
            base=request.base,
            inspirations=request.inspirations,
            truncate_limit=self._truncate_limit,
            max_metrics=4,
        )
        prompt = f"""
You are the planning agent inside Loreley's evolution worker.

Produce the next implementation plan for the coding agent.

{shared_packet}

Planning Instructions:
- Use the base commit and inspiration commits to identify the most promising next move.
- Optimize for one coherent next step, not a broad rewrite.
- If the context is incomplete, make a reasonable assumption and proceed.
- Do not restate all context; synthesize it into a concrete plan.

Return:
- A short Markdown document.
- Use these sections: `## Summary`, `## Steps`, `## Validation`, `## Notes` (optional).
- In `## Steps`, use 3-6 concrete numbered steps.
- Mention file paths in backticks.
- Avoid fenced code blocks.
"""
        return textwrap.dedent(prompt).strip()

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

        focus_metrics = tuple(metric.name for metric in request.base.metrics)[:4]
        guardrails = WORKER_CONTRACT_GUARDRAILS

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
                "iteration_context": {
                    "seed_job": bool(request.iteration_context.seed_job)
                    if request.iteration_context
                    else False,
                    "sampling_strategy": (
                        request.iteration_context.sampling_strategy
                        if request.iteration_context
                        else None
                    ),
                    "facts": list(request.iteration_context.facts)
                    if request.iteration_context
                    else [],
                },
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
