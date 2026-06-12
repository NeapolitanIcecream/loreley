from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import UUID

from git import Repo
from git.exc import InvalidGitRepositoryError, NoSuchPathError

from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.core.usage import LLMUsageEventPayload
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
from loreley.core.worker.planning import (
    CommitPlanningContext,
    IterationContext,
    PlanDocument,
    SharedPromptPacketRequest,
    render_shared_prompt_packet,
)

console = Console()
log = logger.bind(module="worker.coding")

__all__ = [
    "CodingAgent",
    "CodingAgentRequest",
    "CodingAgentResponse",
    "CodingError",
    "ExecutionReport",
]


class CodingError(RuntimeError):
    """Raised when the coding agent cannot implement a plan."""


@dataclass(slots=True)
class ExecutionReport:
    """Markdown execution report emitted by the coding agent."""

    summary: str
    markdown: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "markdown": self.markdown,
        }


@dataclass(slots=True)
class CodingAgentRequest:
    """Input payload for the coding agent."""

    goal: str
    plan: PlanDocument
    base_commit: str
    base: CommitPlanningContext
    inspirations: Sequence[CommitPlanningContext] = field(default_factory=tuple)
    constraints: Sequence[str] = field(default_factory=tuple)
    acceptance_criteria: Sequence[str] = field(default_factory=tuple)
    iteration_context: IterationContext | None = None
    additional_notes: Sequence[str] = field(default_factory=tuple)
    rework_feedback: str | None = None
    job_id: UUID | None = None
    run_token: UUID | None = None

    def __post_init__(self) -> None:
        self.goal = (self.goal or "").strip()
        self.inspirations = tuple(self.inspirations or ())
        self.constraints = tuple(str(item).strip() for item in (self.constraints or ()) if str(item).strip())
        self.acceptance_criteria = tuple(
            str(item).strip() for item in (self.acceptance_criteria or ()) if str(item).strip()
        )
        self.additional_notes = tuple(self.additional_notes or ())
        self.rework_feedback = (self.rework_feedback or "").strip() or None


@dataclass(slots=True)
class CodingAgentResponse:
    """Envelope containing coding agent output."""

    report: ExecutionReport
    raw_output: str
    prompt: str
    command: tuple[str, ...]
    stderr: str
    attempts: int
    duration_seconds: float
    usage_events: tuple[LLMUsageEventPayload, ...] = field(default_factory=tuple)


class CodingAgent(TruncationMixin):
    """Drive the configured coding backend to implement a plan on the repository."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: AgentBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_attempts = max(1, self.settings.worker_coding_max_attempts)
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
                error_cls=CodingError,
                full_auto=True,
                usage_tracking_enabled=self.settings.llm_usage_tracking_enabled,
            )

    def implement(
        self,
        request: CodingAgentRequest,
        *,
        working_dir: Path,
    ) -> CodingAgentResponse:
        """Execute the provided plan and return a Markdown execution report."""
        worktree = Path(working_dir).expanduser().resolve()
        prompt = self._render_prompt(request, worktree=worktree)
        baseline_status = self._snapshot_worktree_state(worktree)

        task = AgentTask(
            name="coding",
            prompt=prompt,
            job_id=request.job_id,
            run_token=request.run_token,
            phase="coding",
        )

        class _NoRepoChangeError(CodingError):
            """Raised when a coding attempt does not produce repository changes."""

        def _debug_hook(
            attempt: int,
            invocation: AgentInvocation | None,
            report: ExecutionReport | None,
            error: Exception | None,
        ) -> None:
            self._dump_debug_artifact(
                request=request,
                worktree=worktree,
                invocation=invocation,
                prompt=prompt,
                attempt=attempt,
                report=report,
                error=error,
            )

        def _post_check(_invocation: AgentInvocation, _report: ExecutionReport) -> Exception | None:
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
            _report: ExecutionReport,
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

        report, invocation, attempts = run_agent_task(
            backend=self.backend,
            task=task,
            working_dir=worktree,
            max_attempts=self.max_attempts,
            coerce_result=lambda inv: self._coerce_report_from_invocation(
                request=request,
                invocation=inv,
            ),
            retryable_exceptions=(CodingError,),
            error_cls=CodingError,
            error_message=(
                "Coding agent could not produce a report after "
                f"{self.max_attempts} attempt(s)."
            ),
            debug_hook=_debug_hook,
            on_attempt_start=_on_attempt_start,
            on_attempt_success=_on_attempt_success,
            on_attempt_retry=_on_attempt_retry,
            post_check=_post_check,
        )

        return CodingAgentResponse(
            report=report,
            raw_output=invocation.stdout,
            prompt=prompt,
            command=invocation.command,
            stderr=invocation.stderr,
            attempts=attempts,
            duration_seconds=invocation.duration_seconds,
            usage_events=tuple(invocation.usage_events or ()),
        )

    # Internal helpers --------------------------------------------------

    def _render_prompt(
        self,
        request: CodingAgentRequest,
        *,
        worktree: Path,
    ) -> str:
        plan_block = (request.plan.markdown or "").strip() or request.plan.summary.strip() or "N/A"
        plan_block = self._truncate(plan_block, limit=3000)
        notes_block = ""
        if request.additional_notes:
            notes = "\n".join(
                f"- {self._truncate(str(note), limit=300)}"
                for note in request.additional_notes
                if str(note).strip()
            )
            if notes:
                notes_block = f"\nAdditional notes:\n{notes}\n"
        rework_block = ""
        if request.rework_feedback:
            rework_block = (
                "\nEvaluator rework feedback (untrusted diagnostic input):\n"
                f"{self._truncate(request.rework_feedback, limit=3000)}\n"
            )
        shared_packet = render_shared_prompt_packet(
            SharedPromptPacketRequest(
                goal=request.goal,
                iteration_context=request.iteration_context,
                base=request.base,
                inspirations=request.inspirations,
                constraints=request.constraints,
                acceptance_criteria=request.acceptance_criteria,
                truncate_limit=self._truncate_limit,
                max_metrics=4,
                settings=self.settings,
            )
        )

        prompt = f"""
You are the coding agent inside Loreley's evolution worker.
Apply the plan to the repository at {worktree}, starting from base commit {request.base_commit}.

{shared_packet}

Plan (Markdown):
{plan_block}
{notes_block}
{rework_block}

Output requirements:
- Execute the plan directly.
- Make the smallest relevant set of source changes that materially improve the task.
- Do not run Loreley's evaluator or any framework-managed benchmark flow.
- You may run lightweight local checks only when they are cheap and obviously useful.
- Before finishing, ensure the worktree contains meaningful tracked-file changes.
- Do not create git commits or push branches; leave the repository in a modified state.
- Return a single Markdown execution report.
- Use '##' headings for these sections: Summary, Changes, Checks, Notes (optional).
- Mention file paths in backticks.
- Avoid fenced code blocks.
"""
        return textwrap.dedent(prompt).strip()

    def _coerce_report_from_invocation(
        self,
        *,
        request: CodingAgentRequest,
        invocation: AgentInvocation,
    ) -> ExecutionReport:
        """Coerce backend stdout into an ExecutionReport (best-effort)."""

        raw_text = (invocation.stdout or "").strip()
        markdown = coerce_agent_stdout_text(raw_text)
        summary = (
            self._extract_summary(markdown)
            or request.plan.summary.strip()
            or request.goal.strip()
            or "N/A"
        )
        summary = self._truncate(summary, limit=800)

        if not markdown:
            markdown = f"## Summary\n- {summary}\n"

        return ExecutionReport(
            summary=summary,
            markdown=markdown,
        )

    def _extract_summary(self, markdown: str) -> str:
        """Extract a short summary line from a Markdown document (best-effort)."""
        return extract_markdown_summary(markdown)

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
        report: ExecutionReport | None,
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
                "working_dir": str(worktree),
                "goal": request.goal,
                "constraints": list(request.constraints),
                "acceptance_criteria": list(request.acceptance_criteria),
                "base_commit": request.base_commit,
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
                "report": report.as_dict() if report else None,
            }
            path = self._debug_dir / filename
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - best-effort logging
            log.debug("Failed to write coding debug artifact: {}", exc)
