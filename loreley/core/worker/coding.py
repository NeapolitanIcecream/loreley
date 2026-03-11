from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from git import Repo
from git.exc import InvalidGitRepositoryError, NoSuchPathError

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
from loreley.core.worker.planning import PlanDocument

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

    report: ExecutionReport
    raw_output: str
    prompt: str
    command: tuple[str, ...]
    stderr: str
    attempts: int
    duration_seconds: float


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

        task = AgentTask(name="coding", prompt=prompt)

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
        )

    # Internal helpers --------------------------------------------------

    def _render_prompt(
        self,
        request: CodingAgentRequest,
        *,
        worktree: Path,
    ) -> str:
        constraints = "\n".join(f"- {item}" for item in request.constraints) or "None"
        acceptance = "\n".join(f"- {item}" for item in request.acceptance_criteria) or "None"
        notes = "\n".join(f"- {item}" for item in request.additional_notes) or "None"
        iteration_hint = request.iteration_hint or "None provided"

        plan_block = (request.plan.markdown or "").strip() or request.plan.summary.strip() or "N/A"
        plan_block = self._truncate(plan_block, limit=3000)

        prompt = f"""
You are the coding agent inside Loreley's evolution worker.
Apply the plan to the repository at {worktree}, starting from base commit {request.base_commit}.

Goal:
{request.goal.strip()}

Constraints:
{constraints}

Acceptance criteria:
{acceptance}

Iteration hint:
{iteration_hint}

Additional notes:
{notes}

Plan (Markdown):
{plan_block}

Output requirements:
- Apply the required changes.
- Operate non-interactively: do not ask for clarification, approval, or confirmation; make reasonable assumptions and proceed.
- Treat explicit speed or runtime requirements in the goal as hard constraints, not nice-to-haves; avoid shipping multi-second algorithms when the goal calls for fast execution.
- Before finishing, ensure `git status --porcelain` is non-empty; if the worktree is still clean, keep editing until you have at least one meaningful tracked-file change.
- Prefer the smallest relevant set of source-file edits needed to execute the plan; avoid unrelated documentation, config, or formatting churn unless it is directly required for correctness.
- Do not delete or rename existing tracked files unless the plan explicitly requires it, and never leave a required tracked file absent or half-rewritten at the end of the attempt.
- Do not create git commits or push branches; leave the repository in a modified state.
- Return a single Markdown execution report.
- Use '##' headings for these sections: Summary, Changes, Tests, Follow-ups (optional).
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
                "report": report.as_dict() if report else None,
            }
            path = self._debug_dir / filename
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - best-effort logging
            log.debug("Failed to write coding debug artifact: {}", exc)
