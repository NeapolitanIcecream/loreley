from __future__ import annotations

import hashlib
import json
import os
import stat
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
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
_HASH_CHUNK_SIZE = 1024 * 1024

__all__ = [
    "CodingAgent",
    "CodingAgentRequest",
    "CodingAgentResponse",
    "CodingError",
    "ExecutionReport",
]


class CodingError(RuntimeError):
    """Raised when the coding agent cannot implement a plan."""


class _NoEffectiveRepoChangeError(CodingError):
    """Raised when an attempt leaves no content-level repository delta."""

    def __init__(self, message: str, *, has_output: bool) -> None:
        super().__init__(message)
        self.has_output = has_output


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
    working_directory: str | None = None


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
        baseline_snapshot = self._snapshot_worktree_state(worktree)

        task = AgentTask(
            name="coding",
            prompt=prompt,
            job_id=request.job_id,
            run_token=request.run_token,
            phase="coding",
        )

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

        def _post_check(invocation: AgentInvocation, _report: ExecutionReport) -> Exception | None:
            current_snapshot = self._snapshot_worktree_state(worktree)
            if current_snapshot == baseline_snapshot:
                has_output = bool((invocation.stdout or "").strip())
                output_detail = "produced output but" if has_output else "produced no output and"
                return _NoEffectiveRepoChangeError(
                    f"Coding agent {output_detail} no effective repository changes.",
                    has_output=has_output,
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
            if isinstance(exc, _NoEffectiveRepoChangeError):
                console.log(
                    "[yellow]Coding agent[/] produced no effective repository changes; retrying…",
                )
                log.warning(
                    "Coding attempt {} produced no effective repository changes output_present={}",
                    attempt,
                    exc.has_output,
                )
                return
            log.warning("Coding attempt {} failed: {}", attempt, exc)

        try:
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
        except CodingError as exc:
            self._raise_no_effective_change_error(exc)
            raise

        return CodingAgentResponse(
            report=report,
            raw_output=invocation.stdout,
            prompt=prompt,
            command=invocation.command,
            stderr=invocation.stderr,
            attempts=attempts,
            duration_seconds=invocation.duration_seconds,
            usage_events=tuple(invocation.usage_events or ()),
            working_directory=invocation.working_directory,
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
        """Return a content-aware snapshot of Git-visible worktree changes."""
        try:
            repo = Repo(worktree)
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:  # pragma: no cover - defensive
            raise CodingError(f"Invalid git worktree for coding agent: {worktree}") from exc

        try:
            status_output = repo.git.status(
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise CodingError("Failed to inspect worktree status during coding run.") from exc

        status_entries, content_paths = self._parse_worktree_status(
            status_output,
            clean_exclude_patterns=self._snapshot_clean_exclude_patterns(),
        )
        content_entries = [
            f"content\0{path}\0{self._fingerprint_worktree_path(worktree, path)}"
            for path in content_paths
        ]
        return tuple(sorted((*status_entries, *content_entries)))

    def _parse_worktree_status(
        self,
        status_output: str,
        *,
        clean_exclude_patterns: Sequence[str] = (),
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        status_entries: list[str] = []
        content_paths: list[str] = []
        for status, path, old_path in self._iter_worktree_status_records(status_output):
            if self._skip_snapshot_status_path(path, status, clean_exclude_patterns):
                continue
            status_entries.append(f"status\0{status}\0{path}")
            content_paths.append(path)
            if old_path:
                status_entries.append(f"status-source\0{status}\0{old_path}")
        return tuple(status_entries), tuple(dict.fromkeys(content_paths))

    def _raise_no_effective_change_error(self, exc: CodingError) -> None:
        cause = exc.__cause__
        if not isinstance(cause, _NoEffectiveRepoChangeError):
            return
        output_detail = "produced output but" if cause.has_output else "produced no output and"
        error = CodingError(
            f"Coding agent {output_detail} no effective repository changes "
            f"after {self.max_attempts} attempt(s)."
        )
        usage_events = getattr(exc, "usage_events", ())
        if usage_events:
            setattr(error, "usage_events", usage_events)
        raise error from cause

    def _iter_worktree_status_records(
        self,
        status_output: str,
    ) -> tuple[tuple[str, str, str | None], ...]:
        tokens = [token for token in status_output.split("\0") if token]
        records: list[tuple[str, str, str | None]] = []
        index = 0
        while index < len(tokens):
            entry = tokens[index]
            if len(entry) < 4:
                index += 1
                continue
            status = entry[:2]
            path = entry[3:]
            index += 1
            old_path = None
            if ("R" in status or "C" in status) and index < len(tokens):
                old_path = tokens[index] or None
                index += 1
            if path:
                records.append((status, path, old_path))
        return tuple(records)

    def _skip_snapshot_status_path(
        self,
        path: str,
        status: str,
        clean_exclude_patterns: Sequence[str],
    ) -> bool:
        return status == "??" and self._matches_snapshot_clean_exclude(
            path,
            clean_exclude_patterns,
        )

    def _fingerprint_worktree_path(self, worktree: Path, repo_path: str) -> str:
        unsafe_reason = self._unsafe_status_path_reason(repo_path)
        if unsafe_reason is not None:
            return f"unsafe-path:{unsafe_reason}"

        candidate = worktree / repo_path
        try:
            path_stat = os.lstat(candidate)
        except FileNotFoundError:
            return "missing"
        except OSError as exc:
            return f"unreadable:{type(exc).__name__}:{getattr(exc, 'errno', '')}"

        permissions = stat.S_IMODE(path_stat.st_mode)
        if stat.S_ISLNK(path_stat.st_mode):
            try:
                target = os.readlink(candidate)
            except OSError as exc:
                return f"symlink-unreadable:{permissions:o}:{type(exc).__name__}"
            return f"symlink:{permissions:o}:{target}"

        if stat.S_ISREG(path_stat.st_mode):
            digest = hashlib.sha256()
            try:
                with candidate.open("rb") as file:
                    for chunk in iter(lambda: file.read(_HASH_CHUNK_SIZE), b""):
                        digest.update(chunk)
            except OSError as exc:
                return (
                    f"file-unreadable:{permissions:o}:{path_stat.st_size}:"
                    f"{type(exc).__name__}:{getattr(exc, 'errno', '')}"
                )
            return f"file:{permissions:o}:{path_stat.st_size}:{digest.hexdigest()}"

        file_type = stat.S_IFMT(path_stat.st_mode)
        if stat.S_ISDIR(path_stat.st_mode):
            return f"dir:{permissions:o}"
        return f"special:{file_type:o}:{permissions:o}:{path_stat.st_size}"

    def _snapshot_clean_exclude_patterns(self) -> tuple[str, ...]:
        patterns: list[str] = []
        for raw_pattern in self.settings.worker_repo_clean_excludes:
            pattern = self._normalize_snapshot_clean_exclude(raw_pattern)
            if pattern is not None:
                patterns.append(pattern)
        return tuple(dict.fromkeys(patterns))

    def _normalize_snapshot_clean_exclude(self, raw_pattern: str) -> str | None:
        pattern = str(raw_pattern or "").strip()
        if not pattern:
            return None
        while pattern.startswith("./"):
            pattern = pattern[2:]
        pattern = pattern.rstrip("/")
        if not pattern or self._unsafe_status_path_reason(pattern) is not None:
            return None
        return pattern

    def _matches_snapshot_clean_exclude(
        self,
        repo_path: str,
        patterns: Sequence[str],
    ) -> bool:
        normalized_path = self._normalize_repo_status_path(repo_path)
        if normalized_path is None:
            return False
        return any(
            self._repo_path_matches_clean_exclude(normalized_path, pattern)
            for pattern in patterns
        )

    def _normalize_repo_status_path(self, repo_path: str) -> str | None:
        path = str(repo_path or "")
        while path.startswith("./"):
            path = path[2:]
        path = path.rstrip("/")
        if not path or self._unsafe_status_path_reason(path) is not None:
            return None
        return path

    def _repo_path_matches_clean_exclude(self, repo_path: str, pattern: str) -> bool:
        if repo_path == pattern or repo_path.startswith(f"{pattern}/"):
            return True
        if fnmatchcase(repo_path, pattern):
            return True
        path_parts = PurePosixPath(repo_path).parts
        if "/" not in pattern:
            return any(part == pattern or fnmatchcase(part, pattern) for part in path_parts)
        return False

    def _unsafe_status_path_reason(self, repo_path: str) -> str | None:
        if not repo_path:
            return "empty"
        if "\0" in repo_path:
            return "nul_byte"
        if repo_path.startswith("/"):
            return "absolute_path"
        if any(part == ".." for part in PurePosixPath(repo_path).parts):
            return "path_traversal"
        return None

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
