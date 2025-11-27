"""Autonomous evolution worker orchestrating planning, coding, and evaluation."""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import UUID

from loguru import logger
from openai import OpenAI, OpenAIError
from rich.console import Console
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.core.worker.coding import (
    CodingAgent,
    CodingAgentRequest,
    CodingAgentResponse,
    CodingError,
    CodingPlanExecution,
    StepExecutionStatus,
)
from app.core.worker.evaluator import (
    Evaluator,
    EvaluationContext,
    EvaluationError,
    EvaluationResult,
)
from app.core.worker.planning import (
    CommitMetric,
    CommitPlanningContext,
    PlanningAgent,
    PlanningAgentRequest,
    PlanningAgentResponse,
    PlanningError,
    PlanningPlan,
)
from app.core.worker.repository import CheckoutContext, WorkerRepository
from app.db.base import session_scope
from app.db.models import CommitMetadata, EvolutionJob, JobStatus, Metric

console = Console()
log = logger.bind(module="worker.evolution")

__all__ = [
    "EvolutionWorker",
    "EvolutionWorkerError",
    "EvolutionWorkerResult",
    "CommitSummarizer",
]


class EvolutionWorkerError(RuntimeError):
    """Raised when the evolution worker cannot complete a job."""


class CommitSummaryError(RuntimeError):
    """Raised when the commit summarizer cannot produce a subject line."""


@dataclass(slots=True)
class CommitSnapshot:
    """Immutable snapshot of commit data for planning context construction."""

    commit_hash: str
    summary: str
    evaluation_summary: str | None
    highlights: tuple[str, ...]
    metrics: tuple[CommitMetric, ...]
    extra_context: dict[str, Any] = field(default_factory=dict)

    def to_planning_context(self) -> CommitPlanningContext:
        return CommitPlanningContext(
            commit_hash=self.commit_hash,
            summary=self.summary,
            highlights=self.highlights,
            evaluation_summary=self.evaluation_summary,
            metrics=self.metrics,
            extra_context=self.extra_context,
        )


@dataclass(slots=True)
class JobContext:
    """Loaded job information used across the worker stages."""

    job_id: UUID
    base_commit_hash: str
    island_id: str | None
    payload: dict[str, Any]
    base_snapshot: CommitSnapshot
    inspiration_snapshots: tuple[CommitSnapshot, ...]
    goal: str
    constraints: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    iteration_hint: str | None
    notes: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(slots=True)
class EvolutionWorkerResult:
    """Structured success payload returned by the worker."""

    job_id: UUID
    base_commit_hash: str
    candidate_commit_hash: str
    plan: PlanningAgentResponse
    coding: CodingAgentResponse
    evaluation: EvaluationResult
    checkout: CheckoutContext
    commit_message: str


class CommitSummarizer:
    """LLM-powered helper that derives concise commit subjects."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or OpenAI()
        self._model = self.settings.worker_evolution_commit_model
        self._temperature = self.settings.worker_evolution_commit_temperature
        self._max_tokens = max(32, self.settings.worker_evolution_commit_max_output_tokens)
        self._max_retries = max(1, self.settings.worker_evolution_commit_max_retries)
        self._retry_backoff = max(
            0.0,
            self.settings.worker_evolution_commit_retry_backoff_seconds,
        )
        self._subject_limit = max(32, self.settings.worker_evolution_commit_subject_max_chars)
        self._truncate_limit = 1200

    def generate(
        self,
        *,
        job: JobContext,
        plan: PlanningPlan,
        coding: CodingPlanExecution,
    ) -> str:
        """Return a commit subject line grounded in plan and coding context."""
        prompt = self._build_prompt(job=job, plan=plan, coding=coding)
        attempt = 0
        while attempt < self._max_retries:
            attempt += 1
            try:
                response = self._client.responses.create(
                    model=self._model,
                    input=prompt,
                    temperature=self._temperature,
                    max_output_tokens=self._max_tokens,
                    instructions=(
                        "Respond with a single concise git commit subject line "
                        "in imperative mood (<=72 characters)."
                    ),
                )
                subject = (response.output_text or "").strip()
                if not subject:
                    raise CommitSummaryError("Commit summarizer returned empty output.")
                cleaned = self._normalise_subject(subject)
                log.info("Commit summarizer produced subject after attempt {}", attempt)
                return cleaned
            except (OpenAIError, CommitSummaryError) as exc:
                if attempt >= self._max_retries:
                    raise CommitSummaryError(
                        f"Commit summarizer failed after {attempt} attempt(s): {exc}",
                    ) from exc
                delay = self._retry_backoff * attempt
                log.warning(
                    "Commit summarizer attempt {} failed: {}. Retrying in {:.1f}s",
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise CommitSummaryError("Commit summarizer exhausted retries without success.")

    def _build_prompt(
        self,
        *,
        job: JobContext,
        plan: PlanningPlan,
        coding: CodingPlanExecution,
    ) -> str:
        goal = job.goal.strip()
        plan_summary = plan.summary.strip()
        plan_rationale = plan.rationale.strip()
        step_lines = "\n".join(
            f"- {step.step_id} ({step.status.value}): {self._truncate(step.summary)}"
            for step in coding.step_results
        ) or "- No detailed step results."
        tests = "\n".join(f"- {item}" for item in coding.tests_executed) or "- None"
        focus_metrics = "\n".join(f"- {metric}" for metric in plan.focus_metrics) or "- None"
        guardrails = "\n".join(f"- {guardrail}" for guardrail in plan.guardrails) or "- None"
        constraints = "\n".join(f"- {entry}" for entry in job.constraints) or "- None"
        acceptance = "\n".join(f"- {entry}" for entry in job.acceptance_criteria) or "- None"
        notes = "\n".join(f"- {entry}" for entry in job.notes) or "- None"
        coding_summary = coding.implementation_summary.strip()
        fallback_commit_message = (coding.commit_message or "").strip() or "N/A"

        prompt = f"""
You generate precise git commit subjects for an autonomous evolution worker.
Summaries must stay under {self._subject_limit} characters and follow imperative mood.

Global goal:
{goal}

Plan summary:
{plan_summary}

Plan rationale:
{plan_rationale}

Plan focus metrics:
{focus_metrics}

Plan guardrails:
{guardrails}

Constraints to respect:
{constraints}

Acceptance criteria:
{acceptance}

Worker notes:
{notes}

Coding execution summary:
{coding_summary}

Step outcomes:
{step_lines}

Tests executed:
{tests}

Coding agent suggested commit message:
{fallback_commit_message}

Respond with a single subject line without surrounding quotes.
"""
        return textwrap.dedent(prompt).strip()

    def _normalise_subject(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) > self._subject_limit:
            return f"{cleaned[: self._subject_limit - 1].rstrip()}…"
        return cleaned

    def coerce_subject(self, text: str | None, *, default: str) -> str:
        """Clamp arbitrary text into a valid git subject."""
        baseline = " ".join((text or "").split()).strip()
        candidate = baseline or default.strip()
        return self._normalise_subject(candidate or default)

    def _truncate(self, text: str, limit: int | None = None) -> str:
        active = limit or self._truncate_limit
        snippet = (text or "").strip()
        if len(snippet) <= active:
            return snippet
        return f"{snippet[:active]}…"


class EvolutionWorker:
    """Service-layer entry point for executing evolution jobs synchronously."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        repository: WorkerRepository | None = None,
        planning_agent: PlanningAgent | None = None,
        coding_agent: CodingAgent | None = None,
        evaluator: Evaluator | None = None,
        summarizer: CommitSummarizer | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or WorkerRepository(self.settings)
        self.planning_agent = planning_agent or PlanningAgent(self.settings)
        self.coding_agent = coding_agent or CodingAgent(self.settings)
        self.evaluator = evaluator or Evaluator(self.settings)
        self.summarizer = summarizer or CommitSummarizer(settings=self.settings)

    def run(self, job_id: str | UUID) -> EvolutionWorkerResult:
        """Execute the full evolution loop for the requested job."""
        job_uuid = self._coerce_uuid(job_id)
        job_ctx = self._start_job(job_uuid)
        checkout: CheckoutContext | None = None
        plan_response: PlanningAgentResponse | None = None
        coding_response: CodingAgentResponse | None = None
        evaluation_result: EvaluationResult | None = None
        commit_message: str | None = None
        candidate_commit: str | None = None

        console.log(
            f"[bold cyan]Evolution worker[/] starting job={job_uuid} "
            f"base={job_ctx.base_commit_hash}",
        )
        try:
            checkout = self.repository.checkout_for_job(
                job_id=job_uuid,
                base_commit=job_ctx.base_commit_hash,
            )
            plan_response = self._run_planning(job_ctx, checkout)
            coding_response = self._run_coding(job_ctx, plan_response, checkout)
            commit_message = self._prepare_commit_message(
                job_ctx=job_ctx,
                plan=plan_response,
                coding=coding_response,
            )
            candidate_commit = self._create_commit(
                checkout=checkout,
                commit_message=commit_message,
            )
            evaluation_result = self._run_evaluation(
                job_ctx=job_ctx,
                checkout=checkout,
                plan=plan_response,
                candidate_commit=candidate_commit,
            )
            self._persist_success(
                job_ctx=job_ctx,
                plan=plan_response,
                coding=coding_response,
                evaluation=evaluation_result,
                commit_hash=candidate_commit,
                commit_message=commit_message,
            )
            console.log(
                f"[bold green]Evolution worker[/] job={job_uuid} "
                f"produced commit={candidate_commit}",
            )
            return EvolutionWorkerResult(
                job_id=job_uuid,
                base_commit_hash=job_ctx.base_commit_hash,
                candidate_commit_hash=candidate_commit,
                plan=plan_response,
                coding=coding_response,
                evaluation=evaluation_result,
                checkout=checkout,
                commit_message=commit_message,
            )
        except Exception as exc:
            self._mark_job_failed(job_uuid, exc)
            raise

    # Internal orchestration helpers -------------------------------------

    def _start_job(self, job_id: UUID) -> JobContext:
        try:
            with session_scope() as session:
                job_stmt = (
                    select(EvolutionJob)
                    .where(EvolutionJob.id == job_id)
                    .with_for_update(nowait=True)
                )
                job = session.execute(job_stmt).scalar_one_or_none()
                if not job:
                    raise EvolutionWorkerError(f"Evolution job {job_id} does not exist.")
                if not job.base_commit_hash:
                    raise EvolutionWorkerError("Evolution job is missing base_commit_hash.")

                allowed_statuses = {JobStatus.PENDING, JobStatus.QUEUED}
                if job.status not in allowed_statuses:
                    raise EvolutionWorkerError(
                        f"Evolution job {job_id} is {job.status} and cannot run.",
                    )

                job.status = JobStatus.RUNNING
                job.started_at = _utc_now()
                job.last_error = None

                payload = dict(job.payload or {})
                base_payload = self._match_record_payload(payload.get("base"))
                inspirations_payload = self._map_inspiration_payloads(payload.get("inspirations"))
                base_snapshot = self._load_commit_snapshot(
                    session=session,
                    commit_hash=job.base_commit_hash,
                    fallback=base_payload,
                )
                inspiration_snapshots = tuple(
                    self._load_commit_snapshot(
                        session=session,
                        commit_hash=commit_hash,
                        fallback=inspirations_payload.get(commit_hash),
                    )
                    for commit_hash in tuple(job.inspiration_commit_hashes or [])
                )
                goal = self._extract_goal(
                    payload=payload,
                    job_id=job_id,
                    default=base_snapshot.summary,
                )
                constraints = self._coerce_str_sequence(
                    payload.get("constraints")
                    or payload.get("guardrails")
                    or payload.get("extra_context", {}).get("constraints"),
                )
                acceptance = self._coerce_str_sequence(
                    payload.get("acceptance_criteria")
                    or payload.get("definition_of_done")
                    or payload.get("extra_context", {}).get("acceptance_criteria"),
                )
                iteration_hint = self._extract_iteration_hint(payload)
                notes = self._coerce_str_sequence(
                    payload.get("notes") or payload.get("extra_context", {}).get("notes"),
                )
                tags = self._coerce_str_sequence(
                    payload.get("tags") or payload.get("extra_context", {}).get("tags"),
                )

                return JobContext(
                    job_id=job_id,
                    base_commit_hash=job.base_commit_hash,
                    island_id=job.island_id,
                    payload=payload,
                    base_snapshot=base_snapshot,
                    inspiration_snapshots=inspiration_snapshots,
                    goal=goal,
                    constraints=constraints,
                    acceptance_criteria=acceptance,
                    iteration_hint=iteration_hint,
                    notes=notes,
                    tags=tags,
                )
        except SQLAlchemyError as exc:
            raise EvolutionWorkerError(f"Failed to start job {job_id}: {exc}") from exc

    def _run_planning(
        self,
        job_ctx: JobContext,
        checkout: CheckoutContext,
    ) -> PlanningAgentResponse:
        request = PlanningAgentRequest(
            base=job_ctx.base_snapshot.to_planning_context(),
            inspirations=tuple(
                snapshot.to_planning_context() for snapshot in job_ctx.inspiration_snapshots
            ),
            goal=job_ctx.goal,
            constraints=job_ctx.constraints,
            acceptance_criteria=job_ctx.acceptance_criteria,
            iteration_hint=job_ctx.iteration_hint,
        )
        try:
            return self.planning_agent.plan(request, working_dir=checkout.worktree)
        except PlanningError as exc:
            raise EvolutionWorkerError(f"Planning agent failed for job {job_ctx.job_id}: {exc}") from exc

    def _run_coding(
        self,
        job_ctx: JobContext,
        plan: PlanningAgentResponse,
        checkout: CheckoutContext,
    ) -> CodingAgentResponse:
        request = CodingAgentRequest(
            goal=job_ctx.goal,
            plan=plan.plan,
            base_commit=job_ctx.base_commit_hash,
            constraints=job_ctx.constraints,
            acceptance_criteria=job_ctx.acceptance_criteria,
            iteration_hint=job_ctx.iteration_hint,
            additional_notes=job_ctx.notes,
        )
        try:
            return self.coding_agent.implement(request, working_dir=checkout.worktree)
        except CodingError as exc:
            raise EvolutionWorkerError(f"Coding agent failed for job {job_ctx.job_id}: {exc}") from exc

    def _prepare_commit_message(
        self,
        *,
        job_ctx: JobContext,
        plan: PlanningAgentResponse,
        coding: CodingAgentResponse,
    ) -> str:
        try:
            return self.summarizer.generate(
                job=job_ctx,
                plan=plan.plan,
                coding=coding.execution,
            )
        except CommitSummaryError as exc:
            log.warning("Commit summarizer failed; falling back to coding message: {}", exc)
            fallback = (
                coding.execution.commit_message
                or plan.plan.summary
                or f"Evolution job {job_ctx.job_id}"
            )
            return self.summarizer.coerce_subject(
                fallback,
                default=f"Evolution job {job_ctx.job_id}",
            )

    def _create_commit(
        self,
        *,
        checkout: CheckoutContext,
        commit_message: str,
    ) -> str:
        if not checkout.branch_name:
            raise EvolutionWorkerError(
                "Checkout context is detached; cannot publish commit without a branch.",
            )
        status = self.repository._run_git("status", "--porcelain").stdout.strip()
        if not status:
            raise EvolutionWorkerError("Coding agent produced no changes to commit.")
        self.repository._run_git("add", "--all")
        self.repository._run_git("commit", "-m", commit_message)
        commit_hash = self.repository.current_commit()
        self.repository.push_branch(checkout.branch_name)
        console.log(
            f"[green]Created worker commit[/] hash={commit_hash} "
            f"branch={checkout.branch_name or 'detached'}",
        )
        return commit_hash

    def _run_evaluation(
        self,
        *,
        job_ctx: JobContext,
        checkout: CheckoutContext,
        plan: PlanningAgentResponse,
        candidate_commit: str,
    ) -> EvaluationResult:
        payload = {
            "job": {
                "id": str(job_ctx.job_id),
                "island_id": job_ctx.island_id,
                "goal": job_ctx.goal,
                "constraints": list(job_ctx.constraints),
                "acceptance_criteria": list(job_ctx.acceptance_criteria),
                "notes": list(job_ctx.notes),
            },
            "plan": self._plan_payload(plan),
        }
        try:
            context = EvaluationContext(
                worktree=checkout.worktree,
                base_commit_hash=job_ctx.base_commit_hash,
                candidate_commit_hash=candidate_commit,
                job_id=str(job_ctx.job_id),
                goal=job_ctx.goal,
                payload=payload,
                plan_summary=plan.plan.summary,
                metadata=dict(job_ctx.payload or {}),
            )
            return self.evaluator.evaluate(context)
        except EvaluationError as exc:
            raise EvolutionWorkerError(f"Evaluator failed for job {job_ctx.job_id}: {exc}") from exc

    def _persist_success(
        self,
        *,
        job_ctx: JobContext,
        plan: PlanningAgentResponse,
        coding: CodingAgentResponse,
        evaluation: EvaluationResult,
        commit_hash: str,
        commit_message: str,
    ) -> None:
        commit_extra = {
            "job": {
                "id": str(job_ctx.job_id),
                "island_id": job_ctx.island_id,
                "goal": job_ctx.goal,
                "constraints": list(job_ctx.constraints),
                "acceptance_criteria": list(job_ctx.acceptance_criteria),
                "notes": list(job_ctx.notes),
                "tags": list(job_ctx.tags),
                "payload": job_ctx.payload,
            },
            "base_commit": job_ctx.base_snapshot.commit_hash,
            "inspirations": [snapshot.commit_hash for snapshot in job_ctx.inspiration_snapshots],
            "plan": self._plan_payload(plan),
            "coding": self._coding_payload(coding),
            "evaluation": self._evaluation_payload(evaluation),
            "worker": {
                "app_name": self.settings.app_name,
                "environment": self.settings.environment,
                "completed_at": _utc_now().isoformat(),
            },
        }
        job_payload = dict(job_ctx.payload)
        job_payload["result"] = {
            "commit_hash": commit_hash,
            "plan_summary": plan.plan.summary,
            "tests_executed": list(coding.execution.tests_executed),
            "tests_recommended": list(coding.execution.tests_recommended),
            "evaluation_summary": evaluation.summary,
            "metrics": [metric.as_dict() for metric in evaluation.metrics],
        }

        try:
            with session_scope() as session:
                job = session.get(EvolutionJob, job_ctx.job_id)
                if not job:
                    raise EvolutionWorkerError(
                        f"Evolution job {job_ctx.job_id} disappeared during persistence.",
                    )
                job.status = JobStatus.SUCCEEDED
                job.completed_at = _utc_now()
                job.plan_summary = plan.plan.summary
                job.payload = job_payload
                job.last_error = None

                metadata = CommitMetadata(
                    commit_hash=commit_hash,
                    parent_commit_hash=job_ctx.base_commit_hash,
                    island_id=job_ctx.island_id,
                    author=self.settings.worker_evolution_commit_author,
                    message=commit_message,
                    evaluation_summary=evaluation.summary,
                    tags=list(job_ctx.tags),
                    extra_context=commit_extra,
                )
                session.add(metadata)
                for metric in evaluation.metrics:
                    session.add(
                        Metric(
                            commit_hash=commit_hash,
                            name=metric.name,
                            value=metric.value,
                            unit=metric.unit,
                            higher_is_better=metric.higher_is_better,
                            details=dict(metric.details or {}),
                        )
                    )
        except SQLAlchemyError as exc:
            raise EvolutionWorkerError(f"Failed to persist results for job {job_ctx.job_id}: {exc}") from exc

    def _mark_job_failed(self, job_id: UUID, exc: Exception) -> None:
        message = str(exc)
        console.log(f"[bold red]Evolution worker[/] job={job_id} failed: {message}")
        try:
            with session_scope() as session:
                job = session.get(EvolutionJob, job_id)
                if not job:
                    return
                if job.status in {JobStatus.SUCCEEDED, JobStatus.CANCELLED}:
                    return
                job.status = JobStatus.FAILED
                job.completed_at = _utc_now()
                job.last_error = message
        except SQLAlchemyError as db_exc:
            log.error("Failed to record failure for job {}: {}", job_id, db_exc)

    # Data extraction utilities -------------------------------------------

    def _load_commit_snapshot(
        self,
        *,
        session: Session,
        commit_hash: str,
        fallback: Mapping[str, Any] | None,
    ) -> CommitSnapshot:
        commit_stmt = select(CommitMetadata).where(
            CommitMetadata.commit_hash == commit_hash,
        )
        commit = session.execute(commit_stmt).scalar_one_or_none()
        fallback_meta = self._extract_mapping(fallback, "metadata")
        extra_context: dict[str, Any] = {}
        if fallback:
            extra_context["map_elites_record"] = dict(fallback)
        if fallback_meta:
            extra_context.setdefault("map_elites_metadata", dict(fallback_meta))
        evaluation_summary = None
        summary_candidates: list[str] = []
        highlights_sources: list[Any] = []

        if commit:
            extra_context.update(dict(commit.extra_context or {}))
            evaluation_summary = commit.evaluation_summary
            if commit.message:
                summary_candidates.append(commit.message)
            highlights_sources.append(extra_context.get("highlights"))
            highlights_sources.append(extra_context.get("snippets"))
        if fallback_meta:
            summary_candidates.append(str(fallback_meta.get("summary") or ""))
            evaluation_summary = evaluation_summary or fallback_meta.get("evaluation_summary")
            highlights_sources.append(fallback_meta.get("highlights"))

        if fallback:
            summary_candidates.append(str(fallback.get("summary") or ""))
        summary_candidates.append(f"Commit {commit_hash}")

        summary = self._first_non_empty(*summary_candidates) or f"Commit {commit_hash}"
        highlights = self._extract_highlights(*highlights_sources)
        metrics: list[CommitMetric] = []
        if commit:
            metric_rows = session.scalars(
                select(Metric).where(Metric.commit_hash == commit_hash)
            ).all()
            metrics.extend(self._metric_from_row(row) for row in metric_rows)
        elif fallback_meta:
            metrics.extend(self._metrics_from_payload(fallback_meta.get("metrics")))

        return CommitSnapshot(
            commit_hash=commit_hash,
            summary=summary,
            evaluation_summary=evaluation_summary,
            highlights=highlights,
            metrics=tuple(metrics),
            extra_context=extra_context,
        )

    def _metric_from_row(self, row: Metric) -> CommitMetric:
        details = dict(row.details or {})
        summary = ""
        if "summary" in details:
            summary = str(details.get("summary"))
        elif "description" in details:
            summary = str(details.get("description"))
        return CommitMetric(
            name=row.name,
            value=row.value,
            unit=row.unit,
            higher_is_better=row.higher_is_better,
            summary=summary or None,
        )

    def _metrics_from_payload(self, payload: Any) -> list[CommitMetric]:
        if not payload:
            return []
        metrics: list[CommitMetric] = []
        candidates: Sequence[Any]
        if isinstance(payload, Mapping):
            candidates = (payload,)
        else:
            try:
                candidates = tuple(payload)
            except TypeError:
                return []
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("metric") or "").strip()
            value = item.get("value")
            if not name or value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            metrics.append(
                CommitMetric(
                    name=name,
                    value=numeric,
                    unit=(item.get("unit") or None),
                    higher_is_better=item.get("higher_is_better"),
                    summary=(item.get("summary") or None),
                )
            )
        return metrics

    def _extract_goal(
        self,
        *,
        payload: Mapping[str, Any],
        job_id: UUID,
        default: str,
    ) -> str:
        goal = self._first_non_empty(
            payload.get("goal"),
            payload.get("objective"),
            payload.get("description"),
            payload.get("extra_context", {}).get("goal") if isinstance(payload.get("extra_context"), Mapping) else None,
        )
        if goal:
            return goal
        return f"Evolution job {job_id} objective: {default}"

    def _extract_iteration_hint(self, payload: Mapping[str, Any]) -> str | None:
        hint = self._first_non_empty(
            payload.get("iteration_hint"),
            payload.get("extra_context", {}).get("iteration_hint")
            if isinstance(payload.get("extra_context"), Mapping)
            else None,
        )
        if hint:
            return hint
        sampling = payload.get("sampling")
        if isinstance(sampling, Mapping):
            selection = sampling.get("selection")
            if isinstance(selection, Mapping):
                radius = selection.get("radius_used")
                fallback = selection.get("initial_radius")
                if radius is not None:
                    return f"MAP-Elites radius {radius} (initial {fallback})"
        return None

    def _match_record_payload(self, payload: Any) -> Mapping[str, Any] | None:
        if isinstance(payload, Mapping):
            return dict(payload)
        return None

    def _map_inspiration_payloads(
        self,
        payload: Any,
    ) -> dict[str, Mapping[str, Any]]:
        mapping: dict[str, Mapping[str, Any]] = {}
        if not payload:
            return mapping
        candidates: Sequence[Any]
        if isinstance(payload, Mapping):
            candidates = (payload,)
        else:
            try:
                candidates = tuple(payload)
            except TypeError:
                return mapping
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            commit_hash = str(item.get("commit_hash") or "").strip()
            if commit_hash:
                mapping[commit_hash] = dict(item)
        return mapping

    def _plan_payload(self, response: PlanningAgentResponse) -> dict[str, Any]:
        plan_dict = response.plan.as_dict()
        plan_dict.update(
            {
                "prompt": response.prompt,
                "raw_output": response.raw_output,
                "command": list(response.command),
                "stderr": response.stderr,
                "attempts": response.attempts,
                "duration_seconds": response.duration_seconds,
            }
        )
        return plan_dict

    def _coding_payload(self, response: CodingAgentResponse) -> dict[str, Any]:
        execution = response.execution
        return {
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
            "raw_output": response.raw_output,
            "prompt": response.prompt,
            "command": list(response.command),
            "stderr": response.stderr,
            "attempts": response.attempts,
            "duration_seconds": response.duration_seconds,
        }

    def _evaluation_payload(self, result: EvaluationResult) -> dict[str, Any]:
        return {
            "summary": result.summary,
            "metrics": [metric.as_dict() for metric in result.metrics],
            "tests_executed": list(result.tests_executed),
            "logs": list(result.logs),
            "extra": dict(result.extra or {}),
        }

    def _extract_mapping(self, payload: Mapping[str, Any] | None, key: str) -> Mapping[str, Any] | None:
        if not payload:
            return None
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return dict(candidate)
        return None

    def _extract_highlights(self, *sources: Any) -> tuple[str, ...]:
        highlights: list[str] = []
        for source in sources:
            if not source:
                continue
            if isinstance(source, str):
                candidate = source.strip()
                if candidate:
                    highlights.append(candidate)
                continue
            if isinstance(source, Mapping):
                highlights.extend(
                    self._extract_highlights(
                        source.get("highlights"),
                        source.get("snippets"),
                        source.get("notes"),
                        source.get("samples"),
                    )
                )
                continue
            if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
                for item in source:
                    highlights.extend(self._extract_highlights(item))
        deduped: list[str] = []
        seen: set[str] = set()
        for entry in highlights:
            if entry not in seen:
                seen.add(entry)
                deduped.append(entry)
        return tuple(deduped)

    def _first_non_empty(self, *values: Any) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _coerce_str_sequence(self, value: Any) -> tuple[str, ...]:
        if not value:
            return tuple()
        if isinstance(value, str):
            text = value.strip()
            return (text,) if text else tuple()
        if isinstance(value, Mapping):
            return tuple()
        try:
            items: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    items.append(text)
            return tuple(items)
        except TypeError:
            text = str(value).strip()
            return (text,) if text else tuple()

    def _coerce_uuid(self, value: str | UUID) -> UUID:
        if isinstance(value, UUID):
            return value
        return UUID(str(value))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

