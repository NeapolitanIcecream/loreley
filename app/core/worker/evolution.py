"""Autonomous evolution worker orchestrating planning, coding, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import UUID

from loguru import logger
from rich.console import Console
from sqlalchemy import select

from app.config import Settings, get_settings
from app.core.worker.coding import (
    CodingAgent,
    CodingAgentRequest,
    CodingAgentResponse,
    CodingError,
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
from app.core.worker.commit_summary import CommitSummarizer, CommitSummaryError
from app.core.worker.job_store import (
    EvolutionJobStore,
    EvolutionWorkerError,
    JobLockConflict,
    JobPreconditionError,
    build_plan_payload,
)
from app.core.worker.repository import CheckoutContext, WorkerRepository, RepositoryError
from app.db.base import session_scope
from app.db.models import CommitMetadata, Metric

console = Console()
log = logger.bind(module="worker.evolution")

__all__ = [
    "EvolutionWorker",
    "EvolutionWorkerResult",
    "CommitSummarizer",
    "CommitSummaryError",
]


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
        job_store: EvolutionJobStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or WorkerRepository(self.settings)
        self.planning_agent = planning_agent or PlanningAgent(self.settings)
        self.coding_agent = coding_agent or CodingAgent(self.settings)
        self.evaluator = evaluator or Evaluator(self.settings)
        self.summarizer = summarizer or CommitSummarizer(settings=self.settings)
        self.job_store = job_store or EvolutionJobStore(settings=self.settings)

    def run(self, job_id: str | UUID) -> EvolutionWorkerResult:
        """Execute the full evolution loop for the requested job."""
        job_uuid = self._coerce_uuid(job_id)
        try:
            job_ctx = self._start_job(job_uuid)
        except JobLockConflict:
            console.log(
                f"[yellow]Evolution worker[/] job={job_uuid} skipped because it is locked elsewhere.",
            )
            log.info("Job {} skipped due to concurrent lock", job_uuid)
            raise
        except JobPreconditionError as exc:
            console.log(
                f"[yellow]Evolution worker[/] job={job_uuid} cannot start: {exc}",
            )
            log.warning("Job {} skipped due to precondition failure: {}", job_uuid, exc)
            raise
        except Exception as exc:
            self._mark_job_failed(job_uuid, exc)
            raise
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
            self.job_store.persist_success(
                job_ctx=job_ctx,
                plan=plan_response,
                coding=coding_response,
                evaluation=evaluation_result,
                commit_hash=candidate_commit,
                commit_message=commit_message,
            )
            self._prune_job_branches()
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
        locked_job = self.job_store.start_job(job_id)
        payload = dict(locked_job.payload or {})
        extra_context = self._extract_mapping(payload, "extra_context") or {}
        base_payload = self._match_record_payload(payload.get("base"))
        inspirations_payload = self._map_inspiration_payloads(payload.get("inspirations"))
        base_snapshot = self._load_commit_snapshot(
            commit_hash=locked_job.base_commit_hash,
            fallback=base_payload,
        )
        inspiration_snapshots = tuple(
            self._load_commit_snapshot(
                commit_hash=commit_hash,
                fallback=inspirations_payload.get(commit_hash),
            )
            for commit_hash in locked_job.inspiration_commit_hashes
        )
        goal = self._extract_goal(
            payload=payload,
            extra_context=extra_context,
            job_id=job_id,
            default=base_snapshot.summary,
        )
        constraints = self._coerce_str_sequence(
            payload.get("constraints") or payload.get("guardrails") or extra_context.get("constraints"),
        )
        acceptance = self._coerce_str_sequence(
            payload.get("acceptance_criteria")
            or payload.get("definition_of_done")
            or extra_context.get("acceptance_criteria"),
        )
        iteration_hint = self._extract_iteration_hint(payload, extra_context=extra_context)
        notes = self._coerce_str_sequence(payload.get("notes") or extra_context.get("notes"))
        tags = self._coerce_str_sequence(payload.get("tags") or extra_context.get("tags"))

        return JobContext(
            job_id=locked_job.job_id,
            base_commit_hash=locked_job.base_commit_hash,
            island_id=locked_job.island_id,
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
        if not self.repository.has_changes():
            raise EvolutionWorkerError("Coding agent produced no changes to commit.")
        self.repository.stage_all()
        commit_hash = self.repository.commit(commit_message)
        self.repository.push_branch(checkout.branch_name, force_with_lease=True)
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
            "plan": build_plan_payload(plan),
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

    def _prune_job_branches(self) -> None:
        try:
            pruned = self.repository.prune_stale_job_branches()
            if pruned:
                console.log(
                    f"[yellow]Evolution worker[/] pruned {pruned} stale job branch"
                    f"{'es' if pruned != 1 else ''}.",
                )
        except RepositoryError as exc:
            log.warning("Skipping job branch pruning: {}", exc)

    def _mark_job_failed(self, job_id: UUID, exc: Exception) -> None:
        message = str(exc)
        console.log(f"[bold red]Evolution worker[/] job={job_id} failed: {message}")
        self.job_store.mark_job_failed(job_id, message)

    # Data extraction utilities -------------------------------------------

    def _load_commit_snapshot(
        self,
        *,
        commit_hash: str,
        fallback: Mapping[str, Any] | None,
    ) -> CommitSnapshot:
        commit: CommitMetadata | None
        metric_rows: Sequence[Metric] = ()
        with session_scope() as session:
            commit_stmt = select(CommitMetadata).where(
                CommitMetadata.commit_hash == commit_hash,
            )
            commit = session.execute(commit_stmt).scalar_one_or_none()
            if commit:
                metric_rows = session.scalars(
                    select(Metric).where(Metric.commit_hash == commit_hash)
                ).all()
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
        extra_context: Mapping[str, Any] | None,
        job_id: UUID,
        default: str,
    ) -> str:
        goal = self._first_non_empty(
            payload.get("goal"),
            payload.get("objective"),
            payload.get("description"),
            (extra_context or {}).get("goal"),
        )
        if goal:
            return goal
        return f"Evolution job {job_id} objective: {default}"

    def _extract_iteration_hint(
        self,
        payload: Mapping[str, Any],
        *,
        extra_context: Mapping[str, Any] | None = None,
    ) -> str | None:
        hint = self._first_non_empty(
            payload.get("iteration_hint"),
            (extra_context or {}).get("iteration_hint"),
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