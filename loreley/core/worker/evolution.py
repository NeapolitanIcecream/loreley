"""Autonomous evolution worker orchestrating planning, coding, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from uuid import UUID

from loguru import logger
from openai import OpenAI
from rich.console import Console
from sqlalchemy import select

from loreley.config import Settings, get_settings
from loreley.core.worker.coding import (
    CodingAgent,
    CodingAgentRequest,
    CodingAgentResponse,
    CodingError,
)
from loreley.core.worker.evaluator import (
    Evaluator,
    EvaluationContext,
    EvaluationError,
    EvaluationResult,
)
from loreley.core.worker.planning import (
    CommitMetric,
    CommitPlanningContext,
    PlanningAgent,
    PlanningAgentRequest,
    PlanningAgentResponse,
    PlanningError,
    PlanningPlan,
)
from loreley.core.worker.commit_summary import CommitSummarizer, CommitSummaryError
from loreley.core.worker.trajectory import build_inspiration_trajectory_rollup
from loreley.core.worker.job_store import (
    EvolutionJobStore,
    EvolutionWorkerError,
    JobLockConflict,
    JobPreconditionError,
)
from loreley.core.worker.repository import CheckoutContext, WorkerRepository, RepositoryError
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, MapElitesArchiveCell, Metric

console = Console()
log = logger.bind(module="worker.evolution")

__all__ = [
    "EvolutionWorker",
    "EvolutionWorkerResult",
    "CommitSummarizer",
    "CommitSummaryError",
]


@dataclass(slots=True)
class JobContext:
    """Loaded job information used across the worker stages."""

    job_id: UUID
    base_commit_hash: str
    island_id: str | None
    experiment_id: UUID | None
    repository_id: UUID | None
    inspiration_commit_hashes: tuple[str, ...]
    goal: str
    constraints: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    iteration_hint: str | None
    notes: tuple[str, ...]
    tags: tuple[str, ...]
    is_seed_job: bool
    sampling_strategy: str | None
    sampling_initial_radius: int | None
    sampling_radius_used: int | None
    sampling_fallback_inspirations: int | None


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
                worktree=checkout.worktree,
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
        goal = (locked_job.goal or "").strip()
        if not goal:
            goal = (self.settings.worker_evolution_global_goal or "").strip()
        if not goal:
            raise EvolutionWorkerError(
                "No evolution goal configured. "
                "Set WORKER_EVOLUTION_GLOBAL_GOAL or provide a per-job goal.",
            )

        iteration_hint = (locked_job.iteration_hint or "").strip() or None
        if not iteration_hint and locked_job.sampling_radius_used is not None:
            iteration_hint = (
                f"MAP-Elites radius {locked_job.sampling_radius_used} "
                f"(initial {locked_job.sampling_initial_radius})"
            )

        is_seed_job = bool(getattr(locked_job, "is_seed_job", False))
        if not is_seed_job:
            root_hash = (self.settings.mapelites_experiment_root_commit or "").strip()
            is_seed_job = bool(
                root_hash
                and locked_job.base_commit_hash == root_hash
                and not locked_job.inspiration_commit_hashes
            )
        if is_seed_job and iteration_hint:
            iteration_hint = (
                f"{iteration_hint} | Seed job: cold-start population design, focus on diverse starting directions."
            )
        elif is_seed_job:
            iteration_hint = (
                "Seed job: cold-start population design, focus on diverse starting directions."
            )

        return JobContext(
            job_id=locked_job.job_id,
            base_commit_hash=locked_job.base_commit_hash,
            island_id=locked_job.island_id,
            experiment_id=locked_job.experiment_id,
            repository_id=locked_job.repository_id,
            inspiration_commit_hashes=tuple(locked_job.inspiration_commit_hashes or ()),
            goal=goal,
            constraints=tuple(locked_job.constraints or ()),
            acceptance_criteria=tuple(locked_job.acceptance_criteria or ()),
            iteration_hint=iteration_hint,
            notes=tuple(locked_job.notes or ()),
            tags=tuple(locked_job.tags or ()),
            is_seed_job=is_seed_job,
            sampling_strategy=locked_job.sampling_strategy,
            sampling_initial_radius=locked_job.sampling_initial_radius,
            sampling_radius_used=locked_job.sampling_radius_used,
            sampling_fallback_inspirations=locked_job.sampling_fallback_inspirations,
        )

    def _run_planning(
        self,
        job_ctx: JobContext,
        checkout: CheckoutContext,
    ) -> PlanningAgentResponse:
        base_context = self._load_commit_planning_context(
            commit_hash=job_ctx.base_commit_hash,
            experiment_id=job_ctx.experiment_id,
            island_id=job_ctx.island_id,
        )
        inspiration_contexts = [
            self._load_commit_planning_context(
                commit_hash=commit_hash,
                experiment_id=job_ctx.experiment_id,
                island_id=job_ctx.island_id,
            )
            for commit_hash in job_ctx.inspiration_commit_hashes
        ]
        if inspiration_contexts:
            shared_client: OpenAI | None = None
            if int(self.settings.worker_planning_trajectory_max_chunks or 0) > 0:
                client_kwargs: dict[str, object] = {}
                if self.settings.openai_api_key:
                    client_kwargs["api_key"] = self.settings.openai_api_key
                if self.settings.openai_base_url:
                    client_kwargs["base_url"] = self.settings.openai_base_url
                shared_client = (
                    OpenAI(**client_kwargs)  # type: ignore[call-arg]
                    if client_kwargs
                    else OpenAI()
                )
            with session_scope() as session:
                for ctx in inspiration_contexts:
                    try:
                        rollup = build_inspiration_trajectory_rollup(
                            base_commit_hash=base_context.commit_hash,
                            inspiration_commit_hash=ctx.commit_hash,
                            session=session,
                            settings=self.settings,
                            client=shared_client,
                        )
                        ctx.trajectory = rollup.lines
                        ctx.trajectory_meta = rollup.meta
                    except Exception as exc:  # pragma: no cover - best-effort enrichment
                        log.warning(
                            "Failed to build trajectory rollup for base={} insp={}: {}",
                            base_context.commit_hash[:12],
                            ctx.commit_hash[:12],
                            exc,
                        )
                        ctx.trajectory = (
                            "  - Trajectory unavailable: internal error while building rollup.",
                        )
                        ctx.trajectory_meta = {"error": str(exc)}
        inspirations = tuple(inspiration_contexts)
        cold_start = False

        if job_ctx.is_seed_job:
            # For seed jobs, hide historical metrics/highlights/evaluation details so the
            # planning agent relies purely on the global objective and constraints.
            base_context = CommitPlanningContext(
                commit_hash=base_context.commit_hash,
                subject=base_context.subject,
                change_summary=base_context.change_summary,
                trajectory=(),
                trajectory_meta=None,
                key_files=base_context.key_files,
                highlights=(),
                evaluation_summary=None,
                metrics=(),
                map_elites_cell_index=base_context.map_elites_cell_index,
                map_elites_objective=base_context.map_elites_objective,
                map_elites_measures=base_context.map_elites_measures,
            )
            inspirations = ()
            cold_start = True

        request = PlanningAgentRequest(
            base=base_context,
            inspirations=inspirations,
            goal=job_ctx.goal,
            constraints=job_ctx.constraints,
            acceptance_criteria=job_ctx.acceptance_criteria,
            iteration_hint=job_ctx.iteration_hint,
            cold_start=cold_start,
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
                "experiment_id": str(job_ctx.experiment_id) if job_ctx.experiment_id else None,
                "repository_id": str(job_ctx.repository_id) if job_ctx.repository_id else None,
                "goal": job_ctx.goal,
                "constraints": list(job_ctx.constraints),
                "acceptance_criteria": list(job_ctx.acceptance_criteria),
                "notes": list(job_ctx.notes),
                "tags": list(job_ctx.tags),
            },
            "plan": {
                "summary": plan.plan.summary,
                "focus_metrics": list(plan.plan.focus_metrics),
                "guardrails": list(plan.plan.guardrails),
            },
        }
        try:
            context = EvaluationContext(  # type: ignore[call-arg]
                worktree=checkout.worktree,
                base_commit_hash=job_ctx.base_commit_hash,
                candidate_commit_hash=candidate_commit,
                job_id=str(job_ctx.job_id),
                goal=job_ctx.goal,
                payload=payload,
                plan_summary=plan.plan.summary,
                metadata={
                    "is_seed_job": bool(job_ctx.is_seed_job),
                    "sampling": {
                        "strategy": job_ctx.sampling_strategy,
                        "initial_radius": job_ctx.sampling_initial_radius,
                        "radius_used": job_ctx.sampling_radius_used,
                        "fallback_inspirations": job_ctx.sampling_fallback_inspirations,
                    },
                },
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

    def _load_commit_planning_context(
        self,
        *,
        commit_hash: str,
        experiment_id: UUID | None,
        island_id: str | None,
    ) -> CommitPlanningContext:
        card: CommitCard | None = None
        metric_rows: Sequence[Metric] = ()
        cell: MapElitesArchiveCell | None = None
        with session_scope() as session:
            card = session.get(CommitCard, commit_hash)
            metric_rows = session.scalars(
                select(Metric).where(Metric.commit_hash == commit_hash)
            ).all()
            if experiment_id and island_id:
                cell = session.execute(
                    select(MapElitesArchiveCell).where(
                        MapElitesArchiveCell.experiment_id == experiment_id,
                        MapElitesArchiveCell.island_id == island_id,
                        MapElitesArchiveCell.commit_hash == commit_hash,
                    )
                ).scalar_one_or_none()

        subject = (getattr(card, "subject", None) or "").strip() or f"Commit {commit_hash}"
        change_summary = (getattr(card, "change_summary", None) or "").strip() or "N/A"
        key_files = tuple(getattr(card, "key_files", None) or ())
        highlights = tuple(getattr(card, "highlights", None) or ())
        if not highlights:
            highlights = ("No highlights available.",)
        evaluation_summary = getattr(card, "evaluation_summary", None)
        metrics = tuple(self._metric_from_row(row) for row in metric_rows)

        return CommitPlanningContext(
            commit_hash=commit_hash,
            subject=subject,
            change_summary=change_summary,
            key_files=key_files,
            highlights=highlights,
            evaluation_summary=evaluation_summary,
            metrics=metrics,
            map_elites_cell_index=int(cell.cell_index) if cell is not None else None,
            map_elites_objective=float(cell.objective) if cell is not None else None,
            map_elites_measures=tuple(float(v) for v in (cell.measures or ())) if cell is not None else (),
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
    def _coerce_uuid(self, value: str | UUID) -> UUID:
        if isinstance(value, UUID):
            return value
        return UUID(str(value))