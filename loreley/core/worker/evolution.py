"""Autonomous evolution worker orchestrating planning, coding, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Sequence
from uuid import UUID

from loguru import logger
from openai import OpenAI
from rich.console import Console
from sqlalchemy import select
from sqlalchemy.exc import MultipleResultsFound

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
    IterationContext,
    PlanningAgent,
    PlanningAgentRequest,
    PlanningAgentResponse,
    PlanningError,
)
from loreley.core.worker.commit_summary import (
    CommitSummarizer,
    CommitSummaryError,
    CommitSummaryUnavailableError,
)
from loreley.core.worker.trajectory import build_inspiration_trajectory_rollup
from loreley.core.worker.job_store import (
    EvolutionJobStore,
    EvolutionWorkerError,
    JobLeaseLost,
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
    run_token: UUID
    base_commit_hash: str
    island_id: str | None
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


@dataclass(slots=True)
class WorkerPromptContext:
    """Shared task packet inputs reused across planning and coding."""

    base: CommitPlanningContext
    inspirations: tuple[CommitPlanningContext, ...]
    iteration_context: IterationContext


class _JobLeaseHeartbeat:
    """Renew a job lease in the background while long-running stages execute."""

    def __init__(
        self,
        *,
        job_store: EvolutionJobStore,
        job_id: UUID,
        run_token: UUID,
        settings: Settings,
    ) -> None:
        self._job_store = job_store
        self._job_id = job_id
        self._run_token = run_token
        interval = max(1, int(settings.worker_job_heartbeat_interval_seconds))
        ttl = max(1, int(settings.worker_job_lease_ttl_seconds))
        self._interval_seconds = min(interval, max(1, ttl // 2))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"job-lease-{str(job_id)[:8]}",
            daemon=True,
        )
        self._lease_lost: JobLeaseLost | None = None

    def __enter__(self) -> _JobLeaseHeartbeat:
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, float(self._interval_seconds) * 2.0))

    def raise_if_lease_lost(self) -> None:
        if self._lease_lost is not None:
            raise self._lease_lost

    def _run(self) -> None:
        while not self._stop.wait(float(self._interval_seconds)):
            try:
                lease_expires_at = self._job_store.renew_job_lease(
                    self._job_id,
                    self._run_token,
                )
                log.debug(
                    "Renewed lease for job {} run_token={} expires_at={}",
                    self._job_id,
                    self._run_token,
                    lease_expires_at,
                )
            except JobLeaseLost as exc:
                self._lease_lost = exc
                log.warning(
                    "Lease lost for job {} run_token={}: {}",
                    self._job_id,
                    self._run_token,
                    exc,
                )
                self._stop.set()
                return
            except Exception as exc:  # pragma: no cover - defensive / transient DB failures
                log.warning(
                    "Lease heartbeat failed for job {} run_token={}: {}",
                    self._job_id,
                    self._run_token,
                    exc,
                )


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
        except JobLeaseLost as exc:
            console.log(
                f"[yellow]Evolution worker[/] job={job_uuid} lost lease before start: {exc}",
            )
            log.warning("Job {} lost lease before start: {}", job_uuid, exc)
            raise
        except Exception as exc:
            self._mark_job_failed(job_uuid, None, exc)
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
            with _JobLeaseHeartbeat(
                job_store=self.job_store,
                job_id=job_uuid,
                run_token=job_ctx.run_token,
                settings=self.settings,
            ) as heartbeat:
                with self.repository.checkout_lease_for_job(
                    job_id=job_uuid,
                    base_commit=job_ctx.base_commit_hash,
                    attempt_token=job_ctx.run_token,
                ) as checkout:
                    heartbeat.raise_if_lease_lost()
                    prompt_context = self._build_prompt_context(job_ctx)
                    heartbeat.raise_if_lease_lost()
                    plan_response = self._run_planning(job_ctx, checkout, prompt_context)
                    heartbeat.raise_if_lease_lost()
                    coding_response = self._run_coding(
                        job_ctx,
                        plan_response,
                        checkout,
                        prompt_context,
                    )
                    heartbeat.raise_if_lease_lost()
                    commit_message = self._prepare_commit_message(
                        job_ctx=job_ctx,
                        plan=plan_response,
                        coding=coding_response,
                    )
                    heartbeat.raise_if_lease_lost()
                    candidate_commit = self._create_commit(
                        checkout=checkout,
                        commit_message=commit_message,
                    )
                    heartbeat.raise_if_lease_lost()
                    evaluation_result = self._run_evaluation(
                        job_ctx=job_ctx,
                        checkout=checkout,
                        plan=plan_response,
                        candidate_commit=candidate_commit,
                    )
                    heartbeat.raise_if_lease_lost()
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
        except JobLeaseLost as exc:
            console.log(f"[yellow]Evolution worker[/] job={job_uuid} lost lease: {exc}")
            log.warning("Job {} lost lease during execution: {}", job_uuid, exc)
            raise
        except Exception as exc:
            self._mark_job_failed(job_uuid, job_ctx.run_token, exc)
            raise

    # Internal orchestration helpers -------------------------------------

    def _start_job(self, job_id: UUID) -> JobContext:
        locked_job = self.job_store.start_job(
            job_id,
        )

        base_commit_hash = (locked_job.base_commit_hash or "").strip()
        if not base_commit_hash:
            raise EvolutionWorkerError(
                f"Evolution job {locked_job.job_id} has no base commit hash configured."
            )
        inspiration_commit_hashes = tuple(
            commit_hash.strip()
            for commit_hash in (locked_job.inspiration_commit_hashes or ())
            if (commit_hash or "").strip()
        )

        goal = (locked_job.goal or "").strip()
        if not goal:
            goal = (self.settings.worker_evolution_global_goal or "").strip()
        if not goal:
            raise EvolutionWorkerError(
                "No evolution goal configured. "
                "Set WORKER_EVOLUTION_GLOBAL_GOAL or provide a per-job goal.",
            )

        iteration_hint = (locked_job.iteration_hint or "").strip() or None
        is_seed_job = bool(getattr(locked_job, "is_seed_job", False))
        if not is_seed_job:
            root_hash = (self.settings.mapelites_experiment_root_commit or "").strip()
            is_seed_job = bool(
                root_hash
                and base_commit_hash == root_hash
                and not inspiration_commit_hashes
            )

        return JobContext(
            job_id=locked_job.job_id,
            run_token=locked_job.run_token,
            base_commit_hash=base_commit_hash,
            island_id=locked_job.island_id,
            inspiration_commit_hashes=inspiration_commit_hashes,
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

    def _build_prompt_context(
        self,
        job_ctx: JobContext,
    ) -> WorkerPromptContext:
        planning_contexts = self._load_commit_planning_contexts(
            commit_hashes=(job_ctx.base_commit_hash, *job_ctx.inspiration_commit_hashes),
            island_id=job_ctx.island_id,
        )
        if not planning_contexts:
            raise EvolutionWorkerError("Planning context loading returned no commit contexts.")
        base_context = planning_contexts[0]
        inspiration_contexts = list(planning_contexts[1:])
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

        if job_ctx.is_seed_job:
            # For seed jobs, hide historical metrics/evaluation details so the
            # worker starts from the global objective and sampling context alone.
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

        return WorkerPromptContext(
            base=base_context,
            inspirations=inspirations,
            iteration_context=self._build_iteration_context(job_ctx),
        )

    def _run_planning(
        self,
        job_ctx: JobContext,
        checkout: CheckoutContext,
        prompt_context: WorkerPromptContext,
    ) -> PlanningAgentResponse:

        request = PlanningAgentRequest(
            base=prompt_context.base,
            inspirations=prompt_context.inspirations,
            goal=job_ctx.goal,
            iteration_context=prompt_context.iteration_context,
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
        prompt_context: WorkerPromptContext,
    ) -> CodingAgentResponse:
        request = CodingAgentRequest(
            goal=job_ctx.goal,
            plan=plan.plan,
            base_commit=job_ctx.base_commit_hash,
            base=prompt_context.base,
            inspirations=prompt_context.inspirations,
            iteration_context=prompt_context.iteration_context,
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
                coding=coding.report,
            )
        except (CommitSummaryError, CommitSummaryUnavailableError) as exc:
            log.warning("Commit summarizer failed; falling back to non-LLM subject: {}", exc)
            fallback = (
                coding.report.summary
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
        if not self.repository.has_changes(worktree=checkout.worktree):
            raise EvolutionWorkerError("Coding agent produced no changes to commit.")
        self.repository.stage_all(worktree=checkout.worktree)
        commit_hash = self.repository.commit(commit_message, worktree=checkout.worktree)
        self.repository.push_branch(
            checkout.branch_name,
            worktree=checkout.worktree,
            force_with_lease=True,
        )
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

    def _mark_job_failed(self, job_id: UUID, run_token: UUID | None, exc: Exception) -> None:
        message = str(exc)
        console.log(f"[bold red]Evolution worker[/] job={job_id} failed: {message}")
        recorded = self.job_store.mark_job_failed(job_id, message, run_token=run_token)
        if not recorded and run_token is not None:
            log.warning(
                "Skipped failure persistence for job {} because run_token={} is no longer active.",
                job_id,
                run_token,
            )

    # Data extraction utilities -------------------------------------------

    def _load_commit_planning_context(
        self,
        *,
        commit_hash: str,
        island_id: str | None,
    ) -> CommitPlanningContext:
        contexts = self._load_commit_planning_contexts(
            commit_hashes=(commit_hash,),
            island_id=island_id,
        )
        if contexts:
            return contexts[0]
        return self._build_commit_planning_context(
            commit_hash=commit_hash,
            card=None,
            metric_rows=(),
            cell=None,
        )

    def _load_commit_planning_contexts(
        self,
        *,
        commit_hashes: Sequence[str],
        island_id: str | None,
    ) -> tuple[CommitPlanningContext, ...]:
        ordered_hashes = tuple(commit_hash for commit_hash in commit_hashes if commit_hash)
        if not ordered_hashes:
            return ()
        unique_hashes = tuple(dict.fromkeys(ordered_hashes))

        cards_by_hash: dict[str, CommitCard] = {}
        metrics_by_card_id: dict[UUID, list[Metric]] = {}
        cells_by_hash: dict[str, MapElitesArchiveCell] = {}
        with session_scope() as session:
            cards = session.scalars(
                select(CommitCard).where(CommitCard.commit_hash.in_(unique_hashes))
            ).all()
            cards_by_hash = {card.commit_hash: card for card in cards}
            card_ids = tuple(card.id for card in cards)
            if card_ids:
                metric_rows = session.scalars(
                    select(Metric).where(Metric.commit_card_id.in_(card_ids))
                ).all()
                for row in metric_rows:
                    metrics_by_card_id.setdefault(row.commit_card_id, []).append(row)
            if island_id:
                cells = session.scalars(
                    select(MapElitesArchiveCell).where(
                        MapElitesArchiveCell.island_id == island_id,
                        MapElitesArchiveCell.commit_hash.in_(unique_hashes),
                    )
                ).all()
                for cell in cells:
                    existing = cells_by_hash.get(cell.commit_hash)
                    if existing is not None:
                        raise MultipleResultsFound(
                            "Multiple map-elites archive cells found for one commit hash "
                            f"(island={island_id}, commit={cell.commit_hash})."
                        )
                    cells_by_hash[cell.commit_hash] = cell

        contexts: list[CommitPlanningContext] = []
        for commit_hash in ordered_hashes:
            card = cards_by_hash.get(commit_hash)
            metric_rows_for_card: Sequence[Metric] = ()
            if card is not None:
                metric_rows_for_card = tuple(metrics_by_card_id.get(card.id, ()))
            contexts.append(
                self._build_commit_planning_context(
                    commit_hash=commit_hash,
                    card=card,
                    metric_rows=metric_rows_for_card,
                    cell=cells_by_hash.get(commit_hash),
                )
            )
        return tuple(contexts)

    def _build_commit_planning_context(
        self,
        *,
        commit_hash: str,
        card: CommitCard | None,
        metric_rows: Sequence[Metric],
        cell: MapElitesArchiveCell | None,
    ) -> CommitPlanningContext:
        subject = (getattr(card, "subject", None) or "").strip() or f"Commit {commit_hash}"
        change_summary = (getattr(card, "change_summary", None) or "").strip() or "N/A"
        key_files = tuple(getattr(card, "key_files", None) or ())
        highlights = tuple(getattr(card, "highlights", None) or ())
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

    def _build_iteration_context(self, job_ctx: JobContext) -> IterationContext:
        facts: list[str] = []
        if job_ctx.sampling_radius_used is not None:
            facts.append(f"radius_used: {job_ctx.sampling_radius_used}")
        if job_ctx.sampling_initial_radius is not None:
            facts.append(f"initial_radius: {job_ctx.sampling_initial_radius}")
        if job_ctx.sampling_fallback_inspirations is not None:
            facts.append(
                f"fallback_inspirations: {job_ctx.sampling_fallback_inspirations}"
            )
        if job_ctx.iteration_hint:
            facts.append(job_ctx.iteration_hint)
        if job_ctx.is_seed_job:
            facts.append("MAP-Elites archive is empty.")
            facts.append("Prioritize diverse starting directions.")
        return IterationContext(
            seed_job=bool(job_ctx.is_seed_job),
            sampling_strategy=job_ctx.sampling_strategy,
            facts=tuple(facts),
        )
    def _coerce_uuid(self, value: str | UUID) -> UUID:
        if isinstance(value, UUID):
            return value
        return UUID(str(value))
