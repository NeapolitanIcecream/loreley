"""Scheduler coordination for persisted campaign-level seed portfolios."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import select

from loreley.config import Settings, resolve_objective_contract
from loreley.core.campaign_program import (
    CampaignProgramSnapshot,
    CampaignProjectionInput,
    apply_campaign_program_projection,
)
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.map_elites.embedding_cache_manifest import (
    REPO_STATE_FILE_EMBEDDING_CACHE_KIND,
)
from loreley.core.model_routes import resolve_effective_routes
from loreley.core.seed_portfolio import (
    SEED_PORTFOLIO_SCHEMA_VERSION,
    SeedPortfolioArtifact,
    SeedPortfolioError,
    SeedPortfolioPlanner,
    SeedPortfolioPlanningRequest,
    SeedRootEvidence,
    canonical_json_hash,
    materialize_seed_portfolio,
)
from loreley.core.usage import persist_usage_events
from loreley.core.worker.repository import WorkerRepository
from loreley.db.base import session_scope
from loreley.db.models import (
    CampaignBaseline,
    CommitCard,
    EmbeddingCacheManifest,
    EvaluationArtifactRecord,
    Metric,
)
from loreley.db.models import SeedDirection as SeedDirectionRow
from loreley.db.models import SeedPortfolio as SeedPortfolioRow

log = logger.bind(module="scheduler.seed_portfolios")

SEED_PORTFOLIO_STATUS_PLANNING = "planning"
SEED_PORTFOLIO_STATUS_READY = "ready"
SEED_PORTFOLIO_STATUS_FAILED = "failed"


class SeedPortfolioStateError(SeedPortfolioError):
    """Raised when a persisted portfolio cannot safely advance automatically."""


@dataclass(frozen=True, slots=True)
class SeedPortfolioEvidence:
    root_evidence: SeedRootEvidence
    fingerprints: dict[str, str]


class SeedPortfolioCoordinator:
    """Create one portfolio after baseline bootstrap and reuse it on restarts."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo_root: Path,
        planner: SeedPortfolioPlanner | None = None,
    ) -> None:
        self.settings = settings
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.planner = planner

    def ensure_ready(
        self,
        *,
        root_commit_hash: str,
        baseline_key_hash: str,
        direction_count: int,
        campaign_program: CampaignProgramSnapshot | None,
    ) -> SeedPortfolioArtifact:
        request = self._build_request(
            root_commit_hash=root_commit_hash,
            baseline_key_hash=baseline_key_hash,
            direction_count=direction_count,
            campaign_program=campaign_program,
        )
        existing = self._load_by_request_fingerprint(request.request_fingerprint)
        if existing is not None:
            return self._artifact_from_existing(existing)

        portfolio_id = self._stage_request(
            request=request,
            baseline_key_hash=baseline_key_hash,
        )
        try:
            planner = self.planner or SeedPortfolioPlanner(settings=self.settings)
            with self._planning_worktree(request.root_commit_hash) as worktree:
                response = planner.plan(request, working_dir=worktree)
            artifact = materialize_seed_portfolio(request, response.draft)
            self._persist_ready_portfolio(
                portfolio_id=portfolio_id,
                artifact=artifact,
                response=response,
            )
            self._persist_usage_events(response.usage_events)
            return artifact
        except Exception as exc:
            self._persist_failure(portfolio_id=portfolio_id, exc=exc)
            self._persist_usage_events(
                tuple(getattr(exc, "usage_events", ()) or ()),
            )
            raise SeedPortfolioStateError(
                "seed portfolio planning failed; the persisted failed request will "
                "block automatic retries to avoid an untracked duplicate model call "
                f"(request_fingerprint={request.request_fingerprint})"
            ) from exc

    @contextmanager
    def _planning_worktree(self, root_commit_hash: str):
        """Yield an isolated detached checkout for read-only portfolio inspection."""

        repository = WorkerRepository(self.settings)
        with repository.checkout_lease_for_job(
            job_id=None,
            base_commit=root_commit_hash,
            create_branch=False,
        ) as checkout:
            yield checkout.worktree

    def _build_request(
        self,
        *,
        root_commit_hash: str,
        baseline_key_hash: str,
        direction_count: int,
        campaign_program: CampaignProgramSnapshot | None,
    ) -> SeedPortfolioPlanningRequest:
        route = dict(resolve_effective_routes(self.settings)["seed_portfolio"])
        model = str(route.get("model") or "").strip()
        if model.rsplit("/", 1)[-1] != "gpt-5.6-sol":
            raise SeedPortfolioStateError(
                "the initial seed portfolio contract is pinned to gpt-5.6-sol "
                f"(configured={model!r})"
            )
        contract = resolve_objective_contract(self.settings)
        evidence = self._load_root_evidence(
            root_commit_hash=root_commit_hash,
            baseline_key_hash=baseline_key_hash,
        )
        default_goal = str(self.settings.worker_evolution_global_goal or "").strip()
        projection = apply_campaign_program_projection(
            CampaignProjectionInput(
                snapshot=campaign_program,
                goal=default_goal,
                constraints=(),
                acceptance_criteria=(),
                notes=(),
                default_goal=default_goal,
            )
        )
        goal = str(projection.goal or "").strip()
        if not goal:
            raise SeedPortfolioStateError(
                "cannot plan a seed portfolio without an evolution goal"
            )
        fingerprints = {
            **evidence.fingerprints,
            "baseline_key": baseline_key_hash,
        }
        return SeedPortfolioPlanningRequest(
            configured_direction_count=int(
                self.settings.mapelites_seed_portfolio_direction_count
            ),
            direction_count=max(1, int(direction_count)),
            root_commit_hash=str(root_commit_hash).strip(),
            campaign_program_hash=(
                campaign_program.raw_sha256 if campaign_program else None
            ),
            campaign_title=campaign_program.title if campaign_program else None,
            goal=goal,
            constraints=tuple(projection.constraints),
            acceptance_criteria=tuple(projection.acceptance_criteria),
            notes=tuple(projection.notes),
            objective_contract=tuple(contract.as_payload()),
            objective_contract_fingerprint=contract.fingerprint,
            root_evidence=evidence.root_evidence,
            input_evidence_fingerprints=fingerprints,
            model_route=route,
            reasoning_effort=str(self.settings.worker_seed_portfolio_reasoning_effort),
            max_pairwise_overlap=float(
                self.settings.mapelites_seed_portfolio_max_pairwise_overlap
            ),
        )

    def _load_root_evidence(
        self,
        *,
        root_commit_hash: str,
        baseline_key_hash: str,
    ) -> SeedPortfolioEvidence:
        contract = resolve_objective_contract(self.settings)
        with session_scope() as session:
            baseline = session.execute(
                select(CampaignBaseline).where(
                    CampaignBaseline.baseline_key_hash == baseline_key_hash
                )
            ).scalar_one_or_none()
            if baseline is None or str(baseline.status) != "valid":
                raise SeedPortfolioStateError(
                    "seed portfolio requires a valid persisted campaign baseline"
                )
            card = session.execute(
                select(CommitCard).where(CommitCard.commit_hash == root_commit_hash)
            ).scalar_one_or_none()
            if card is None:
                raise SeedPortfolioStateError(
                    "seed portfolio requires a persisted root CommitCard"
                )
            metric_rows = list(
                session.scalars(
                    select(Metric)
                    .where(Metric.commit_card_id == card.id)
                    .order_by(Metric.name.asc())
                ).all()
            )
            contract.resolve(metric_rows)
            artifacts = list(
                session.scalars(
                    select(EvaluationArtifactRecord)
                    .where(
                        EvaluationArtifactRecord.commit_hash == root_commit_hash,
                        EvaluationArtifactRecord.visibility == "agent_visible",
                    )
                    .order_by(
                        EvaluationArtifactRecord.created_at.asc(),
                        EvaluationArtifactRecord.id.asc(),
                    )
                ).all()
            )
            manifest = session.execute(
                select(EmbeddingCacheManifest).where(
                    EmbeddingCacheManifest.cache_kind
                    == REPO_STATE_FILE_EMBEDDING_CACHE_KIND
                )
            ).scalar_one_or_none()

        if manifest is None or not str(manifest.fingerprint or "").strip():
            raise SeedPortfolioStateError(
                "seed portfolio requires the persisted repo-state embedding manifest"
            )

        metrics = self._ordered_metric_payload(metric_rows, contract.names)
        diagnostics = self._bounded_diagnostics(artifacts)
        artifact_manifest = [
            {
                "key": str(artifact.key),
                "kind": str(artifact.kind),
                "sha256": str(artifact.sha256 or ""),
                "summary": normalize_single_line(str(artifact.summary or "")),
                "diagnostics": list(artifact.diagnostics or ()),
            }
            for artifact in artifacts
        ]
        summary = normalize_single_line(
            str(card.evaluation_summary or baseline.evaluation_summary or "")
        )
        fingerprints = {
            "root_metrics": canonical_json_hash(metrics),
            "root_evaluation_summary": canonical_json_hash(summary),
            "root_evaluation_artifacts": canonical_json_hash(artifact_manifest),
            "repo_state_embedding_manifest": str(manifest.fingerprint),
        }
        return SeedPortfolioEvidence(
            root_evidence=SeedRootEvidence(
                evaluation_summary=summary or None,
                metrics=tuple(metrics),
                diagnostics=diagnostics,
            ),
            fingerprints=fingerprints,
        )

    @staticmethod
    def _ordered_metric_payload(
        metrics: Sequence[Metric],
        objective_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        by_name = {str(metric.name): metric for metric in metrics}
        ordered_names = [
            *[name for name in objective_names if name in by_name],
            *sorted(name for name in by_name if name not in objective_names),
        ]
        return [
            {
                "name": name,
                "value": float(by_name[name].value),
                "unit": by_name[name].unit,
                "higher_is_better": bool(by_name[name].higher_is_better),
                "summary": normalize_single_line(
                    str(
                        dict(by_name[name].details or {}).get("summary")
                        or dict(by_name[name].details or {}).get("description")
                        or ""
                    )
                )
                or None,
            }
            for name in ordered_names
        ]

    @staticmethod
    def _bounded_diagnostics(
        artifacts: Sequence[EvaluationArtifactRecord],
    ) -> tuple[str, ...]:
        values: list[str] = []
        for artifact in artifacts[:8]:
            summary = normalize_single_line(str(artifact.summary or ""))
            if summary:
                values.append(clamp_text(summary, 500))
            for diagnostic in tuple(artifact.diagnostics or ())[:4]:
                if not isinstance(diagnostic, dict):
                    continue
                message = normalize_single_line(str(diagnostic.get("message") or ""))
                if message:
                    values.append(clamp_text(message, 500))
                if len(values) >= 32:
                    break
            if len(values) >= 32:
                break
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _load_by_request_fingerprint(
        request_fingerprint: str,
    ) -> SeedPortfolioRow | None:
        with session_scope() as session:
            return session.execute(
                select(SeedPortfolioRow).where(
                    SeedPortfolioRow.request_fingerprint == request_fingerprint
                )
            ).scalar_one_or_none()

    @staticmethod
    def _artifact_from_existing(row: SeedPortfolioRow) -> SeedPortfolioArtifact:
        status = str(row.status or "").strip().lower()
        if status == SEED_PORTFOLIO_STATUS_READY:
            try:
                return SeedPortfolioArtifact.model_validate(dict(row.payload or {}))
            except ValueError as exc:
                raise SeedPortfolioStateError(
                    "persisted ready seed portfolio payload is invalid"
                ) from exc
        detail = normalize_single_line(str(row.error_summary or "")) or "n/a"
        raise SeedPortfolioStateError(
            "seed portfolio request already exists in a non-ready terminal/safe-stop "
            f"state (status={status!r} reason={detail})"
        )

    @staticmethod
    def _stage_request(
        *,
        request: SeedPortfolioPlanningRequest,
        baseline_key_hash: str,
    ) -> Any:
        route = request.model_route
        with session_scope() as session:
            row = SeedPortfolioRow(
                request_fingerprint=request.request_fingerprint,
                portfolio_hash=None,
                schema_version=SEED_PORTFOLIO_SCHEMA_VERSION,
                status=SEED_PORTFOLIO_STATUS_PLANNING,
                root_commit_hash=request.root_commit_hash,
                campaign_program_hash=request.campaign_program_hash,
                baseline_key_hash=baseline_key_hash,
                objective_contract_fingerprint=(request.objective_contract_fingerprint),
                input_evidence_fingerprints=dict(request.input_evidence_fingerprints),
                model_backend=str(route.get("backend") or "unknown"),
                model_provider=str(route.get("provider") or "unknown"),
                model_name=str(route.get("model") or "unknown"),
                reasoning_effort=request.reasoning_effort,
                direction_count=request.direction_count,
                payload={},
                planning_started_at=datetime.now(UTC),
            )
            session.add(row)
            session.flush()
            return row.id

    @staticmethod
    def _persist_ready_portfolio(
        *,
        portfolio_id: Any,
        artifact: SeedPortfolioArtifact,
        response: Any,
    ) -> None:
        with session_scope() as session:
            row = session.get(SeedPortfolioRow, portfolio_id)
            if row is None or row.status != SEED_PORTFOLIO_STATUS_PLANNING:
                raise SeedPortfolioStateError(
                    "staged seed portfolio row was lost or changed before persistence"
                )
            row.portfolio_hash = artifact.portfolio_hash
            row.status = SEED_PORTFOLIO_STATUS_READY
            row.payload = artifact.model_dump(mode="json")
            row.planner_prompt_sha256 = response.prompt_sha256
            row.planner_output_sha256 = response.output_sha256
            row.planner_attempts = int(response.attempts)
            row.planner_duration_seconds = float(response.duration_seconds)
            row.error_summary = None
            row.completed_at = datetime.now(UTC)
            for ordinal, direction in enumerate(artifact.directions):
                session.add(
                    SeedDirectionRow(
                        portfolio_id=row.id,
                        direction_id=direction.direction_id,
                        ordinal=ordinal,
                        content_hash=direction.content_hash,
                        title=direction.title,
                        causal_mechanism=direction.causal_mechanism,
                        admission_intent=direction.admission_intent,
                        payload=direction.model_dump(mode="json"),
                    )
                )

    @staticmethod
    def _persist_failure(*, portfolio_id: Any, exc: Exception) -> None:
        summary = clamp_text(
            normalize_single_line(str(exc)) or exc.__class__.__name__,
            4096,
        )
        with session_scope() as session:
            row = session.get(SeedPortfolioRow, portfolio_id)
            if row is None:
                return
            row.status = SEED_PORTFOLIO_STATUS_FAILED
            row.error_summary = summary
            row.completed_at = datetime.now(UTC)

    def _persist_usage_events(self, events: Sequence[Any]) -> None:
        materialized = []
        for event in events:
            with_context = getattr(event, "with_context", None)
            if callable(with_context):
                materialized.append(with_context(phase="seed_portfolio"))
        if materialized:
            persist_usage_events(materialized, settings=self.settings)


__all__ = [
    "SEED_PORTFOLIO_STATUS_FAILED",
    "SEED_PORTFOLIO_STATUS_PLANNING",
    "SEED_PORTFOLIO_STATUS_READY",
    "SeedPortfolioCoordinator",
    "SeedPortfolioEvidence",
    "SeedPortfolioStateError",
]
