from __future__ import annotations

"""Campaign baseline bootstrap and comparability helpers."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from loguru import logger
from rich.console import Console
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from loreley.config import Settings, resolve_default_island_id
from loreley.core.campaign_program import (
    CampaignProgramSnapshot,
    campaign_program_evaluator_payload,
    load_campaign_program_from_repo,
)
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.worker.evaluator import (
    EvaluationContext,
    EvaluationMetric,
    EvaluationOutcome,
    EvaluationResult,
    Evaluator,
)
from loreley.core.worker.evaluator_identity import evaluator_identity_version
from loreley.core.worker.repository import RepositoryError, WorkerRepository
from loreley.db.base import session_scope
from loreley.db.models import CampaignBaseline, CommitCard, EvolutionJob, Metric

log = logger.bind(module="scheduler.baselines")

BASELINE_STATUS_VALID = "valid"
BASELINE_STATUS_FAILED = "failed"
BASELINE_STATUS_DEGRADED = "degraded"

_BASELINE_FAILURE_TEXT_MAX = 4096
_CAMPAIGN_PROGRAM_HASH_UNSET = object()


@dataclass(frozen=True, slots=True)
class BaselineMetricSpec:
    name: str
    higher_is_better: bool


@dataclass(frozen=True, slots=True)
class BaselineKey:
    root_commit_hash: str
    campaign_program_hash: str | None
    evaluator_name: str | None
    evaluator_version: str | None
    primary_metric_name: str
    primary_metric_higher_is_better: bool
    runtime_profile: str | None
    effective_settings_fingerprint: str | None

    def as_payload(self) -> dict[str, Any]:
        return {
            "root_commit_hash": self.root_commit_hash,
            "campaign_program_hash": self.campaign_program_hash,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_higher_is_better": self.primary_metric_higher_is_better,
            "runtime_profile": self.runtime_profile,
            "effective_settings_fingerprint": self.effective_settings_fingerprint,
        }

    @property
    def hash(self) -> str:
        return baseline_key_hash(self.as_payload())


@dataclass(frozen=True, slots=True)
class BaselineValidationResult:
    ok: bool
    metric: EvaluationMetric | None = None
    failure_kind: str | None = None
    failure_summary: str | None = None


@dataclass(frozen=True, slots=True)
class BaselineBootstrapResult:
    can_dispatch_or_schedule: bool
    status: str
    policy: str
    baseline_key_hash: str
    baseline_id: str | None = None
    failure_kind: str | None = None
    failure_summary: str | None = None


@dataclass(frozen=True, slots=True)
class BaselineAttempt:
    validation: BaselineValidationResult
    valid: bool
    status: str
    failure_kind: str | None
    failure_summary: str | None
    metric: EvaluationMetric | None


@dataclass(frozen=True, slots=True)
class CampaignProgramHashResolution:
    known: bool
    campaign_program_hash: str | None = None
    source_path: str | None = None
    failure_summary: str | None = None


@dataclass(frozen=True, slots=True)
class _PersistedCampaignProgramHash:
    campaign_program_hash: str | None
    source: str
    observed_at: datetime | None = None


def baseline_effective_settings_fingerprint(settings: Settings) -> str:
    """Return the narrow non-secret settings fingerprint used for baseline identity."""

    payload = {
        "profile": str(getattr(settings, "profile", "") or ""),
        "worker_evaluator_plugin": str(getattr(settings, "worker_evaluator_plugin", "") or ""),
        "worker_evaluator_version": baseline_evaluator_version(settings) or "",
        "worker_evaluator_python_paths": [
            str(item) for item in tuple(getattr(settings, "worker_evaluator_python_paths", ()) or ())
        ],
        "worker_evaluator_timeout_seconds": int(
            getattr(settings, "worker_evaluator_timeout_seconds", 0) or 0,
        ),
        "worker_evaluator_max_metrics": int(
            getattr(settings, "worker_evaluator_max_metrics", 0) or 0,
        ),
        "mapelites_fitness_metric": str(getattr(settings, "mapelites_fitness_metric", "") or ""),
        "mapelites_fitness_higher_is_better": bool(
            getattr(settings, "mapelites_fitness_higher_is_better", True),
        ),
        "mapelites_fitness_floor": float(getattr(settings, "mapelites_fitness_floor", 0.0) or 0.0),
    }
    return baseline_key_hash(payload)


def baseline_key_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def baseline_evaluator_version(settings: Settings) -> str | None:
    """Return the pre-run evaluator version/fingerprint used for baseline identity."""

    return evaluator_identity_version(
        plugin_ref=getattr(settings, "worker_evaluator_plugin", None),
        explicit_version=getattr(settings, "worker_evaluator_version", None),
        python_paths=tuple(str(item) for item in getattr(settings, "worker_evaluator_python_paths", ()) or ()),
    )


def build_baseline_key(
    *,
    settings: Settings,
    root_commit_hash: str,
    campaign_program: CampaignProgramSnapshot | None,
) -> BaselineKey:
    spec = baseline_metric_spec(settings)
    evaluator_name = normalize_single_line(str(getattr(settings, "worker_evaluator_plugin", "") or "")) or None
    runtime_profile = normalize_single_line(str(getattr(settings, "profile", "") or "")) or None
    return BaselineKey(
        root_commit_hash=normalize_single_line(root_commit_hash),
        campaign_program_hash=campaign_program.raw_sha256 if campaign_program else None,
        evaluator_name=evaluator_name,
        evaluator_version=baseline_evaluator_version(settings),
        primary_metric_name=spec.name,
        primary_metric_higher_is_better=spec.higher_is_better,
        runtime_profile=runtime_profile,
        effective_settings_fingerprint=baseline_effective_settings_fingerprint(settings),
    )


def baseline_metric_spec(settings: Settings) -> BaselineMetricSpec:
    return BaselineMetricSpec(
        name=normalize_single_line(str(getattr(settings, "mapelites_fitness_metric", "") or "")),
        higher_is_better=bool(getattr(settings, "mapelites_fitness_higher_is_better", True)),
    )


def validate_baseline_primary_metric(
    *,
    result: EvaluationResult | None,
    spec: BaselineMetricSpec,
) -> BaselineValidationResult:
    if result is None:
        return BaselineValidationResult(
            ok=False,
            failure_kind="evaluation_missing_result",
            failure_summary="Baseline evaluator did not return a success result.",
        )
    if not spec.name:
        return BaselineValidationResult(
            ok=False,
            failure_kind="primary_metric_not_configured",
            failure_summary="MAPELITES_FITNESS_METRIC must be configured for baseline bootstrap.",
        )
    metric = next((item for item in result.metrics if item.name == spec.name), None)
    if metric is None:
        return BaselineValidationResult(
            ok=False,
            failure_kind="primary_metric_missing",
            failure_summary=f"Baseline result did not include primary metric {spec.name!r}.",
        )
    try:
        value = float(metric.value)
    except (TypeError, ValueError):
        return BaselineValidationResult(
            ok=False,
            metric=metric,
            failure_kind="primary_metric_non_finite",
            failure_summary=f"Baseline primary metric {spec.name!r} is not numeric.",
        )
    if not math.isfinite(value):
        return BaselineValidationResult(
            ok=False,
            metric=metric,
            failure_kind="primary_metric_non_finite",
            failure_summary=f"Baseline primary metric {spec.name!r} is not finite.",
        )
    if bool(metric.higher_is_better) != bool(spec.higher_is_better):
        return BaselineValidationResult(
            ok=False,
            metric=metric,
            failure_kind="primary_metric_direction_conflict",
            failure_summary=(
                f"Baseline primary metric {spec.name!r} direction conflicts with "
                "MAPELITES_FITNESS_HIGHER_IS_BETTER."
            ),
        )
    return BaselineValidationResult(ok=True, metric=metric)


def improvement_from_baseline(
    *,
    candidate_value: float | None,
    baseline: CampaignBaseline | None,
) -> float | None:
    if baseline is None or baseline.status != BASELINE_STATUS_VALID:
        return None
    if candidate_value is None or baseline.metric_value is None:
        return None
    try:
        candidate = float(candidate_value)
        root = float(baseline.metric_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate) or not math.isfinite(root):
        return None
    if bool(baseline.primary_metric_higher_is_better):
        return candidate - root
    return root - candidate


def load_latest_matching_baseline(
    *,
    session: Session,
    settings: Settings,
    campaign_program_hash: Any = _CAMPAIGN_PROGRAM_HASH_UNSET,
    valid_only: bool = False,
) -> CampaignBaseline | None:
    conditions = _matching_baseline_conditions(settings)
    if conditions is None:
        return None
    stmt = (
        select(CampaignBaseline)
        .where(
            *conditions,
        )
        .order_by(CampaignBaseline.updated_at.desc(), CampaignBaseline.created_at.desc())
        .limit(1)
    )
    if campaign_program_hash is not _CAMPAIGN_PROGRAM_HASH_UNSET:
        normalized_program_hash = normalize_single_line(str(campaign_program_hash or "")) or None
        if normalized_program_hash is None:
            stmt = stmt.where(CampaignBaseline.campaign_program_hash.is_(None))
        else:
            stmt = stmt.where(CampaignBaseline.campaign_program_hash == normalized_program_hash)
    if valid_only:
        stmt = stmt.where(CampaignBaseline.status == BASELINE_STATUS_VALID)
    return session.execute(stmt).scalar_one_or_none()


def resolve_status_campaign_program_hash(
    *,
    session: Session,
    settings: Settings,
) -> CampaignProgramHashResolution:
    """Resolve the campaign program hash status should use for baseline lookup.

    Status is an operational view over the scheduler's active campaign contract.
    For long-running schedulers with locked or approve program-change policy, the
    active hash can intentionally differ from the file currently visible on disk.
    Prefer persisted scheduler provenance before falling back to a local file
    inspection for pre-bootstrap databases.
    """

    persisted = _newest_persisted_campaign_program_hash(
        _latest_job_campaign_program_hash(session),
        _latest_baseline_campaign_program_hash(session=session, settings=settings),
    )
    if persisted is not None:
        return CampaignProgramHashResolution(
            known=True,
            campaign_program_hash=persisted.campaign_program_hash,
            source_path=persisted.source,
        )
    return resolve_current_campaign_program_hash(settings)


def resolve_current_campaign_program_hash(settings: Settings) -> CampaignProgramHashResolution:
    """Best-effort read of the campaign program currently visible to local commands."""

    raw_repo_root = (
        str(getattr(settings, "scheduler_repo_root", "") or "").strip()
        or str(getattr(settings, "worker_repo_worktree", "") or "").strip()
    )
    repo_root = Path(raw_repo_root).expanduser().resolve() if raw_repo_root else Path.cwd().resolve()
    try:
        loaded = load_campaign_program_from_repo(repo_root)
    except Exception as exc:
        summary = f"failed to inspect campaign program at {repo_root}: {exc}"
        log.warning("{}", summary)
        return CampaignProgramHashResolution(
            known=False,
            failure_summary=summary,
        )
    if loaded.snapshot is None:
        return CampaignProgramHashResolution(
            known=True,
            campaign_program_hash=None,
            source_path=str(loaded.source_path) if loaded.source_path is not None else None,
        )
    return CampaignProgramHashResolution(
        known=True,
        campaign_program_hash=loaded.snapshot.raw_sha256,
        source_path=str(loaded.snapshot.source_path),
    )


def _matching_baseline_conditions(settings: Settings) -> tuple[Any, ...] | None:
    root_commit = normalize_single_line(str(getattr(settings, "mapelites_experiment_root_commit", "") or ""))
    spec = baseline_metric_spec(settings)
    if not root_commit or not spec.name:
        return None
    root_conditions: list[Any] = [CampaignBaseline.root_commit_hash == root_commit]
    if len(root_commit) < 64:
        root_conditions.append(CampaignBaseline.root_commit_hash.like(f"{root_commit}%"))
    return (
        or_(*root_conditions),
        CampaignBaseline.primary_metric_name == spec.name,
        CampaignBaseline.primary_metric_higher_is_better == spec.higher_is_better,
        CampaignBaseline.effective_settings_fingerprint == baseline_effective_settings_fingerprint(settings),
    )


def _latest_baseline_campaign_program_hash(
    *,
    session: Session,
    settings: Settings,
) -> _PersistedCampaignProgramHash | None:
    conditions = _matching_baseline_conditions(settings)
    if conditions is None:
        return None
    stmt = (
        select(
            CampaignBaseline.campaign_program_hash,
            CampaignBaseline.updated_at,
            CampaignBaseline.created_at,
        )
        .where(*conditions)
        .order_by(CampaignBaseline.updated_at.desc(), CampaignBaseline.created_at.desc())
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None
    return _PersistedCampaignProgramHash(
        campaign_program_hash=_normalize_campaign_program_hash(_row_item(row, 0, "campaign_program_hash")),
        source="database:campaign_baselines",
        observed_at=_first_datetime(
            _row_item(row, 1, "updated_at"),
            _row_item(row, 2, "created_at"),
        ),
    )


def _latest_job_campaign_program_hash(session: Session) -> _PersistedCampaignProgramHash | None:
    stmt = (
        select(
            EvolutionJob.campaign_program_hash,
            EvolutionJob.scheduled_at,
            EvolutionJob.created_at,
            EvolutionJob.updated_at,
        )
        .order_by(
            EvolutionJob.scheduled_at.desc().nullslast(),
            EvolutionJob.created_at.desc().nullslast(),
            EvolutionJob.updated_at.desc().nullslast(),
        )
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None
    return _PersistedCampaignProgramHash(
        campaign_program_hash=_normalize_campaign_program_hash(_row_item(row, 0, "campaign_program_hash")),
        source="database:evolution_jobs",
        observed_at=_first_datetime(
            _row_item(row, 1, "scheduled_at"),
            _row_item(row, 2, "created_at"),
            _row_item(row, 3, "updated_at"),
        ),
    )


def _newest_persisted_campaign_program_hash(
    *candidates: _PersistedCampaignProgramHash | None,
) -> _PersistedCampaignProgramHash | None:
    available = [candidate for candidate in candidates if candidate is not None]
    if not available:
        return None

    def _sort_key(candidate: _PersistedCampaignProgramHash) -> tuple[bool, datetime, int]:
        observed_at = candidate.observed_at
        if observed_at is None:
            comparable_time = datetime.min.replace(tzinfo=timezone.utc)
            has_time = False
        elif observed_at.tzinfo is None:
            comparable_time = observed_at.replace(tzinfo=timezone.utc)
            has_time = True
        else:
            comparable_time = observed_at.astimezone(timezone.utc)
            has_time = True
        source_priority = 1 if candidate.source == "database:campaign_baselines" else 0
        return has_time, comparable_time, source_priority

    return max(available, key=_sort_key)


def _normalize_campaign_program_hash(value: object) -> str | None:
    return normalize_single_line(str(value or "")) or None


def _first_datetime(*values: object) -> datetime | None:
    for value in values:
        if isinstance(value, datetime):
            return value
    return None


def _row_item(row: Any, index: int, name: str) -> object:
    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return getattr(row, name, None)


class BaselineBootstrapService:
    """Ensure the active campaign baseline exists before worker budget is spent."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo_root: Path,
        console: Console,
    ) -> None:
        self.settings = settings
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.console = console

    def ensure_or_load_baseline(
        self,
        *,
        root_commit_hash: str,
        campaign_program: CampaignProgramSnapshot | None,
    ) -> BaselineBootstrapResult:
        key = build_baseline_key(
            settings=self.settings,
            root_commit_hash=root_commit_hash,
            campaign_program=campaign_program,
        )
        existing = self._load_baseline_by_key(key.hash)
        policy = self._policy()
        if existing is not None and existing.status == BASELINE_STATUS_VALID:
            return self._result_from_row(existing, key_hash=key.hash, policy=policy)

        outcome: EvaluationOutcome | None = None
        failure_kind: str | None = None
        failure_summary: str | None = None
        try:
            outcome = self._evaluate_root_baseline(
                root_commit_hash=root_commit_hash,
                campaign_program=campaign_program,
                key_hash=key.hash,
            )
        except Exception as exc:
            failure_kind = "baseline_evaluation_failed"
            failure_summary = _safe_failure_summary(exc)
            log.exception("Baseline evaluation failed key={} root={}: {}", key.hash, root_commit_hash, exc)

        row = self._persist_baseline_attempt(
            key=key,
            outcome=outcome,
            policy=policy,
            failure_kind=failure_kind,
            failure_summary=failure_summary,
        )
        result = self._result_from_row(row, key_hash=key.hash, policy=policy)
        self._log_baseline_result(result)
        return result

    def can_dispatch_or_schedule(self, result: BaselineBootstrapResult) -> bool:
        return bool(result.can_dispatch_or_schedule)

    def _policy(self) -> str:
        policy = str(getattr(self.settings, "baseline_bootstrap_policy", "required") or "required")
        return policy if policy in {"required", "warn"} else "required"

    @staticmethod
    def _load_baseline_by_key(key_hash: str) -> CampaignBaseline | None:
        with session_scope() as session:
            return session.execute(
                select(CampaignBaseline).where(CampaignBaseline.baseline_key_hash == key_hash)
            ).scalar_one_or_none()

    def _evaluate_root_baseline(
        self,
        *,
        root_commit_hash: str,
        campaign_program: CampaignProgramSnapshot | None,
        key_hash: str,
    ) -> EvaluationOutcome:
        try:
            worker_repo = WorkerRepository(self.settings)
        except RepositoryError as exc:
            raise RuntimeError(f"Worker repository is not configured for baseline evaluation: {exc}") from exc

        with worker_repo.checkout_lease_for_job(
            job_id=None,
            base_commit=root_commit_hash,
            create_branch=False,
        ) as checkout:
            evaluator = Evaluator(self.settings)
            return evaluator.evaluate_outcome(
                self._evaluation_context(
                    worktree=checkout.worktree,
                    root_commit_hash=root_commit_hash,
                    campaign_program=campaign_program,
                    key_hash=key_hash,
                )
            )

    def _evaluation_context(
        self,
        *,
        worktree: Path,
        root_commit_hash: str,
        campaign_program: CampaignProgramSnapshot | None,
        key_hash: str,
    ) -> EvaluationContext:
        metric_spec = baseline_metric_spec(self.settings)
        campaign_payload = campaign_program_evaluator_payload(campaign_program)
        goal = f"Baseline evaluation for root commit {root_commit_hash}"
        payload: dict[str, Any] = {
            "job": {
                "id": None,
                "island_id": resolve_default_island_id(self.settings),
                "goal": goal,
                "constraints": [],
                "acceptance_criteria": [],
                "notes": [],
            },
            "plan": {
                "summary": goal,
            },
            "baseline": {
                "kind": "baseline",
                "root_commit_hash": root_commit_hash,
                "campaign_program_hash": campaign_program.raw_sha256 if campaign_program else None,
                "baseline_key_hash": key_hash,
                "primary_metric_name": metric_spec.name,
                "primary_metric_higher_is_better": metric_spec.higher_is_better,
            },
        }
        if campaign_payload is not None:
            payload["campaign_program"] = campaign_payload
        return EvaluationContext(
            worktree=worktree,
            base_commit_hash=None,
            candidate_commit_hash=root_commit_hash,
            job_id=None,
            goal=goal,
            payload=payload,
            plan_summary=goal,
            metadata={
                "kind": "baseline",
                "root_commit_hash": root_commit_hash,
                "campaign_program_hash": campaign_program.raw_sha256 if campaign_program else None,
                "baseline_key_hash": key_hash,
            },
        )

    def _persist_baseline_attempt(
        self,
        *,
        key: BaselineKey,
        outcome: EvaluationOutcome | None,
        policy: str,
        failure_kind: str | None,
        failure_summary: str | None,
    ) -> CampaignBaseline:
        attempt = _baseline_attempt(
            key=key,
            outcome=outcome,
            policy=policy,
            failure_kind=failure_kind,
            failure_summary=failure_summary,
        )

        with session_scope() as session:
            row = _load_or_create_baseline_row(session=session, key_hash=key.hash)
            projection = self._persist_projection_for_outcome(
                session=session,
                key=key,
                outcome=outcome,
            )
            _apply_baseline_row(
                row=row,
                key=key,
                attempt=attempt,
                outcome=outcome,
                projection=projection,
            )
            _flush_if_available(session)
            return row

    def _persist_projection_for_outcome(
        self,
        *,
        session: Session,
        key: BaselineKey,
        outcome: EvaluationOutcome | None,
    ) -> tuple[Any, Any | None] | None:
        if outcome is None or outcome.result is None:
            return None
        return self._persist_compat_projection(
            session=session,
            commit_hash=key.root_commit_hash,
            result=outcome.result,
        )

    def _persist_compat_projection(
        self,
        *,
        session: Session,
        commit_hash: str,
        result: EvaluationResult,
    ) -> tuple[Any, Any | None]:
        commit_row = session.execute(
            select(CommitCard).where(CommitCard.commit_hash == commit_hash)
        ).scalar_one_or_none()
        if commit_row is None:
            commit_row = CommitCard(
                commit_hash=commit_hash,
                parent_commit_hash=None,
                island_id=resolve_default_island_id(self.settings),
                author=None,
                subject=f"Root baseline {commit_hash[:12]}",
                change_summary="Root baseline commit.",
                evaluation_summary=result.summary,
                tags=[],
                key_files=[],
                highlights=["Root baseline commit."],
                job_id=None,
            )
            session.add(commit_row)
            _flush_if_available(session)
        else:
            commit_row.evaluation_summary = result.summary

        primary_metric_row: Metric | None = None
        primary_metric_name = normalize_single_line(str(self.settings.mapelites_fitness_metric or ""))
        for metric in result.metrics:
            try:
                value = float(metric.value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            metric_row = self._upsert_metric_projection(
                session=session,
                commit_row=commit_row,
                metric=metric,
                value=value,
            )
            if metric.name == primary_metric_name:
                primary_metric_row = metric_row
        _flush_if_available(session)
        return getattr(commit_row, "id", None), (
            getattr(primary_metric_row, "id", None) if primary_metric_row is not None else None
        )

    @staticmethod
    def _upsert_metric_projection(
        *,
        session: Session,
        commit_row: CommitCard,
        metric: EvaluationMetric,
        value: float,
    ) -> Metric:
        existing = session.execute(
            select(Metric).where(
                Metric.commit_card_id == commit_row.id,
                Metric.name == metric.name,
            )
        ).scalar_one_or_none()
        if existing is not None:
            existing.value = value
            existing.unit = metric.unit
            existing.higher_is_better = bool(metric.higher_is_better)
            existing.details = dict(metric.details or {})
            return existing

        metric_row = Metric(
            commit=commit_row,
            name=metric.name,
            value=value,
            unit=metric.unit,
            higher_is_better=bool(metric.higher_is_better),
            details=dict(metric.details or {}),
        )
        session.add(metric_row)
        return metric_row

    @staticmethod
    def _result_from_row(
        row: CampaignBaseline,
        *,
        key_hash: str,
        policy: str,
    ) -> BaselineBootstrapResult:
        can_run = row.status == BASELINE_STATUS_VALID or policy == "warn"
        return BaselineBootstrapResult(
            can_dispatch_or_schedule=can_run,
            status=str(row.status),
            policy=policy,
            baseline_key_hash=key_hash,
            baseline_id=str(row.id) if getattr(row, "id", None) is not None else None,
            failure_kind=row.failure_kind,
            failure_summary=row.failure_summary,
        )

    def _log_baseline_result(self, result: BaselineBootstrapResult) -> None:
        if result.status == BASELINE_STATUS_VALID:
            self.console.log(
                "[green]Campaign baseline ready[/] key={} id={}".format(
                    result.baseline_key_hash[:12],
                    result.baseline_id or "n/a",
                ),
            )
            log.info("Campaign baseline valid key={} id={}", result.baseline_key_hash, result.baseline_id)
            return
        message = result.failure_summary or result.failure_kind or result.status
        if result.can_dispatch_or_schedule:
            self.console.log(
                "[yellow]Campaign baseline degraded[/] key={} reason={}".format(
                    result.baseline_key_hash[:12],
                    message,
                ),
            )
            log.warning(
                "Campaign baseline degraded key={} failure_kind={} reason={}",
                result.baseline_key_hash,
                result.failure_kind,
                message,
            )
            return
        self.console.log(
            "[bold red]Campaign baseline blocked[/] key={} reason={}".format(
                result.baseline_key_hash[:12],
                message,
            ),
        )
        log.error(
            "Campaign baseline blocked key={} failure_kind={} reason={}",
            result.baseline_key_hash,
            result.failure_kind,
            message,
        )


def _baseline_attempt(
    *,
    key: BaselineKey,
    outcome: EvaluationOutcome | None,
    policy: str,
    failure_kind: str | None,
    failure_summary: str | None,
) -> BaselineAttempt:
    spec = BaselineMetricSpec(
        name=key.primary_metric_name,
        higher_is_better=key.primary_metric_higher_is_better,
    )
    validation = _baseline_validation_for_outcome(outcome=outcome, spec=spec)
    valid = outcome is not None and outcome.outcome_kind == "passed" and validation.ok
    return BaselineAttempt(
        validation=validation,
        valid=valid,
        status=_baseline_attempt_status(valid=valid, policy=policy),
        failure_kind=failure_kind or validation.failure_kind,
        failure_summary=failure_summary or validation.failure_summary,
        metric=validation.metric,
    )


def _baseline_validation_for_outcome(
    *,
    outcome: EvaluationOutcome | None,
    spec: BaselineMetricSpec,
) -> BaselineValidationResult:
    validation = validate_baseline_primary_metric(
        result=outcome.result if outcome is not None else None,
        spec=spec,
    )
    if outcome is None or outcome.outcome_kind == "passed":
        return validation
    return BaselineValidationResult(
        ok=False,
        failure_kind=outcome.failure.failure_kind if outcome.failure else outcome.outcome_kind,
        failure_summary=(
            outcome.failure.safe_failure_summary
            if outcome.failure
            else f"Evaluator returned outcome_kind={outcome.outcome_kind}."
        ),
    )


def _baseline_attempt_status(*, valid: bool, policy: str) -> str:
    if valid:
        return BASELINE_STATUS_VALID
    return BASELINE_STATUS_DEGRADED if policy == "warn" else BASELINE_STATUS_FAILED


def _load_or_create_baseline_row(*, session: Session, key_hash: str) -> CampaignBaseline:
    row = session.execute(
        select(CampaignBaseline).where(
            CampaignBaseline.baseline_key_hash == key_hash,
        )
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = CampaignBaseline(baseline_key_hash=key_hash)
    session.add(row)
    return row


def _apply_baseline_row(
    *,
    row: CampaignBaseline,
    key: BaselineKey,
    attempt: BaselineAttempt,
    outcome: EvaluationOutcome | None,
    projection: tuple[Any, Any | None] | None,
) -> None:
    row.root_commit_hash = key.root_commit_hash
    row.campaign_program_hash = key.campaign_program_hash
    row.evaluator_name = _baseline_evaluator_name(key=key, outcome=outcome)
    row.evaluator_version = _baseline_evaluator_version(key=key, outcome=outcome)
    row.primary_metric_name = key.primary_metric_name
    row.primary_metric_higher_is_better = key.primary_metric_higher_is_better
    row.runtime_profile = key.runtime_profile
    row.effective_settings_fingerprint = key.effective_settings_fingerprint
    row.status = attempt.status
    row.metric_value = _finite_metric_value(attempt.metric)
    row.metric_unit = attempt.metric.unit if attempt.metric is not None else None
    row.evaluation_summary = _baseline_evaluation_summary(outcome)
    row.failure_kind = None if attempt.valid else _bounded_line(attempt.failure_kind, 64)
    row.failure_summary = None if attempt.valid else _bounded_failure_text(attempt.failure_summary)
    row.commit_card_id = projection[0] if projection is not None else None
    row.metric_id = projection[1] if projection is not None and attempt.metric is not None else None
    row.started_at = outcome.started_at if outcome is not None else datetime.now(timezone.utc)
    row.finished_at = outcome.finished_at if outcome is not None else datetime.now(timezone.utc)


def _baseline_evaluator_name(*, key: BaselineKey, outcome: EvaluationOutcome | None) -> str | None:
    if outcome is None:
        return key.evaluator_name
    return _bounded_line(outcome.evaluator_name, 128)


def _baseline_evaluator_version(*, key: BaselineKey, outcome: EvaluationOutcome | None) -> str | None:
    if key.evaluator_version:
        return key.evaluator_version
    if outcome is None:
        return None
    return _bounded_line(outcome.evaluator_version, 128)


def _baseline_evaluation_summary(outcome: EvaluationOutcome | None) -> str | None:
    if outcome is None or outcome.result is None:
        return None
    return _bounded_failure_text(outcome.result.summary)


def _safe_failure_summary(exc: Exception) -> str:
    text = normalize_single_line(str(exc)) or exc.__class__.__name__
    return _bounded_failure_text(text) or exc.__class__.__name__


def _bounded_failure_text(value: Any) -> str | None:
    if value is None:
        return None
    return clamp_text(normalize_single_line(str(value)), _BASELINE_FAILURE_TEXT_MAX) or None


def _bounded_line(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    return clamp_text(normalize_single_line(str(value)), limit) or None


def _finite_metric_value(metric: EvaluationMetric | None) -> float | None:
    if metric is None:
        return None
    try:
        value = float(metric.value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _flush_if_available(session: Any) -> None:
    flush = getattr(session, "flush", None)
    if callable(flush):
        flush()
