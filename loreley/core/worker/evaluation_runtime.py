"""Database-backed evaluator capacity and accepted-measurement coordination."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import socket
from time import monotonic, sleep
from typing import Any, Mapping
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import create_engine, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import NullPool

from loreley.config import Settings
from loreley.core.worker.evaluator import EvaluationMeasurement, EvaluationPreparation
from loreley.db.base import session_scope
from loreley.db.models import (
    EvaluationAttempt,
    EvaluationArtifactRecord,
    EvaluationConcurrencyContract,
    EvaluationMeasurement as EvaluationMeasurementRow,
    EvaluationResourceLease,
)

log = logger.bind(module="worker.evaluation_runtime")

_EVALUATOR_SLOT_NAMESPACE = "loreley:evaluator-slot:v1"
_MEASUREMENT_LOCK_NAMESPACE = "loreley:measurement-lock:v1"


class EvaluationRuntimeError(RuntimeError):
    """Raised when evaluator capacity or measurement coordination cannot proceed."""


@dataclass(frozen=True, slots=True)
class CachedMeasurement:
    id: UUID
    cache_key: str
    measurement: EvaluationMeasurement
    source_evaluation_attempt_id: UUID | None
    source_job_id: UUID | None
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class _LeaseRequest:
    resource_kind: str
    resource_key: str
    contract_key: str
    job_id: UUID
    run_token: UUID
    deadline: float
    slot_count: int
    namespace: str
    scope: str


@dataclass(slots=True)
class AdvisoryLease:
    """Held PostgreSQL advisory lock plus its durable observability row."""

    connection: Connection
    advisory_key: int
    lease_id: UUID
    slot_index: int | None
    scope: str
    wait_seconds: float
    acquired_at: datetime | None = None
    released_at: datetime | None = None
    _released: bool = False

    def release(self, reason: str = "completed") -> datetime | None:
        if self._released:
            return self.released_at
        unlock_ok = False
        try:
            unlock_ok = bool(
                self.connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": self.advisory_key},
                ).scalar()
            )
        except Exception as exc:  # connection death itself releases the lock
            log.warning(
                "Evaluator advisory unlock failed lease={}: {}", self.lease_id, exc
            )
        finally:
            self.connection.close()
            self._released = True
        self.released_at = _mark_lease_released(
            self.lease_id,
            reason=reason if unlock_ok else f"{reason}_connection_closed",
        )
        return self.released_at

    def __enter__(self) -> "AdvisoryLease":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release("error" if exc is not None else "completed")


@lru_cache(maxsize=8)
def _lock_engine(database_dsn: str) -> Engine:
    """Use connections outside the application pool for long-held session locks."""

    return create_engine(
        database_dsn,
        poolclass=NullPool,
        pool_pre_ping=True,
        isolation_level="AUTOCOMMIT",
        future=True,
    )


def evaluation_contract_key(
    *,
    experiment_id: str,
    evaluator_name: str,
    evaluator_version: str,
    campaign_program_hash: str,
) -> str:
    return _canonical_digest(
        {
            "campaign_program_hash": campaign_program_hash,
            "evaluator_name": evaluator_name,
            "evaluator_version": evaluator_version,
            "experiment_id": experiment_id,
        }
    )


def measurement_cache_key(
    *,
    preparation: EvaluationPreparation,
    evaluator_name: str,
    evaluator_version: str,
    campaign_program_hash: str,
) -> str:
    """Return the complete, non-null cache contract for a phased measurement."""

    fields = {
        "candidate_identity": preparation.candidate_identity,
        "measurement_contract_fingerprint": preparation.measurement_contract_fingerprint,
        "evaluator_name": evaluator_name,
        "evaluator_version": evaluator_version,
        "campaign_program_hash": campaign_program_hash,
    }
    missing = [name for name, value in fields.items() if not str(value or "").strip()]
    if missing:
        raise EvaluationRuntimeError(
            "Persistent measurement reuse requires non-empty contract field(s): "
            + ", ".join(missing)
        )
    return _canonical_digest(fields)


def measurement_payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


class EvaluationRuntimeCoordinator:
    """Coordinates E slots and first measurements without target-specific logic."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.poll_seconds = max(
            0.01, float(settings.worker_evaluator_slot_poll_seconds)
        )

    def ensure_contract(
        self,
        *,
        evaluator_name: str,
        evaluator_version: str,
        campaign_program_hash: str,
        limit_scope: str,
    ) -> str:
        if limit_scope not in {"whole", "measurement"}:
            raise EvaluationRuntimeError(
                f"Unsupported evaluator limit scope: {limit_scope!r}."
            )
        experiment_id = (
            str(self.settings.experiment_id or "default").strip() or "default"
        )
        key = evaluation_contract_key(
            experiment_id=experiment_id,
            evaluator_name=evaluator_name,
            evaluator_version=evaluator_version,
            campaign_program_hash=campaign_program_hash,
        )
        configured_limit = self.settings.worker_evaluator_max_concurrency
        with session_scope() as session:
            bind = session.get_bind()
            if bind.dialect.name != "postgresql":
                raise EvaluationRuntimeError(
                    "Evaluator runtime contracts require PostgreSQL."
                )
            session.execute(
                pg_insert(EvaluationConcurrencyContract)
                .values(
                    contract_key=key,
                    experiment_id=experiment_id,
                    evaluator_name=evaluator_name,
                    evaluator_version=evaluator_version,
                    campaign_program_hash=campaign_program_hash,
                    max_concurrency=configured_limit,
                    limit_scope=limit_scope,
                )
                .on_conflict_do_nothing(index_elements=["contract_key"])
            )
            row = session.get(EvaluationConcurrencyContract, key)
            if row is None:  # pragma: no cover - insert/get invariant
                raise EvaluationRuntimeError(
                    "Evaluator runtime contract was not persisted."
                )
            if row.limit_scope != limit_scope:
                raise EvaluationRuntimeError(
                    "Evaluator limit scope disagrees with the persisted contract "
                    f"({limit_scope!r} != {row.limit_scope!r})."
                )
            if row.max_concurrency != configured_limit:
                raise EvaluationRuntimeError(
                    "WORKER_EVALUATOR_MAX_CONCURRENCY disagrees with the persisted "
                    f"contract ({configured_limit!r} != {row.max_concurrency!r})."
                )
        return key

    def acquire_evaluator_slot(
        self,
        *,
        contract_key: str,
        job_id: UUID,
        run_token: UUID,
        deadline: float,
    ) -> AdvisoryLease | None:
        limit = self.settings.worker_evaluator_max_concurrency
        if limit is None:
            return None
        return self._acquire_any(
            _LeaseRequest(
                resource_kind="evaluator_slot",
                resource_key=contract_key,
                contract_key=contract_key,
                job_id=job_id,
                run_token=run_token,
                deadline=deadline,
                slot_count=int(limit),
                namespace=_EVALUATOR_SLOT_NAMESPACE,
                scope="evaluator",
            )
        )

    def acquire_measurement_lock(
        self,
        *,
        cache_key: str,
        contract_key: str,
        job_id: UUID,
        run_token: UUID,
        deadline: float,
    ) -> AdvisoryLease:
        lease = self._acquire_any(
            _LeaseRequest(
                resource_kind="measurement_key",
                resource_key=cache_key,
                contract_key=contract_key,
                job_id=job_id,
                run_token=run_token,
                deadline=deadline,
                slot_count=1,
                namespace=_MEASUREMENT_LOCK_NAMESPACE,
                scope="measurement_key",
            )
        )
        if (
            lease is None
        ):  # pragma: no cover - slot_count=1 always yields a lease or raises
            raise EvaluationRuntimeError("Failed to acquire measurement lock.")
        return lease

    def lookup_measurement(self, cache_key: str) -> CachedMeasurement | None:
        with session_scope() as session:
            row = session.execute(
                select(EvaluationMeasurementRow).where(
                    EvaluationMeasurementRow.cache_key == cache_key
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            payload = dict(row.payload or {})
            expected = measurement_payload_sha256(payload)
            if expected != row.payload_sha256:
                raise EvaluationRuntimeError(
                    f"Accepted measurement {row.id} payload hash does not match its manifest."
                )
            measurement = EvaluationMeasurement.from_cache_payload(payload)
            if not measurement.cacheable:
                raise EvaluationRuntimeError(
                    f"Accepted measurement {row.id} is not marked cacheable."
                )
            payload_evidence = measurement.cache_payload()["evidence"]
            if payload_evidence != list(row.evidence_manifest or ()):
                raise EvaluationRuntimeError(
                    f"Accepted measurement {row.id} evidence manifest does not match its payload."
                )
            if row.source_evaluation_attempt_id is None:
                raise EvaluationRuntimeError(
                    f"Accepted measurement {row.id} has no original evaluation attempt."
                )
            source_attempt = session.get(
                EvaluationAttempt,
                row.source_evaluation_attempt_id,
            )
            if not (
                source_attempt is not None
                and source_attempt.outcome_kind == "passed"
                and source_attempt.measurement_executed
                and source_attempt.measurement_id == row.id
            ):
                raise EvaluationRuntimeError(
                    f"Accepted measurement {row.id} original attempt is incomplete."
                )
            _require_intact_cached_evidence(
                session=session,
                measurement=row,
                source_attempt=source_attempt,
            )
            return CachedMeasurement(
                id=row.id,
                cache_key=row.cache_key,
                measurement=measurement,
                source_evaluation_attempt_id=row.source_evaluation_attempt_id,
                source_job_id=row.source_job_id,
                payload_sha256=row.payload_sha256,
            )

    def _acquire_any(
        self,
        request: _LeaseRequest,
    ) -> AdvisoryLease:
        if request.slot_count < 1:
            raise EvaluationRuntimeError("Evaluator slot_count must be at least one.")
        lease_id = uuid4()
        requested_at = monotonic()
        _create_waiting_lease(lease_id=lease_id, request=request)
        connection = _lock_engine(self.settings.database_dsn).connect()
        _require_postgres_lock_connection(connection=connection, lease_id=lease_id)
        try:
            while True:
                lease = _try_acquire_requested_slot(
                    connection=connection,
                    lease_id=lease_id,
                    request=request,
                    requested_at=requested_at,
                )
                if lease is not None:
                    return lease
                self._wait_for_slot(request)
        except Exception:
            connection.close()
            _mark_lease_released(
                lease_id, reason="acquisition_failed", status="cancelled"
            )
            raise

    def _wait_for_slot(self, request: _LeaseRequest) -> None:
        now = monotonic()
        if now >= request.deadline:
            raise EvaluationRuntimeError(
                f"Timed out waiting for {request.resource_kind} capacity."
            )
        sleep(min(self.poll_seconds, max(0.01, request.deadline - now)))


def _create_waiting_lease(*, lease_id: UUID, request: _LeaseRequest) -> None:
    with session_scope() as session:
        session.add(
            EvaluationResourceLease(
                id=lease_id,
                resource_kind=request.resource_kind,
                resource_key=request.resource_key,
                contract_key=request.contract_key,
                job_id=request.job_id,
                run_token=request.run_token,
                worker_id=f"{socket.gethostname()}:{os.getpid()}",
                status="waiting",
            )
        )


def _require_postgres_lock_connection(
    *, connection: Connection, lease_id: UUID
) -> None:
    if connection.dialect.name == "postgresql":
        return
    connection.close()
    _mark_lease_released(lease_id, reason="unsupported_database", status="cancelled")
    raise EvaluationRuntimeError(
        "Evaluator concurrency and first-measurement locking require PostgreSQL."
    )


def _try_acquire_requested_slot(
    *,
    connection: Connection,
    lease_id: UUID,
    request: _LeaseRequest,
    requested_at: float,
) -> AdvisoryLease | None:
    for slot in range(request.slot_count):
        advisory_key = _advisory_key(request.namespace, request.resource_key, slot)
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": advisory_key},
            ).scalar()
        )
        if acquired:
            return _acquired_lease(
                connection=connection,
                lease_id=lease_id,
                request=request,
                slot=slot,
                advisory_key=advisory_key,
                requested_at=requested_at,
            )
    return None


def _acquired_lease(
    *,
    connection: Connection,
    lease_id: UUID,
    request: _LeaseRequest,
    slot: int,
    advisory_key: int,
    requested_at: float,
) -> AdvisoryLease:
    waited = monotonic() - requested_at
    acquired_at = _mark_lease_acquired(
        lease_id,
        slot_index=slot,
        wait_seconds=waited,
    )
    return AdvisoryLease(
        connection=connection,
        advisory_key=advisory_key,
        lease_id=lease_id,
        slot_index=slot,
        scope=request.scope,
        wait_seconds=waited,
        acquired_at=acquired_at,
    )


def _require_intact_cached_evidence(
    *,
    session: Any,
    measurement: EvaluationMeasurementRow,
    source_attempt: EvaluationAttempt,
) -> None:
    expected = {
        str(item.get("key") or ""): item
        for item in (measurement.evidence_manifest or ())
        if isinstance(item, Mapping) and str(item.get("key") or "")
    }
    rows = list(
        session.execute(
            select(EvaluationArtifactRecord).where(
                EvaluationArtifactRecord.evaluation_attempt_id == source_attempt.id,
                EvaluationArtifactRecord.key.in_(tuple(expected)),
            )
        ).scalars()
    )
    by_key = {str(row.key): row for row in rows}
    for key, item in expected.items():
        artifact = by_key.get(key)
        if artifact is None:
            raise EvaluationRuntimeError(
                f"Accepted measurement {measurement.id} evidence {key!r} has no artifact record."
            )
        _require_cached_artifact_metadata(
            measurement_id=measurement.id,
            key=key,
            expected=item,
            artifact=artifact,
        )
        _require_cached_artifact_bytes(
            measurement_id=measurement.id,
            key=key,
            artifact=artifact,
        )


def _require_cached_artifact_metadata(
    *,
    measurement_id: UUID,
    key: str,
    expected: Mapping[str, Any],
    artifact: EvaluationArtifactRecord,
) -> None:
    expected_sha = str(expected.get("sha256") or "").lower()
    expected_size = expected.get("size_bytes")
    sha_matches = str(artifact.sha256 or "").lower() == expected_sha
    size_matches = expected_size is None or int(artifact.size_bytes or -1) == int(
        expected_size
    )
    if not (sha_matches and size_matches):
        raise EvaluationRuntimeError(
            f"Accepted measurement {measurement_id} evidence {key!r} artifact metadata drifted."
        )


def _require_cached_artifact_bytes(
    *,
    measurement_id: UUID,
    key: str,
    artifact: EvaluationArtifactRecord,
) -> None:
    path = Path(str(artifact.storage_path or ""))
    if not path.is_file():
        raise EvaluationRuntimeError(
            f"Accepted measurement {measurement_id} evidence {key!r} payload is unavailable."
        )
    if path.stat().st_size != int(artifact.size_bytes or -1):
        raise EvaluationRuntimeError(
            f"Accepted measurement {measurement_id} evidence {key!r} payload size drifted."
        )
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != str(artifact.sha256 or "").lower()
    ):
        raise EvaluationRuntimeError(
            f"Accepted measurement {measurement_id} evidence {key!r} payload hash drifted."
        )


def _mark_lease_acquired(
    lease_id: UUID,
    *,
    slot_index: int,
    wait_seconds: float,
) -> datetime | None:
    with session_scope() as session:
        row = session.get(EvaluationResourceLease, lease_id)
        if row is None:
            return None
        acquired_at = _db_now(session)
        row.status = "acquired"
        row.slot_index = int(slot_index)
        row.wait_seconds = float(wait_seconds)
        row.acquired_at = acquired_at
        return acquired_at


def _mark_lease_released(
    lease_id: UUID,
    *,
    reason: str,
    status: str = "released",
) -> datetime | None:
    try:
        with session_scope() as session:
            row = session.get(EvaluationResourceLease, lease_id)
            if row is None:
                return None
            released_at = _db_now(session)
            row.status = status
            row.release_reason = str(reason)[:64]
            row.released_at = released_at
            return released_at
    except Exception as exc:  # lock release must not mask the evaluator result
        log.warning("Failed to persist evaluator lease release {}: {}", lease_id, exc)
        return None


def _db_now(session: Any) -> Any:
    return session.execute(text("SELECT CURRENT_TIMESTAMP")).scalar_one()


def _advisory_key(namespace: str, resource_key: str, slot: int) -> int:
    digest = hashlib.sha256(
        f"{namespace}:{resource_key}:{slot}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationRuntimeError(
            "Measurement payload must be canonical JSON."
        ) from exc


__all__ = [
    "AdvisoryLease",
    "CachedMeasurement",
    "EvaluationRuntimeCoordinator",
    "EvaluationRuntimeError",
    "evaluation_contract_key",
    "measurement_cache_key",
    "measurement_payload_sha256",
]
