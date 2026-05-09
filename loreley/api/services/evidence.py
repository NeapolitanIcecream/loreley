"""Evaluation evidence queries and projection helpers."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterable
from uuid import UUID

from sqlalchemy import select

from loreley.config import Settings, get_settings
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.worker.planning import (
    CommitEvaluationArtifactFeedback,
    EvaluationDiagnosticBrief,
    render_evaluation_agent_feedback,
)
from loreley.db.base import session_scope
from loreley.db.models import EvaluationArtifactRecord


@dataclass(frozen=True, slots=True)
class EvidenceIndicator:
    has_evaluation_evidence: bool = False
    agent_visible_evidence_count: int = 0
    top_evaluation_diagnosis: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "has_evaluation_evidence": self.has_evaluation_evidence,
            "agent_visible_evidence_count": self.agent_visible_evidence_count,
            "top_evaluation_diagnosis": self.top_evaluation_diagnosis,
        }


def evaluation_artifact_download_path(*, job_id: UUID, artifact_key: str) -> str:
    return f"/api/v1/jobs/{job_id}/evaluation-artifacts/{artifact_key}"


def evaluation_artifact_filename(row: EvaluationArtifactRecord) -> str:
    extension = mimetypes.guess_extension(str(row.mime_type or "")) or ".bin"
    return f"{row.key}{extension}"


def list_evaluation_artifacts_for_job(
    *,
    job_id: UUID,
    include_hidden: bool = False,
    visibility: str | None = None,
) -> list[EvaluationArtifactRecord]:
    normalized_visibility = normalize_single_line(str(visibility or ""))
    with session_scope() as session:
        stmt = (
            select(EvaluationArtifactRecord)
            .where(EvaluationArtifactRecord.job_id == job_id)
            .order_by(EvaluationArtifactRecord.created_at.asc(), EvaluationArtifactRecord.id.asc())
        )
        if normalized_visibility:
            stmt = stmt.where(EvaluationArtifactRecord.visibility == normalized_visibility)
        elif not include_hidden:
            stmt = stmt.where(EvaluationArtifactRecord.visibility != "hidden")
        return list(session.execute(stmt).scalars().all())


def list_evaluation_artifacts_for_commit(
    *,
    commit_hash: str,
    include_hidden: bool = False,
    visibility: str | None = None,
) -> list[EvaluationArtifactRecord]:
    normalized_hash = str(commit_hash or "").strip()
    if not normalized_hash:
        return []
    normalized_visibility = normalize_single_line(str(visibility or ""))
    with session_scope() as session:
        stmt = (
            select(EvaluationArtifactRecord)
            .where(EvaluationArtifactRecord.commit_hash == normalized_hash)
            .order_by(EvaluationArtifactRecord.created_at.asc(), EvaluationArtifactRecord.id.asc())
        )
        if normalized_visibility:
            stmt = stmt.where(EvaluationArtifactRecord.visibility == normalized_visibility)
        elif not include_hidden:
            stmt = stmt.where(EvaluationArtifactRecord.visibility != "hidden")
        return list(session.execute(stmt).scalars().all())


def get_downloadable_evaluation_artifact(
    *,
    job_id: UUID,
    artifact_key: str,
) -> EvaluationArtifactRecord | None:
    key = str(artifact_key or "").strip()
    if not key:
        return None
    with session_scope() as session:
        stmt = select(EvaluationArtifactRecord).where(
            EvaluationArtifactRecord.job_id == job_id,
            EvaluationArtifactRecord.key == key,
            EvaluationArtifactRecord.visibility != "hidden",
        )
        return session.execute(stmt).scalar_one_or_none()


def load_evidence_indicators_by_commit_hash(
    commit_hashes: Collection[str],
) -> dict[str, EvidenceIndicator]:
    ordered = tuple(dict.fromkeys(str(value or "").strip() for value in commit_hashes if str(value or "").strip()))
    if not ordered:
        return {}
    indicators: dict[str, EvidenceIndicator] = {
        commit_hash: EvidenceIndicator() for commit_hash in ordered
    }
    with session_scope() as session:
        rows = list(
            session.execute(
                select(EvaluationArtifactRecord)
                .where(
                    EvaluationArtifactRecord.commit_hash.in_(ordered),
                    EvaluationArtifactRecord.visibility != "hidden",
                )
                .order_by(
                    EvaluationArtifactRecord.commit_hash.asc(),
                    EvaluationArtifactRecord.created_at.asc(),
                    EvaluationArtifactRecord.id.asc(),
                )
            ).scalars().all()
        )
    grouped: dict[str, list[EvaluationArtifactRecord]] = {}
    for row in rows:
        if row.visibility == "hidden":
            continue
        grouped.setdefault(str(row.commit_hash), []).append(row)
    for commit_hash, commit_rows in grouped.items():
        agent_rows = [row for row in commit_rows if row.visibility == "agent_visible"]
        indicators[commit_hash] = EvidenceIndicator(
            has_evaluation_evidence=bool(commit_rows),
            agent_visible_evidence_count=len(agent_rows),
            top_evaluation_diagnosis=_top_agent_visible_diagnosis(agent_rows),
        )
    return indicators


def build_evaluation_artifact_payload(row: EvaluationArtifactRecord) -> dict[str, object]:
    return {
        "id": row.id,
        "job_id": row.job_id,
        "commit_hash": row.commit_hash,
        "key": row.key,
        "kind": row.kind,
        "mime_type": row.mime_type,
        "label": row.label,
        "summary": row.summary,
        "visibility": row.visibility,
        "agent_projection": row.agent_projection,
        "size_bytes": row.size_bytes,
        "sha256": row.sha256,
        "diagnostics": _diagnostic_payloads(row.diagnostics or ()),
        "download_url": (
            evaluation_artifact_download_path(job_id=row.job_id, artifact_key=row.key)
            if row.storage_path
            else None
        ),
    }


def build_agent_feedback_payload(
    rows: Iterable[EvaluationArtifactRecord],
    *,
    settings: Settings | None = None,
) -> dict[str, object] | None:
    materialized = tuple(_feedback_from_row(row) for row in rows)
    if not materialized:
        return None
    projection = render_evaluation_agent_feedback(
        materialized,
        settings=settings or get_settings(),
    )
    return {
        "mode": projection.mode,
        "budget_chars": projection.budget_chars,
        "text": projection.text,
        "included_artifact_keys": list(projection.included_artifact_keys),
        "omitted_artifact_count": projection.omitted_artifact_count,
        "omitted_reasons": list(projection.omitted_reasons),
    }


def artifact_file_path(row: EvaluationArtifactRecord) -> Path | None:
    if not row.storage_path:
        return None
    path = Path(str(row.storage_path))
    if not path.exists() or not path.is_file():
        return None
    return path


def _feedback_from_row(row: EvaluationArtifactRecord) -> CommitEvaluationArtifactFeedback:
    return CommitEvaluationArtifactFeedback(
        key=row.key,
        kind=row.kind,
        mime_type=row.mime_type,
        label=row.label,
        summary=row.summary,
        diagnostics=tuple(
            EvaluationDiagnosticBrief(**item)
            for item in _diagnostic_payloads(row.diagnostics or ())
        ),
        projection=row.agent_projection,
        visibility=row.visibility,
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        artifact_uri=(
            f"loreley://evaluation-artifacts/{row.job_id}/{row.key}"
            if row.storage_path
            else None
        ),
    )


def _diagnostic_payloads(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list | tuple):
        return []
    payloads: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        payloads.append(
            {
                "kind": normalize_single_line(str(item.get("kind") or "")) or "diagnostic",
                "message": clamp_text(
                    normalize_single_line(str(item.get("message") or "")),
                    512,
                ),
                "severity": normalize_single_line(str(item.get("severity") or "info")) or "info",
                "location": _optional_line(item.get("location"), 256),
                "metric": _optional_line(item.get("metric"), 128),
                "value": _optional_float(item.get("value")),
                "unit": _optional_line(item.get("unit"), 32),
            }
        )
    return payloads


def _top_agent_visible_diagnosis(rows: list[EvaluationArtifactRecord]) -> str | None:
    for row in rows:
        for diagnostic in _diagnostic_payloads(row.diagnostics or ()):
            message = normalize_single_line(str(diagnostic.get("message") or ""))
            if message:
                return clamp_text(message, 200)
    for row in rows:
        summary = normalize_single_line(str(row.summary or ""))
        if summary:
            return clamp_text(summary, 200)
    return None


def _optional_line(value: object, limit: int) -> str | None:
    if value is None:
        return None
    text = clamp_text(normalize_single_line(str(value)), limit)
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
