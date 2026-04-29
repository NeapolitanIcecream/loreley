"""Evaluation evidence API schemas."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from loreley.api.schemas import OrmOutModel


class EvaluationDiagnosticOut(OrmOutModel):
    kind: str
    message: str
    severity: str
    location: str | None = None
    metric: str | None = None
    value: float | None = None
    unit: str | None = None


class EvaluationArtifactOut(OrmOutModel):
    id: UUID
    job_id: UUID
    commit_hash: str
    key: str
    kind: str
    mime_type: str
    label: str | None = None
    summary: str | None = None
    visibility: str
    agent_projection: str
    size_bytes: int | None = None
    sha256: str | None = None
    diagnostics: list[EvaluationDiagnosticOut] = Field(default_factory=list)
    download_url: str | None = None


class EvaluationAgentFeedbackOut(OrmOutModel):
    mode: str
    budget_chars: int
    text: str
    included_artifact_keys: list[str] = Field(default_factory=list)
    omitted_artifact_count: int = 0
    omitted_reasons: list[str] = Field(default_factory=list)


class EvaluationEvidenceIndicatorOut(OrmOutModel):
    has_evaluation_evidence: bool = False
    agent_visible_evidence_count: int = 0
    top_evaluation_diagnosis: str | None = None
