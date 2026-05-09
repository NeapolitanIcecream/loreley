"""Agent REST facade API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from loreley.api.schemas import OrmOutModel
from loreley.api.schemas.evidence import (
    EvaluationAgentFeedbackOut,
    EvaluationArtifactOut,
)
from loreley.api.schemas.operator import OperatorStatusOut


class AgentErrorOut(OrmOutModel):
    error_code: str
    message: str
    retryable: bool = False
    resource: dict[str, str] | None = None
    suggested_next_actions: list[dict[str, Any]] = Field(default_factory=list)


class AgentActionRequest(BaseModel):
    action_type: str
    dry_run: bool = True
    idempotency_key: str | None = None
    reason: str | None = None
    expected_state: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_state", "params", mode="before")
    @classmethod
    def _dict_default(cls, value: object) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("must be an object")
        return dict(value)


class AgentActionOut(OrmOutModel):
    action_id: UUID
    status: str
    dry_run: bool
    action_type: str
    risk: str
    preconditions: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: AgentErrorOut | None = None
    created_at: datetime
    completed_at: datetime | None = None


class AgentCapabilityActionOut(OrmOutModel):
    action_type: str
    risk: str
    dry_run_supported: bool = True
    reason_expected: bool = True
    idempotency_key_expected: bool = True
    required_params: list[str] = Field(default_factory=list)
    expected_state_fields: list[str] = Field(default_factory=list)


class AgentCapabilitiesOut(OrmOutModel):
    schema_version: str
    database_schema_version: int
    auth: dict[str, Any] = Field(default_factory=dict)
    read_resources: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[AgentCapabilityActionOut] = Field(default_factory=list)
    error_shape: dict[str, Any] = Field(default_factory=dict)


class AgentNextActionOut(OrmOutModel):
    action_type: str
    reason: str
    risk: str
    dry_run: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    expected_state: dict[str, Any] = Field(default_factory=dict)
    resource: dict[str, str] | None = None


class AgentBlockingIssueOut(OrmOutModel):
    issue_type: str
    message: str
    resource: dict[str, str] | None = None
    suggested_next_actions: list[dict[str, Any]] = Field(default_factory=list)


class AgentStatusOut(OrmOutModel):
    operator_status: OperatorStatusOut
    health: str
    blocking_issues: list[AgentBlockingIssueOut] = Field(default_factory=list)
    safe_next_actions: list[AgentNextActionOut] = Field(default_factory=list)


class AgentFeedbackOut(OrmOutModel):
    resource: dict[str, str]
    artifact_count: int = 0
    artifacts: list[EvaluationArtifactOut] = Field(default_factory=list)
    feedback: EvaluationAgentFeedbackOut | None = None
