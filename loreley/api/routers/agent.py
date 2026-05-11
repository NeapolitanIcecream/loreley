"""Agent REST control facade endpoints."""

from __future__ import annotations

import hmac
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Header
from loguru import logger

from loreley.api.agent_errors import AgentAPIError
from loreley.api.schemas.agent import (
    AgentActionOut,
    AgentActionRequest,
    AgentCapabilitiesOut,
    AgentFeedbackOut,
    AgentNextActionOut,
    AgentStatusOut,
)
from loreley.api.services.agent import (
    agent_capabilities,
    agent_feedback_payload,
    agent_next_actions,
    agent_status,
    get_agent_action_record,
    run_agent_action,
)
from loreley.api.services.evidence import (
    list_evaluation_artifacts_for_commit,
    list_evaluation_artifacts_for_job,
)
from loreley.config import get_settings

router = APIRouter(prefix="/agent")
log = logger.bind(module="api.agent")


def require_agent_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Return the audit actor after applying optional agent-token auth."""

    token = str(get_settings().loreley_agent_api_token or "").strip()
    if not token:
        log.bind(auth_scope="agent_api", reason="token_unconfigured", status_code=503).warning(
            "Agent API request rejected by auth"
        )
        raise AgentAPIError(
            status_code=503,
            error_code="agent_auth_not_configured",
            message="LORELEY_AGENT_API_TOKEN is required for agent routes.",
            retryable=False,
            resource={"type": "agent_api", "id": "configuration"},
        )
    raw = str(authorization or "").strip()
    if not raw:
        log.bind(auth_scope="agent_api", reason="missing_authorization", status_code=401).warning(
            "Agent API request rejected by auth"
        )
        raise AgentAPIError(
            status_code=401,
            error_code="unauthorized",
            message="Missing Authorization bearer token.",
            retryable=False,
            resource={"type": "agent_api", "id": "auth"},
        )
    scheme, separator, supplied = raw.partition(" ")
    if not separator or scheme.lower() != "bearer":
        log.bind(auth_scope="agent_api", reason="invalid_scheme", status_code=401).warning(
            "Agent API request rejected by auth"
        )
        raise AgentAPIError(
            status_code=401,
            error_code="unauthorized",
            message="Authorization must use Bearer token credentials.",
            retryable=False,
            resource={"type": "agent_api", "id": "auth"},
        )
    supplied = supplied.strip()
    if not supplied:
        log.bind(auth_scope="agent_api", reason="missing_token", status_code=401).warning(
            "Agent API request rejected by auth"
        )
        raise AgentAPIError(
            status_code=401,
            error_code="unauthorized",
            message="Missing Authorization bearer token.",
            retryable=False,
            resource={"type": "agent_api", "id": "auth"},
        )
    if not hmac.compare_digest(supplied, token):
        log.bind(auth_scope="agent_api", reason="invalid_token", status_code=403).warning(
            "Agent API request rejected by auth"
        )
        raise AgentAPIError(
            status_code=403,
            error_code="forbidden",
            message="Authorization bearer token is invalid.",
            retryable=False,
            resource={"type": "agent_api", "id": "auth"},
        )
    return "agent-token"


@router.get("/capabilities", response_model=AgentCapabilitiesOut)
def get_agent_capabilities(
    _actor: str = Depends(require_agent_auth),
) -> AgentCapabilitiesOut:
    return AgentCapabilitiesOut.model_validate(agent_capabilities())


@router.get("/status", response_model=AgentStatusOut)
def get_agent_status(
    _actor: str = Depends(require_agent_auth),
) -> AgentStatusOut:
    return AgentStatusOut.model_validate(agent_status())


@router.get("/next-actions", response_model=list[AgentNextActionOut])
def get_agent_next_actions(
    _actor: str = Depends(require_agent_auth),
) -> list[AgentNextActionOut]:
    return [AgentNextActionOut.model_validate(item) for item in agent_next_actions()]


@router.post("/actions", response_model=AgentActionOut)
def post_agent_action(
    body: AgentActionRequest,
    background_tasks: BackgroundTasks,
    actor: str = Depends(require_agent_auth),
) -> AgentActionOut:
    return AgentActionOut.model_validate(
        run_agent_action(body, actor=actor, background_tasks=background_tasks)
    )


@router.get("/actions/{action_id}", response_model=AgentActionOut)
def get_agent_action(
    action_id: UUID,
    _actor: str = Depends(require_agent_auth),
) -> AgentActionOut:
    return AgentActionOut.model_validate(get_agent_action_record(action_id=action_id))


@router.get("/jobs/{job_id}/feedback", response_model=AgentFeedbackOut)
def get_agent_job_feedback(
    job_id: UUID,
    _actor: str = Depends(require_agent_auth),
) -> AgentFeedbackOut:
    rows = list_evaluation_artifacts_for_job(
        job_id=job_id,
        visibility="agent_visible",
    )
    return AgentFeedbackOut.model_validate(
        agent_feedback_payload(resource_type="job", resource_id=str(job_id), rows=rows)
    )


@router.get("/commits/{commit_hash}/feedback", response_model=AgentFeedbackOut)
def get_agent_commit_feedback(
    commit_hash: str,
    _actor: str = Depends(require_agent_auth),
) -> AgentFeedbackOut:
    rows = list_evaluation_artifacts_for_commit(
        commit_hash=commit_hash,
        visibility="agent_visible",
    )
    return AgentFeedbackOut.model_validate(
        agent_feedback_payload(
            resource_type="commit",
            resource_id=str(commit_hash or "").strip(),
            rows=rows,
        )
    )
