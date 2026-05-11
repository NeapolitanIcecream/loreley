"""Authentication helpers for mutating UI API routes."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Header, HTTPException
from loguru import logger

from loreley.config import get_settings

log = logger.bind(module="api.auth")


def require_write_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Require the configured UI API write bearer token."""

    token = str(get_settings().loreley_api_write_token or "").strip()
    if not token:
        _log_auth_rejected(reason="token_unconfigured", status_code=503)
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "write_auth_not_configured",
                "message": "LORELEY_API_WRITE_TOKEN is required for UI API write routes.",
                "retryable": False,
                "resource": {"type": "api_write_auth", "id": "configuration"},
            },
        )

    raw = str(authorization or "").strip()
    if not raw:
        _log_auth_rejected(reason="missing_authorization", status_code=401)
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": "unauthorized",
                "message": "Missing Authorization bearer token.",
                "retryable": False,
                "resource": {"type": "api_write_auth", "id": "authorization"},
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, separator, supplied = raw.partition(" ")
    if not separator or scheme.lower() != "bearer":
        _log_auth_rejected(reason="invalid_scheme", status_code=401)
        raise HTTPException(
            status_code=401,
            detail={
                "error_code": "unauthorized",
                "message": "Authorization must use Bearer token credentials.",
                "retryable": False,
                "resource": {"type": "api_write_auth", "id": "authorization"},
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(supplied.strip(), token):
        _log_auth_rejected(reason="invalid_token", status_code=403)
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "forbidden",
                "message": "Authorization bearer token is invalid.",
                "retryable": False,
                "resource": {"type": "api_write_auth", "id": "authorization"},
            },
        )
    return "api-write-token"


def _log_auth_rejected(*, reason: str, status_code: int) -> None:
    log.bind(
        auth_scope="ui_api_write",
        reason=reason,
        status_code=int(status_code),
    ).warning("UI API write request rejected by auth")
