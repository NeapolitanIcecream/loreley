"""Structured errors for the agent REST facade."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AgentAPIError(Exception):
    """Exception rendered as the agent facade structured error shape."""

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        message: str,
        retryable: bool = False,
        resource: dict[str, str] | None = None,
        suggested_next_actions: list[dict[str, Any]] | None = None,
        preconditions: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.error_code = str(error_code)
        self.message = str(message)
        self.retryable = bool(retryable)
        self.resource = resource
        self.suggested_next_actions = suggested_next_actions or []
        self.preconditions = preconditions or []

    def payload(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
            "resource": self.resource,
            "suggested_next_actions": self.suggested_next_actions,
        }


async def agent_api_error_handler(_request: Request, exc: AgentAPIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.payload())


async def agent_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Use structured request-validation errors only for agent routes."""

    if request.url.path.startswith("/api/v1/agent"):
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "invalid_request",
                "message": "Agent request validation failed.",
                "retryable": False,
                "resource": None,
                "suggested_next_actions": [],
            },
        )
    return await request_validation_exception_handler(request, exc)
