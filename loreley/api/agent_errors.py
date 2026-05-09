"""Structured errors for the agent REST facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@dataclass(slots=True)
class AgentAPIError(Exception):
    """Exception rendered as the agent facade structured error shape."""

    status_code: int
    error_code: str
    message: str
    retryable: bool = False
    resource: dict[str, str] | None = None
    suggested_next_actions: list[dict[str, Any]] = field(default_factory=list)
    preconditions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)
        self.status_code = int(self.status_code)
        self.error_code = str(self.error_code)
        self.message = str(self.message)
        self.retryable = bool(self.retryable)

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
