"""Shared HTTP helpers built on top of httpx.

This module centralizes default timeout/redirect behavior and provides a small
sync wrapper that maps httpx exceptions into Loreley-friendly errors.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import httpx

__all__ = ["HttpCallError", "HttpClient"]


class HttpCallError(RuntimeError):
    """Raised when an HTTP request fails or returns a non-success response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = str(message)
        self.status_code = int(status_code) if status_code is not None else None

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.status_code is None:
            return self.message
        return f"{self.message} (status={self.status_code})"


def _safe_response_text(response: httpx.Response) -> str:
    """Best-effort extraction of response text for error messages."""
    try:
        return (response.text or "").strip()
    except Exception:
        try:
            return response.content.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""


class HttpClient:
    """Small sync HTTP client with consistent defaults and error mapping."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 10.0,
        follow_redirects: bool = True,
        user_agent: str | None = None,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/") + "/" if base_url else None
        self.timeout_seconds = float(timeout_seconds)
        self.follow_redirects = bool(follow_redirects)
        self.transport = transport

        merged: dict[str, str] = dict(headers or {})
        if user_agent and "User-Agent" not in merged:
            merged["User-Agent"] = user_agent
        self.headers = merged

    def _build_client(self, *, timeout_seconds: float | None = None) -> httpx.Client:
        timeout = float(max(0.1, timeout_seconds if timeout_seconds is not None else self.timeout_seconds))
        kwargs: dict[str, object] = {
            "timeout": timeout,
            "follow_redirects": self.follow_redirects,
            "headers": self.headers,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.transport is not None:
            kwargs["transport"] = self.transport
        return httpx.Client(**kwargs)  # type: ignore[arg-type]

    def request(
        self,
        method: str,
        url_or_path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        content: bytes | None = None,
        json_body: Any | None = None,
    ) -> httpx.Response:
        """Send an HTTP request and return the response.

        Raises:
            HttpCallError: When the request fails or returns a 4xx/5xx response.
        """
        method = (method or "GET").strip().upper()
        target = (url_or_path or "").strip()
        if not target:
            raise ValueError("url_or_path must be non-empty.")

        try:
            with self._build_client() as client:
                response = client.request(
                    method,
                    target,
                    params=dict(params) if params else None,
                    headers=dict(headers) if headers else None,
                    content=content,
                    json=json_body,
                )
                response.raise_for_status()
                return response
        except httpx.HTTPStatusError as exc:
            message = _safe_response_text(exc.response) or "HTTP request failed"
            raise HttpCallError(message, status_code=int(exc.response.status_code)) from exc
        except httpx.RequestError as exc:
            raise HttpCallError(f"HTTP request failed: {exc}") from exc

    def get_json(
        self,
        url_or_path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """GET a JSON response, returning parsed payload (or None on empty body)."""
        merged_headers: dict[str, str] = {"Accept": "application/json"}
        if headers:
            merged_headers.update(dict(headers))
        response = self.request("GET", url_or_path, params=params, headers=merged_headers)
        if not response.content:
            return None
        try:
            text = response.content.decode("utf-8", errors="replace")
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise HttpCallError(
                f"Invalid JSON response: {exc}",
                status_code=int(response.status_code),
            ) from exc

    def get_bytes(
        self,
        url_or_path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[bytes, str | None]:
        """GET raw bytes, returning (content, content_type)."""
        response = self.request("GET", url_or_path, params=params, headers=headers)
        content_type = response.headers.get("Content-Type")
        return response.content, content_type

    def is_reachable(
        self,
        url_or_path: str,
        *,
        timeout_seconds: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bool:
        """Return True when a GET request returns a 2xx response."""
        target = (url_or_path or "").strip()
        if not target:
            return False
        try:
            with self._build_client(timeout_seconds=timeout_seconds) as client:
                response = client.get(target, headers=dict(headers) if headers else None)
                return 200 <= int(response.status_code) < 300
        except httpx.HTTPError:
            return False

