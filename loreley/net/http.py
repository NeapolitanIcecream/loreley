"""Shared HTTP helpers built on top of httpx.

This module centralizes default timeout/redirect behavior and provides a small
sync wrapper that maps httpx exceptions into Loreley-friendly errors.
"""

from __future__ import annotations

import json
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Mapping

import httpx

__all__ = ["HttpCallError", "HttpClient"]

_MIN_TIMEOUT_SECONDS = 0.1
_MAX_ERROR_TEXT_CHARS = 2048


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


def _truncate(text: str, *, limit: int) -> str:
    """Return a truncated string with an ellipsis when needed."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = text[: max(0, limit - 3)].rstrip()
    return f"{head}..."


def _safe_response_text(response: httpx.Response) -> str:
    """Best-effort extraction of response text for error messages.

    The returned value is trimmed and truncated to keep logs and UI readable.
    """
    try:
        text = (response.text or "").strip()
    except Exception:
        try:
            text = response.content.decode("utf-8", errors="replace").strip()
        except Exception:
            text = ""
    return _truncate(text, limit=_MAX_ERROR_TEXT_CHARS)


class HttpClient:
    """Small sync HTTP client with consistent defaults and error mapping.

    Notes:
        - By default, a short-lived `httpx.Client` is created per request.
        - When `reuse_connections=True`, an internal persistent `httpx.Client` is
          used to enable connection pooling. Call `close()` (or use this object
          as a context manager) to release resources deterministically.
        - Timeouts are clamped to at least `_MIN_TIMEOUT_SECONDS`.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float = 10.0,
        follow_redirects: bool = True,
        user_agent: str | None = None,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        reuse_connections: bool = False,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/") + "/" if base_url else None
        self.timeout_seconds = float(timeout_seconds)
        self.follow_redirects = bool(follow_redirects)
        self.transport = transport
        self.reuse_connections = bool(reuse_connections)

        merged: dict[str, str] = dict(headers or {})
        if user_agent and "User-Agent" not in merged:
            merged["User-Agent"] = user_agent
        self.headers = merged
        self._client: httpx.Client | None = None
        self._finalizer: weakref.finalize | None = None

    def _build_client(self, *, timeout_seconds: float | None = None) -> httpx.Client:
        timeout = float(
            max(
                _MIN_TIMEOUT_SECONDS,
                timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
            )
        )
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

    def open(self) -> None:
        """Open an internal persistent `httpx.Client` when reuse is enabled."""
        if not self.reuse_connections:
            return
        if self._client is not None:
            return
        self._client = self._build_client()
        # Ensure we don't leak open pools if callers forget to close explicitly.
        self._finalizer = weakref.finalize(self, self._client.close)

    def close(self) -> None:
        """Close any internal persistent `httpx.Client`."""
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._client is None:
            return
        self._client.close()
        self._client = None

    def __enter__(self) -> "HttpClient":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @contextmanager
    def _client_ctx(self) -> Iterator[httpx.Client]:
        if self.reuse_connections:
            self.open()
            assert self._client is not None
            yield self._client
            return
        with self._build_client() as client:
            yield client

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
            with self._client_ctx() as client:
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
            with self._client_ctx() as client:
                if timeout_seconds is None:
                    response = client.get(target, headers=dict(headers) if headers else None)
                else:
                    timeout = float(max(_MIN_TIMEOUT_SECONDS, timeout_seconds))
                    response = client.get(
                        target,
                        headers=dict(headers) if headers else None,
                        timeout=timeout,
                    )
                return 200 <= int(response.status_code) < 300
        except httpx.HTTPError:
            return False

