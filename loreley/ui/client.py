"""Minimal HTTP client for calling the Loreley UI API.

This module delegates HTTP calls to `loreley.net.http` so that timeout,
redirect, and error mapping behavior remains consistent across call sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loreley.net.http import HttpCallError, HttpClient

if TYPE_CHECKING:
    import httpx


class APIError(HttpCallError):
    """Raised when the UI API returns an error or cannot be reached."""


class LoreleyAPIClient:
    """Small JSON client for the read-only UI API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        transport: "httpx.BaseTransport | None" = None,
        reuse_connections: bool = False,
    ) -> None:
        base_url = (base_url or "").strip()
        if not base_url:
            raise APIError("Invalid API base URL: value is empty.")
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = float(timeout_seconds)
        self._http = HttpClient(
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
            user_agent="loreley-ui",
            transport=transport,
            reuse_connections=bool(reuse_connections),
        )

    def close(self) -> None:
        """Close any underlying persistent HTTP resources."""
        self._http.close()

    def __enter__(self) -> "LoreleyAPIClient":
        self._http.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """GET a JSON response from the API."""
        try:
            return self._http.get_json(path, params=params)
        except HttpCallError as exc:  # pragma: no cover - network dependent
            raise APIError(exc.message, status_code=exc.status_code) from exc

    def get_bytes(self, path: str, *, params: dict[str, Any] | None = None) -> tuple[bytes, str | None]:
        """GET raw bytes from the API (used for downloading artifacts)."""
        try:
            return self._http.get_bytes(path, params=params)
        except HttpCallError as exc:  # pragma: no cover - network dependent
            raise APIError(exc.message, status_code=exc.status_code) from exc


