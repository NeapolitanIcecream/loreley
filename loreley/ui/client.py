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

    def post_json(self, path: str, *, json_body: Any | None = None) -> Any:
        """POST JSON to the API and return a JSON response."""
        try:
            response = self._http.request(
                "POST",
                path,
                headers={"Accept": "application/json"},
                json_body=json_body if json_body is not None else {},
            )
        except HttpCallError as exc:  # pragma: no cover - network dependent
            raise APIError(exc.message, status_code=exc.status_code) from exc
        if not response.content:
            return None
        try:
            import json

            return json.loads(response.content.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise APIError(
                f"Invalid JSON response: {exc}",
                status_code=int(response.status_code),
            ) from exc

    def get_json_page(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """GET a paginated JSON payload with `items` and `next_cursor` keys."""

        payload = self.get_json(path, params=params)
        if not isinstance(payload, dict):
            raise APIError("Invalid paginated API response: expected an object.")
        items = payload.get("items")
        if items is None:
            payload["items"] = []
        elif not isinstance(items, list):
            raise APIError("Invalid paginated API response: `items` must be a list.")
        next_cursor = payload.get("next_cursor")
        if next_cursor is not None and not isinstance(next_cursor, str):
            raise APIError("Invalid paginated API response: `next_cursor` must be a string or null.")
        return payload

    def get_bytes(self, path: str, *, params: dict[str, Any] | None = None) -> tuple[bytes, str | None]:
        """GET raw bytes from the API (used for downloading artifacts)."""
        try:
            return self._http.get_bytes(path, params=params)
        except HttpCallError as exc:  # pragma: no cover - network dependent
            raise APIError(exc.message, status_code=exc.status_code) from exc
