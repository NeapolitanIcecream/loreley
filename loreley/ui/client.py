"""Minimal HTTP client for calling the Loreley UI API.

We intentionally avoid extra HTTP dependencies to keep the UI extras small.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


class APIError(RuntimeError):
    """Raised when the UI API returns an error or cannot be reached."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = str(message)
        self.status_code = int(status_code) if status_code is not None else None

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.status_code is None:
            return self.message
        return f"{self.message} (status={self.status_code})"


class LoreleyAPIClient:
    """Small JSON client for the read-only UI API."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_seconds = float(timeout_seconds)

    def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """GET a JSON response from the API."""

        relative = path.lstrip("/")
        url = urljoin(self.base_url, relative)
        if params:
            url = f"{url}?{urlencode(params)}"

        req = Request(url, method="GET", headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                status = getattr(resp, "status", 200)
                body = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - network dependent
            raise APIError(f"Failed to call API: {exc}") from exc

        if status >= 400:
            raise APIError(body or "API request failed", status_code=int(status))

        try:
            return json.loads(body) if body else None
        except json.JSONDecodeError as exc:
            raise APIError(f"Invalid JSON response: {exc}", status_code=int(status)) from exc

    def get_bytes(self, path: str, *, params: dict[str, Any] | None = None) -> tuple[bytes, str | None]:
        """GET raw bytes from the API (used for downloading artifacts)."""

        relative = path.lstrip("/")
        url = urljoin(self.base_url, relative)
        if params:
            url = f"{url}?{urlencode(params)}"

        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                status = getattr(resp, "status", 200)
                content_type = None
                try:
                    content_type = resp.headers.get("Content-Type")
                except Exception:
                    content_type = None
                body = resp.read()
        except Exception as exc:  # pragma: no cover - network dependent
            raise APIError(f"Failed to call API: {exc}") from exc

        if status >= 400:
            try:
                message = body.decode("utf-8", errors="replace")
            except Exception:
                message = "API request failed"
            raise APIError(message or "API request failed", status_code=int(status))

        return body, content_type


