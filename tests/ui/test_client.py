from __future__ import annotations

import httpx
import pytest

from loreley.ui.client import APIError, LoreleyAPIClient


def test_ui_client_get_json_uses_base_url_and_params() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.scheme == "http"
        assert request.url.host == "example.local"
        assert request.url.path == "/api/v1/health"
        assert dict(request.url.params) == {"q": "1"}
        return httpx.Response(200, json={"status": "ok"}, request=request)

    transport = httpx.MockTransport(handler)
    client = LoreleyAPIClient("http://example.local", transport=transport)
    payload = client.get_json("/api/v1/health", params={"q": 1})
    assert payload == {"status": "ok"}


def test_ui_client_maps_http_errors_to_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"nope", request=request)

    transport = httpx.MockTransport(handler)
    client = LoreleyAPIClient("http://example.local", transport=transport)
    with pytest.raises(APIError) as excinfo:
        client.get_json("/missing")
    assert excinfo.value.status_code == 404
    assert "nope" in str(excinfo.value)


def test_ui_client_get_bytes_returns_content_and_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"payload",
            headers={"Content-Type": "application/octet-stream"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = LoreleyAPIClient("http://example.local", transport=transport)
    body, content_type = client.get_bytes("/artifact")
    assert body == b"payload"
    assert content_type == "application/octet-stream"


def test_ui_client_post_json_sends_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/jobs/retry-failed-stale"
        assert request.read() == b'{"limit":1}'
        return httpx.Response(200, json={"count": 1}, request=request)

    transport = httpx.MockTransport(handler)
    client = LoreleyAPIClient("http://example.local", transport=transport)
    payload = client.post_json("/api/v1/jobs/retry-failed-stale", json_body={"limit": 1})
    assert payload == {"count": 1}


def test_ui_client_get_json_page_validates_paginated_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"items": [{"id": 1}], "next_cursor": "abc"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = LoreleyAPIClient("http://example.local", transport=transport)
    payload = client.get_json_page("/api/v1/jobs/page")
    assert payload == {"items": [{"id": 1}], "next_cursor": "abc"}


def test_ui_client_get_json_page_rejects_non_list_items() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"items": {"id": 1}, "next_cursor": None},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = LoreleyAPIClient("http://example.local", transport=transport)
    with pytest.raises(APIError, match="`items` must be a list"):
        client.get_json_page("/api/v1/jobs/page")
