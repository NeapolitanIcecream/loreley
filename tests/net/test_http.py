from __future__ import annotations

import httpx
import pytest

from loreley.net.http import HttpCallError, HttpClient


def test_get_json_encodes_params_and_parses_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.scheme == "http"
        assert request.url.host == "example.local"
        assert request.url.path == "/api/v1/data"
        assert dict(request.url.params) == {"x": "1", "y": "two"}
        assert request.headers.get("accept") == "application/json"
        assert request.headers.get("user-agent") == "test-agent"
        return httpx.Response(200, json={"ok": True}, request=request)

    transport = httpx.MockTransport(handler)
    client = HttpClient(
        base_url="http://example.local",
        timeout_seconds=1.0,
        follow_redirects=True,
        user_agent="test-agent",
        transport=transport,
    )
    payload = client.get_json("/api/v1/data", params={"x": 1, "y": "two"})
    assert payload == {"ok": True}


def test_get_json_returns_none_on_empty_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"", request=request)

    transport = httpx.MockTransport(handler)
    client = HttpClient(base_url="http://example.local", transport=transport)
    assert client.get_json("/empty") is None


def test_get_json_raises_on_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{", request=request)

    transport = httpx.MockTransport(handler)
    client = HttpClient(base_url="http://example.local", transport=transport)
    with pytest.raises(HttpCallError, match="Invalid JSON response"):
        client.get_json("/invalid-json")


def test_request_raises_with_status_code_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"not found", request=request)

    transport = httpx.MockTransport(handler)
    client = HttpClient(base_url="http://example.local", transport=transport)
    with pytest.raises(HttpCallError) as excinfo:
        client.get_json("/missing")
    assert excinfo.value.status_code == 404
    assert "not found" in str(excinfo.value)


def test_get_bytes_returns_content_and_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"abc",
            headers={"Content-Type": "application/octet-stream"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    client = HttpClient(base_url="http://example.local", transport=transport)
    body, content_type = client.get_bytes("/artifact")
    assert body == b"abc"
    assert content_type == "application/octet-stream"


def test_is_reachable_true_for_2xx_false_for_non_2xx_and_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ok":
            return httpx.Response(204, request=request)
        if request.url.path == "/bad":
            return httpx.Response(500, request=request)
        raise httpx.ConnectError("boom", request=request)

    transport = httpx.MockTransport(handler)
    client = HttpClient(base_url="http://example.local", transport=transport)
    assert client.is_reachable("/ok") is True
    assert client.is_reachable("/bad") is False
    assert client.is_reachable("/error") is False

