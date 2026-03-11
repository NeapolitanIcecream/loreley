from __future__ import annotations

import importlib
import sys
import types

import pytest


class _FakeStreamlitModule:
    def __init__(self) -> None:
        self.link_calls: list[dict[str, object]] = []
        self.markdown_calls: list[str] = []
        self.write_calls: list[str] = []
        self.error_calls: list[str] = []

    @staticmethod
    def cache_resource(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    @staticmethod
    def cache_data(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    def link_button(self, label: str, url: str, **kwargs) -> None:
        self.link_calls.append({"label": label, "url": url, "kwargs": kwargs})

    def markdown(self, value: str) -> None:
        self.markdown_calls.append(value)

    def write(self, value: str) -> None:
        self.write_calls.append(value)

    def error(self, value: str) -> None:
        self.error_calls.append(value)

    @staticmethod
    def stop() -> None:
        raise RuntimeError("st.stop")


@pytest.fixture
def ui_api_module(monkeypatch: pytest.MonkeyPatch):
    fake_streamlit = _FakeStreamlitModule()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    sys.modules.pop("loreley.ui.components.api", None)
    module = importlib.import_module("loreley.ui.components.api")
    yield module, fake_streamlit
    sys.modules.pop("loreley.ui.components.api", None)


def test_api_get_all_pages_without_max_items_follows_cursor(ui_api_module) -> None:
    module, _fake_streamlit = ui_api_module
    calls: list[dict[str, object]] = []

    def _fake_page(base_url: str, path: str, *, params: dict[str, object] | None = None) -> dict[str, object]:
        current = dict(params or {})
        calls.append(current)
        cursor = current.get("cursor")
        if cursor is None:
            return {"items": [{"id": 1}], "next_cursor": "page-2"}
        if cursor == "page-2":
            return {"items": [{"id": 2}], "next_cursor": None}
        raise AssertionError(f"unexpected cursor: {cursor}")

    module.api_get_page_or_stop = _fake_page

    items = module.api_get_all_pages_or_stop(
        "http://example.local",
        "/api/v1/archive/records/page",
        page_limit=100,
        max_items=None,
    )

    assert items == [{"id": 1}, {"id": 2}]
    assert calls == [{"limit": 100}, {"limit": 100, "cursor": "page-2"}]


def test_render_artifact_downloads_uses_links_without_prefetch(ui_api_module) -> None:
    module, fake_streamlit = ui_api_module

    def _unexpected_fetch(*_args, **_kwargs):
        raise AssertionError("artifact bytes should not be fetched while rendering links")

    module.api_get_bytes_or_stop = _unexpected_fetch

    module.render_artifact_downloads(
        api_base_url="http://example.local/root/",
        artifacts={
            "planning_prompt_url": "/api/v1/jobs/123/artifacts/planning_prompt",
            "evaluation_json_url": "/api/v1/jobs/123/artifacts/evaluation_json",
        },
        key_prefix="artifact_test",
    )

    assert [call["label"] for call in fake_streamlit.link_calls] == [
        "Open: Planning prompt",
        "Open: Evaluation JSON",
    ]
    assert [call["url"] for call in fake_streamlit.link_calls] == [
        "http://example.local/api/v1/jobs/123/artifacts/planning_prompt",
        "http://example.local/api/v1/jobs/123/artifacts/evaluation_json",
    ]

