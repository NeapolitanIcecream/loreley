from __future__ import annotations

import importlib
import sys

import pytest


class _FakeStreamlitModule:
    def __init__(self) -> None:
        self.button_calls: list[dict[str, object]] = []
        self.button_values: dict[str, bool] = {}
        self.download_calls: list[dict[str, object]] = []
        self.write_calls: list[str] = []
        self.markdown_calls: list[str] = []
        self.caption_calls: list[str] = []
        self.text_area_calls: list[dict[str, object]] = []
        self.error_calls: list[str] = []
        self.session_state: dict[str, object] = {}

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

    def button(self, label: str, **kwargs) -> bool:
        payload = {"label": label, "kwargs": kwargs}
        self.button_calls.append(payload)
        key = str(kwargs.get("key") or "")
        return bool(self.button_values.get(key, False))

    def download_button(self, label: str, **kwargs) -> None:
        self.download_calls.append({"label": label, "kwargs": kwargs})

    def write(self, value: str) -> None:
        self.write_calls.append(value)

    def markdown(self, value: str) -> None:
        self.markdown_calls.append(value)

    def caption(self, value: str) -> None:
        self.caption_calls.append(value)

    def text_area(self, label: str, **kwargs) -> None:
        self.text_area_calls.append({"label": label, "kwargs": kwargs})

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


def test_render_artifact_downloads_defers_fetch_until_prepare_clicked(ui_api_module) -> None:
    module, fake_streamlit = ui_api_module
    fetch_calls: list[tuple[str, str]] = []

    def _fake_fetch(base_url: str, path: str, *, params=None):
        fetch_calls.append((base_url, path))
        return b"payload", "application/json"

    module.api_get_bytes_or_stop = _fake_fetch

    module.render_artifact_downloads(
        api_base_url="http://example.local/root/",
        artifacts={
            "planning_prompt_url": "/api/v1/jobs/123/artifacts/planning_prompt",
        },
        key_prefix="artifact_test",
    )

    assert fetch_calls == []
    assert fake_streamlit.download_calls == []
    assert [call["label"] for call in fake_streamlit.button_calls] == [
        "Prepare: Planning prompt",
    ]


def test_render_artifact_downloads_fetches_server_side_after_prepare(ui_api_module) -> None:
    module, fake_streamlit = ui_api_module
    fetch_calls: list[tuple[str, str]] = []
    fake_streamlit.button_values["artifact_test_planning_prompt_prepare"] = True

    def _fake_fetch(base_url: str, path: str, *, params=None):
        fetch_calls.append((base_url, path))
        return b"payload", "application/json"

    module.api_get_bytes_or_stop = _fake_fetch

    module.render_artifact_downloads(
        api_base_url="http://example.local/root/",
        artifacts={
            "planning_prompt_url": "/api/v1/jobs/123/artifacts/planning_prompt",
        },
        key_prefix="artifact_test",
    )

    assert fetch_calls == [
        ("http://example.local/root/", "/api/v1/jobs/123/artifacts/planning_prompt"),
    ]
    assert len(fake_streamlit.download_calls) == 1
    assert fake_streamlit.download_calls[0]["label"] == "Download: Planning prompt"
    assert fake_streamlit.download_calls[0]["kwargs"]["data"] == b"payload"
    assert fake_streamlit.download_calls[0]["kwargs"]["mime"] == "application/json"


def test_render_evaluation_evidence_defers_fetch_until_prepare_clicked(ui_api_module) -> None:
    module, fake_streamlit = ui_api_module
    fetch_calls: list[tuple[str, str]] = []

    def _fake_fetch(base_url: str, path: str, *, params=None):
        fetch_calls.append((base_url, path))
        return b"payload", "application/json"

    module.api_get_bytes_or_stop = _fake_fetch

    module.render_evaluation_evidence(
        api_base_url="http://example.local/root/",
        artifacts=[
            {
                "key": "benchmark_report",
                "kind": "benchmark_json",
                "mime_type": "application/json",
                "summary": "Parser throughput improved.",
                "visibility": "agent_visible",
                "download_url": "/api/v1/jobs/123/evaluation-artifacts/benchmark_report",
            }
        ],
        key_prefix="evidence_test",
    )

    assert fetch_calls == []
    assert fake_streamlit.download_calls == []
    assert fake_streamlit.markdown_calls == ["**benchmark_report**  `benchmark_report`"]
    assert [call["label"] for call in fake_streamlit.button_calls] == [
        "Prepare: benchmark_report",
    ]


def test_render_evaluation_evidence_uses_unique_widget_keys_for_duplicate_artifact_keys(
    ui_api_module,
) -> None:
    module, fake_streamlit = ui_api_module

    module.render_evaluation_evidence(
        api_base_url="http://example.local/root/",
        artifacts=[
            {
                "key": "benchmark_report",
                "kind": "benchmark_json",
                "mime_type": "application/json",
                "download_url": "/api/v1/jobs/1/evaluation-artifacts/benchmark_report",
            },
            {
                "key": "benchmark_report",
                "kind": "benchmark_json",
                "mime_type": "application/json",
                "download_url": "/api/v1/jobs/2/evaluation-artifacts/benchmark_report",
            },
        ],
        key_prefix="evidence_test",
    )

    button_keys = [call["kwargs"]["key"] for call in fake_streamlit.button_calls]
    assert button_keys == [
        "evidence_test_0_benchmark_report_prepare",
        "evidence_test_1_benchmark_report_prepare",
    ]


def test_render_agent_feedback_preview_renders_bounded_text(ui_api_module) -> None:
    module, fake_streamlit = ui_api_module

    module.render_agent_feedback_preview(
        {
            "mode": "summary",
            "budget_chars": 2000,
            "text": "Evaluation Evidence:\n- `bench`: summary",
            "included_artifact_keys": ["bench"],
            "omitted_artifact_count": 0,
        }
    )

    assert fake_streamlit.text_area_calls[0]["label"] == "Agent feedback preview"
    assert "Evaluation Evidence" in fake_streamlit.text_area_calls[0]["kwargs"]["value"]
