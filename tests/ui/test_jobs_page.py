from __future__ import annotations

import importlib
import sys

import pandas as pd
import pytest


class _FakeStreamlitModule:
    session_state: dict[str, object] = {}
    buttons: list[dict[str, object]] = []
    captions: list[str] = []
    infos: list[str] = []
    selectbox_values: dict[str, object] = {}
    checkbox_values: dict[str, bool] = {}

    @classmethod
    def reset(cls) -> None:
        cls.session_state = {}
        cls.buttons = []
        cls.captions = []
        cls.infos = []
        cls.selectbox_values = {}
        cls.checkbox_values = {}

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

    @classmethod
    def title(cls, _value: str) -> None:
        return None

    @classmethod
    def subheader(cls, _value: str) -> None:
        return None

    @classmethod
    def error(cls, _value: str) -> None:
        return None

    @classmethod
    def info(cls, value: str) -> None:
        cls.infos.append(value)

    @classmethod
    def caption(cls, value: str) -> None:
        cls.captions.append(value)

    @classmethod
    def selectbox(cls, label: str, options, index: int = 0, **_kwargs):
        return cls.selectbox_values.get(label, options[index])

    @classmethod
    def checkbox(cls, _label: str, *, key: str, **_kwargs) -> bool:
        return cls.checkbox_values.get(key, False)

    @classmethod
    def button(cls, label: str, *, key: str, disabled: bool = False, **_kwargs) -> bool:
        cls.buttons.append({"label": label, "key": key, "disabled": disabled})
        return False

    @classmethod
    def columns(cls, count: int):
        return [cls() for _ in range(count)]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.fixture
def jobs_module(monkeypatch: pytest.MonkeyPatch):
    _FakeStreamlitModule.reset()
    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlitModule())
    sys.modules.pop("loreley.ui.components.api", None)
    sys.modules.pop("loreley.ui.components.aggrid", None)
    sys.modules.pop("loreley.ui.pages.jobs", None)
    module = importlib.import_module("loreley.ui.pages.jobs")
    yield module
    sys.modules.pop("loreley.ui.pages.jobs", None)


def test_jobs_fate_unknown_filter_includes_missing_and_empty_labels(jobs_module) -> None:
    df = pd.DataFrame(
        [
            {"id": "missing", "candidate_fate_label": None},
            {"id": "empty", "candidate_fate_label": ""},
            {"id": "blank", "candidate_fate_label": "   "},
            {"id": "known", "candidate_fate_label": "elite_inserted"},
        ]
    )

    filtered = jobs_module._filter_jobs_df(  # noqa: SLF001
        df,
        fate_filter="unknown",
        evidence_filter="all",
    )

    assert filtered["id"].tolist() == ["missing", "empty", "blank"]


def test_jobs_decoration_normalizes_missing_fate_to_unknown(jobs_module) -> None:
    df = pd.DataFrame(
        [
            {"id": "missing", "candidate_fate_label": None},
            {"id": "empty", "candidate_fate_label": ""},
            {"id": "known", "candidate_fate_label": "elite_inserted"},
        ]
    )

    decorated = jobs_module._decorate_jobs_df(df)  # noqa: SLF001

    assert decorated["fate"].tolist() == ["unknown", "unknown", "elite_inserted"]


def test_jobs_global_retry_payload_is_not_tied_to_page_size(jobs_module) -> None:
    assert jobs_module._retry_failed_stale_payload() == {"all": True}  # noqa: SLF001


def test_jobs_render_keeps_pager_available_when_client_filter_empties_page(
    jobs_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeStreamlitModule.session_state[jobs_module.API_BASE_URL_KEY] = "http://api.local"
    _FakeStreamlitModule.selectbox_values = {
        "Page size": 50,
        "Status filter": "all",
        "Job kind": "all",
        "Fate": "unknown",
        "Evidence": "all",
    }
    monkeypatch.setattr(
        jobs_module,
        "api_get_page_or_stop",
        lambda *_args, **_kwargs: {
            "items": [{"id": "job-1", "candidate_fate_label": "elite_inserted"}],
            "next_cursor": "next-page",
        },
    )

    jobs_module.render()

    assert "page=1 items=0" in _FakeStreamlitModule.captions
    assert "No jobs found." in _FakeStreamlitModule.infos
    assert {
        "label": "Next page",
        "key": "jobs_next_page",
        "disabled": False,
    } in _FakeStreamlitModule.buttons
