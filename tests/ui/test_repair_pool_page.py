from __future__ import annotations

import importlib
import sys

import pytest


class _FakeStreamlitModule:
    session_state: dict[str, object] = {}
    buttons: list[dict[str, object]] = []
    captions: list[str] = []
    infos: list[str] = []
    selectbox_values: dict[str, object] = {}
    text_input_values: dict[str, str] = {}

    @classmethod
    def reset(cls) -> None:
        cls.session_state = {}
        cls.buttons = []
        cls.captions = []
        cls.infos = []
        cls.selectbox_values = {}
        cls.text_input_values = {}

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
    def error(cls, _value: str) -> None:
        return None

    @classmethod
    def info(cls, value: str) -> None:
        cls.infos.append(value)

    @classmethod
    def caption(cls, value: str) -> None:
        cls.captions.append(value)

    @classmethod
    def metric(cls, *_args, **_kwargs) -> None:
        return None

    @classmethod
    def selectbox(cls, label: str, options, index: int = 0, **_kwargs):
        return cls.selectbox_values.get(label, options[index])

    @classmethod
    def text_input(cls, label: str, value: str = "", **_kwargs) -> str:
        return cls.text_input_values.get(label, value)

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
def repair_pool_module(monkeypatch: pytest.MonkeyPatch):
    _FakeStreamlitModule.reset()
    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlitModule())
    sys.modules.pop("loreley.ui.components.api", None)
    sys.modules.pop("loreley.ui.components.aggrid", None)
    sys.modules.pop("loreley.ui.pages.repair_pool", None)
    module = importlib.import_module("loreley.ui.pages.repair_pool")
    yield module
    sys.modules.pop("loreley.ui.pages.repair_pool", None)


def test_repair_pool_keeps_pager_available_when_current_page_is_empty(
    repair_pool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_base_url = "http://api.local"
    _FakeStreamlitModule.session_state[repair_pool_module.API_BASE_URL_KEY] = api_base_url
    _FakeStreamlitModule.selectbox_values = {
        "Page size": 50,
        "Repair state": "all",
        "Lifecycle": "all",
    }
    signature = repair_pool_module.pager_signature((api_base_url, 50, "all", "all", "", ""))
    _FakeStreamlitModule.session_state[repair_pool_module._REPAIR_CURSOR_SIGNATURE_KEY] = signature  # noqa: SLF001
    _FakeStreamlitModule.session_state[repair_pool_module._REPAIR_CURSOR_KEY] = [None, "cursor-2"]  # noqa: SLF001
    _FakeStreamlitModule.session_state[repair_pool_module._REPAIR_CURSOR_INDEX_KEY] = 1  # noqa: SLF001
    monkeypatch.setattr(
        repair_pool_module,
        "api_get_page_or_stop",
        lambda *_args, **_kwargs: {
            "items": [],
            "next_cursor": None,
            "summary": {},
        },
    )

    repair_pool_module.render()

    assert "page=2 items=0" in _FakeStreamlitModule.captions
    assert "No failed candidates match these filters." in _FakeStreamlitModule.infos
    assert {
        "label": "Previous page",
        "key": "repair_prev_page",
        "disabled": False,
    } in _FakeStreamlitModule.buttons
