from __future__ import annotations

import importlib
import sys

import pytest


class _FakeCacheData:
    def __call__(self, *_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    @staticmethod
    def clear() -> None:
        return None


class _FakeStreamlitModule:
    cache_data = _FakeCacheData()
    session_state: dict[str, object] = {}
    buttons: list[dict[str, object]] = []
    infos: list[str] = []
    successes: list[str] = []
    checkbox_values: dict[str, bool] = {}
    button_clicks: dict[str, bool] = {}

    @classmethod
    def reset(cls) -> None:
        cls.session_state = {}
        cls.buttons = []
        cls.infos = []
        cls.successes = []
        cls.checkbox_values = {}
        cls.button_clicks = {}

    @staticmethod
    def cache_resource(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    @classmethod
    def success(cls, value: str) -> None:
        cls.successes.append(value)

    @classmethod
    def subheader(cls, _value: str) -> None:
        return None

    @classmethod
    def info(cls, value: str) -> None:
        cls.infos.append(value)

    @classmethod
    def checkbox(cls, _label: str, *, key: str, **_kwargs) -> bool:
        return cls.checkbox_values.get(key, False)

    @classmethod
    def button(cls, label: str, *, key: str, disabled: bool = False, **_kwargs) -> bool:
        cls.buttons.append({"label": label, "key": key, "disabled": disabled})
        return bool(cls.button_clicks.get(key, False)) and not disabled

    @classmethod
    def columns(cls, count: int):
        return [cls() for _ in range(count)]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.fixture
def campaign_module(monkeypatch: pytest.MonkeyPatch):
    _FakeStreamlitModule.reset()
    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlitModule())
    sys.modules.pop("loreley.ui.components.api", None)
    sys.modules.pop("loreley.ui.pages.campaign", None)
    module = importlib.import_module("loreley.ui.pages.campaign")
    monkeypatch.setattr(module, "api_get_or_stop", lambda *_args, **_kwargs: {"items": []})
    yield module
    sys.modules.pop("loreley.ui.pages.campaign", None)


def test_campaign_baseline_ensure_button_requires_confirmation(campaign_module) -> None:
    campaign_module._render_baseline_tasks("http://api.local")  # noqa: SLF001

    assert {
        "label": "Create baseline ensure task",
        "key": "campaign_baseline_ensure",
        "disabled": True,
    } in _FakeStreamlitModule.buttons

    _FakeStreamlitModule.reset()
    _FakeStreamlitModule.checkbox_values["campaign_baseline_ensure_confirm_0"] = True
    campaign_module._render_baseline_tasks("http://api.local")  # noqa: SLF001

    assert {
        "label": "Create baseline ensure task",
        "key": "campaign_baseline_ensure",
        "disabled": False,
    } in _FakeStreamlitModule.buttons


def test_campaign_baseline_ensure_click_posts_when_confirmed(
    campaign_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[tuple[str, str, dict[str, object]]] = []

    def _post(api_base_url: str, path: str, *, json_body: dict[str, object]):
        posts.append((api_base_url, path, json_body))
        return {"id": "task-1"}

    monkeypatch.setattr(campaign_module, "api_post_or_stop", _post)

    _FakeStreamlitModule.button_clicks["campaign_baseline_ensure"] = True
    campaign_module._render_baseline_tasks("http://api.local")  # noqa: SLF001

    assert posts == []

    _FakeStreamlitModule.reset()
    _FakeStreamlitModule.checkbox_values["campaign_baseline_ensure_confirm_0"] = True
    _FakeStreamlitModule.button_clicks["campaign_baseline_ensure"] = True
    campaign_module._render_baseline_tasks("http://api.local")  # noqa: SLF001

    assert posts == [
        (
            "http://api.local",
            "/api/v1/operator/tasks/baseline-ensure",
            {},
        )
    ]
