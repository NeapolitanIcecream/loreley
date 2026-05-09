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
    checkboxes: list[dict[str, object]] = []
    captions: list[str] = []
    infos: list[str] = []
    successes: list[str] = []
    writes: list[object] = []
    text_areas: list[dict[str, object]] = []
    selectbox_values: dict[str, object] = {}
    text_input_values: dict[str, str] = {}
    checkbox_values: dict[str, bool] = {}
    button_clicks: dict[str, bool] = {}

    @classmethod
    def reset(cls) -> None:
        cls.session_state = {}
        cls.buttons = []
        cls.checkboxes = []
        cls.captions = []
        cls.infos = []
        cls.successes = []
        cls.writes = []
        cls.text_areas = []
        cls.selectbox_values = {}
        cls.text_input_values = {}
        cls.checkbox_values = {}
        cls.button_clicks = {}

    @staticmethod
    def cache_resource(*_args, **_kwargs):
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
    def subheader(cls, _value: str) -> None:
        return None

    @classmethod
    def info(cls, value: str) -> None:
        cls.infos.append(value)

    @classmethod
    def success(cls, value: str) -> None:
        cls.successes.append(value)

    @classmethod
    def caption(cls, value: str) -> None:
        cls.captions.append(value)

    @classmethod
    def metric(cls, *_args, **_kwargs) -> None:
        return None

    @classmethod
    def write(cls, value: object) -> None:
        cls.writes.append(value)

    @classmethod
    def text_area(cls, label: str, **kwargs) -> None:
        cls.text_areas.append({"label": label, "kwargs": kwargs})

    @classmethod
    def selectbox(cls, label: str, options, index: int = 0, **_kwargs):
        return cls.selectbox_values.get(label, options[index])

    @classmethod
    def text_input(cls, label: str, value: str = "", **_kwargs) -> str:
        return cls.text_input_values.get(label, value)

    @classmethod
    def checkbox(cls, label: str, *, key: str, **_kwargs) -> bool:
        cls.checkboxes.append({"label": label, "key": key})
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


def test_repair_pool_schedule_one_button_requires_confirmation(repair_pool_module) -> None:
    repair_pool_module._render_schedule_controls("http://api.local")  # noqa: SLF001

    assert {
        "label": "Schedule one repair",
        "key": "repair_schedule_one",
        "disabled": True,
    } in _FakeStreamlitModule.buttons

    _FakeStreamlitModule.reset()
    _FakeStreamlitModule.checkbox_values["repair_schedule_one_confirm_0"] = True
    repair_pool_module._render_schedule_controls("http://api.local")  # noqa: SLF001

    assert {
        "label": "Schedule one repair",
        "key": "repair_schedule_one",
        "disabled": False,
    } in _FakeStreamlitModule.buttons


def test_repair_pool_schedule_one_click_posts_when_confirmed(
    repair_pool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[tuple[str, str, dict[str, object]]] = []

    def _post(api_base_url: str, path: str, *, json_body: dict[str, object]):
        posts.append((api_base_url, path, json_body))
        return {"scheduled": True, "job_id": "repair-job-1"}

    monkeypatch.setattr(repair_pool_module, "api_post_or_stop", _post)

    _FakeStreamlitModule.button_clicks["repair_schedule_one"] = True
    repair_pool_module._render_schedule_controls("http://api.local")  # noqa: SLF001

    assert posts == []

    _FakeStreamlitModule.reset()
    _FakeStreamlitModule.checkbox_values["repair_schedule_one_confirm_0"] = True
    _FakeStreamlitModule.button_clicks["repair_schedule_one"] = True
    repair_pool_module._render_schedule_controls("http://api.local")  # noqa: SLF001

    assert posts == [
        (
            "http://api.local",
            "/api/v1/repair/schedule-one",
            {},
        )
    ]


def test_repair_pool_candidate_buttons_use_candidate_action_confirmation_keys(
    repair_pool_module,
) -> None:
    repair_pool_module._render_selected_candidate(  # noqa: SLF001
        api_base_url="http://api.local",
        selected=[{"id": "candidate-123"}],
    )

    assert _FakeStreamlitModule.checkboxes == [
        {
            "label": "Confirm quarantine for candidate candidate-123",
            "key": "repair_candidate_action_confirm_candidate-123_quarantine_0",
        },
        {
            "label": "Confirm discard for candidate candidate-123",
            "key": "repair_candidate_action_confirm_candidate-123_discard_0",
        },
        {
            "label": "Confirm restore for candidate candidate-123",
            "key": "repair_candidate_action_confirm_candidate-123_restore_0",
        },
    ]
    assert {
        "label": "Quarantine",
        "key": "repair_quarantine",
        "disabled": True,
    } in _FakeStreamlitModule.buttons

    _FakeStreamlitModule.reset()
    _FakeStreamlitModule.checkbox_values[
        "repair_candidate_action_confirm_candidate-123_quarantine_0"
    ] = True
    repair_pool_module._render_selected_candidate(  # noqa: SLF001
        api_base_url="http://api.local",
        selected=[{"id": "candidate-123"}],
    )

    assert {
        "label": "Quarantine",
        "key": "repair_quarantine",
        "disabled": False,
    } in _FakeStreamlitModule.buttons
    assert {
        "label": "Discard",
        "key": "repair_discard",
        "disabled": True,
    } in _FakeStreamlitModule.buttons


def test_repair_pool_candidate_confirmation_key_rotates_after_write(
    repair_pool_module,
) -> None:
    base_key = repair_pool_module._candidate_action_confirm_base_key(  # noqa: SLF001
        "candidate-123",
        "discard",
    )

    assert (
        repair_pool_module.operator_confirmation_key(base_key)
        == "repair_candidate_action_confirm_candidate-123_discard_0"
    )

    repair_pool_module.expire_operator_confirmation(base_key)

    assert (
        repair_pool_module.operator_confirmation_key(base_key)
        == "repair_candidate_action_confirm_candidate-123_discard_1"
    )


def test_repair_pool_candidate_action_click_posts_when_confirmed(
    repair_pool_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[tuple[str, str, dict[str, object]]] = []

    def _post(api_base_url: str, path: str, *, json_body: dict[str, object]):
        posts.append((api_base_url, path, json_body))
        return {"ok": True}

    monkeypatch.setattr(repair_pool_module, "api_post_or_stop", _post)

    _FakeStreamlitModule.button_clicks["repair_quarantine"] = True
    repair_pool_module._render_selected_candidate(  # noqa: SLF001
        api_base_url="http://api.local",
        selected=[{"id": "candidate-123"}],
    )

    assert posts == []

    _FakeStreamlitModule.reset()
    _FakeStreamlitModule.checkbox_values[
        "repair_candidate_action_confirm_candidate-123_quarantine_0"
    ] = True
    _FakeStreamlitModule.button_clicks["repair_quarantine"] = True
    repair_pool_module._render_selected_candidate(  # noqa: SLF001
        api_base_url="http://api.local",
        selected=[{"id": "candidate-123"}],
    )

    assert {
        "label": "Confirm quarantine for candidate candidate-123",
        "key": "repair_candidate_action_confirm_candidate-123_quarantine_0",
    } in _FakeStreamlitModule.checkboxes
    assert posts == [
        (
            "http://api.local",
            "/api/v1/repair/candidates/candidate-123/quarantine",
            {},
        )
    ]
