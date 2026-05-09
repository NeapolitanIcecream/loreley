from __future__ import annotations

import importlib
import sys

import pandas as pd
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
    captions: list[str] = []
    infos: list[str] = []
    successes: list[str] = []
    writes: list[object] = []
    text_areas: list[dict[str, object]] = []
    selectbox_values: dict[str, object] = {}
    checkbox_values: dict[str, bool] = {}
    button_clicks: dict[str, bool] = {}

    @classmethod
    def reset(cls) -> None:
        cls.session_state = {}
        cls.buttons = []
        cls.captions = []
        cls.infos = []
        cls.successes = []
        cls.writes = []
        cls.text_areas = []
        cls.selectbox_values = {}
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
    def subheader(cls, _value: str) -> None:
        return None

    @classmethod
    def error(cls, _value: str) -> None:
        return None

    @classmethod
    def info(cls, value: str) -> None:
        cls.infos.append(value)

    @classmethod
    def success(cls, value: str) -> None:
        cls.successes.append(value)

    @classmethod
    def write(cls, value: object) -> None:
        cls.writes.append(value)

    @classmethod
    def caption(cls, value: str) -> None:
        cls.captions.append(value)

    @classmethod
    def divider(cls) -> None:
        return None

    @classmethod
    def text_area(cls, label: str, **kwargs) -> None:
        cls.text_areas.append({"label": label, "kwargs": kwargs})

    @classmethod
    def selectbox(cls, label: str, options, index: int = 0, **_kwargs):
        return cls.selectbox_values.get(label, options[index])

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

    @classmethod
    def expander(cls, *_args, **_kwargs):
        return cls()

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


def test_jobs_page_params_send_server_side_fate_and_evidence_filters(jobs_module) -> None:
    params = jobs_module._job_page_params(  # noqa: SLF001
        {
            "page_size": 50,
            "selected_status": "failed",
            "job_kind": "repair",
            "fate_filter": "unknown",
            "evidence_filter": "agent-visible",
        }
    )

    assert params == {
        "limit": 50,
        "status": "failed",
        "job_kind": "repair",
        "candidate_fate": "unknown",
        "evidence": "agent_visible",
    }


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


def test_jobs_global_failed_stale_retry_button_requires_confirmation(jobs_module) -> None:
    jobs_module._render_jobs_actions(api_base_url="http://api.local")  # noqa: SLF001

    assert {
        "label": "Retry global failed-stale jobs",
        "key": "jobs_retry_failed_stale",
        "disabled": True,
    } in _FakeStreamlitModule.buttons

    _FakeStreamlitModule.reset()
    _FakeStreamlitModule.checkbox_values["jobs_retry_failed_stale_confirm_0"] = True
    jobs_module._render_jobs_actions(api_base_url="http://api.local")  # noqa: SLF001

    assert {
        "label": "Retry global failed-stale jobs",
        "key": "jobs_retry_failed_stale",
        "disabled": False,
    } in _FakeStreamlitModule.buttons


def test_jobs_single_job_retry_button_requires_job_confirmation(
    jobs_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs_module, "_render_evidence_sections", lambda **_kwargs: None)

    jobs_module._render_job_detail(  # noqa: SLF001
        api_base_url="http://api.local",
        job_id="job-123",
        detail={"id": "job-123", "goal": ""},
    )

    assert {
        "label": "Retry this job",
        "key": "retry_job_job-123",
        "disabled": True,
    } in _FakeStreamlitModule.buttons

    _FakeStreamlitModule.reset()
    _FakeStreamlitModule.checkbox_values["retry_job_job-123_confirm_0"] = True
    jobs_module._render_job_detail(  # noqa: SLF001
        api_base_url="http://api.local",
        job_id="job-123",
        detail={"id": "job-123", "goal": ""},
    )

    assert {
        "label": "Retry this job",
        "key": "retry_job_job-123",
        "disabled": False,
    } in _FakeStreamlitModule.buttons


def test_jobs_single_job_retry_click_posts_when_confirmed(
    jobs_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[tuple[str, str, dict[str, object]]] = []

    def _post(api_base_url: str, path: str, *, json_body: dict[str, object]):
        posts.append((api_base_url, path, json_body))
        return {"job_id": "job-123"}

    monkeypatch.setattr(jobs_module, "_render_evidence_sections", lambda **_kwargs: None)
    monkeypatch.setattr(jobs_module, "api_post_or_stop", _post)

    _FakeStreamlitModule.button_clicks["retry_job_job-123"] = True
    jobs_module._render_job_detail(  # noqa: SLF001
        api_base_url="http://api.local",
        job_id="job-123",
        detail={"id": "job-123", "goal": ""},
    )

    assert posts == []

    _FakeStreamlitModule.reset()
    _FakeStreamlitModule.checkbox_values["retry_job_job-123_confirm_0"] = True
    _FakeStreamlitModule.button_clicks["retry_job_job-123"] = True
    jobs_module._render_job_detail(  # noqa: SLF001
        api_base_url="http://api.local",
        job_id="job-123",
        detail={"id": "job-123", "goal": ""},
    )

    assert posts == [
        (
            "http://api.local",
            "/api/v1/jobs/job-123/retry",
            {},
        )
    ]


def test_jobs_fate_filter_options_cover_canonical_labels(jobs_module) -> None:
    from loreley.core.candidate_fate import CANDIDATE_FATE_LABELS

    options = jobs_module._fate_filter_options()  # noqa: SLF001

    assert options == ["all", *sorted(CANDIDATE_FATE_LABELS)]
    assert "policy_failed" in options
    assert "valid_not_considered" in options


def test_jobs_render_uses_server_filters_without_current_page_post_filter(
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
    captured_params: list[dict[str, object]] = []

    def _get_page(_api_base_url: str, _path: str, *, params: dict[str, object]):
        captured_params.append(dict(params))
        return {
            "items": [{"id": "job-1", "candidate_fate_label": "elite_inserted"}],
            "next_cursor": "next-page",
        }

    monkeypatch.setattr(
        jobs_module,
        "api_get_page_or_stop",
        _get_page,
    )
    monkeypatch.setattr(jobs_module, "render_table", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(jobs_module, "selected_rows", lambda _grid: [])

    jobs_module.render()

    assert captured_params == [{"limit": 50, "candidate_fate": "unknown"}]
    assert "page=1 items=1" in _FakeStreamlitModule.captions
    assert "No jobs found." not in _FakeStreamlitModule.infos
    assert {
        "label": "Next page",
        "key": "jobs_next_page",
        "disabled": False,
    } in _FakeStreamlitModule.buttons


def test_jobs_pager_signature_includes_fate_and_evidence_filters(jobs_module) -> None:
    state = jobs_module._sync_jobs_pager(  # noqa: SLF001
        api_base_url="http://api.local",
        filters={
            "page_size": 50,
            "selected_status": "all",
            "job_kind": "all",
            "fate_filter": "all",
            "evidence_filter": "all",
        },
    )
    _FakeStreamlitModule.session_state[jobs_module._JOBS_CURSOR_KEY] = [None, "next"]  # noqa: SLF001
    _FakeStreamlitModule.session_state[jobs_module._JOBS_CURSOR_INDEX_KEY] = 1  # noqa: SLF001

    changed = jobs_module._sync_jobs_pager(  # noqa: SLF001
        api_base_url="http://api.local",
        filters={
            "page_size": 50,
            "selected_status": "all",
            "job_kind": "all",
            "fate_filter": "unknown",
            "evidence_filter": "none",
        },
    )

    assert changed.signature != state.signature
    assert changed.cursors == (None,)
    assert changed.index == 0
