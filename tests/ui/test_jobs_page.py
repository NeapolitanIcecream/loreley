from __future__ import annotations

import importlib
import sys

import pandas as pd
import pytest


class _FakeStreamlitModule:
    session_state: dict[str, object] = {}

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


@pytest.fixture
def jobs_module(monkeypatch: pytest.MonkeyPatch):
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
