from __future__ import annotations

import importlib
import sys

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
def overview_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlitModule())
    sys.modules.pop("loreley.ui.components.api", None)
    sys.modules.pop("loreley.ui.pages.overview", None)
    module = importlib.import_module("loreley.ui.pages.overview")
    yield module
    sys.modules.pop("loreley.ui.pages.overview", None)


def test_overview_kpis_shape_jobs_and_selected_island_stats(overview_module) -> None:
    kpis = overview_module._build_overview_kpis(  # noqa: SLF001
        island_id="main",
        jobs=[
            {"status": "succeeded"},
            {"status": "failed"},
            {"status": "succeeded"},
        ],
        islands=[
            {
                "island_id": "main",
                "metric_name": "score",
                "higher_is_better": True,
                "best_fitness": 2.5,
                "coverage": 0.25,
                "qd_score": 10.0,
                "norm_qd_score": 0.5,
                "occupied": 2,
                "cells": 8,
            },
            {
                "island_id": "side",
                "metric_name": "score",
                "higher_is_better": True,
                "best_fitness": 1.5,
            },
        ],
    )

    assert kpis["status_counts"] == {"succeeded": 2, "failed": 1}
    assert kpis["total_jobs"] == 3
    assert kpis["succeeded"] == 2
    assert kpis["failed"] == 1
    assert kpis["metric_name"] == "score"
    assert kpis["best_fitness"] == 2.5
    assert kpis["coverage"] == 0.25
    assert kpis["occupied"] == 2
    assert kpis["cells"] == 8
