from __future__ import annotations

from contextlib import contextmanager

from fastapi.testclient import TestClient

import loreley.api.app as api_app


def test_api_startup_marks_interrupted_operator_tasks(monkeypatch) -> None:
    calls: list[str] = []

    class _Session:
        pass

    @contextmanager
    def _scope():
        yield _Session()

    monkeypatch.setattr(api_app, "ensure_database_schema", lambda **_kwargs: calls.append("schema"))
    monkeypatch.setattr(api_app, "session_scope", _scope)
    monkeypatch.setattr(
        api_app,
        "validate_instance_marker_schema",
        lambda **_kwargs: calls.append("marker"),
    )
    monkeypatch.setattr(
        api_app,
        "mark_interrupted_baseline_ensure_tasks_failed",
        lambda: calls.append("operator_tasks") or 0,
    )

    with TestClient(api_app.create_app()):
        pass

    assert calls == ["schema", "marker", "operator_tasks"]
