from __future__ import annotations

import importlib
import sys

import dramatiq

from loreley.config import Settings


def test_broker_import_has_no_side_effect(monkeypatch) -> None:
    calls: list[object] = []

    def _record(broker: object) -> None:
        calls.append(broker)

    monkeypatch.setattr(dramatiq, "set_broker", _record)
    sys.modules.pop("loreley.tasks.broker", None)

    module = importlib.import_module("loreley.tasks.broker")

    assert calls == []
    assert module.broker is None


def test_setup_broker_configures_dramatiq(monkeypatch) -> None:
    calls: list[object] = []

    def _record(broker: object) -> None:
        calls.append(broker)

    monkeypatch.setattr(dramatiq, "set_broker", _record)
    monkeypatch.setenv("EXPERIMENT_ID", "test")
    sys.modules.pop("loreley.tasks.broker", None)

    module = importlib.import_module("loreley.tasks.broker")
    settings = Settings(_env_file=None)
    broker = module.setup_broker(settings=settings)

    assert calls and calls[-1] is broker
    assert module.broker is broker
