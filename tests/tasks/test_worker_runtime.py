from __future__ import annotations

import importlib
import sys

import loreley.config as config_module
import loreley.entrypoints as entrypoints
import loreley.tasks.broker as broker_module
import loreley.tasks.workers as workers_module


def test_worker_runtime_bootstraps_one_process_local_actor(
    monkeypatch,
    settings,
) -> None:
    calls: list[tuple[str, object]] = []
    broker = object()
    actor = object()

    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        entrypoints,
        "configure_process_logging",
        lambda **kwargs: calls.append(("logging", kwargs["settings"])),
    )
    monkeypatch.setattr(
        entrypoints,
        "_apply_dramatiq_prefetch_settings",
        lambda **kwargs: calls.append(("prefetch", kwargs["settings"])),
    )
    monkeypatch.setattr(
        broker_module,
        "setup_broker",
        lambda *, settings: calls.append(("broker", settings)) or broker,
    )
    monkeypatch.setattr(
        workers_module,
        "build_evolution_job_worker_actor",
        lambda *, settings, broker: (
            calls.append(("actor", (settings, broker))) or actor
        ),
    )
    sys.modules.pop("loreley.tasks.worker_runtime", None)

    try:
        runtime = importlib.import_module("loreley.tasks.worker_runtime")
        assert runtime.broker is broker
        assert runtime.run_evolution_job is actor
        assert calls == [
            ("logging", settings),
            ("prefetch", settings),
            ("broker", settings),
            ("actor", (settings, broker)),
        ]
    finally:
        sys.modules.pop("loreley.tasks.worker_runtime", None)
