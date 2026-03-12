from __future__ import annotations

from rich.console import Console

import loreley.entrypoints as entrypoints


def test_apply_dramatiq_prefetch_settings_uses_loreley_configuration(
    monkeypatch,
    settings,
) -> None:
    import dramatiq.worker as dramatiq_worker

    monkeypatch.setattr(dramatiq_worker, "QUEUE_PREFETCH", 99)
    monkeypatch.setattr(dramatiq_worker, "DELAY_QUEUE_PREFETCH", 999)
    settings.tasks_queue_prefetch = 1
    settings.tasks_delay_queue_prefetch = 3

    entrypoints._apply_dramatiq_prefetch_settings(  # noqa: SLF001 - spec-level assertion
        settings=settings,
        console=Console(record=True),
    )

    assert dramatiq_worker.QUEUE_PREFETCH == 1
    assert dramatiq_worker.DELAY_QUEUE_PREFETCH == 3


def test_apply_dramatiq_prefetch_settings_preserves_explicit_zero(
    monkeypatch,
    settings,
) -> None:
    import dramatiq.worker as dramatiq_worker

    monkeypatch.setattr(dramatiq_worker, "QUEUE_PREFETCH", 99)
    monkeypatch.setattr(dramatiq_worker, "DELAY_QUEUE_PREFETCH", 999)
    settings.tasks_queue_prefetch = 0
    settings.tasks_delay_queue_prefetch = 0

    entrypoints._apply_dramatiq_prefetch_settings(  # noqa: SLF001 - spec-level assertion
        settings=settings,
        console=Console(record=True),
    )

    assert dramatiq_worker.QUEUE_PREFETCH == 0
    assert dramatiq_worker.DELAY_QUEUE_PREFETCH == 0
