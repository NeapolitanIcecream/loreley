from __future__ import annotations

from dramatiq.brokers.stub import StubBroker

import loreley.tasks.workers as workers_module


class _DummyEvolutionWorker:
    def __init__(self, settings=None) -> None:  # noqa: ANN001 - test double
        self.settings = settings

    def run(self, job_id: str):  # pragma: no cover - not executed in this spec
        return job_id


def test_build_worker_actor_declares_queue_on_supplied_broker(
    monkeypatch,
    settings,
) -> None:
    broker = StubBroker()
    monkeypatch.setattr(workers_module, "EvolutionWorker", _DummyEvolutionWorker)
    monkeypatch.setattr(
        workers_module,
        "setup_broker",
        lambda settings=None: (_ for _ in ()).throw(AssertionError("unexpected setup_broker call")),
    )

    actor = workers_module.build_evolution_job_worker_actor(settings=settings, broker=broker)

    assert actor.broker is broker
    assert actor.queue_name in broker.get_declared_queues()
    assert actor.actor_name in broker.actors


def test_build_sender_actor_declares_queue_on_supplied_broker(
    monkeypatch,
    settings,
) -> None:
    broker = StubBroker()
    monkeypatch.setattr(
        workers_module,
        "setup_broker",
        lambda settings=None: (_ for _ in ()).throw(AssertionError("unexpected setup_broker call")),
    )

    actor = workers_module.build_evolution_job_sender_actor(settings=settings, broker=broker)

    assert actor.broker is broker
    assert actor.queue_name in broker.get_declared_queues()
    assert actor.actor_name in broker.actors
