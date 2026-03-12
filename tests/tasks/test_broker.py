from __future__ import annotations

import loreley.tasks.broker as broker_module


class _DummyClient:
    def __init__(self, keys: list[bytes]) -> None:
        self._keys = list(keys)
        self.deleted: tuple[bytes, ...] = ()

    def scan_iter(self, *, match: str):  # noqa: ANN001 - test double
        assert match == "test.ns:*"
        return iter(self._keys)

    def delete(self, *keys: bytes) -> int:
        self.deleted = tuple(keys)
        return len(keys)


class _DummyBroker:
    def __init__(self, keys: list[bytes]) -> None:
        self.namespace = "test.ns"
        self.client = _DummyClient(keys)


def test_reset_redis_namespace_deletes_all_matching_keys(monkeypatch) -> None:
    broker = _DummyBroker([b"test.ns:q.msgs", b"test.ns:__heartbeats__"])
    monkeypatch.setattr(broker_module, "build_redis_broker", lambda settings=None: broker)

    deleted = broker_module.reset_redis_namespace()

    assert deleted == 2
    assert broker.client.deleted == (b"test.ns:q.msgs", b"test.ns:__heartbeats__")


def test_reset_redis_namespace_is_noop_when_namespace_empty(monkeypatch) -> None:
    broker = _DummyBroker([])
    monkeypatch.setattr(broker_module, "build_redis_broker", lambda settings=None: broker)

    deleted = broker_module.reset_redis_namespace()

    assert deleted == 0
    assert broker.client.deleted == ()
