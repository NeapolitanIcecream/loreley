from __future__ import annotations

import uuid
from typing import Any, cast

import pytest
from rich.console import Console

import loreley.scheduler.job_scheduler as job_scheduler
from loreley.config import Settings
from loreley.core.map_elites.sampler import MapElitesSampler
from loreley.scheduler.job_scheduler import JobScheduler


class DummySenderActor:
    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.sent: list[str] = []

    def send(self, job_id: str) -> None:
        if job_id in self.fail_on:
            raise RuntimeError("send failed")
        self.sent.append(job_id)


class DummyLog:
    def __init__(self) -> None:
        self.debug_calls: list[tuple[str, tuple[object, ...]]] = []

    def debug(self, message: str, *args: object) -> None:
        self.debug_calls.append((message, args))

    def exception(self, _message: str, *_args: object) -> None:  # pragma: no cover - not used here
        return


def test_enqueue_jobs_marks_only_sent_jobs(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_ids = [uuid.uuid4(), uuid.uuid4()]
    sender = DummySenderActor(fail_on={str(job_ids[1])})
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    marked: list[uuid.UUID] = []
    monkeypatch.setattr(
        JobScheduler,
        "_mark_jobs_queued",
        lambda _self, ids: marked.extend(list(ids)) or list(ids),
    )

    dispatched = scheduler._enqueue_jobs(job_ids)

    assert dispatched == 1
    assert sender.sent == [str(job_ids[0])]
    assert marked == [job_ids[0]]


def test_enqueue_jobs_logs_marked_vs_sent_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_ids = [uuid.uuid4(), uuid.uuid4()]
    sender = DummySenderActor()
    monkeypatch.setattr(
        job_scheduler,
        "build_evolution_job_sender_actor",
        lambda **_kwargs: sender,
    )
    dummy_log = DummyLog()
    monkeypatch.setattr(job_scheduler, "log", dummy_log)
    scheduler = cast(Any, JobScheduler)(
        settings=settings,
        console=Console(record=True),
        sampler=cast(MapElitesSampler, object()),
    )
    monkeypatch.setattr(JobScheduler, "_mark_jobs_queued", lambda _self, _ids: [])

    dispatched = scheduler._enqueue_jobs(job_ids)

    assert dispatched == len(job_ids)
    assert len(dummy_log.debug_calls) == 1
    message, args = dummy_log.debug_calls[0]
    assert "marked {} job(s) as QUEUED" in message
    assert args[0] == len(job_ids)
    assert args[1] == 0
