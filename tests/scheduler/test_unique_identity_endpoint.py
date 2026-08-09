from __future__ import annotations

from typing import Any, cast

from rich.console import Console

from loreley.core.progress import CampaignProgress
from loreley.scheduler.main import EvolutionScheduler
from tests.support import TestSettings


class _EndpointIngestion:
    def __init__(self, events: list[str], *, pending: int = 0) -> None:
        self.events = events
        self.pending = pending

    def ingest_completed_jobs(self) -> int:
        self.events.append("ingest")
        return 1

    def count_pending_ingestion_jobs(self) -> int:
        self.events.append("count_pending_ingestion")
        return self.pending


class _EndpointJobScheduler:
    def __init__(self, events: list[str], *, unfinished: int) -> None:
        self.events = events
        self.unfinished = unfinished

    def reclaim_stale_running_jobs(self) -> object:
        self.events.append("reclaim")
        return object()

    def cancel_pending_for_identity_endpoint(self) -> int:
        self.events.append("cancel_pending")
        return 2

    def count_unfinished_jobs(self) -> int:
        self.events.append("count_unfinished")
        return self.unfinished

    def dispatch_pending_jobs(self) -> int:
        raise AssertionError("identity endpoint must stop dispatch")

    def schedule_jobs(self, *_args: object, **_kwargs: object) -> int:
        raise AssertionError("identity endpoint must stop scheduling")


def _progress(*, identities: int, target: int) -> CampaignProgress:
    return CampaignProgress(
        terminal_jobs=12,
        succeeded_jobs=12,
        failed_jobs=0,
        cancelled_jobs=0,
        running_jobs=0,
        queued_jobs=0,
        pending_jobs=0,
        distinct_passed_source_trees=12,
        distinct_passed_evaluation_identities=identities,
        passed_candidates_without_identity=0,
        real_measurements=identities,
        measurement_reuses=0,
        exact_tree_reuses=0,
        archive_entries=3,
        archive_unique_evaluation_identities=3,
        occupied_coordinates=3,
        scheduler_max_unfinished_jobs=4,
        identity_target=target,
        identity_target_reached=identities >= target,
        identity_overshoot=max(0, identities - target),
    )


def _scheduler(
    events: list[str],
    *,
    identities: int,
    target: int,
    unfinished: int = 0,
    pending_ingestion: int = 0,
) -> EvolutionScheduler:
    scheduler = cast(Any, EvolutionScheduler.__new__(EvolutionScheduler))
    scheduler.settings = TestSettings(
        MAPELITES_EXPERIMENT_ROOT_COMMIT="root",
        SCHEDULER_MAX_UNIQUE_EVALUATION_IDENTITIES=target,
    )
    scheduler.console = Console(record=True)
    scheduler._stop_requested = False
    scheduler.ingestion = _EndpointIngestion(events, pending=pending_ingestion)
    scheduler.job_scheduler = _EndpointJobScheduler(events, unfinished=unfinished)
    scheduler._identity_endpoint_progress = lambda: _progress(
        identities=identities,
        target=target,
    )
    scheduler._create_primary_objective_branch_if_possible = (
        lambda: events.append("create_primary_branch")
    )
    scheduler.stop = lambda: events.append("stop")
    return cast(EvolutionScheduler, scheduler)


def test_reached_identity_endpoint_cancels_only_pending_and_drains() -> None:
    events: list[str] = []
    scheduler = _scheduler(events, identities=5, target=4, unfinished=1)

    stats = scheduler.tick()

    assert stats == {
        "ingested": 1,
        "reclaimed_pending": 0,
        "reclaimed_failed": 0,
        "identity_endpoint_reached": 1,
        "identity_count": 5,
        "identity_overshoot": 1,
        "endpoint_cancelled_pending": 2,
        "dispatched": 0,
        "seed_scheduled": 0,
        "scheduled": 0,
        "unfinished": 1,
        "pending_ingestion": 0,
    }
    assert events == [
        "ingest",
        "reclaim",
        "cancel_pending",
        "count_unfinished",
        "count_pending_ingestion",
    ]


def test_identity_endpoint_is_restart_stable_and_stops_after_drain() -> None:
    for _restart in range(2):
        events: list[str] = []
        scheduler = _scheduler(events, identities=4, target=4)

        stats = scheduler.tick()

        assert stats["identity_endpoint_reached"] == 1
        assert stats["scheduled"] == 0
        assert events == [
            "ingest",
            "reclaim",
            "cancel_pending",
            "count_unfinished",
            "count_pending_ingestion",
            "create_primary_branch",
            "stop",
        ]
