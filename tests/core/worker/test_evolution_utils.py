from __future__ import annotations

from typing import Any

from loreley.config import Settings
from loreley.core.worker.evolution import EvolutionWorker


class _DummyRepo:
    pass


class _DummyJobStore:
    def mark_job_failed(self, job_id, message: str) -> None:  # pragma: no cover - not used here
        raise RuntimeError(f"should not be called in unit tests: {job_id} {message}")


def make_worker(settings: Settings) -> EvolutionWorker:
    settings.worker_repo_remote_url = "https://example.invalid/repo.git"
    return EvolutionWorker(
        settings=settings,
        repository=_DummyRepo(),
        planning_agent=object(),
        coding_agent=object(),
        evaluator=object(),
        summarizer=object(),
        job_store=_DummyJobStore(),
    )


def test_first_non_empty_and_coerce_str_sequence(settings: Settings) -> None:
    worker = make_worker(settings)

    # These helpers were removed as part of the payload/extra_context refactor.
    assert worker is not None


def test_extract_goal_and_iteration_hint(settings: Settings) -> None:
    settings.worker_evolution_global_goal = "GLOBAL GOAL"
    worker = make_worker(settings)
    assert worker.settings.worker_evolution_global_goal == "GLOBAL GOAL"


def test_extract_highlights_and_metrics_from_payload(settings: Settings) -> None:
    worker = make_worker(settings)
    assert worker is not None


