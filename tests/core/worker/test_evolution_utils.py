from __future__ import annotations

from typing import Any, Mapping, Sequence

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

    assert worker._first_non_empty(None, "", " a ", "b") == "a"  # type: ignore[attr-defined]
    assert worker._first_non_empty(None, " ", None) is None  # type: ignore[attr-defined]

    assert worker._coerce_str_sequence(None) == ()  # type: ignore[attr-defined]
    assert worker._coerce_str_sequence("  x  ") == ("x",)  # type: ignore[attr-defined]
    assert worker._coerce_str_sequence(["a", " ", None, "b"]) == ("a", "b")  # type: ignore[attr-defined]
    assert worker._coerce_str_sequence(42) == ("42",)  # type: ignore[attr-defined]


def test_extract_goal_and_iteration_hint(settings: Settings) -> None:
    settings.worker_evolution_global_goal = "GLOBAL GOAL"
    worker = make_worker(settings)

    payload: Mapping[str, Any] = {"goal": "G"}
    goal = worker._extract_goal(  # type: ignore[attr-defined]
        payload=payload,
        extra_context={},
    )
    assert goal == "G"

    goal2 = worker._extract_goal(  # type: ignore[attr-defined]
        payload={},
        extra_context={"goal": "EG"},
    )
    assert goal2 == "EG"

    fallback = worker._extract_goal(  # type: ignore[attr-defined]
        payload={},
        extra_context={},
    )
    assert fallback == "GLOBAL GOAL"

    hint = worker._extract_iteration_hint(  # type: ignore[attr-defined]
        payload={"iteration_hint": "explicit"},
        extra_context={},
    )
    assert hint == "explicit"

    hint2 = worker._extract_iteration_hint(  # type: ignore[attr-defined]
        payload={
            "sampling": {
                "selection": {
                    "radius_used": 2,
                    "initial_radius": 1,
                }
            }
        },
        extra_context={},
    )
    assert "MAP-Elites radius 2" in hint2  # type: ignore[operator]


def test_extract_highlights_and_metrics_from_payload(settings: Settings) -> None:
    worker = make_worker(settings)

    highlights = worker._extract_highlights(  # type: ignore[attr-defined]
        "single",
        ["a", "b"],
        {"highlights": ["b", "c"], "snippets": ["d"]},
        ({"notes": ["e"]},),
    )
    assert highlights == ("single", "a", "b", "c", "d", "e")

    metrics = worker._metrics_from_payload(  # type: ignore[attr-defined]
        [
            {"name": "score", "value": 1.5, "unit": "pt", "higher_is_better": True},
            {"metric": "latency", "value": "20.5", "unit": "ms"},
            {"name": "invalid", "value": "not-number"},
        ]
    )
    assert len(metrics) == 2
    names = {m.name for m in metrics}
    assert {"score", "latency"} <= names


