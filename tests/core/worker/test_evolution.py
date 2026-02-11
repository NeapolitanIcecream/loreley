from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import loreley.core.worker.evolution as evolution_module
from loreley.config import Settings
from loreley.core.worker.evolution import EvolutionWorker, JobContext
from loreley.core.worker.planning import PlanDocument, PlanningAgentResponse
from loreley.core.worker.repository import CheckoutContext
from loreley.db.models import CommitCard, MapElitesArchiveCell, Metric


class _ResultList:
    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def all(self) -> list[Any]:
        return list(self._values)


class _ResultSingle:
    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def scalar_one_or_none(self) -> Any:
        if not self._values:
            return None
        if len(self._values) > 1:
            raise AssertionError("Expected at most one row.")
        return self._values[0]


class _FakePlanningAgent:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def plan(self, request: Any, *, working_dir: Path) -> PlanningAgentResponse:
        self.requests.append((request, working_dir))
        return PlanningAgentResponse(
            plan=PlanDocument(
                summary="ok",
                markdown="## Summary\n- ok\n",
                focus_metrics=(),
                guardrails=(),
            ),
            raw_output="raw",
            prompt="prompt",
            command=("cmd",),
            stderr="",
            attempts=1,
            duration_seconds=0.1,
        )


def _extract_values(params: dict[str, Any], prefix: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for key, value in params.items():
        if not key.startswith(prefix):
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        else:
            values.append(value)
    return tuple(values)


class _FakeSession:
    def __init__(self, cards: dict[str, Any], metrics: dict[uuid.UUID, list[Any]], cells: dict[str, Any]) -> None:
        self.cards = cards
        self.metrics = metrics
        self.cells = cells
        self.query_count = 0

    def _entity(self, stmt: Any) -> Any:
        descriptions = getattr(stmt, "column_descriptions", ())
        if not descriptions:
            raise AssertionError("Unexpected statement without entity descriptions.")
        return descriptions[0].get("entity")

    def execute(self, stmt: Any) -> _ResultSingle:
        self.query_count += 1
        entity = self._entity(stmt)
        params = stmt.compile().params
        if entity is CommitCard:
            hashes = _extract_values(params, "commit_hash")
            rows = [self.cards[h] for h in hashes if h in self.cards]
            return _ResultSingle(rows)
        if entity is MapElitesArchiveCell:
            hashes = _extract_values(params, "commit_hash")
            rows = [self.cells[h] for h in hashes if h in self.cells]
            return _ResultSingle(rows)
        raise AssertionError(f"Unexpected execute entity: {entity!r}")

    def scalars(self, stmt: Any) -> _ResultList:
        self.query_count += 1
        entity = self._entity(stmt)
        params = stmt.compile().params
        if entity is Metric:
            card_ids = _extract_values(params, "commit_card_id")
            rows: list[Any] = []
            for card_id in card_ids:
                rows.extend(self.metrics.get(card_id, ()))
            return _ResultList(rows)
        if entity is CommitCard:
            hashes = _extract_values(params, "commit_hash")
            rows = [self.cards[h] for h in hashes if h in self.cards]
            return _ResultList(rows)
        if entity is MapElitesArchiveCell:
            hashes = _extract_values(params, "commit_hash")
            rows = [self.cells[h] for h in hashes if h in self.cells]
            return _ResultList(rows)
        raise AssertionError(f"Unexpected scalars entity: {entity!r}")


def _make_job_context() -> JobContext:
    return JobContext(
        job_id=uuid.uuid4(),
        base_commit_hash="base",
        island_id="island-1",
        inspiration_commit_hashes=("insp-a", "insp-b"),
        goal="Ship value",
        constraints=(),
        acceptance_criteria=(),
        iteration_hint=None,
        notes=(),
        tags=(),
        is_seed_job=False,
        sampling_strategy=None,
        sampling_initial_radius=None,
        sampling_radius_used=None,
        sampling_fallback_inspirations=None,
    )


def test_run_planning_batches_context_queries_for_base_and_inspirations_gh_n_plus_1(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: planning context loading should avoid per-commit N+1 queries."""
    settings.worker_planning_trajectory_max_chunks = 0
    planning_agent = _FakePlanningAgent()
    worker = EvolutionWorker(
        settings=settings,
        repository=object(),  # type: ignore[arg-type]
        planning_agent=planning_agent,  # type: ignore[arg-type]
        coding_agent=object(),  # type: ignore[arg-type]
        evaluator=object(),  # type: ignore[arg-type]
        summarizer=object(),  # type: ignore[arg-type]
        job_store=object(),  # type: ignore[arg-type]
    )

    base_card_id = uuid.uuid4()
    insp_a_card_id = uuid.uuid4()
    cards = {
        "base": type(
            "Card",
            (),
            {
                "id": base_card_id,
                "commit_hash": "base",
                "subject": "Base",
                "change_summary": "base summary",
                "key_files": ["base.py"],
                "highlights": ["base highlight"],
                "evaluation_summary": "good",
            },
        )(),
        "insp-a": type(
            "Card",
            (),
            {
                "id": insp_a_card_id,
                "commit_hash": "insp-a",
                "subject": "Insp A",
                "change_summary": "insp-a summary",
                "key_files": ["a.py"],
                "highlights": ["a highlight"],
                "evaluation_summary": None,
            },
        )(),
    }
    metrics = {
        base_card_id: [
            type(
                "MetricRow",
                (),
                {
                    "name": "quality",
                    "value": 1.0,
                    "unit": None,
                    "higher_is_better": True,
                    "details": {"summary": "q"},
                    "commit_card_id": base_card_id,
                },
            )()
        ],
        insp_a_card_id: [
            type(
                "MetricRow",
                (),
                {
                    "name": "speed",
                    "value": 2.0,
                    "unit": None,
                    "higher_is_better": False,
                    "details": {"description": "s"},
                    "commit_card_id": insp_a_card_id,
                },
            )()
        ],
    }
    cells = {
        "base": type(
            "Cell",
            (),
            {
                "cell_index": 7,
                "commit_hash": "base",
                "objective": 0.8,
                "measures": [0.1, 0.2],
            },
        )(),
        "insp-a": type(
            "Cell",
            (),
            {
                "cell_index": 11,
                "commit_hash": "insp-a",
                "objective": 0.9,
                "measures": [0.3, 0.4],
            },
        )(),
    }
    fake_session = _FakeSession(cards=cards, metrics=metrics, cells=cells)

    @contextmanager
    def fake_session_scope() -> Any:
        yield fake_session

    monkeypatch.setattr(evolution_module, "session_scope", fake_session_scope)
    monkeypatch.setattr(
        evolution_module,
        "build_inspiration_trajectory_rollup",
        lambda **_kwargs: type("Rollup", (), {"lines": (), "meta": None})(),
    )

    job_ctx = _make_job_context()
    checkout = CheckoutContext(
        job_id=str(job_ctx.job_id),
        branch_name="branch",
        base_commit="base",
        worktree=Path("."),
    )
    worker._run_planning(job_ctx, checkout)

    assert fake_session.query_count == 3
    assert len(planning_agent.requests) == 1
    request, _working_dir = planning_agent.requests[0]
    assert request.base.commit_hash == "base"
    assert tuple(ctx.commit_hash for ctx in request.inspirations) == ("insp-a", "insp-b")
    assert request.inspirations[1].highlights == ("No highlights available.",)


def test_start_job_requires_non_empty_base_commit_hash(settings: Settings) -> None:
    """Regression: worker should fail fast when base_commit_hash is missing."""

    locked_job_id = uuid.uuid4()
    locked_job = type(
        "LockedJob",
        (),
        {
            "job_id": locked_job_id,
            "base_commit_hash": "   ",
            "island_id": None,
            "inspiration_commit_hashes": (),
            "goal": "Goal",
            "constraints": (),
            "acceptance_criteria": (),
            "iteration_hint": None,
            "notes": (),
            "tags": (),
            "is_seed_job": False,
            "sampling_strategy": None,
            "sampling_initial_radius": None,
            "sampling_radius_used": None,
            "sampling_fallback_inspirations": None,
        },
    )()

    class _FakeJobStore:
        def start_job(self, _job_id: uuid.UUID) -> Any:  # noqa: ANN401 - test stub
            return locked_job

    worker = EvolutionWorker(
        settings=settings,
        repository=object(),  # type: ignore[arg-type]
        planning_agent=object(),  # type: ignore[arg-type]
        coding_agent=object(),  # type: ignore[arg-type]
        evaluator=object(),  # type: ignore[arg-type]
        summarizer=object(),  # type: ignore[arg-type]
        job_store=_FakeJobStore(),  # type: ignore[arg-type]
    )

    with pytest.raises(evolution_module.EvolutionWorkerError, match="no base commit hash"):
        worker._start_job(locked_job_id)
