from __future__ import annotations

import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import loreley.core.worker.evolution as evolution_module
from loreley.config import Settings
from loreley.core.campaign_program import parse_campaign_program
from loreley.core.worker.coding import CodingAgentResponse, CodingError, ExecutionReport
from loreley.core.worker.evaluator import (
    EvalFail,
    EvalPass,
    EvaluationArtifact,
    EvaluationFailureResult,
    EvaluationOutcome,
    EvaluationResult,
)
from loreley.core.worker.evolution import EvolutionWorker, JobContext
from loreley.core.worker.planning import PlanDocument, PlanningAgentResponse
from loreley.core.worker.repository import CheckoutContext
from loreley.core.worker.scope_gate import ScopeGateResult
from loreley.core.usage import LLMUsageEventPayload
from loreley.db.models import CommitCard, EvaluationArtifactRecord, MapElitesArchiveCell, Metric


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


class _FakeCodingAgent:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def implement(self, request: Any, *, working_dir: Path) -> CodingAgentResponse:
        self.requests.append((request, working_dir))
        return CodingAgentResponse(
            report=ExecutionReport(
                summary="implemented",
                markdown="## Summary\n- implemented\n",
            ),
            raw_output="raw",
            prompt="prompt",
            command=("cmd",),
            stderr="",
            attempts=1,
            duration_seconds=0.1,
        )


class _UsageFakeCodingAgent(_FakeCodingAgent):
    def implement(self, request: Any, *, working_dir: Path) -> CodingAgentResponse:
        response = super().implement(request, working_dir=working_dir)
        invocation = int(request.invocation)
        response.usage_events = (
            LLMUsageEventPayload(
                source="fake",
                phase="coding",
                total_tokens=invocation,
                external_usage_id=f"coding-invocation-{invocation}",
            ),
        )
        return response


class _SecondPassFailingCodingAgent(_FakeCodingAgent):
    def implement(self, request: Any, *, working_dir: Path) -> CodingAgentResponse:
        self.requests.append((request, working_dir))
        if len(self.requests) > 1:
            raise CodingError("simulated retry coding crash")
        return CodingAgentResponse(
            report=ExecutionReport(
                summary="implemented",
                markdown="## Summary\n- implemented\n",
            ),
            raw_output="raw",
            prompt="prompt",
            command=("cmd",),
            stderr="",
            attempts=1,
            duration_seconds=0.1,
        )


class _ScopeViolatingCodingAgent:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def implement(self, request: Any, *, working_dir: Path) -> CodingAgentResponse:
        self._events.append("coding.implement")
        (working_dir / "README.md").write_text("changed outside editable scope\n", encoding="utf-8")
        return CodingAgentResponse(
            report=ExecutionReport(
                summary="implemented",
                markdown="## Summary\n- changed README\n",
            ),
            raw_output="raw",
            prompt="prompt",
            command=("cmd",),
            stderr="",
            attempts=1,
            duration_seconds=0.1,
        )


class _FakeEvaluator:
    def evaluate(self, _context: Any) -> EvaluationResult:
        return EvaluationResult(summary="evaluation ok")


class _EventCapturingEvaluator:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def evaluate_outcome(self, context: Any) -> EvaluationOutcome:
        self._events.append("evaluator.evaluate")
        return EvaluationOutcome(
            evaluator_name="fake",
            candidate_commit_hash=context.candidate_commit_hash,
            outcome_kind="passed",
            result=EvaluationResult(summary="ok"),
        )


class _PathArtifactEvaluator:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def evaluate_outcome(self, context: Any) -> EvaluationOutcome:
        self._events.append("evaluator.evaluate")
        report = context.worktree / "eval-report.txt"
        report.write_text("diagnostic report\n", encoding="utf-8")
        return EvaluationOutcome(
            evaluator_name="fake",
            candidate_commit_hash=context.candidate_commit_hash,
            outcome_kind="passed",
            result=EvaluationResult(
                summary="ok",
                artifacts=(
                    EvaluationArtifact(
                        key="eval_report",
                        kind="report",
                        mime_type="text/plain",
                        path="eval-report.txt",
                    ),
                ),
            ),
        )


class _CapturingOutcomeEvaluator:
    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def evaluate_outcome(self, context: Any) -> EvaluationOutcome:
        self.contexts.append(context)
        return EvaluationOutcome(
            evaluator_name="fake",
            candidate_commit_hash=context.candidate_commit_hash,
            outcome_kind="passed",
            result=EvaluationResult(summary="ok"),
        )


class _FakeCandidateFailureEvaluator:
    def evaluate_outcome(self, context: Any) -> EvaluationOutcome:
        return EvaluationOutcome(
            evaluator_name="fake",
            candidate_commit_hash=context.candidate_commit_hash,
            outcome_kind="candidate_failed",
            failure=EvaluationFailureResult(
                failure_stage="evaluation",
                failure_kind="typecheck_failed",
                repairability="repairable",
                safe_failure_summary="typecheck failed",
            ),
        )


class _SequenceEvaluator:
    def __init__(self, outcomes: list[Any], events: list[str]) -> None:
        self.outcomes = list(outcomes)
        self.events = events
        self.contexts: list[Any] = []

    def evaluate_outcome(self, context: Any) -> EvaluationOutcome | EvalPass | EvalFail:
        self.events.append("evaluator.evaluate")
        self.contexts.append(context)
        if not self.outcomes:
            raise AssertionError("No evaluator outcome configured.")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, EvaluationOutcome):
            outcome.candidate_commit_hash = outcome.candidate_commit_hash or context.candidate_commit_hash
        return outcome


def _candidate_failed_outcome(
    *,
    failure_kind: str = "typecheck_failed",
    summary: str = "typecheck failed",
) -> EvaluationOutcome:
    return EvaluationOutcome(
        evaluator_name="fake",
        outcome_kind="candidate_failed",
        failure=EvaluationFailureResult(
            failure_stage="evaluation",
            failure_kind=failure_kind,
            repairability="repairable",
            safe_failure_summary=summary,
            compiler_errors_summary=summary,
        ),
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
    def __init__(
        self,
        cards: dict[str, Any],
        metrics: dict[uuid.UUID, list[Any]],
        cells: dict[str, Any],
        artifacts: dict[str, list[Any]] | None = None,
    ) -> None:
        self.cards = cards
        self.metrics = metrics
        self.cells = cells
        self.artifacts = artifacts or {}
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
        if entity is EvaluationArtifactRecord:
            hashes = _extract_values(params, "commit_hash")
            rows = []
            for commit_hash in hashes:
                rows.extend(self.artifacts.get(commit_hash, ()))
            return _ResultList(rows)
        raise AssertionError(f"Unexpected scalars entity: {entity!r}")


class _FakeRepositoryForRun:
    def __init__(self, *, worktree: Path, events: list[str]) -> None:
        self._worktree = worktree
        self._events = events
        self._commit_count = 0
        self._current_commit = "base123"

    @contextmanager
    def checkout_lease_for_job(
        self,
        *,
        job_id: Any,
        base_commit: str,
        create_branch: bool = True,
        attempt_token: Any = None,
    ) -> Any:  # noqa: ANN401
        self._events.append("repo.checkout")
        yield CheckoutContext(
            job_id=str(job_id),
            branch_name="exp/job-branch" if create_branch else None,
            base_commit=base_commit,
            worktree=self._worktree,
        )

    def has_changes(self, *, worktree: Path | None = None) -> bool:
        self._events.append("repo.has_changes")
        return True

    def stage_all(self, *, worktree: Path | None = None) -> None:
        self._events.append("repo.stage_all")

    def commit(self, message: str, *, worktree: Path | None = None) -> str:
        self._events.append("repo.commit")
        self._commit_count += 1
        self._current_commit = "candidate123" if self._commit_count == 1 else f"candidate{self._commit_count}"
        return self._current_commit

    def push_branch(
        self,
        branch_name: str,
        *,
        worktree: Path | None = None,
        remote: str = "origin",
        force_with_lease: bool = False,
    ) -> None:
        self._events.append("repo.push_branch")

    def prune_stale_job_branches(self) -> int:
        self._events.append("repo.prune")
        return 0

    def clean_worktree(self, *, worktree: Path | None = None) -> None:
        self._events.append("repo.clean_worktree")

    def current_commit(self, *, worktree: Path | None = None) -> str:
        self._events.append("repo.current_commit")
        return self._current_commit

    def tree_hash(
        self,
        commit_hash: str,
        *,
        worktree: Path | None = None,
    ) -> str:
        del worktree
        return f"tree-{commit_hash}"

    def reset_mixed_to_commit(self, commit_hash: str, *, worktree: Path | None = None) -> None:
        self._events.append(f"repo.reset_mixed_to_commit[{commit_hash}]")
        self._current_commit = commit_hash

    def diff_summary_between_commits(
        self,
        *,
        base_commit: str,
        candidate_commit: str,
        worktree: Path | None = None,
    ) -> str:
        self._events.append("repo.diff_summary")
        return f"{base_commit}..{candidate_commit}"


class _CleaningRepositoryForRun(_FakeRepositoryForRun):
    def clean_worktree(self, *, worktree: Path | None = None) -> None:
        super().clean_worktree(worktree=worktree)
        target = (worktree or self._worktree) / "eval-report.txt"
        target.unlink(missing_ok=True)


class _SuppliedRepositoryForRun(_FakeRepositoryForRun):
    def ensure_remote_commit(self, *, commit_hash: str, remote_ref: str) -> None:
        self._events.append(f"repo.ensure_remote_commit[{remote_ref}]")
        self._current_commit = commit_hash

    @contextmanager
    def checkout_lease_for_job(
        self,
        *,
        job_id: Any,
        base_commit: str,
        create_branch: bool = True,
        attempt_token: Any = None,
    ) -> Any:  # noqa: ANN401
        assert create_branch is False
        self._current_commit = base_commit
        with super().checkout_lease_for_job(
            job_id=job_id,
            base_commit=base_commit,
            create_branch=create_branch,
            attempt_token=attempt_token,
        ) as checkout:
            yield checkout


class _FakeJobStoreForPublishFailure:
    def __init__(
        self,
        *,
        job_id: uuid.UUID,
        events: list[str],
        persist_error: Exception,
        campaign_program_hash: str | None = None,
    ) -> None:
        self._job_id = job_id
        self._events = events
        self._persist_error = persist_error
        self._campaign_program_hash = campaign_program_hash
        self.recorded_candidates: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.reusable_evaluation: EvaluationOutcome | None = None
        self.reuse_queries: list[dict[str, Any]] = []

    def start_job(self, job_id: uuid.UUID) -> Any:
        assert job_id == self._job_id
        return type(
            "LockedJob",
            (),
            {
                "job_id": job_id,
                "run_token": uuid.uuid4(),
                "worker_id": "worker-01",
                "base_commit_hash": "base123",
                "island_id": "island-1",
                "inspiration_commit_hashes": (),
                "goal": "Ship value",
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
                "campaign_program_hash": self._campaign_program_hash,
            },
        )()

    def renew_job_lease(self, _job_id: uuid.UUID, _run_token: uuid.UUID) -> None:
        return None

    def find_reusable_evaluation(self, **kwargs: Any) -> EvaluationOutcome | None:
        self.reuse_queries.append(kwargs)
        return self.reusable_evaluation

    def record_candidate_commit(
        self,
        record: Any,
    ) -> None:
        self._events.append(
            f"store.record_candidate[published={record.published}]"
        )
        self.recorded_candidates.append(
            {
                "job_id": record.job_id,
                "commit_hash": record.commit_hash,
                "branch_name": record.branch_name,
                "run_token": record.run_token,
                "published": record.published,
                "source_tree_hash": record.source_tree_hash,
            }
        )

    def persist_success(self, **_kwargs: Any) -> None:
        self._events.append("store.persist_success")
        raise self._persist_error

    def mark_job_failed(
        self,
        job_id: uuid.UUID,
        message: str,
        *,
        run_token: uuid.UUID | None = None,
    ) -> None:
        self._events.append("store.mark_job_failed")
        self.failures.append(
            {
                "job_id": job_id,
                "message": message,
                "run_token": run_token,
            }
        )


class _FakeJobStoreForSuccess(_FakeJobStoreForPublishFailure):
    def __init__(
        self,
        *,
        job_id: uuid.UUID,
        events: list[str],
        campaign_program_hash: str | None = None,
    ) -> None:
        super().__init__(
            job_id=job_id,
            events=events,
            persist_error=evolution_module.EvolutionWorkerError("unused"),
            campaign_program_hash=campaign_program_hash,
        )
        self.success_calls: list[dict[str, Any]] = []

    def persist_success(self, **kwargs: Any) -> None:
        self._events.append("store.persist_success")
        self.success_calls.append(kwargs)


class _ManualSeedJobStore(_FakeJobStoreForSuccess):
    def __init__(self, *, job_id: uuid.UUID, events: list[str], commit_hash: str) -> None:
        super().__init__(job_id=job_id, events=events)
        self._commit_hash = commit_hash

    def start_job(self, job_id: uuid.UUID) -> Any:
        assert job_id == self._job_id
        return type(
            "LockedJob",
            (),
            {
                "job_id": job_id,
                "run_token": uuid.uuid4(),
                "worker_id": "worker-01",
                "base_commit_hash": "a" * 40,
                "island_id": "island-1",
                "inspiration_commit_hashes": (),
                "goal": "Evaluate the supplied seed",
                "constraints": (),
                "acceptance_criteria": (),
                "iteration_hint": None,
                "notes": (),
                "tags": ("manual_seed",),
                "is_seed_job": True,
                "job_kind": "manual_seed",
                "execution_mode": "evaluate_existing",
                "input_candidate_commit_hash": self._commit_hash,
                "input_candidate_summary": "Independent allocation fast path",
                "external_submission_key": "b" * 64,
                "input_provenance": {
                    "source_type": "manual_seed_manifest",
                    "remote_ref": "refs/heads/loreley-seeds/allocation",
                },
                "archive_ingestion_enabled": True,
                "sampling_strategy": "manual_seed",
                "sampling_initial_radius": None,
                "sampling_radius_used": None,
                "sampling_fallback_inspirations": None,
                "campaign_program_hash": None,
            },
        )()


class _FakeJobStoreForEvaluationFailureRetry(_FakeJobStoreForPublishFailure):
    def __init__(self, *, job_id: uuid.UUID, events: list[str]) -> None:
        super().__init__(
            job_id=job_id,
            events=events,
            persist_error=evolution_module.EvolutionWorkerError("unused"),
        )
        self.persist_failure_outcomes: list[EvaluationOutcome] = []

    def persist_failure(self, **kwargs: Any) -> bool:
        self._events.append("store.persist_failure")
        self.persist_failure_outcomes.append(kwargs["outcome"])
        return len(self.persist_failure_outcomes) > 1


class _FakeJobStoreForScopeGateFailure(_FakeJobStoreForPublishFailure):
    def __init__(self, *, job_id: uuid.UUID, events: list[str], campaign_program_hash: str) -> None:
        super().__init__(
            job_id=job_id,
            events=events,
            persist_error=evolution_module.EvolutionWorkerError("unused"),
            campaign_program_hash=campaign_program_hash,
        )
        self.persist_failure_calls: list[dict[str, Any]] = []

    def persist_failure(self, **kwargs: Any) -> bool:
        self._events.append("store.persist_failure")
        self.persist_failure_calls.append(kwargs)
        return True

    def persist_success(self, **_kwargs: Any) -> None:
        raise AssertionError("scope-gate failures must not persist success")


def _patch_empty_planning_context_session(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _FakeSession(cards={}, metrics={}, cells={})

    @contextmanager
    def fake_session_scope() -> Any:
        yield fake_session

    monkeypatch.setattr(evolution_module, "session_scope", fake_session_scope)


def _make_job_context() -> JobContext:
    return JobContext(
        job_id=uuid.uuid4(),
        run_token=uuid.uuid4(),
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


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_scope_repo(repo: Path) -> None:
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")


def test_manual_seed_skips_model_agents_and_uses_detached_supplied_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    commit_hash = "c" * 40
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    store = _ManualSeedJobStore(
        job_id=job_id,
        events=events,
        commit_hash=commit_hash,
    )
    planning = _FakePlanningAgent()
    coding = _FakeCodingAgent()
    worker = EvolutionWorker(
        settings=settings,
        repository=_SuppliedRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=planning,  # type: ignore[arg-type]
        coding_agent=coding,  # type: ignore[arg-type]
        evaluator=_FakeEvaluator(),  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    result = worker.run(job_id)

    assert result.candidate_commit_hash == commit_hash
    assert planning.requests == []
    assert coding.requests == []
    assert "repo.commit" not in events
    assert "repo.push_branch" not in events
    assert "repo.ensure_remote_commit[refs/heads/loreley-seeds/allocation]" in events
    assert store.recorded_candidates == [
        {
            "job_id": job_id,
            "commit_hash": commit_hash,
            "branch_name": "",
            "run_token": store.recorded_candidates[0]["run_token"],
            "published": False,
            "source_tree_hash": f"tree-{commit_hash}",
        }
    ]
    persisted = store.success_calls[0]
    assert result.plan is None
    assert result.coding is None
    assert persisted["plan"] is None
    assert persisted["coding"] is None


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
    prompt_context = worker._build_prompt_context(job_ctx)
    worker._run_planning(job_ctx, checkout, prompt_context)

    assert fake_session.query_count == 3
    assert len(planning_agent.requests) == 1
    request, _working_dir = planning_agent.requests[0]
    assert request.base.commit_hash == "base"
    assert tuple(ctx.commit_hash for ctx in request.inspirations) == ("insp-a", "insp-b")
    assert request.inspirations[1].highlights == ()
    assert request.iteration_context.seed_job is False


def test_seed_job_prompt_context_suppresses_historical_evaluation_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    worker = EvolutionWorker(
        settings=settings,
        repository=object(),  # type: ignore[arg-type]
        planning_agent=object(),  # type: ignore[arg-type]
        coding_agent=object(),  # type: ignore[arg-type]
        evaluator=object(),  # type: ignore[arg-type]
        job_store=object(),  # type: ignore[arg-type]
    )
    card_id = uuid.uuid4()
    cards = {
        "base": type(
            "Card",
            (),
            {
                "id": card_id,
                "commit_hash": "base",
                "subject": "Base",
                "change_summary": "base summary",
                "key_files": ["base.py"],
                "highlights": ["base highlight"],
                "evaluation_summary": "historical summary",
            },
        )(),
    }
    artifacts = {
        "base": [
            type(
                "ArtifactRow",
                (),
                {
                        "job_id": uuid.uuid4(),
                        "commit_hash": "base",
                        "key": "benchmark_report",
                    "kind": "benchmark_json",
                    "mime_type": "application/json",
                    "label": None,
                    "summary": "historical artifact",
                    "diagnostics": [],
                    "agent_projection": "summary",
                    "visibility": "agent_visible",
                    "size_bytes": None,
                    "sha256": None,
                    "storage_path": None,
                },
            )()
        ]
    }
    fake_session = _FakeSession(cards=cards, metrics={}, cells={}, artifacts=artifacts)

    @contextmanager
    def fake_session_scope() -> Any:
        yield fake_session

    monkeypatch.setattr(evolution_module, "session_scope", fake_session_scope)
    job_ctx = _make_job_context()
    job_ctx.is_seed_job = True
    job_ctx.inspiration_commit_hashes = ()

    prompt_context = worker._build_prompt_context(job_ctx)

    assert prompt_context.base.evaluation_summary is None
    assert prompt_context.base.metrics == ()
    assert prompt_context.base.evaluation_artifacts == ()


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
        job_store=_FakeJobStore(),  # type: ignore[arg-type]
    )

    with pytest.raises(evolution_module.EvolutionWorkerError, match="no base commit hash"):
        worker._start_job(locked_job_id)


def test_run_records_candidate_before_push_when_persist_success_fails_gh_candidate_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: worker must durably record the candidate before publishing it."""

    job_id = uuid.uuid4()
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    store = _FakeJobStoreForPublishFailure(
        job_id=job_id,
        events=events,
        persist_error=evolution_module.EvolutionWorkerError("simulated persistence failure"),
    )
    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
        evaluator=_FakeEvaluator(),  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    with pytest.raises(evolution_module.EvolutionWorkerError, match="simulated persistence failure"):
        worker.run(job_id)

    assert "repo.push_branch" in events
    assert "store.record_candidate[published=False]" in events
    assert events.index("store.record_candidate[published=False]") < events.index("repo.push_branch")


def test_run_reuses_exact_source_tree_without_calling_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    store = _FakeJobStoreForSuccess(job_id=job_id, events=events)
    store.reusable_evaluation = EvaluationOutcome(
        evaluator_name="fake-evaluator",
        evaluator_version="v1",
        candidate_commit_hash="candidate123",
        outcome_kind="passed",
        result=EvaluationResult(
            summary="reused",
            extra={"evaluation_reused": True},
        ),
    )

    class NeverCalledEvaluator:
        plugin_ref = "fake-evaluator"
        evaluator_version = "v1"

        def evaluate_outcome(self, _context: Any) -> EvaluationOutcome:
            raise AssertionError("identical source trees must bypass the evaluator")

    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
        evaluator=NeverCalledEvaluator(),  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    result = worker.run(job_id)

    assert result.evaluation.extra["evaluation_reused"] is True
    assert store.reuse_queries == [
        {
            "source_tree_hash": "tree-candidate123",
            "evaluator_name": "fake-evaluator",
            "evaluator_version": "v1",
            "campaign_program_hash": None,
            "candidate_commit_hash": "candidate123",
        }
    ]
    assert store.recorded_candidates[0]["source_tree_hash"] == "tree-candidate123"


def test_run_keeps_candidate_metadata_when_post_push_persistence_fails_gh_candidate_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: a pushed candidate must remain discoverable after post-push failure."""

    job_id = uuid.uuid4()
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    store = _FakeJobStoreForPublishFailure(
        job_id=job_id,
        events=events,
        persist_error=evolution_module.EvolutionWorkerError("simulated persistence failure"),
    )
    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
        evaluator=_FakeEvaluator(),  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    with pytest.raises(evolution_module.EvolutionWorkerError, match="simulated persistence failure"):
        worker.run(job_id)

    assert store.recorded_candidates == [
        {
            "job_id": job_id,
            "commit_hash": "candidate123",
            "branch_name": "exp/job-branch",
            "run_token": store.recorded_candidates[0]["run_token"],
            "published": False,
            "source_tree_hash": "tree-candidate123",
        },
        {
            "job_id": job_id,
            "commit_hash": "candidate123",
            "branch_name": "exp/job-branch",
            "run_token": store.recorded_candidates[1]["run_token"],
            "published": True,
            "source_tree_hash": "tree-candidate123",
        },
    ]
    assert store.failures == [
        {
            "job_id": job_id,
            "message": "simulated persistence failure",
            "run_token": store.recorded_candidates[1]["run_token"],
        }
    ]


def test_run_reworks_candidate_failure_and_publishes_only_final_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    program = parse_campaign_program(b"## Editable scope\n- src/**\n")
    store = _FakeJobStoreForSuccess(
        job_id=job_id,
        events=events,
        campaign_program_hash=program.raw_sha256,
    )
    coding_agent = _UsageFakeCodingAgent()
    evaluator = _SequenceEvaluator(
        [
            EvalFail(kind="typecheck", summary="typecheck failed in src/foo.py"),
            EvaluationOutcome(
                evaluator_name="fake",
                outcome_kind="passed",
                result=EvaluationResult(summary="ok"),
            ),
        ],
        events,
    )
    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=coding_agent,  # type: ignore[arg-type]
        evaluator=evaluator,  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(worker, "_load_campaign_program", lambda _program_hash: program)

    def _scope_pass(**_kwargs: Any) -> ScopeGateResult:
        events.append("scope_gate")
        return ScopeGateResult(checked_paths=("src/foo.py",))

    monkeypatch.setattr(evolution_module, "validate_campaign_scope", _scope_pass)

    result = worker.run(job_id)

    assert result.candidate_commit_hash == "candidate2"
    assert [row["commit_hash"] for row in store.recorded_candidates] == [
        "candidate2",
        "candidate2",
    ]
    assert [row["published"] for row in store.recorded_candidates] == [False, True]
    assert events.count("repo.push_branch") == 1
    assert events.count("evaluator.evaluate") == 2
    assert events.count("scope_gate") == 2
    assert events.count("repo.clean_worktree") == 2
    assert events.count("repo.current_commit") == 2
    rework_reset_index = events.index("repo.reset_mixed_to_commit[base123]")
    cleanup_indexes = [
        index for index, event in enumerate(events) if event == "repo.clean_worktree"
    ]
    assert cleanup_indexes[0] < rework_reset_index
    assert cleanup_indexes[-1] < events.index("store.record_candidate[published=False]")
    assert len(coding_agent.requests) == 2
    assert [request.invocation for request, _worktree in coding_agent.requests] == [1, 2]
    second_request, _worktree = coding_agent.requests[1]
    assert "typecheck failed in src/foo.py" in str(second_request.rework_feedback)
    persisted_coding = store.success_calls[0]["coding"]
    assert [
        event.external_usage_id for event in persisted_coding.usage_events
    ] == ["coding-invocation-1", "coding-invocation-2"]
    success_outcome = store.success_calls[0]["evaluation_outcome"]
    assert success_outcome.artifacts[-1].key == "evaluator_rework_attempts"


def test_run_preserves_path_backed_evaluation_artifacts_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    store = _FakeJobStoreForSuccess(job_id=job_id, events=events)
    worker = EvolutionWorker(
        settings=settings,
        repository=_CleaningRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
        evaluator=_PathArtifactEvaluator(events),  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    worker.run(job_id)

    assert events.index("repo.clean_worktree") < events.index("store.persist_success")
    assert not (tmp_path / "eval-report.txt").exists()
    outcome = store.success_calls[0]["evaluation_outcome"]
    artifact = outcome.result.artifacts[0]
    assert artifact.key == "eval_report"
    assert artifact.path is None
    assert artifact.inline_payload == b"diagnostic report\n"


def test_run_preserves_rework_history_when_retry_coding_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    store = _FakeJobStoreForScopeGateFailure(job_id=job_id, events=events, campaign_program_hash="")
    coding_agent = _SecondPassFailingCodingAgent()
    evaluator = _SequenceEvaluator(
        [_candidate_failed_outcome(summary="first evaluator failure")],
        events,
    )
    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=coding_agent,  # type: ignore[arg-type]
        evaluator=evaluator,  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    with pytest.raises(evolution_module.EvolutionWorkerError, match="simulated retry coding crash"):
        worker.run(job_id)

    assert len(coding_agent.requests) == 2
    assert "store.mark_job_failed" not in events
    failure_call = store.persist_failure_calls[-1]
    assert failure_call["candidate_commit_hash"] is None
    outcome = failure_call["outcome"]
    assert outcome.outcome_kind == "infrastructure_failed"
    artifact = outcome.artifacts[-1]
    assert artifact.key == "evaluator_rework_attempts"
    assert artifact.inline_payload[0]["summary"] == "first evaluator failure"


def test_run_exhausts_rework_without_publishing_failed_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    store = _FakeJobStoreForScopeGateFailure(job_id=job_id, events=events, campaign_program_hash="")
    evaluator = _SequenceEvaluator(
        [
            _candidate_failed_outcome(summary="first failure"),
            _candidate_failed_outcome(summary="second failure"),
        ],
        events,
    )
    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
        evaluator=evaluator,  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    with pytest.raises(evolution_module.EvolutionWorkerError, match="second failure"):
        worker.run(job_id)

    assert store.recorded_candidates == []
    assert "repo.push_branch" not in events
    assert events.count("repo.commit") == 2
    failure_outcome = store.persist_failure_calls[-1]["outcome"]
    assert failure_outcome.artifacts[-1].key == "evaluator_rework_attempts"


def test_run_does_not_rework_non_allowlisted_evalfail_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    job_id = uuid.uuid4()
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    store = _FakeJobStoreForScopeGateFailure(job_id=job_id, events=events, campaign_program_hash="")
    evaluator = _SequenceEvaluator(
        [_candidate_failed_outcome(failure_kind="benchmark_failed", summary="benchmark regressed")],
        events,
    )
    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
        evaluator=evaluator,  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    with pytest.raises(evolution_module.EvolutionWorkerError, match="benchmark regressed"):
        worker.run(job_id)

    assert events.count("evaluator.evaluate") == 1
    assert events.count("repo.commit") == 1
    assert "repo.push_branch" not in events
    assert store.recorded_candidates == []


def test_run_preserves_evaluator_failure_outcome_when_structured_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: fallback persistence must not relabel evaluator failures as infrastructure failures."""

    job_id = uuid.uuid4()
    events: list[str] = []
    _patch_empty_planning_context_session(monkeypatch)
    store = _FakeJobStoreForEvaluationFailureRetry(job_id=job_id, events=events)
    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
        evaluator=_FakeCandidateFailureEvaluator(),  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    with pytest.raises(evolution_module.EvolutionWorkerError, match="typecheck failed"):
        worker.run(job_id)

    assert [outcome.outcome_kind for outcome in store.persist_failure_outcomes] == [
        "candidate_failed",
        "candidate_failed",
    ]
    assert [outcome.failure.failure_kind for outcome in store.persist_failure_outcomes if outcome.failure] == [
        "typecheck_failed",
        "typecheck_failed",
    ]
    assert store.failures == []


def test_run_persists_campaign_scope_violation_before_commit_push_or_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Campaign scope violations stop before candidate publication or evaluation."""

    _init_scope_repo(tmp_path)
    job_id = uuid.uuid4()
    events: list[str] = []
    program = parse_campaign_program(b"## Editable scope\n- src/**\n")
    _patch_empty_planning_context_session(monkeypatch)
    store = _FakeJobStoreForScopeGateFailure(
        job_id=job_id,
        events=events,
        campaign_program_hash=program.raw_sha256,
    )
    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=_ScopeViolatingCodingAgent(events),  # type: ignore[arg-type]
        evaluator=_EventCapturingEvaluator(events),  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(worker, "_load_campaign_program", lambda _program_hash: program)

    with pytest.raises(evolution_module.EvolutionWorkerError, match="Campaign scope gate rejected"):
        worker.run(job_id)

    assert "coding.implement" in events
    assert "store.persist_failure" in events
    assert "repo.commit" not in events
    assert "repo.push_branch" not in events
    assert "store.record_candidate[published=False]" not in events
    assert "evaluator.evaluate" not in events
    assert "store.mark_job_failed" not in events

    failure_call = store.persist_failure_calls[0]
    outcome = failure_call["outcome"]
    assert failure_call["candidate_commit_hash"] is None
    assert outcome.outcome_kind == "candidate_failed"
    assert outcome.failure is not None
    assert outcome.failure.failure_stage == "policy"
    assert outcome.failure.failure_kind == "campaign_scope_violation"
    artifact = outcome.artifacts[0]
    assert artifact.key == "campaign_scope_violation"
    assert artifact.kind == "policy_failure"
    assert artifact.inline_payload["violations"][0]["code"] == "outside_editable_scope"
    assert artifact.inline_payload["violations"][0]["path"] == "README.md"


def test_run_fails_closed_when_referenced_campaign_program_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    """Regression: missing campaign program rows must not bypass scope policy."""

    job_id = uuid.uuid4()
    events: list[str] = []
    missing_hash = "a" * 64
    store = _FakeJobStoreForPublishFailure(
        job_id=job_id,
        events=events,
        persist_error=evolution_module.EvolutionWorkerError("unused"),
        campaign_program_hash=missing_hash,
    )
    worker = EvolutionWorker(
        settings=settings,
        repository=_FakeRepositoryForRun(worktree=tmp_path, events=events),  # type: ignore[arg-type]
        planning_agent=_FakePlanningAgent(),  # type: ignore[arg-type]
        coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
        evaluator=_EventCapturingEvaluator(events),  # type: ignore[arg-type]
        job_store=store,  # type: ignore[arg-type]
    )

    @contextmanager
    def fake_session_scope() -> Any:
        yield object()

    monkeypatch.setattr(evolution_module, "session_scope", fake_session_scope)
    monkeypatch.setattr(
        evolution_module,
        "load_campaign_program_snapshot_by_hash",
        lambda **_kwargs: None,
    )

    with pytest.raises(evolution_module.EvolutionWorkerError, match="refusing to run without contract"):
        worker.run(job_id)

    assert "repo.checkout" not in events
    assert "coding.implement" not in events
    assert "evaluator.evaluate" not in events
    assert store.failures[0]["job_id"] == job_id
    assert missing_hash in store.failures[0]["message"]


def test_run_evaluation_payload_includes_campaign_program_and_runtime_provenance(
    tmp_path: Path,
    settings: Settings,
) -> None:
    evaluator = _CapturingOutcomeEvaluator()
    worker = EvolutionWorker(
        settings=settings,
        repository=object(),  # type: ignore[arg-type]
        planning_agent=object(),  # type: ignore[arg-type]
        coding_agent=object(),  # type: ignore[arg-type]
        evaluator=evaluator,  # type: ignore[arg-type]
        job_store=object(),  # type: ignore[arg-type]
    )
    program = parse_campaign_program(
        b"""## Goal
Program goal.

## Primary metric
name: throughput
direction: higher_is_better
unit: req/s
"""
    )
    job_ctx = _make_job_context()
    job_ctx.campaign_program_hash = program.raw_sha256
    job_ctx.campaign_program = program
    job_ctx.constraints = ("Correctness gate: pytest",)
    job_ctx.acceptance_criteria = ("Primary metric: throughput",)
    plan = PlanningAgentResponse(
        plan=PlanDocument(summary="plan", markdown="## Summary\n- plan\n"),
        raw_output="raw",
        prompt="prompt",
        command=("cmd",),
        stderr="",
        attempts=1,
        duration_seconds=0.1,
    )
    checkout = CheckoutContext(
        job_id=str(job_ctx.job_id),
        branch_name="branch",
        base_commit=job_ctx.base_commit_hash,
        worktree=tmp_path,
    )

    outcome = worker._run_evaluation(  # noqa: SLF001
        job_ctx=job_ctx,
        checkout=checkout,
        plan=plan,
        candidate_commit="candidate",
    )

    assert outcome.outcome_kind == "passed"
    context = evaluator.contexts[0]
    assert context.payload["campaign_program"]["hash"] == program.raw_sha256
    assert context.payload["campaign_program"]["snapshot"]["primary_metric"]["name"] == "throughput"
    assert context.payload["job"]["constraints"] == ["Correctness gate: pytest"]
    assert context.metadata["campaign_program_hash"] == program.raw_sha256
    assert context.metadata["runtime_profile"] == settings.profile
    assert len(context.metadata["effective_settings_fingerprint"]) == 64
