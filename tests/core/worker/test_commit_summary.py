from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from openai import OpenAIError

from loreley.config import Settings
from loreley.core.worker.coding import CodingPlanExecution, CodingStepReport, StepExecutionStatus
from loreley.core.worker.commit_summary import CommitSummarizer, CommitSummaryError
from loreley.core.worker.evolution import JobContext
from loreley.core.worker.planning import PlanStep, PlanningPlan


def _make_plan() -> PlanningPlan:
    step = PlanStep(
        step_id="s1",
        title="Implement",
        intent="Do work",
        actions=("edit",),
        files=(),
        dependencies=(),
        validation=("tests",),
        risks=(),
        references=(),
    )
    return PlanningPlan(
        summary="plan summary",
        rationale="plan rationale",
        focus_metrics=("quality",),
        guardrails=("guard",),
        risks=("risk1",),
        validation=("val",),
        steps=(step,),
        handoff_notes=(),
        fallback_plan=None,
    )


def _make_coding_execution() -> CodingPlanExecution:
    step_report = CodingStepReport(
        step_id="s1",
        status=StepExecutionStatus.COMPLETED,
        summary="done",
        files=("file.py",),
        commands=(),
    )
    return CodingPlanExecution(
        implementation_summary="implemented feature",
        commit_message="fallback commit",
        step_results=(step_report,),
        tests_executed=("pytest",),
        tests_recommended=(),
        follow_up_items=(),
        notes=(),
    )


def _make_job_context() -> JobContext:
    return JobContext(
        job_id=uuid4(),
        base_commit_hash="abc",
        island_id=None,
        experiment_id=None,
        repository_id=None,
        inspiration_commit_hashes=(),
        goal="Improve docs",
        constraints=("c1",),
        acceptance_criteria=("a1",),
        iteration_hint=None,
        notes=("note",),
        tags=(),
        is_seed_job=False,
        sampling_strategy=None,
        sampling_initial_radius=None,
        sampling_radius_used=None,
        sampling_fallback_inspirations=None,
    )


def test_generate_subject_with_responses_api(settings: Settings) -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(output_text="  Fix bugs  ")

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    client = FakeClient()
    summarizer = CommitSummarizer(settings=settings, client=client)

    subject = summarizer.generate(
        job=_make_job_context(),
        plan=_make_plan(),
        coding=_make_coding_execution(),
    )

    assert subject == "Fix bugs"
    assert client.responses.calls


def test_generate_subject_truncates_for_chat_api(settings: Settings) -> None:
    settings.openai_api_spec = "chat_completions"
    settings.worker_evolution_commit_subject_max_chars = 20

    long_text = "This subject is intentionally longer than allowed characters."

    class FakeChatCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=long_text))])

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeChatCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    summarizer = CommitSummarizer(settings=settings, client=FakeClient())

    subject = summarizer.generate(
        job=_make_job_context(),
        plan=_make_plan(),
        coding=_make_coding_execution(),
    )

    assert subject.endswith("…")
    assert len(subject) <= summarizer._subject_limit  # type: ignore[attr-defined]


def test_generate_retries_and_raises_after_failures(settings: Settings, monkeypatch) -> None:
    settings.worker_evolution_commit_max_retries = 2
    settings.worker_evolution_commit_retry_backoff_seconds = 0

    class FailingResponses:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise OpenAIError("boom")  # type: ignore[arg-type]

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FailingResponses()

    summarizer = CommitSummarizer(settings=settings, client=FakeClient())

    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(CommitSummaryError) as excinfo:
        summarizer.generate(
            job=_make_job_context(),
            plan=_make_plan(),
            coding=_make_coding_execution(),
        )

    assert "2 attempt" in str(excinfo.value)


def test_coerce_subject_trims_and_defaults(settings: Settings) -> None:
    summarizer = CommitSummarizer(settings=settings, client=SimpleNamespace())

    assert summarizer.coerce_subject("  spaced  subject  ", default="fallback") == "spaced subject"
    assert summarizer.coerce_subject("", default="fallback value") == "fallback value"


def test_extract_chat_completion_text_merges_parts() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[SimpleNamespace(text="hello"), " ", SimpleNamespace(text="world")]
                )
            )
        ]
    )
    text = CommitSummarizer._extract_chat_completion_text(response)
    assert text == "hello world"
