from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from openai import OpenAIError

from loreley.config import Settings
import loreley.core.worker.commit_summary as commit_summary_module
from loreley.core.worker.coding import CodingAgentResponse, ExecutionReport
from loreley.core.worker.commit_summary import (
    CommitSummarizer,
    CommitSummaryError,
    CommitSummaryUnavailableError,
)
from loreley.core.worker.evolution import EvolutionWorker, JobContext
from loreley.core.worker.planning import PlanDocument, PlanningAgentResponse


def _make_plan() -> PlanDocument:
    return PlanDocument(
        summary="plan summary",
        markdown="## Summary\n- plan summary\n",
        focus_metrics=("quality",),
        guardrails=("guard",),
    )


def _make_coding_execution() -> ExecutionReport:
    return ExecutionReport(
        summary="implemented feature",
        markdown="## Summary\n- implemented feature\n",
    )


def _make_job_context() -> JobContext:
    return JobContext(
        job_id=uuid4(),
        run_token=uuid4(),
        base_commit_hash="abc",
        island_id=None,
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


def _make_planning_response() -> PlanningAgentResponse:
    return PlanningAgentResponse(
        plan=_make_plan(),
        raw_output="raw plan",
        prompt="prompt",
        command=("planner",),
        stderr="",
        attempts=1,
        duration_seconds=0.1,
    )


def _make_coding_response() -> CodingAgentResponse:
    return CodingAgentResponse(
        report=_make_coding_execution(),
        raw_output="raw coding",
        prompt="prompt",
        command=("coder",),
        stderr="",
        attempts=1,
        duration_seconds=0.1,
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


def test_default_provider_inherits_kilocode_openai_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.model_validate(
        {
            "WORKER_KILOCODE_OPENAI_API_KEY": "sk-worker",
            "WORKER_KILOCODE_OPENAI_BASE_URL": "https://worker.example.com/v1",
            "WORKER_KILOCODE_OPENAI_MODEL": "worker-summary-model",
            "WORKER_KILOCODE_OPENAI_API_SPEC": "chat_completions",
            "OPENAI_API_KEY": "sk-global",
            "OPENAI_BASE_URL": "https://global.example.com/v1",
            "OPENAI_API_SPEC": "responses",
        }
    )
    client_kwargs: list[dict[str, object]] = []
    chat_calls: list[dict[str, object]] = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            chat_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Inherit worker"))]
            )

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeChatCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    monkeypatch.setattr(
        commit_summary_module,
        "OpenAI",
        lambda **kwargs: client_kwargs.append(kwargs) or FakeClient(),
    )

    summarizer = CommitSummarizer(settings=settings)
    subject = summarizer.generate(
        job=_make_job_context(),
        plan=_make_plan(),
        coding=_make_coding_execution(),
    )

    assert subject == "Inherit worker"
    assert client_kwargs == [
        {
            "api_key": "sk-worker",
            "base_url": "https://worker.example.com/v1",
        }
    ]
    assert chat_calls[0]["model"] == "worker-summary-model"


def test_custom_provider_overrides_worker_and_global_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.model_validate(
        {
            "WORKER_EVOLUTION_COMMIT_PROVIDER_MODE": "custom",
            "WORKER_EVOLUTION_COMMIT_API_KEY": "sk-commit",
            "WORKER_EVOLUTION_COMMIT_BASE_URL": "https://commit.example.com/v1",
            "WORKER_EVOLUTION_COMMIT_API_SPEC": "chat_completions",
            "WORKER_EVOLUTION_COMMIT_MODEL": "commit-model",
            "WORKER_KILOCODE_OPENAI_API_KEY": "sk-worker",
            "WORKER_KILOCODE_OPENAI_BASE_URL": "https://worker.example.com/v1",
            "WORKER_KILOCODE_OPENAI_MODEL": "worker-model",
            "OPENAI_API_KEY": "sk-global",
            "OPENAI_BASE_URL": "https://global.example.com/v1",
        }
    )
    client_kwargs: list[dict[str, object]] = []
    chat_calls: list[dict[str, object]] = []

    class FakeChatCompletions:
        def create(self, **kwargs):
            chat_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Use commit provider"))]
            )

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeChatCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    monkeypatch.setattr(
        commit_summary_module,
        "OpenAI",
        lambda **kwargs: client_kwargs.append(kwargs) or FakeClient(),
    )

    summarizer = CommitSummarizer(settings=settings)
    subject = summarizer.generate(
        job=_make_job_context(),
        plan=_make_plan(),
        coding=_make_coding_execution(),
    )

    assert subject == "Use commit provider"
    assert client_kwargs == [
        {
            "api_key": "sk-commit",
            "base_url": "https://commit.example.com/v1",
        }
    ]
    assert chat_calls[0]["model"] == "commit-model"


def test_disabled_provider_mode_does_not_call_llm(settings: Settings) -> None:
    settings.worker_evolution_commit_provider_mode = "disabled"
    calls = {"count": 0}

    class FakeResponses:
        def create(self, **kwargs):
            calls["count"] += 1
            return SimpleNamespace(output_text="should not happen")

    summarizer = CommitSummarizer(
        settings=settings,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    with pytest.raises(CommitSummaryUnavailableError, match="disabled"):
        summarizer.generate(
            job=_make_job_context(),
            plan=_make_plan(),
            coding=_make_coding_execution(),
        )

    assert calls["count"] == 0


def test_custom_provider_without_api_key_does_not_call_llm(settings: Settings) -> None:
    settings.worker_evolution_commit_provider_mode = "custom"
    settings.worker_evolution_commit_api_key = None
    calls = {"count": 0}

    class FakeResponses:
        def create(self, **kwargs):
            calls["count"] += 1
            return SimpleNamespace(output_text="should not happen")

    summarizer = CommitSummarizer(
        settings=settings,
        client=SimpleNamespace(responses=FakeResponses()),
    )

    with pytest.raises(CommitSummaryUnavailableError, match="API_KEY"):
        summarizer.generate(
            job=_make_job_context(),
            plan=_make_plan(),
            coding=_make_coding_execution(),
        )

    assert calls["count"] == 0


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


def test_generate_does_not_retry_non_retryable_4xx(settings: Settings) -> None:
    settings.worker_evolution_commit_max_retries = 3
    settings.worker_evolution_commit_retry_backoff_seconds = 0

    class PermissionDenied(OpenAIError):
        status_code = 403

    class FailingResponses:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise PermissionDenied("model unavailable")

    responses = FailingResponses()
    summarizer = CommitSummarizer(
        settings=settings,
        client=SimpleNamespace(responses=responses),
    )

    with pytest.raises(CommitSummaryUnavailableError, match="status=403"):
        summarizer.generate(
            job=_make_job_context(),
            plan=_make_plan(),
            coding=_make_coding_execution(),
        )

    assert responses.calls == 1


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


def test_generate_surfaces_client_initialization_failures_without_retry(
    settings: Settings,
) -> None:
    calls = {"count": 0}
    summarizer = CommitSummarizer(settings=settings)
    summarizer._client_factory = lambda: (
        calls.__setitem__("count", calls["count"] + 1),
        (_ for _ in ()).throw(RuntimeError("missing key")),
    )[1]  # type: ignore[method-assign]  # noqa: SLF001

    with pytest.raises(CommitSummaryUnavailableError, match="missing key"):
        summarizer.generate(
            job=_make_job_context(),
            plan=_make_plan(),
            coding=_make_coding_execution(),
        )
    assert calls["count"] == 1


def test_generate_rebuilds_client_with_current_runtime_api_key_for_each_retry(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.worker_evolution_commit_provider_mode = "global_openai"
    settings.worker_evolution_commit_max_retries = 2
    settings.worker_evolution_commit_retry_backoff_seconds = 0

    seen_api_keys: list[str] = []
    api_keys = iter(["dyn-1", "dyn-2"])

    class _FailingResponses:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            raise OpenAIError("boom")  # type: ignore[arg-type]

    class _SuccessResponses:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(output_text="Fix dynamic auth")

    clients = iter(
        [
            SimpleNamespace(responses=_FailingResponses()),
            SimpleNamespace(responses=_SuccessResponses()),
        ]
    )

    monkeypatch.setattr(
        commit_summary_module,
        "get_internal_openai_api_key",
        lambda _settings: next(api_keys),
    )
    monkeypatch.setattr(
        commit_summary_module,
        "OpenAI",
        lambda **kwargs: seen_api_keys.append(str(kwargs["api_key"])) or next(clients),
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    summarizer = CommitSummarizer(settings=settings)

    subject = summarizer.generate(
        job=_make_job_context(),
        plan=_make_plan(),
        coding=_make_coding_execution(),
    )

    assert subject == "Fix dynamic auth"
    assert seen_api_keys == ["dyn-1", "dyn-2"]


def test_worker_commit_message_falls_back_when_summarizer_disabled(settings: Settings) -> None:
    settings.worker_evolution_commit_provider_mode = "disabled"
    worker = object.__new__(EvolutionWorker)
    worker.summarizer = CommitSummarizer(settings=settings)

    subject = worker._prepare_commit_message(  # noqa: SLF001
        job_ctx=_make_job_context(),
        plan=_make_planning_response(),
        coding=_make_coding_response(),
    )

    assert subject == "implemented feature"
