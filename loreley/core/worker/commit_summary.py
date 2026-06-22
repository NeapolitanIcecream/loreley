"""Commit summarization utilities for the evolution worker."""

from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import TYPE_CHECKING, Any, Callable, Literal

from loguru import logger
from openai import OpenAI, OpenAIError
from tenacity import RetryError

from loreley.config import Settings, get_settings
from loreley.core.openai_auth import get_agent_openai_api_key, get_internal_openai_api_key
from loreley.core.openai_retry import openai_retrying, retry_error_details
from loreley.core.usage import normalize_openai_usage_event, record_usage_event
from loreley.core.worker.coding import ExecutionReport
from loreley.core.worker.planning import PlanDocument

if TYPE_CHECKING:
    from loreley.core.worker.evolution import JobContext

log = logger.bind(module="worker.commit_summary")

__all__ = ["CommitSummarizer", "CommitSummaryError", "CommitSummaryUnavailableError"]

_ApiSpec = Literal["responses", "chat_completions"]
_ProviderMode = Literal["inherit_worker", "global_openai", "custom", "disabled"]
_NON_RETRYABLE_4XX_EXCEPTIONS = {408, 409, 425, 429}


class CommitSummaryError(RuntimeError):
    """Raised when the commit summarizer cannot produce a subject line."""


class CommitSummaryUnavailableError(RuntimeError):
    """Raised when the commit summarizer cannot initialize its client."""


@dataclass(frozen=True, slots=True)
class _CommitProviderConfig:
    mode: _ProviderMode
    model: str
    api_spec: _ApiSpec
    base_url: str | None = None
    api_key_required: bool = False
    unavailable_reason: str | None = None


class CommitSummarizer:
    """LLM-powered helper that derives concise commit subjects."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client: Any | None = client
        self._client_factory: Callable[[], Any] | None = None
        self._provider = _resolve_commit_provider(self.settings)
        self._model = self._provider.model
        self._temperature = self.settings.worker_evolution_commit_temperature
        self._max_tokens = max(32, self.settings.worker_evolution_commit_max_output_tokens)
        self._max_retries = max(1, self.settings.worker_evolution_commit_max_retries)
        self._retry_backoff = max(
            0.0,
            self.settings.worker_evolution_commit_retry_backoff_seconds,
        )
        self._subject_limit = max(32, self.settings.worker_evolution_commit_subject_max_chars)
        self._truncate_limit = 1200
        self._api_spec = self._provider.api_spec

    def generate(
        self,
        *,
        job: JobContext,
        plan: PlanDocument,
        coding: ExecutionReport,
    ) -> str:
        """Return a commit subject line grounded in plan and coding context."""
        if self._provider.mode == "disabled":
            raise CommitSummaryUnavailableError(
                self._provider.unavailable_reason
                or "Commit summarizer is disabled by configuration.",
            )
        prompt = self._build_prompt(job=job, plan=plan, coding=coding)
        retryer = openai_retrying(
            max_attempts=self._max_retries,
            backoff_seconds=self._retry_backoff,
            retry_on=(OpenAIError, CommitSummaryError),
            log=log,
            operation="Commit summarizer",
        )
        try:
            for attempt in retryer:
                with attempt:
                    attempt_number = int(getattr(attempt.retry_state, "attempt_number", 0) or 0)
                    client = self._get_client()
                    instructions = (
                        "Respond with a single concise git commit subject line "
                        f"in imperative mood (<= {self._subject_limit} characters), "
                        "without surrounding quotes."
                    )
                    if self._api_spec == "responses":
                        try:
                            response = client.responses.create(
                                model=self._model,
                                input=prompt,
                                temperature=self._temperature,
                                max_output_tokens=self._max_tokens,
                                instructions=instructions,
                            )
                        except OpenAIError as exc:
                            self._raise_if_non_retryable_provider_error(exc)
                            raise
                        self._record_usage(response, job=job, api_surface="responses")
                        subject = (response.output_text or "").strip()
                    else:
                        try:
                            response = client.chat.completions.create(
                                model=self._model,
                                messages=[
                                    {"role": "system", "content": instructions},
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=self._temperature,
                                max_tokens=self._max_tokens,
                            )
                        except OpenAIError as exc:
                            self._raise_if_non_retryable_provider_error(exc)
                            raise
                        self._record_usage(response, job=job, api_surface="chat_completions")
                        subject = self._extract_chat_completion_text(response).strip()
                    if not subject:
                        raise CommitSummaryError("Commit summarizer returned empty output.")
                    cleaned = self._normalise_subject(subject)
                    log.info("Commit summarizer produced subject after attempt {}", attempt_number)
                    return cleaned
            raise CommitSummaryError("Commit summarizer exhausted retries without success.")
        except RetryError as exc:
            attempts, last_exc = retry_error_details(exc, default_attempts=self._max_retries)
            raise CommitSummaryError(
                f"Commit summarizer failed after {attempts} attempt(s): {last_exc}",
            ) from last_exc

    def _get_client(self) -> Any:
        if self._provider.mode == "custom" and self._provider.unavailable_reason:
            raise CommitSummaryUnavailableError(self._provider.unavailable_reason)
        if self._client is not None:
            return self._client
        try:
            if self._client_factory is not None:
                return self._client_factory()
            if self._provider.unavailable_reason:
                raise CommitSummaryUnavailableError(self._provider.unavailable_reason)
            client_kwargs = self._client_kwargs()
            return OpenAI(**client_kwargs) if client_kwargs else OpenAI()
        except CommitSummaryUnavailableError:
            raise
        except Exception as exc:
            raise CommitSummaryUnavailableError(
                f"Commit summarizer could not initialize an OpenAI client: {exc}",
            ) from exc

    def _client_kwargs(self) -> dict[str, object]:
        client_kwargs: dict[str, object] = {}
        api_key = self._resolve_api_key()
        if api_key:
            client_kwargs["api_key"] = api_key
        elif self._provider.api_key_required:
            raise CommitSummaryUnavailableError(
                "Commit summarizer provider requires an API key, but none was resolved.",
            )
        if self._provider.base_url:
            client_kwargs["base_url"] = self._provider.base_url
        return client_kwargs

    def _resolve_api_key(self) -> str | None:
        if self._provider.mode == "custom":
            api_key = _stripped_setting(self.settings, "worker_evolution_commit_api_key")
            if api_key:
                return api_key
            return None
        if self._provider.mode == "inherit_worker":
            return get_agent_openai_api_key(self.settings)
        if self._provider.mode == "global_openai":
            return get_internal_openai_api_key(self.settings)
        return None

    def _raise_if_non_retryable_provider_error(self, exc: OpenAIError) -> None:
        status_code = _openai_status_code(exc)
        if not _is_non_retryable_openai_status(status_code):
            return
        raise CommitSummaryUnavailableError(
            "Commit summarizer provider rejected the request without retry "
            f"(status={status_code}): {exc}",
        ) from exc

    def _record_usage(self, response: object, *, job: JobContext, api_surface: str) -> None:
        event = normalize_openai_usage_event(
            response,
            phase="commit_summary",
            model=self._model,
            api_surface=api_surface,
            job_id=job.job_id,
            run_token=job.run_token,
            settings=self.settings,
        )
        record_usage_event(event, settings=self.settings)

    def _build_prompt(
        self,
        *,
        job: JobContext,
        plan: PlanDocument,
        coding: ExecutionReport,
    ) -> str:
        goal = job.goal.strip()
        plan_summary = (plan.summary or "").strip() or "N/A"
        coding_summary = (coding.summary or "").strip()
        if not coding_summary:
            coding_summary = "N/A"
        coding_summary = self._truncate(" ".join(coding_summary.split()))
        report_excerpt = self._truncate(" ".join((coding.markdown or "").split()), limit=1200) or "N/A"

        prompt = f"""
You generate precise git commit subjects for an autonomous evolution worker.

Goal:
{goal}

Plan summary:
{plan_summary}

Coding summary:
{coding_summary}

Coding report (excerpt):
{report_excerpt}

Respond with a single subject line without surrounding quotes.
"""
        return textwrap.dedent(prompt).strip()

    def _normalise_subject(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) > self._subject_limit:
            return f"{cleaned[: self._subject_limit - 1].rstrip()}…"
        return cleaned

    def coerce_subject(self, text: str | None, *, default: str) -> str:
        """Clamp arbitrary text into a valid git subject."""
        baseline = " ".join((text or "").split()).strip()
        candidate = baseline or default.strip()
        return self._normalise_subject(candidate or default)

    def _truncate(self, text: str, limit: int | None = None) -> str:
        active = limit or self._truncate_limit
        snippet = (text or "").strip()
        if len(snippet) <= active:
            return snippet
        return f"{snippet[:active]}…"

    @staticmethod
    def _extract_chat_completion_text(response: Any) -> str:
        """Extract assistant text content from a chat completion response."""

        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return str(content or "")


def _resolve_commit_provider(settings: Settings) -> _CommitProviderConfig:
    mode = _commit_provider_mode(settings)
    if mode == "disabled":
        return _CommitProviderConfig(
            mode=mode,
            model=_commit_model(settings),
            api_spec=_commit_api_spec(settings),
            unavailable_reason=(
                "Commit summarizer disabled by "
                "WORKER_EVOLUTION_COMMIT_PROVIDER_MODE=disabled."
            ),
        )
    if mode == "custom":
        return _custom_commit_provider(settings)
    if mode == "global_openai":
        return _global_openai_commit_provider(settings)
    return _inherited_worker_commit_provider(settings)


def _commit_provider_mode(settings: Settings) -> _ProviderMode:
    raw = _stripped_setting(settings, "worker_evolution_commit_provider_mode")
    normalized = raw.lower() or "inherit_worker"
    if normalized in {"inherit_worker", "global_openai", "custom", "disabled"}:
        return normalized  # type: ignore[return-value]
    return "inherit_worker"


def _custom_commit_provider(settings: Settings) -> _CommitProviderConfig:
    api_key = _stripped_setting(settings, "worker_evolution_commit_api_key")
    unavailable_reason = None
    if not api_key:
        unavailable_reason = (
            "Commit summarizer custom provider requires "
            "WORKER_EVOLUTION_COMMIT_API_KEY."
        )
    return _CommitProviderConfig(
        mode="custom",
        model=_commit_model(settings),
        api_spec=_commit_api_spec(settings),
        base_url=_stripped_setting(settings, "worker_evolution_commit_base_url") or None,
        api_key_required=True,
        unavailable_reason=unavailable_reason,
    )


def _global_openai_commit_provider(settings: Settings) -> _CommitProviderConfig:
    return _CommitProviderConfig(
        mode="global_openai",
        model=_commit_model(settings),
        api_spec=getattr(settings, "openai_api_spec", "responses"),
        base_url=_stripped_setting(settings, "openai_base_url") or None,
    )


def _inherited_worker_commit_provider(settings: Settings) -> _CommitProviderConfig:
    model = _commit_model(settings)
    api_spec = getattr(settings, "openai_api_spec", "responses")
    if not _uses_kilocode_backend(settings):
        return _CommitProviderConfig(
            mode="inherit_worker",
            model=model,
            api_spec=api_spec,
            unavailable_reason=(
                "Commit summarizer provider mode inherit_worker found no "
                "OpenAI-compatible worker provider to inherit."
            ),
        )
    if _stripped_setting(settings, "worker_kilocode_provider_config_mode").lower() == "none":
        return _CommitProviderConfig(
            mode="inherit_worker",
            model=model,
            api_spec=api_spec,
            unavailable_reason=(
                "Commit summarizer provider mode inherit_worker cannot inherit "
                "Kilo persisted auth/config when WORKER_KILOCODE_PROVIDER_CONFIG_MODE=none."
            ),
        )

    worker_base_url = _stripped_setting(settings, "worker_kilocode_openai_base_url")
    base_url = worker_base_url or _stripped_setting(settings, "openai_base_url")
    worker_model = _stripped_setting(settings, "worker_kilocode_openai_model")
    worker_api_spec = getattr(settings, "worker_kilocode_openai_api_spec", None)
    if worker_api_spec:
        api_spec = worker_api_spec
    if worker_model and not _commit_model_was_configured(settings):
        model = worker_model

    has_api_key_source = _has_inheritable_kilocode_api_key_source(settings)
    has_provider_config = any((has_api_key_source, base_url, worker_model, worker_api_spec))
    if not has_provider_config:
        return _CommitProviderConfig(
            mode="inherit_worker",
            model=model,
            api_spec=api_spec,
            unavailable_reason=(
                "Commit summarizer provider mode inherit_worker found no "
                "WORKER_KILOCODE_OPENAI_* or global OpenAI-compatible provider settings."
            ),
        )
    if not has_api_key_source:
        return _CommitProviderConfig(
            mode="inherit_worker",
            model=model,
            api_spec=api_spec,
            base_url=base_url or None,
            unavailable_reason=(
                "Commit summarizer provider mode inherit_worker found a Kilo "
                "provider but no API key source."
            ),
        )
    return _CommitProviderConfig(
        mode="inherit_worker",
        model=model,
        api_spec=api_spec,
        base_url=base_url or None,
        api_key_required=True,
    )


def _commit_model(settings: Settings) -> str:
    return _stripped_setting(settings, "worker_evolution_commit_model") or "gpt-4.1-mini"


def _commit_api_spec(settings: Settings) -> _ApiSpec:
    raw = _stripped_setting(settings, "worker_evolution_commit_api_spec") or "responses"
    if raw == "chat_completions":
        return "chat_completions"
    return "responses"


def _commit_model_was_configured(settings: Settings) -> bool:
    fields_set = getattr(settings, "model_fields_set", set())
    return "worker_evolution_commit_model" in fields_set


def _uses_kilocode_backend(settings: Settings) -> bool:
    backend_refs = (
        _stripped_setting(settings, "worker_coding_backend"),
        _stripped_setting(settings, "worker_planning_backend"),
    )
    return any("kilocode_cli" in ref or ":kilocode_" in ref for ref in backend_refs)


def _has_inheritable_kilocode_api_key_source(settings: Settings) -> bool:
    values = (
        _stripped_setting(settings, "worker_kilocode_openai_api_key"),
        _stripped_setting(settings, "openai_api_key"),
        _stripped_setting(settings, "openai_dynamic_api_key_provider"),
    )
    return any(values)


def _stripped_setting(settings: Settings, name: str) -> str:
    return str(getattr(settings, name, "") or "").strip()


def _openai_status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _is_non_retryable_openai_status(status_code: int | None) -> bool:
    if status_code is None:
        return False
    if not 400 <= status_code < 500:
        return False
    return status_code not in _NON_RETRYABLE_4XX_EXCEPTIONS
