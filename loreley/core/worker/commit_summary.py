"""Commit summarization utilities for the evolution worker."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger
from openai import OpenAI, OpenAIError
from tenacity import RetryError

from loreley.config import Settings, get_settings
from loreley.core.openai_auth import get_internal_openai_api_key
from loreley.core.openai_retry import openai_retrying, retry_error_details
from loreley.core.worker.coding import ExecutionReport
from loreley.core.worker.planning import PlanDocument

if TYPE_CHECKING:
    from loreley.core.worker.evolution import JobContext

log = logger.bind(module="worker.commit_summary")

__all__ = ["CommitSummarizer", "CommitSummaryError", "CommitSummaryUnavailableError"]


class CommitSummaryError(RuntimeError):
    """Raised when the commit summarizer cannot produce a subject line."""


class CommitSummaryUnavailableError(RuntimeError):
    """Raised when the commit summarizer cannot initialize its client."""


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
        self._model = self.settings.worker_evolution_commit_model
        self._temperature = self.settings.worker_evolution_commit_temperature
        self._max_tokens = max(32, self.settings.worker_evolution_commit_max_output_tokens)
        self._max_retries = max(1, self.settings.worker_evolution_commit_max_retries)
        self._retry_backoff = max(
            0.0,
            self.settings.worker_evolution_commit_retry_backoff_seconds,
        )
        self._subject_limit = max(32, self.settings.worker_evolution_commit_subject_max_chars)
        self._truncate_limit = 1200
        self._api_spec = self.settings.openai_api_spec

    def generate(
        self,
        *,
        job: JobContext,
        plan: PlanDocument,
        coding: ExecutionReport,
    ) -> str:
        """Return a commit subject line grounded in plan and coding context."""
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
                        response = client.responses.create(
                            model=self._model,
                            input=prompt,
                            temperature=self._temperature,
                            max_output_tokens=self._max_tokens,
                            instructions=instructions,
                        )
                        subject = (response.output_text or "").strip()
                    else:
                        response = client.chat.completions.create(
                            model=self._model,
                            messages=[
                                {"role": "system", "content": instructions},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=self._temperature,
                            max_tokens=self._max_tokens,
                        )
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
        if self._client is not None:
            return self._client
        try:
            if self._client_factory is not None:
                return self._client_factory()
            client_kwargs: dict[str, object] = {}
            api_key = get_internal_openai_api_key(self.settings)
            if api_key:
                client_kwargs["api_key"] = api_key
            if self.settings.openai_base_url:
                client_kwargs["base_url"] = self.settings.openai_base_url
            return OpenAI(**client_kwargs) if client_kwargs else OpenAI()
        except Exception as exc:
            raise CommitSummaryUnavailableError(
                f"Commit summarizer could not initialize an OpenAI client: {exc}",
            ) from exc

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
