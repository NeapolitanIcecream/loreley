"""Commit summarization utilities for the evolution worker."""

from __future__ import annotations

import textwrap
import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import OpenAI, OpenAIError

from loreley.config import Settings, get_settings
from loreley.core.worker.coding import CodingPlanExecution
from loreley.core.worker.planning import PlanningPlan

if TYPE_CHECKING:
    from loreley.core.worker.evolution import JobContext

log = logger.bind(module="worker.commit_summary")

__all__ = ["CommitSummarizer", "CommitSummaryError"]


class CommitSummaryError(RuntimeError):
    """Raised when the commit summarizer cannot produce a subject line."""


class CommitSummarizer:
    """LLM-powered helper that derives concise commit subjects."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if client is not None:
            self._client = client
        else:
            client_kwargs: dict[str, object] = {}
            if self.settings.openai_api_key:
                client_kwargs["api_key"] = self.settings.openai_api_key
            if self.settings.openai_base_url:
                client_kwargs["base_url"] = self.settings.openai_base_url
            self._client = (
                OpenAI(**client_kwargs)  # type: ignore[call-arg]
                if client_kwargs
                else OpenAI()
            )
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
        plan: PlanningPlan,
        coding: CodingPlanExecution,
    ) -> str:
        """Return a commit subject line grounded in plan and coding context."""
        prompt = self._build_prompt(job=job, plan=plan, coding=coding)
        attempt = 0
        while attempt < self._max_retries:
            attempt += 1
            try:
                instructions = (
                    "Respond with a single concise git commit subject line "
                    f"in imperative mood (<= {self._subject_limit} characters)."
                )
                if self._api_spec == "responses":
                    response = self._client.responses.create(
                        model=self._model,
                        input=prompt,
                        temperature=self._temperature,
                        max_output_tokens=self._max_tokens,
                        instructions=instructions,
                    )
                    subject = (response.output_text or "").strip()
                else:
                    response = self._client.chat.completions.create(
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
                log.info("Commit summarizer produced subject after attempt {}", attempt)
                return cleaned
            except (OpenAIError, CommitSummaryError) as exc:
                if attempt >= self._max_retries:
                    raise CommitSummaryError(
                        f"Commit summarizer failed after {attempt} attempt(s): {exc}",
                    ) from exc
                delay = self._retry_backoff * attempt
                log.warning(
                    "Commit summarizer attempt {} failed: {}. Retrying in {:.1f}s",
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise CommitSummaryError("Commit summarizer exhausted retries without success.")

    def _build_prompt(
        self,
        *,
        job: JobContext,
        plan: PlanningPlan,
        coding: CodingPlanExecution,
    ) -> str:
        goal = job.goal.strip()
        plan_summary = plan.summary.strip()
        plan_rationale = plan.rationale.strip()
        step_lines = "\n".join(
            f"- {step.step_id} ({step.status.value}): {self._truncate(step.summary)}"
            for step in coding.step_results
        ) or "- No detailed step results."
        tests = "\n".join(f"- {item}" for item in coding.tests_executed) or "- None"
        focus_metrics = "\n".join(f"- {metric}" for metric in plan.focus_metrics) or "- None"
        guardrails = "\n".join(f"- {guardrail}" for guardrail in plan.guardrails) or "- None"
        constraints = "\n".join(f"- {entry}" for entry in job.constraints) or "- None"
        acceptance = "\n".join(f"- {entry}" for entry in job.acceptance_criteria) or "- None"
        notes = "\n".join(f"- {entry}" for entry in job.notes) or "- None"
        coding_summary = coding.implementation_summary.strip()
        fallback_commit_message = (coding.commit_message or "").strip() or "N/A"

        prompt = f"""
You generate precise git commit subjects for an autonomous evolution worker.
Summaries must stay under {self._subject_limit} characters and follow imperative mood.

Global goal:
{goal}

Plan summary:
{plan_summary}

Plan rationale:
{plan_rationale}

Plan focus metrics:
{focus_metrics}

Plan guardrails:
{guardrails}

Constraints to respect:
{constraints}

Acceptance criteria:
{acceptance}

Worker notes:
{notes}

Coding execution summary:
{coding_summary}

Step outcomes:
{step_lines}

Tests executed:
{tests}

Coding agent suggested commit message:
{fallback_commit_message}

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
