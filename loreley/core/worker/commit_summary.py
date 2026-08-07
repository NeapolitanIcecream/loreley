"""Deterministic commit messages derived from existing worker reports."""

from __future__ import annotations

from uuid import UUID

from loreley.core.contracts import normalize_single_line
from loreley.core.worker.coding import ExecutionReport
from loreley.core.worker.planning import PlanDocument

__all__ = ["build_commit_message"]


def build_commit_message(
    *,
    job_id: UUID,
    plan: PlanDocument,
    coding: ExecutionReport,
) -> str:
    """Reuse the coding summary, then the plan summary, without another LLM call."""

    default = f"Evolution job {job_id}"
    for value in (coding.summary, plan.summary, default):
        message = normalize_single_line(value)
        if message and "```" not in message and not message.startswith(("{", "[")):
            return message
    return default
