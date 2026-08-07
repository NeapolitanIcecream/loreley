from __future__ import annotations

from uuid import uuid4

from loreley.core.worker.coding import ExecutionReport
from loreley.core.worker.commit_summary import build_commit_message
from loreley.core.worker.planning import PlanDocument


def _plan(summary: str = "plan summary") -> PlanDocument:
    return PlanDocument(
        summary=summary,
        markdown="## Summary\n- plan summary\n",
        focus_metrics=("quality",),
        guardrails=("guard",),
    )


def _coding(summary: str = "implemented feature") -> ExecutionReport:
    return ExecutionReport(
        summary=summary,
        markdown="## Summary\n- implemented feature\n",
    )


def test_commit_message_reuses_coding_summary_without_truncation() -> None:
    summary = "Implement the measured hot-path improvement " + ("safely " * 20)

    message = build_commit_message(
        job_id=uuid4(),
        plan=_plan(),
        coding=_coding(summary),
    )

    assert message == " ".join(summary.split())
    assert len(message) > 72


def test_commit_message_uses_plan_when_coding_summary_is_empty() -> None:
    message = build_commit_message(
        job_id=uuid4(),
        plan=_plan("Use the plan summary"),
        coding=_coding("  "),
    )

    assert message == "Use the plan summary"


def test_commit_message_rejects_structured_payloads() -> None:
    job_id = uuid4()

    message = build_commit_message(
        job_id=job_id,
        plan=_plan("[not a subject]"),
        coding=_coding("{not a subject}"),
    )

    assert message == f"Evolution job {job_id}"


def test_commit_message_collapses_newlines() -> None:
    message = build_commit_message(
        job_id=uuid4(),
        plan=_plan(),
        coding=_coding("Improve\n\nparser   throughput"),
    )

    assert message == "Improve parser throughput"
