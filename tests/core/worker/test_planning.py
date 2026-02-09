from __future__ import annotations

import json

from loreley.config import Settings
from loreley.core.worker.agent import AgentInvocation
from loreley.core.worker.planning import (
    CommitMetric,
    CommitPlanningContext,
    PlanDocument,
    PlanningAgent,
    PlanningAgentRequest,
)


class _DummyBackend:
    def run(self, *_args, **_kwargs):  # pragma: no cover - not used here
        raise AssertionError("backend should not be invoked in these tests")


def _make_request(goal: str) -> PlanningAgentRequest:
    base = CommitPlanningContext(
        commit_hash="base",
        subject="Base subject",
        change_summary="base summary",
        highlights=("Touched files: foo.py",),
        metrics=(
            CommitMetric(name="quality", value=1.0),
            CommitMetric(name="speed", value=2.0),
        ),
    )
    return PlanningAgentRequest(
        base=base,
        inspirations=(),
        goal=goal,
        constraints=("guard",),
        acceptance_criteria=("verify",),
    )


def test_coerce_plan_from_invocation_extracts_summary(settings: Settings) -> None:
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = _make_request("Improve docs")
    markdown = """# Plan

## Summary
- Improve docs and defaults.

## Steps
1. Update config
2. Update docs
"""
    invocation = AgentInvocation(
        command=("echo",),
        stdout=markdown,
        stderr="",
        duration_seconds=1.0,
    )

    plan = agent._coerce_plan_from_invocation(  # type: ignore[attr-defined]
        request=request,
        invocation=invocation,
    )

    assert isinstance(plan, PlanDocument)
    assert plan.summary.startswith("Improve docs")
    assert plan.markdown.strip() == markdown.strip()
    assert plan.guardrails == ("guard",)
    assert plan.focus_metrics == ("quality", "speed")


def test_coerce_plan_from_invocation_unwraps_json_stdout(settings: Settings) -> None:
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = _make_request("Improve docs")
    markdown = """## Summary
- Improve docs and defaults.

## Steps
1. Update config
2. Update docs
"""
    invocation = AgentInvocation(
        command=("echo",),
        stdout=json.dumps({"output": markdown}),
        stderr="",
        duration_seconds=1.0,
    )

    plan = agent._coerce_plan_from_invocation(  # type: ignore[attr-defined]
        request=request,
        invocation=invocation,
    )

    assert plan.markdown.strip() == markdown.strip()
    assert plan.summary.startswith("Improve docs")


def test_coerce_plan_from_invocation_falls_back_to_goal(settings: Settings) -> None:
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = _make_request("Ship the feature")
    invocation = AgentInvocation(
        command=("echo",),
        stdout="",
        stderr="",
        duration_seconds=1.0,
    )

    plan = agent._coerce_plan_from_invocation(  # type: ignore[attr-defined]
        request=request,
        invocation=invocation,
    )

    assert plan.summary == "Ship the feature"
    assert "## Summary" in plan.markdown


def test_planning_prompt_requests_markdown_deliverable(settings: Settings) -> None:
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = _make_request("Improve docs")

    prompt = agent._render_prompt(request)  # type: ignore[attr-defined]

    assert "Markdown plan document" in prompt
    assert "Output requirements" in prompt
    assert "Use '##' headings" in prompt

