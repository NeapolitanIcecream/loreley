from __future__ import annotations

import json

from loreley.config import Settings
from loreley.core.worker.agent_backend import AgentInvocation
from loreley.core.worker.planning import (
    CommitPlanningContext,
    PlanningAgent,
    PlanningAgentRequest,
    PlanningPlan,
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
    )
    return PlanningAgentRequest(
        base=base,
        inspirations=(),
        goal=goal,
        constraints=("guard",),
        acceptance_criteria=("verify",),
    )


def test_coerce_plan_from_invocation_strict_parses_structured_output(settings: Settings) -> None:
    settings.worker_planning_validation_mode = "strict"
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = _make_request("Improve docs")
    payload = {
        "plan_summary": "summary",
        "rationale": "because",
        "focus_metrics": ["metric"],
        "guardrails": ["guard"],
        "risks": ["risk"],
        "validation": ["run tests"],
        "steps": [
            {
                "id": "s1",
                "title": "T1",
                "intent": "Do work",
                "actions": ["a1"],
                "files": ["f1"],
                "validation": ["v1"],
                "dependencies": ["dep"],
                "risks": ["r1"],
                "references": ["ref"],
            }
        ],
        "handoff_notes": ["note"],
        "fallback_plan": "fallback",
    }
    invocation = AgentInvocation(
        command=("echo",),
        stdout=json.dumps(payload),
        stderr="",
        duration_seconds=1.0,
    )

    plan = agent._coerce_plan_from_invocation(  # type: ignore[attr-defined]
        request=request,
        invocation=invocation,
    )

    assert isinstance(plan, PlanningPlan)
    assert plan.summary == "summary"
    assert plan.guardrails == ("guard",)
    assert plan.steps[0].actions == ("a1",)
    assert plan.handoff_notes == ("note",)
    assert plan.fallback_plan == "fallback"


def test_coerce_plan_from_invocation_strict_strips_markdown_fences(settings: Settings) -> None:
    settings.worker_planning_validation_mode = "strict"
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = _make_request("Improve docs")
    payload = {
        "plan_summary": "summary",
        "rationale": "because",
        "focus_metrics": ["metric"],
        "guardrails": ["guard"],
        "risks": ["risk"],
        "validation": ["run tests"],
        "steps": [
            {
                "id": "s1",
                "title": "T1",
                "intent": "Do work",
                "actions": ["a1"],
                "files": ["f1"],
                "validation": ["v1"],
                "dependencies": ["dep"],
                "risks": ["r1"],
                "references": ["ref"],
            }
        ],
        "handoff_notes": ["note"],
        "fallback_plan": "fallback",
    }
    fenced = f"```json\n{json.dumps(payload)}\n```"
    invocation = AgentInvocation(
        command=("echo",),
        stdout=fenced,
        stderr="",
        duration_seconds=1.0,
    )

    plan = agent._coerce_plan_from_invocation(  # type: ignore[attr-defined]
        request=request,
        invocation=invocation,
    )

    assert isinstance(plan, PlanningPlan)
    assert plan.summary == "summary"


def test_coerce_plan_from_invocation_lenient_falls_back_to_freeform(settings: Settings) -> None:
    settings.worker_planning_validation_mode = "lenient"
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = _make_request("Goal text")

    invocation = AgentInvocation(
        command=("echo",),
        stdout="freeform output that is not json",
        stderr="",
        duration_seconds=0.5,
    )

    plan = agent._coerce_plan_from_invocation(  # type: ignore[attr-defined]
        request=request,
        invocation=invocation,
    )

    assert plan.summary == "freeform output that is not json"
    assert plan.guardrails == ("guard",)
    assert plan.validation == ("verify",)
    assert plan.steps[0].step_id == "lenient-1"
    assert plan.handoff_notes


def test_build_plan_from_freeform_uses_goal_when_output_empty(settings: Settings) -> None:
    settings.worker_planning_validation_mode = "none"
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = _make_request("Ship the feature")

    plan = agent._build_plan_from_freeform_output(  # type: ignore[attr-defined]
        request=request,
        raw_output="",
    )

    assert plan.summary == "Ship the feature"
    assert plan.validation == ("verify",)
    assert plan.guardrails == ("guard",)
