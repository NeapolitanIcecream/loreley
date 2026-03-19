from __future__ import annotations

import json

from loreley.config import Settings
from loreley.core.worker.agent import AgentInvocation
from loreley.core.worker.planning import (
    CommitMetric,
    CommitPlanningContext,
    IterationContext,
    PlanDocument,
    PlanningAgent,
    PlanningAgentRequest,
)


class _DummyBackend:
    def run(self, *_args, **_kwargs):  # pragma: no cover - not used here
        raise AssertionError("backend should not be invoked in these tests")


def _make_base() -> CommitPlanningContext:
    return CommitPlanningContext(
        commit_hash="base",
        subject="Base subject",
        change_summary="base summary",
        highlights=("Touched files: foo.py",),
        metrics=(
            CommitMetric(name="quality", value=1.0),
            CommitMetric(name="runtime_ms", value=12.0, higher_is_better=False),
        ),
    )


def _make_request(goal: str) -> PlanningAgentRequest:
    base = _make_base()
    return PlanningAgentRequest(
        base=base,
        inspirations=(),
        goal=goal,
        iteration_context=IterationContext(
            seed_job=False,
            sampling_strategy="map_elites",
            facts=("radius_used: 3", "initial_radius: 7"),
        ),
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
    assert plan.guardrails == (
        "non_interactive_worker",
        "framework_managed_evaluation",
        "leave_modified_worktree",
        "no_git_commits",
    )
    assert plan.focus_metrics == ("quality", "runtime_ms")


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


def test_coerce_plan_from_invocation_unwraps_jsonl_stdout(settings: Settings) -> None:
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
        stdout="\n".join(
            [
                json.dumps({"event": "started", "message": "planning"}),
                json.dumps({"event": "completed", "output": markdown}),
            ]
        ),
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

    assert "Evolution Goal:" in prompt
    assert "Worker Contract:" in prompt
    assert "Iteration Context:" in prompt
    assert "Base Commit Context:" in prompt
    assert "Markdown document" in prompt
    assert "Operate non-interactively" in prompt
    assert "Do not run Loreley's evaluator" in prompt
    assert "Constraints:" not in prompt
    assert "Acceptance criteria:" not in prompt
    assert "Use these sections" in prompt


def test_planning_prompt_formats_inspirations_as_transfer_cards(settings: Settings) -> None:
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    base = _make_base()
    inspiration = CommitPlanningContext(
        commit_hash="insp",
        subject="Inspiration subject",
        change_summary="switch to deterministic construction",
        trajectory=(
            "  - unique_steps_count: 3 (lca=abc123)",
            "  - Earliest unique steps (raw, up to 2):",
            "    - a1b2c3d4e5f6: introduce deterministic seed layout",
            "  - Recent unique steps (raw, last 1):",
            "    - b2c3d4e5f6a7: cache the final 26-circle arrangement",
        ),
        evaluation_summary="Higher quality with lower runtime.",
        metrics=(
            CommitMetric(name="quality", value=1.8, higher_is_better=True),
            CommitMetric(name="runtime_ms", value=4.0, higher_is_better=False),
        ),
        key_files=("solver.py", "cache.py"),
    )
    request = PlanningAgentRequest(
        base=base,
        inspirations=(inspiration,),
        goal="Improve docs",
        iteration_context=IterationContext(seed_job=False, sampling_strategy="map_elites"),
    )

    prompt = agent._render_prompt(request)  # type: ignore[attr-defined]

    assert "Inspiration #1" in prompt
    assert "why_it_matters:" in prompt
    assert "Improves `runtime_ms`" in prompt
    assert "`quality`" in prompt
    assert "distinctive_changes_vs_base:" in prompt
    assert "introduce deterministic seed layout" in prompt
    assert "cache the final 26-circle arrangement" in prompt
    assert "No highlights available." not in prompt
