from __future__ import annotations

import json

from loreley.config import Settings
from loreley.core.worker.agent import AgentInvocation
from loreley.core.worker.planning import (
    CommitEvaluationArtifactFeedback,
    CommitMetric,
    CommitPlanningContext,
    EvaluationDiagnosticBrief,
    IterationContext,
    PlanDocument,
    PlanningAgent,
    PlanningAgentRequest,
    render_evaluation_agent_feedback,
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


def test_planning_prompt_projects_only_agent_visible_evaluation_evidence(settings: Settings) -> None:
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    base = CommitPlanningContext(
        commit_hash="base",
        subject="Base",
        change_summary="base summary",
        evaluation_artifacts=(
            CommitEvaluationArtifactFeedback(
                key="benchmark_report",
                kind="benchmark_json",
                mime_type="application/json",
                summary="Parser throughput improved.",
                diagnostics=(
                    EvaluationDiagnosticBrief(
                        kind="improvement",
                        severity="info",
                        message="throughput improved",
                    ),
                ),
                visibility="agent_visible",
            ),
            CommitEvaluationArtifactFeedback(
                key="full_stderr",
                kind="log",
                mime_type="text/plain",
                summary="raw stderr",
                visibility="human_only",
            ),
        ),
    )
    request = PlanningAgentRequest(
        base=base,
        inspirations=(),
        goal="Improve docs",
        iteration_context=IterationContext(seed_job=False),
    )

    prompt = agent._render_prompt(request)  # type: ignore[attr-defined]

    assert "Evaluation Evidence:" in prompt
    assert "Parser throughput improved." in prompt
    assert "throughput improved" in prompt
    assert "Evidence Guardrail:" in prompt
    assert "full_stderr" not in prompt
    assert "raw stderr" not in prompt


def test_planning_prompt_manifest_mode_omits_diagnostic_prose(settings: Settings) -> None:
    settings.worker_evaluation_agent_feedback_mode = "manifest"
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = PlanningAgentRequest(
        base=CommitPlanningContext(
            commit_hash="base",
            subject="Base",
            change_summary="base summary",
            evaluation_artifacts=(
                CommitEvaluationArtifactFeedback(
                    key="profile_hotspots",
                    kind="flamegraph",
                    mime_type="text/plain",
                    summary="tokenizer._scan is hot",
                    diagnostics=(
                        EvaluationDiagnosticBrief(
                            kind="hotspot",
                            message="tokenizer._scan accounts for 37%",
                        ),
                    ),
                    visibility="agent_visible",
                    size_bytes=120,
                ),
            ),
        ),
        inspirations=(),
        goal="Improve docs",
    )

    prompt = agent._render_prompt(request)  # type: ignore[attr-defined]

    assert "profile_hotspots" in prompt
    assert "mime=text/plain" in prompt
    assert "tokenizer._scan is hot" not in prompt
    assert "accounts for 37%" not in prompt


def test_planning_prompt_artifact_manifest_projection_omits_diagnostic_prose(
    settings: Settings,
) -> None:
    settings.worker_evaluation_agent_feedback_mode = "summary"
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    request = PlanningAgentRequest(
        base=CommitPlanningContext(
            commit_hash="base",
            subject="Base",
            change_summary="base summary",
            evaluation_artifacts=(
                CommitEvaluationArtifactFeedback(
                    key="manifest_only",
                    kind="benchmark_json",
                    mime_type="application/json",
                    summary="summary prose must not reach agents",
                    diagnostics=(
                        EvaluationDiagnosticBrief(
                            kind="regression",
                            message="diagnostic prose must not reach agents",
                        ),
                    ),
                    projection="manifest",
                    visibility="agent_visible",
                    size_bytes=120,
                ),
            ),
        ),
        inspirations=(),
        goal="Improve docs",
    )

    prompt = agent._render_prompt(request)  # type: ignore[attr-defined]

    assert "manifest_only" in prompt
    assert "mime=application/json" in prompt
    assert "summary prose must not reach agents" not in prompt
    assert "diagnostic prose must not reach agents" not in prompt
    assert "Evidence Guardrail:" in prompt


def test_planning_prompt_path_mode_uses_stable_uri_without_filesystem_path(settings: Settings) -> None:
    settings.worker_evaluation_agent_feedback_mode = "path"
    agent = PlanningAgent(settings=settings, backend=_DummyBackend())
    local_path = "/tmp/evaluator/benchmark.json"
    request = PlanningAgentRequest(
        base=CommitPlanningContext(
            commit_hash="base",
            subject="Base",
            change_summary="base summary",
            evaluation_artifacts=(
                CommitEvaluationArtifactFeedback(
                    key="benchmark_report",
                    kind="benchmark_json",
                    mime_type="application/json",
                    summary="bounded summary",
                    projection="path",
                    visibility="agent_visible",
                    size_bytes=128,
                    artifact_uri="loreley://evaluation-artifacts/job-1/benchmark_report",
                ),
            ),
        ),
        inspirations=(),
        goal="Improve docs",
    )

    prompt = agent._render_prompt(request)  # type: ignore[attr-defined]

    assert "loreley://evaluation-artifacts/job-1/benchmark_report" in prompt
    assert local_path not in prompt


def test_agent_feedback_budget_keeps_guardrail_and_accurate_omissions(
    settings: Settings,
) -> None:
    settings.worker_evaluation_agent_feedback_max_chars = 850
    artifacts = tuple(
        CommitEvaluationArtifactFeedback(
            key=f"bench_{idx}",
            kind="benchmark_json",
            mime_type="application/json",
            summary=f"summary {idx} " + ("x" * 900),
            visibility="agent_visible",
        )
        for idx in range(4)
    )

    projection = render_evaluation_agent_feedback(artifacts, settings=settings)

    assert len(projection.text) <= 850
    assert "Evidence Guardrail:" in projection.text
    assert "char_budget" in projection.omitted_reasons
    assert projection.omitted_artifact_count == 4 - len(projection.included_artifact_keys)
    for key in projection.included_artifact_keys:
        assert key in projection.text
    for idx in range(4):
        key = f"bench_{idx}"
        if key not in projection.included_artifact_keys:
            assert key not in projection.text


def test_agent_feedback_omits_evidence_when_budget_cannot_fit_guardrail(
    settings: Settings,
) -> None:
    settings.worker_evaluation_agent_feedback_max_chars = 40

    projection = render_evaluation_agent_feedback(
        (
            CommitEvaluationArtifactFeedback(
                key="bench",
                kind="benchmark_json",
                mime_type="application/json",
                summary="summary prose must not appear without a guardrail",
                visibility="agent_visible",
            ),
        ),
        settings=settings,
    )

    assert projection.text == ""
    assert projection.included_artifact_keys == ()
    assert projection.omitted_artifact_count == 1
    assert "char_budget" in projection.omitted_reasons
