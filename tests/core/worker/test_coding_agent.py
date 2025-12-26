from __future__ import annotations

import json
from pathlib import Path

import pytest

from loreley.config import Settings
from loreley.core.worker.agent_backend import AgentInvocation
from loreley.core.worker.coding import (
    CodingAgent,
    CodingAgentRequest,
    CodingError,
    StepExecutionStatus,
)
from loreley.core.worker.planning import PlanStep, PlanningPlan


def _make_plan() -> PlanningPlan:
    step = PlanStep(
        step_id="s1",
        title="Implement feature",
        intent="Do work",
        actions=("edit files",),
        files=("file.py",),
        dependencies=(),
        validation=("tests pass",),
        risks=("bug",),
        references=("link",),
    )
    return PlanningPlan(
        summary="summary",
        rationale="rationale",
        focus_metrics=("speed",),
        guardrails=("guard",),
        risks=("risk1",),
        validation=("validate",),
        steps=(step,),
        handoff_notes=(),
        fallback_plan=None,
    )


class _DummyBackend:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.calls: list[tuple[object, Path]] = []

    def run(self, task, working_dir):  # noqa: ANN001
        self.calls.append((task, working_dir))
        return AgentInvocation(
            command=("dummy",),
            stdout=self.stdout,
            stderr="",
            duration_seconds=1.0,
        )


def test_coding_agent_implements_plan_with_strict_validation(tmp_path: Path, settings: Settings, monkeypatch) -> None:
    settings.worker_coding_validation_mode = "strict"

    payload = {
        "implementation_summary": "done",
        "commit_message": "message",
        "step_results": [
            {
                "step_id": "s1",
                "status": StepExecutionStatus.COMPLETED.value,
                "summary": "ok",
                "files": ["file.py"],
                "commands": ["cmd"],
            }
        ],
        "tests_executed": ["pytest"],
        "tests_recommended": ["tox"],
        "follow_up_items": ["docs"],
        "notes": ["note"],
    }

    backend = _DummyBackend(json.dumps(payload))
    agent = CodingAgent(settings=settings, backend=backend)

    states = iter([("clean",), ("dirty",)])

    def fake_snapshot(_worktree: Path) -> tuple[str, ...]:
        try:
            return next(states)
        except StopIteration:
            return ("dirty",)

    monkeypatch.setattr(agent, "_snapshot_worktree_state", fake_snapshot)
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **kwargs: None)

    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")

    response = agent.implement(request, working_dir=tmp_path)

    assert response.execution.implementation_summary == "done"
    assert response.execution.commit_message == "message"
    assert response.execution.step_results[0].status is StepExecutionStatus.COMPLETED
    assert response.command == ("dummy",)
    assert response.attempts == 1
    assert backend.calls


def test_coding_agent_strict_strips_markdown_fences(tmp_path: Path, settings: Settings, monkeypatch) -> None:
    settings.worker_coding_validation_mode = "strict"

    payload = {
        "implementation_summary": "done",
        "commit_message": "message",
        "step_results": [
            {
                "step_id": "s1",
                "status": StepExecutionStatus.COMPLETED.value,
                "summary": "ok",
                "files": ["file.py"],
                "commands": ["cmd"],
            }
        ],
        "tests_executed": ["pytest"],
        "tests_recommended": ["tox"],
        "follow_up_items": ["docs"],
        "notes": ["note"],
    }

    fenced = f"```json\n{json.dumps(payload)}\n```"
    backend = _DummyBackend(fenced)
    agent = CodingAgent(settings=settings, backend=backend)

    states = iter([("clean",), ("dirty",)])
    monkeypatch.setattr(agent, "_snapshot_worktree_state", lambda _w: next(states, ("dirty",)))
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **kwargs: None)

    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")
    response = agent.implement(request, working_dir=tmp_path)

    assert response.execution.implementation_summary == "done"
    assert response.execution.commit_message == "message"
    assert response.execution.step_results[0].status is StepExecutionStatus.COMPLETED


def test_coding_agent_lenient_falls_back_to_freeform_output(tmp_path: Path, settings: Settings, monkeypatch) -> None:
    settings.worker_coding_validation_mode = "lenient"

    backend = _DummyBackend("  freeform output  ")
    agent = CodingAgent(settings=settings, backend=backend)

    states = iter([("clean",), ("dirty",)])
    monkeypatch.setattr(agent, "_snapshot_worktree_state", lambda _w: next(states, ("dirty",)))
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **kwargs: None)

    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")
    response = agent.implement(request, working_dir=tmp_path)

    execution = response.execution
    assert execution.step_results[0].step_id == "lenient-1"
    assert execution.step_results[0].status is StepExecutionStatus.PARTIAL
    assert execution.implementation_summary == "freeform output"
    assert execution.notes
    assert response.raw_output.strip() == "freeform output"


def test_coding_agent_raises_when_no_changes_after_attempts(tmp_path: Path, settings: Settings, monkeypatch) -> None:
    settings.worker_coding_validation_mode = "strict"
    settings.worker_coding_max_attempts = 2

    payload = {
        "implementation_summary": "done",
        "commit_message": None,
        "step_results": [
            {
                "step_id": "s1",
                "status": StepExecutionStatus.COMPLETED.value,
                "summary": "ok",
                "files": [],
                "commands": [],
            }
        ],
        "tests_executed": [],
        "tests_recommended": [],
        "follow_up_items": [],
        "notes": [],
    }

    backend = _DummyBackend(json.dumps(payload))
    agent = CodingAgent(settings=settings, backend=backend)

    monkeypatch.setattr(agent, "_snapshot_worktree_state", lambda _w: ("same",))
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **kwargs: None)

    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")

    with pytest.raises(CodingError):
        agent.implement(request, working_dir=tmp_path)
