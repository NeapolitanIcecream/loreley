from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from loreley.config import Settings
from loreley.core.worker.agent import AgentInvocation
from loreley.core.worker.coding import CodingAgent, CodingAgentRequest, CodingError
from loreley.core.worker.planning import PlanDocument


class _DummyBackend:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.calls: list[tuple[object, Path]] = []

    def run(self, task, *, working_dir: Path) -> AgentInvocation:  # noqa: ANN001
        self.calls.append((task, working_dir))
        return AgentInvocation(
            command=("dummy",),
            stdout=self.stdout,
            stderr="",
            duration_seconds=1.0,
        )


def _make_plan() -> PlanDocument:
    return PlanDocument(
        summary="plan summary",
        markdown="""## Summary
- Do the thing.

## Steps
1. Edit `file.py`
""",
        focus_metrics=(),
        guardrails=("guard",),
    )


def test_coding_agent_returns_report_and_extracts_summary(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _DummyBackend(
        """## Summary
- Implemented the change.

## Changes
- Updated `file.py`
"""
    )
    agent = CodingAgent(settings=settings, backend=backend)

    states = iter([("clean",), ("dirty",)])
    monkeypatch.setattr(agent, "_snapshot_worktree_state", lambda _w: next(states, ("dirty",)))
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **_kwargs: None)

    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")
    response = agent.implement(request, working_dir=tmp_path)

    assert response.report.summary.startswith("Implemented")
    assert not hasattr(response.report, "commit_message")
    assert response.raw_output.strip().startswith("## Summary")
    assert response.command == ("dummy",)
    assert response.attempts == 1
    assert backend.calls


def test_coding_agent_unwraps_json_stdout_and_preserves_markdown(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = """## Summary
- Implemented the change.

## Changes
- Updated `file.py`
"""
    backend = _DummyBackend(json.dumps({"output": markdown}))
    agent = CodingAgent(settings=settings, backend=backend)

    states = iter([("clean",), ("dirty",)])
    monkeypatch.setattr(agent, "_snapshot_worktree_state", lambda _w: next(states, ("dirty",)))
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **_kwargs: None)

    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")
    response = agent.implement(request, working_dir=tmp_path)

    assert not hasattr(response.report, "commit_message")
    assert response.report.markdown.strip() == markdown.strip()
    assert response.raw_output.strip().startswith("{")


def test_coding_agent_raises_when_no_changes_after_attempts(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.worker_coding_max_attempts = 2
    backend = _DummyBackend("## Summary\n- Did nothing.\n")
    agent = CodingAgent(settings=settings, backend=backend)

    monkeypatch.setattr(agent, "_snapshot_worktree_state", lambda _w: ("same",))
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **_kwargs: None)

    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")

    with pytest.raises(CodingError):
        agent.implement(request, working_dir=tmp_path)


def test_coding_prompt_includes_markdown_contract(tmp_path: Path, settings: Settings) -> None:
    agent = CodingAgent(settings=settings, backend=_DummyBackend("ok"))
    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")
    prompt = agent._render_prompt(request, worktree=tmp_path)  # type: ignore[attr-defined]

    assert "Plan (Markdown):" in prompt
    assert "Output requirements:" in prompt
    assert "Do not create git commits" in prompt
    assert "Commit message" not in prompt


# ---------------------------------------------------------------------------
# Observability: failure signal tests
# ---------------------------------------------------------------------------


def test_coding_agent_logs_warning_when_no_changes_produced(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    captured_logs: list[dict[str, Any]],
) -> None:
    """Observability: a WARNING log is emitted when coding attempt yields no repo changes."""
    settings.worker_coding_max_attempts = 2
    backend = _DummyBackend("## Summary\n- Did nothing.\n")
    agent = CodingAgent(settings=settings, backend=backend)

    # Worktree snapshot never changes → triggers "no repository changes" path.
    monkeypatch.setattr(agent, "_snapshot_worktree_state", lambda _w: ("same",))
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **_kwargs: None)

    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")

    with pytest.raises(CodingError):
        agent.implement(request, working_dir=tmp_path)

    # Assert the expected diagnostic signal was emitted.
    warn_logs = [
        r
        for r in captured_logs
        if r["level"] == "WARNING" and r["module"] == "worker.coding"
    ]
    assert any("no repository changes" in str(r["message"]).lower() for r in warn_logs)


def test_coding_agent_creates_debug_artifact_on_attempt(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observability: a JSON debug artifact is written for each coding attempt."""
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()

    backend = _DummyBackend("## Summary\n- Done.\n")
    agent = CodingAgent(settings=settings, backend=backend)
    agent._debug_dir = debug_dir

    states = iter([("clean",), ("dirty",)])
    monkeypatch.setattr(agent, "_snapshot_worktree_state", lambda _w: next(states, ("dirty",)))

    request = CodingAgentRequest(goal="goal", plan=_make_plan(), base_commit="abc123")
    agent.implement(request, working_dir=tmp_path)

    # At least one debug artifact should be present.
    artifacts = list(debug_dir.glob("coding-*.json"))
    assert len(artifacts) >= 1

    with artifacts[0].open() as f:
        payload = json.load(f)

    assert payload["goal"] == "goal"
    assert payload["base_commit"] == "abc123"
    assert payload["status"] == "ok"

