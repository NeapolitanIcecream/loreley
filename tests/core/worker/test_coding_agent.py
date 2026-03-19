from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from loreley.config import Settings
from loreley.core.worker.agent import AgentInvocation
from loreley.core.worker.coding import CodingAgent, CodingAgentRequest, CodingError
from loreley.core.worker.planning import (
    CommitMetric,
    CommitPlanningContext,
    IterationContext,
    PlanDocument,
)


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


def _make_base_context() -> CommitPlanningContext:
    return CommitPlanningContext(
        commit_hash="base",
        subject="Base subject",
        change_summary="base summary",
        metrics=(
            CommitMetric(name="quality", value=1.0, higher_is_better=True),
            CommitMetric(name="runtime_ms", value=10.0, higher_is_better=False),
        ),
        key_files=("solver.py",),
    )


def _make_request() -> CodingAgentRequest:
    inspiration = CommitPlanningContext(
        commit_hash="insp",
        subject="Inspiration subject",
        change_summary="switch to deterministic construction",
        trajectory=(
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
    return CodingAgentRequest(
        goal="goal",
        plan=_make_plan(),
        base_commit="abc123",
        base=_make_base_context(),
        inspirations=(inspiration,),
        iteration_context=IterationContext(
            seed_job=False,
            sampling_strategy="map_elites",
            facts=("radius_used: 3",),
        ),
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

    request = _make_request()
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

    request = _make_request()
    response = agent.implement(request, working_dir=tmp_path)

    assert not hasattr(response.report, "commit_message")
    assert response.report.markdown.strip() == markdown.strip()
    assert response.raw_output.strip().startswith("{")


def test_coding_agent_unwraps_jsonl_stdout_and_preserves_markdown(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = """## Summary
- Implemented the change.

## Changes
- Updated `file.py`
"""
    backend = _DummyBackend(
        "\n".join(
            [
                json.dumps({"event": "started", "message": "coding"}),
                json.dumps({"event": "completed", "output": markdown}),
            ]
        )
    )
    agent = CodingAgent(settings=settings, backend=backend)

    states = iter([("clean",), ("dirty",)])
    monkeypatch.setattr(agent, "_snapshot_worktree_state", lambda _w: next(states, ("dirty",)))
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **_kwargs: None)

    request = _make_request()
    response = agent.implement(request, working_dir=tmp_path)

    assert response.report.markdown.strip() == markdown.strip()
    assert response.raw_output.strip().startswith('{"event"')


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

    request = _make_request()

    with pytest.raises(CodingError):
        agent.implement(request, working_dir=tmp_path)


def test_coding_prompt_includes_markdown_contract(tmp_path: Path, settings: Settings) -> None:
    agent = CodingAgent(settings=settings, backend=_DummyBackend("ok"))
    request = _make_request()
    prompt = agent._render_prompt(request, worktree=tmp_path)  # type: ignore[attr-defined]

    assert "Evolution Goal:" in prompt
    assert "Worker Contract:" in prompt
    assert "Iteration Context:" in prompt
    assert "Base Commit Context:" in prompt
    assert "Inspiration Commits:" in prompt
    assert "Plan (Markdown):" in prompt
    assert "Output requirements:" in prompt
    assert "Operate non-interactively" in prompt
    assert "Do not run Loreley's evaluator" in prompt
    assert "worktree contains meaningful tracked-file changes" in prompt
    assert "smallest relevant set of source changes" in prompt
    assert "Do not create git commits" in prompt
    assert "Constraints:" not in prompt
    assert "Acceptance criteria:" not in prompt
    assert "Additional notes:" not in prompt
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

    request = _make_request()

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

    request = _make_request()
    agent.implement(request, working_dir=tmp_path)

    # At least one debug artifact should be present.
    artifacts = list(debug_dir.glob("coding-*.json"))
    assert len(artifacts) >= 1

    with artifacts[0].open() as f:
        payload = json.load(f)

    assert payload["goal"] == "goal"
    assert payload["base_commit"] == "abc123"
    assert payload["status"] == "ok"
