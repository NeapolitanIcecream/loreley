from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

from loreley.config import Settings
from loreley.core.worker.agent import AgentInvocation
from loreley.core.worker.coding import CodingAgent, CodingAgentRequest, CodingError
from loreley.core.worker.planning import (
    CommitEvaluationArtifactFeedback,
    CommitMetric,
    CommitPlanningContext,
    EvaluationDiagnosticBrief,
    IterationContext,
    PlanDocument,
)


class _DummyBackend:
    def __init__(self, stdout: str, on_run: Callable[[Path], None] | None = None) -> None:
        self.stdout = stdout
        self.on_run = on_run
        self.calls: list[tuple[object, Path]] = []

    def run(self, task, *, working_dir: Path) -> AgentInvocation:  # noqa: ANN001
        self.calls.append((task, working_dir))
        if self.on_run is not None:
            self.on_run(working_dir)
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


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo_with_file(repo: Path, path: str = "solver.py") -> None:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / path).write_text("clean\n", encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", "initial")


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


def test_coding_agent_detects_content_change_to_preexisting_dirty_tracked_file(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    dirty_file = repo / "solver.py"
    dirty_file.write_text("dirty before agent\n", encoding="utf-8")

    backend = _DummyBackend(
        "## Summary\n- Updated the dirty tracked file.\n",
        on_run=lambda worktree: (worktree / "solver.py").write_text(
            "dirty after agent\n",
            encoding="utf-8",
        ),
    )
    agent = CodingAgent(settings=settings, backend=backend)
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **_kwargs: None)

    response = agent.implement(_make_request(), working_dir=repo)

    assert response.attempts == 1
    assert dirty_file.read_text(encoding="utf-8") == "dirty after agent\n"


def test_coding_agent_detects_content_change_to_preexisting_untracked_file(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    untracked_file = repo / "notes.txt"
    untracked_file.write_text("untracked before agent\n", encoding="utf-8")

    backend = _DummyBackend(
        "## Summary\n- Updated the dirty untracked file.\n",
        on_run=lambda worktree: (worktree / "notes.txt").write_text(
            "untracked after agent\n",
            encoding="utf-8",
        ),
    )
    agent = CodingAgent(settings=settings, backend=backend)
    monkeypatch.setattr(agent, "_dump_debug_artifact", lambda **_kwargs: None)

    response = agent.implement(_make_request(), working_dir=repo)

    assert response.attempts == 1
    assert untracked_file.read_text(encoding="utf-8") == "untracked after agent\n"


def test_coding_agent_snapshot_ignores_clean_excluded_untracked_paths(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    cache_file = repo / ".venv" / "lib" / "site-packages" / "cache.py"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("expensive local cache\n", encoding="utf-8")
    settings.worker_repo_clean_excludes = [".venv"]

    agent = CodingAgent(settings=settings, backend=_DummyBackend("ok"))

    def fail_on_fingerprint(_worktree: Path, repo_path: str) -> str:
        raise AssertionError(f"clean-excluded path was fingerprinted: {repo_path}")

    monkeypatch.setattr(agent, "_fingerprint_worktree_path", fail_on_fingerprint)

    assert agent._snapshot_worktree_state(repo) == ()  # type: ignore[attr-defined]


def test_coding_agent_snapshot_preserves_significant_status_path_whitespace(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    spaced_cache_file = repo / " .venv" / "cache.py"
    spaced_cache_file.parent.mkdir()
    spaced_cache_file.write_text("real change with leading space\n", encoding="utf-8")
    trailing_space_file = repo / ".python-version "
    trailing_space_file.write_text("real change with trailing space\n", encoding="utf-8")
    settings.worker_repo_clean_excludes = [".venv", ".python-version"]

    agent = CodingAgent(settings=settings, backend=_DummyBackend("ok"))
    fingerprinted_paths: list[str] = []

    def record_fingerprint(_worktree: Path, repo_path: str) -> str:
        fingerprinted_paths.append(repo_path)
        return "fingerprinted"

    monkeypatch.setattr(agent, "_fingerprint_worktree_path", record_fingerprint)

    snapshot = agent._snapshot_worktree_state(repo)  # type: ignore[attr-defined]

    assert " .venv/cache.py" in fingerprinted_paths
    assert ".python-version " in fingerprinted_paths
    assert any(entry == "status\0??\0 .venv/cache.py" for entry in snapshot)
    assert any(entry == "status\0??\0.python-version " for entry in snapshot)


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

    with pytest.raises(CodingError, match="no effective repository changes") as exc_info:
        agent.implement(request, working_dir=tmp_path)
    assert "report" not in str(exc_info.value).lower()


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
    assert "Constraints:" in prompt
    assert "Acceptance Criteria:" in prompt
    assert "Additional notes:" not in prompt
    assert "Commit message" not in prompt


def test_coding_prompt_bounds_rework_feedback(tmp_path: Path, settings: Settings) -> None:
    agent = CodingAgent(settings=settings, backend=_DummyBackend("ok"))
    request = _make_request()
    request.rework_feedback = "typecheck failed\n" + ("x" * 3500) + "TAIL_SENTINEL"

    prompt = agent._render_prompt(request, worktree=tmp_path)  # type: ignore[attr-defined]

    assert "Evaluator rework feedback (untrusted diagnostic input):" in prompt
    assert "typecheck failed" in prompt
    assert "TAIL_SENTINEL" not in prompt


def test_coding_prompt_artifact_manifest_projection_omits_diagnostic_prose(
    tmp_path: Path,
    settings: Settings,
) -> None:
    settings.worker_evaluation_agent_feedback_mode = "path"
    agent = CodingAgent(settings=settings, backend=_DummyBackend("ok"))
    request = CodingAgentRequest(
        goal="goal",
        plan=_make_plan(),
        base_commit="abc123",
        base=CommitPlanningContext(
            commit_hash="base",
            subject="Base subject",
            change_summary="base summary",
            evaluation_artifacts=(
                CommitEvaluationArtifactFeedback(
                    key="manifest_only",
                    kind="benchmark_json",
                    mime_type="application/json",
                    summary="summary prose must not reach coding",
                    diagnostics=(
                        EvaluationDiagnosticBrief(
                            kind="regression",
                            message="diagnostic prose must not reach coding",
                        ),
                    ),
                    projection="manifest",
                    visibility="agent_visible",
                    size_bytes=128,
                    artifact_uri="loreley://evaluation-artifacts/job-1/manifest_only",
                ),
            ),
        ),
        inspirations=(),
        iteration_context=IterationContext(seed_job=False),
    )

    prompt = agent._render_prompt(request, worktree=tmp_path)  # type: ignore[attr-defined]

    assert "manifest_only" in prompt
    assert "mime=application/json" in prompt
    assert "summary prose must not reach coding" not in prompt
    assert "diagnostic prose must not reach coding" not in prompt
    assert "loreley://evaluation-artifacts/job-1/manifest_only" not in prompt
    assert "Evidence Guardrail:" in prompt


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
    assert any(
        "no effective repository changes" in str(r["message"]).lower()
        for r in warn_logs
    )


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
