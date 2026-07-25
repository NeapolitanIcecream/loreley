from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from loreley.core.worker.agent.contracts import AgentInvocation, AgentTask
from tools.v15_experiment_backend import V15ExperimentBackend, V15TracingBackend


def _task(phase: str) -> AgentTask:
    return AgentTask(
        name=phase,
        prompt="Base: " + "a" * 40,
        job_id=uuid.uuid4(),
        run_token=uuid.uuid4(),
        phase=phase,
    )


def test_planning_backend_traces_without_editing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "trace.jsonl"
    solution = tmp_path / "solution.py"
    solution.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv("V15_EXPERIMENT_TRACE_PATH", str(trace))
    monkeypatch.setenv("V15_EXPERIMENT_PLANNING_DELAY_SECONDS", "0")

    result = V15ExperimentBackend("planning").run(
        _task("planning"),
        working_dir=tmp_path,
    )

    assert "## Summary" in result.stdout
    assert solution.read_text(encoding="utf-8") == "VALUE = 1\n"
    event = json.loads(trace.read_text(encoding="utf-8"))
    assert event["phase"] == "planning"
    assert event["prompt_commit_hashes"] == ["a" * 40]


def test_coding_backend_edits_solution_and_records_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "trace.jsonl"
    solution = tmp_path / "solution.py"
    solution.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv("V15_EXPERIMENT_TRACE_PATH", str(trace))
    monkeypatch.setenv("V15_EXPERIMENT_CODING_DELAY_SECONDS", "0")
    task = _task("coding")

    result = V15ExperimentBackend("coding").run(task, working_dir=tmp_path)

    assert "## Changes" in result.stdout
    assert f"# loreley-v15-experiment-job: {task.job_id}" in solution.read_text(
        encoding="utf-8"
    )
    event = json.loads(trace.read_text(encoding="utf-8"))
    assert event["phase"] == "coding"
    assert event["working_directory"] == str(tmp_path.resolve())


def test_tracing_backend_preserves_delegate_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("V15_EXPERIMENT_TRACE_PATH", str(trace))
    expected = AgentInvocation(
        command=("delegate",),
        stdout="done",
        stderr="",
        duration_seconds=0.25,
        working_directory=str(tmp_path),
    )

    class Delegate:
        def run(self, task: AgentTask, *, working_dir: Path) -> AgentInvocation:
            assert task.phase == "planning"
            assert working_dir == tmp_path
            return expected

    result = V15TracingBackend("planning", Delegate()).run(
        _task("planning"),
        working_dir=tmp_path,
    )

    assert result is expected
    event = json.loads(trace.read_text(encoding="utf-8"))
    assert event["outcome"] == "succeeded"
    assert event["error_type"] is None


def test_tracing_backend_records_delegate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("V15_EXPERIMENT_TRACE_PATH", str(trace))

    class Delegate:
        def run(self, task: AgentTask, *, working_dir: Path) -> AgentInvocation:
            raise RuntimeError("expected")

    with pytest.raises(RuntimeError, match="expected"):
        V15TracingBackend("coding", Delegate()).run(
            _task("coding"),
            working_dir=tmp_path,
        )

    event = json.loads(trace.read_text(encoding="utf-8"))
    assert event["outcome"] == "failed"
    assert event["error_type"] == "RuntimeError"
