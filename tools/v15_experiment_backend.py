"""Traceable agent backends used only by v15 system experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from loreley.core.worker.agent.backends.kilocode_cli import (
    kilocode_coding_backend as _kilocode_coding_backend,
)
from loreley.core.worker.agent.backends.kilocode_cli import (
    kilocode_planning_backend as _kilocode_planning_backend,
)
from loreley.core.worker.agent.contracts import AgentBackend, AgentInvocation, AgentTask

_COMMIT_PATTERN = re.compile(r"\b[0-9a-f]{40}\b")


def _delay_for(phase: str) -> float:
    name = f"V15_EXPERIMENT_{phase.upper()}_DELAY_SECONDS"
    return max(0.0, float(os.environ.get(name, "0")))


def _trace_path() -> Path | None:
    raw = os.environ.get("V15_EXPERIMENT_TRACE_PATH", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _append_trace(payload: dict[str, object]) -> None:
    path = _trace_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)


class V15ExperimentBackend:
    """Return deterministic Markdown and make a harmless tracked coding change."""

    def __init__(self, phase: str) -> None:
        if phase not in {"planning", "coding"}:
            raise ValueError(f"Unsupported v15 experiment phase: {phase}")
        self.phase = phase

    def run(self, task: AgentTask, *, working_dir: Path) -> AgentInvocation:
        started_wall = datetime.now(UTC)
        started = time.monotonic()
        delay = _delay_for(self.phase)
        if delay:
            time.sleep(delay)

        if self.phase == "coding":
            self._edit_solution(task, working_dir=working_dir)
            stdout = (
                "## Summary\n"
                "- Applied one deterministic experiment marker.\n\n"
                "## Changes\n"
                "- Updated `solution.py` without changing packing behavior.\n\n"
                "## Checks\n"
                "- Deferred validation to Loreley's configured evaluator.\n"
            )
        else:
            stdout = (
                "## Summary\n"
                "- Preserve behavior and add one uniquely attributable marker.\n\n"
                "## Steps\n"
                "1. Inspect `solution.py`.\n"
                "2. Add a job-specific comment.\n"
                "3. Leave the packing algorithm unchanged.\n\n"
                "## Validation\n"
                "- Run the configured Loreley evaluator.\n"
            )

        duration = time.monotonic() - started
        commit_hashes = tuple(dict.fromkeys(_COMMIT_PATTERN.findall(task.prompt)))
        _append_trace(
            {
                "phase": self.phase,
                "job_id": str(task.job_id) if task.job_id is not None else None,
                "run_token": str(task.run_token)
                if task.run_token is not None
                else None,
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "working_directory": str(Path(working_dir).resolve()),
                "started_at": started_wall.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "duration_seconds": duration,
                "configured_delay_seconds": delay,
                "prompt_sha256": hashlib.sha256(
                    task.prompt.encode("utf-8")
                ).hexdigest(),
                "prompt_commit_hashes": list(commit_hashes),
            }
        )
        return AgentInvocation(
            command=("v15-experiment-backend", self.phase),
            stdout=stdout,
            stderr="",
            duration_seconds=duration,
            working_directory=str(Path(working_dir).resolve()),
        )

    @staticmethod
    def _edit_solution(task: AgentTask, *, working_dir: Path) -> None:
        solution_path = Path(working_dir) / "solution.py"
        source = solution_path.read_text(encoding="utf-8")
        marker = f"# loreley-v15-experiment-job: {task.job_id}\n"
        if marker in source:
            marker = f"# loreley-v15-experiment-run: {task.run_token}\n"
        separator = "" if source.endswith("\n") else "\n"
        solution_path.write_text(f"{source}{separator}{marker}", encoding="utf-8")


def planning_backend() -> V15ExperimentBackend:
    return V15ExperimentBackend("planning")


def coding_backend() -> V15ExperimentBackend:
    return V15ExperimentBackend("coding")


class V15TracingBackend:
    """Record process/worktree evidence around an unchanged delegate backend."""

    def __init__(self, phase: str, delegate: AgentBackend) -> None:
        if phase not in {"planning", "coding"}:
            raise ValueError(f"Unsupported v15 experiment phase: {phase}")
        self.phase = phase
        self.delegate = delegate

    def run(self, task: AgentTask, *, working_dir: Path) -> AgentInvocation:
        started_wall = datetime.now(UTC)
        started = time.monotonic()
        outcome = "failed"
        error_type: str | None = None
        try:
            invocation = self.delegate.run(task, working_dir=working_dir)
            outcome = "succeeded"
            return invocation
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            commit_hashes = tuple(dict.fromkeys(_COMMIT_PATTERN.findall(task.prompt)))
            _append_trace(
                {
                    "phase": self.phase,
                    "job_id": str(task.job_id) if task.job_id is not None else None,
                    "run_token": (
                        str(task.run_token) if task.run_token is not None else None
                    ),
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "working_directory": str(Path(working_dir).resolve()),
                    "started_at": started_wall.isoformat(),
                    "finished_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": time.monotonic() - started,
                    "configured_delay_seconds": 0.0,
                    "prompt_sha256": hashlib.sha256(
                        task.prompt.encode("utf-8")
                    ).hexdigest(),
                    "prompt_commit_hashes": list(commit_hashes),
                    "outcome": outcome,
                    "error_type": error_type,
                }
            )


def kilocode_planning_backend() -> V15TracingBackend:
    return V15TracingBackend("planning", _kilocode_planning_backend())


def kilocode_coding_backend() -> V15TracingBackend:
    return V15TracingBackend("coding", _kilocode_coding_backend())


__all__ = [
    "V15ExperimentBackend",
    "V15TracingBackend",
    "coding_backend",
    "kilocode_coding_backend",
    "kilocode_planning_backend",
    "planning_backend",
]
