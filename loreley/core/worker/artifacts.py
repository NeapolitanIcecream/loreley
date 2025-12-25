"""Cold-path artifact store for the evolution worker.

Artifacts are large, audit/debug oriented payloads (prompts, raw outputs, logs).
They must not be embedded in primary DB rows. Instead, store them on disk and
persist only their paths in the database.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID

from loguru import logger

from loreley.config import Settings, get_settings
from loreley.core.worker.coding import CodingAgentResponse
from loreley.core.worker.evaluator import EvaluationResult
from loreley.core.worker.planning import PlanningAgentResponse

log = logger.bind(module="worker.artifacts")

__all__ = ["write_job_artifacts"]


def _resolve_artifacts_dir(settings: Settings, job_id: UUID) -> Path:
    if settings.logs_base_dir:
        base_dir = Path(settings.logs_base_dir).expanduser()
    else:
        base_dir = Path.cwd()
    root = base_dir / "logs" / "worker" / "artifacts" / str(job_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_job_artifacts(
    *,
    job_id: UUID,
    plan: PlanningAgentResponse,
    coding: CodingAgentResponse,
    evaluation: EvaluationResult,
    base_commit_hash: str,
    candidate_commit_hash: str,
    commit_message: str,
    settings: Settings | None = None,
) -> dict[str, str]:
    """Write artifacts to disk and return a dict of absolute paths."""

    settings = settings or get_settings()
    root = _resolve_artifacts_dir(settings, job_id)

    paths: dict[str, str] = {}

    planning_prompt = root / "planning_prompt.txt"
    planning_raw = root / "planning_raw_output.txt"
    planning_plan = root / "planning_plan.json"
    _write_text(planning_prompt, plan.prompt)
    _write_text(planning_raw, plan.raw_output)
    _write_json(
        planning_plan,
        {
            "job_id": str(job_id),
            "base_commit_hash": base_commit_hash,
            "candidate_commit_hash": candidate_commit_hash,
            "commit_message": commit_message,
            "plan": plan.plan.as_dict(),
            "backend": {
                "command": list(plan.command),
                "stderr": plan.stderr,
                "attempts": plan.attempts,
                "duration_seconds": plan.duration_seconds,
            },
        },
    )
    paths["planning_prompt_path"] = str(planning_prompt)
    paths["planning_raw_output_path"] = str(planning_raw)
    paths["planning_plan_json_path"] = str(planning_plan)

    coding_prompt = root / "coding_prompt.txt"
    coding_raw = root / "coding_raw_output.txt"
    coding_exec = root / "coding_execution.json"
    _write_text(coding_prompt, coding.prompt)
    _write_text(coding_raw, coding.raw_output)
    _write_json(
        coding_exec,
        {
            "job_id": str(job_id),
            "base_commit_hash": base_commit_hash,
            "candidate_commit_hash": candidate_commit_hash,
            "commit_message": commit_message,
            "execution": {
                "implementation_summary": coding.execution.implementation_summary,
                "commit_message": coding.execution.commit_message,
                "step_results": [
                    asdict(step) | {"status": step.status.value}
                    for step in coding.execution.step_results
                ],
                "tests_executed": list(coding.execution.tests_executed),
                "tests_recommended": list(coding.execution.tests_recommended),
                "follow_up_items": list(coding.execution.follow_up_items),
                "notes": list(coding.execution.notes),
            },
            "backend": {
                "command": list(coding.command),
                "stderr": coding.stderr,
                "attempts": coding.attempts,
                "duration_seconds": coding.duration_seconds,
            },
        },
    )
    paths["coding_prompt_path"] = str(coding_prompt)
    paths["coding_raw_output_path"] = str(coding_raw)
    paths["coding_execution_json_path"] = str(coding_exec)

    evaluation_json = root / "evaluation.json"
    evaluation_logs = root / "evaluation_logs.txt"
    _write_json(
        evaluation_json,
        {
            "job_id": str(job_id),
            "base_commit_hash": base_commit_hash,
            "candidate_commit_hash": candidate_commit_hash,
            "commit_message": commit_message,
            "summary": evaluation.summary,
            "metrics": [metric.as_dict() for metric in evaluation.metrics],
            "tests_executed": list(evaluation.tests_executed),
            "logs": list(evaluation.logs),
            "extra": dict(evaluation.extra or {}),
        },
    )
    _write_text(evaluation_logs, "\n".join(str(line) for line in evaluation.logs))
    paths["evaluation_json_path"] = str(evaluation_json)
    paths["evaluation_logs_path"] = str(evaluation_logs)

    log.info("Wrote {} artifact file(s) for job {}", len(paths), job_id)
    return paths


