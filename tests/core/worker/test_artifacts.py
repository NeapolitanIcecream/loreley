from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from loreley.core.worker.artifacts import write_job_artifacts
from loreley.core.worker.coding import CodingAgentResponse, ExecutionReport
from loreley.core.worker.evaluator import EvaluationMetric, EvaluationResult
from loreley.core.worker.planning import PlanDocument, PlanningAgentResponse


@pytest.mark.usefixtures("settings")
def test_write_job_artifacts_includes_worker_metadata(settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings.logs_base_dir = str(tmp_path)
    monkeypatch.setenv("LORELEY_WORKER_INSTANCE_ID", "worker-03")
    run_token = uuid.uuid4()

    plan = PlanningAgentResponse(
        plan=PlanDocument(
            summary="plan",
            markdown="## Summary\n- plan\n",
            focus_metrics=("sum_radii",),
            guardrails=("keep deterministic",),
        ),
        raw_output="plan raw",
        prompt="plan prompt",
        command=("codex", "exec"),
        stderr="",
        attempts=1,
        duration_seconds=1.5,
    )
    coding = CodingAgentResponse(
        report=ExecutionReport(
            summary="coding",
            markdown="## Summary\n- coding\n",
        ),
        raw_output="coding raw",
        prompt="coding prompt",
        command=("codex", "exec"),
        stderr="",
        attempts=2,
        duration_seconds=3.0,
    )
    evaluation = EvaluationResult(
        summary="ok",
        metrics=(EvaluationMetric(name="sum_radii", value=1.23),),
        extra={"evaluator_duration_seconds": 0.25},
    )

    paths = write_job_artifacts(
        job_id=uuid.uuid4(),
        run_token=run_token,
        plan=plan,
        coding=coding,
        evaluation=evaluation,
        base_commit_hash="base",
        candidate_commit_hash="candidate",
        commit_message="message",
        settings=settings,
    )

    planning_payload = json.loads(Path(paths["planning_plan_json_path"]).read_text(encoding="utf-8"))
    coding_payload = json.loads(Path(paths["coding_execution_json_path"]).read_text(encoding="utf-8"))
    evaluation_payload = json.loads(Path(paths["evaluation_json_path"]).read_text(encoding="utf-8"))

    assert planning_payload["worker"]["instance_id"] == "worker-03"
    assert coding_payload["worker"]["instance_id"] == "worker-03"
    assert evaluation_payload["worker"]["instance_id"] == "worker-03"
    assert planning_payload["worker"]["pid"] == os.getpid()
    assert coding_payload["worker"]["pid"] == os.getpid()
    assert evaluation_payload["worker"]["pid"] == os.getpid()
    assert str(run_token) in paths["planning_plan_json_path"]
