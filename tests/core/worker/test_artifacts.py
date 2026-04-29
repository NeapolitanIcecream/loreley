from __future__ import annotations

import json
import hashlib
import os
import uuid
from pathlib import Path

import pytest

from loreley.core.worker.artifacts import (
    JobArtifactWriteRequest,
    JobArtifactWriteResult,
    write_job_artifacts,
)
from loreley.core.worker.coding import CodingAgentResponse, ExecutionReport
from loreley.core.worker.evaluator import (
    EvaluationArtifact,
    EvaluationDiagnostic,
    EvaluationMetric,
    EvaluationResult,
)
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

    result = write_job_artifacts(
        JobArtifactWriteRequest(
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
    )
    assert isinstance(result, JobArtifactWriteResult)
    paths = result.fixed.as_dict()

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


def test_write_job_artifacts_materializes_evaluator_artifacts_under_worker_root(
    settings,
    tmp_path: Path,
) -> None:
    settings.logs_base_dir = str(tmp_path)
    worktree = tmp_path / "worktree"
    reports = worktree / "reports"
    reports.mkdir(parents=True)
    report_path = reports / "benchmark.json"
    report_path.write_text('{"score": 1.5}', encoding="utf-8")

    evaluation = EvaluationResult(
        summary="ok",
        artifacts=(
            EvaluationArtifact(
                key="benchmark_report",
                kind="benchmark_json",
                mime_type="application/json",
                path="reports/benchmark.json",
                summary="Parser throughput improved.",
                visibility="agent_visible",
                diagnostics=(
                    EvaluationDiagnostic(
                        kind="improvement",
                        severity="info",
                        message="throughput improved",
                        metric="throughput",
                        value=1.5,
                    ),
                ),
            ),
            EvaluationArtifact(
                key="stderr_excerpt",
                kind="log",
                mime_type="text/plain",
                inline_payload="stderr excerpt",
                summary="Human audit excerpt.",
                visibility="human_only",
            ),
        ),
    )

    result = write_job_artifacts(
        JobArtifactWriteRequest(
            job_id=uuid.uuid4(),
            run_token=uuid.uuid4(),
            plan=_plan_response(),
            coding=_coding_response(),
            evaluation=evaluation,
            base_commit_hash="base",
            candidate_commit_hash="candidate",
            commit_message="message",
            worktree=worktree,
            settings=settings,
        )
    )

    assert len(result.evaluation_artifacts) == 2
    benchmark = result.evaluation_artifacts[0]
    assert benchmark.key == "benchmark_report"
    assert benchmark.size_bytes == report_path.stat().st_size
    assert benchmark.sha256 == hashlib.sha256(report_path.read_bytes()).hexdigest()
    stored = Path(str(benchmark.storage_path))
    assert stored.exists()
    assert stored.read_text(encoding="utf-8") == report_path.read_text(encoding="utf-8")
    assert str(stored).startswith(str(tmp_path / "logs"))
    assert str(report_path) != str(stored)

    evaluation_payload = json.loads(Path(result.fixed.evaluation_json_path or "").read_text(encoding="utf-8"))
    assert evaluation_payload["evaluation_artifacts"][0]["key"] == "benchmark_report"
    assert "storage_path" not in evaluation_payload["evaluation_artifacts"][0]
    assert evaluation_payload["artifact_validation_warnings"] == []


def test_write_job_artifacts_records_sanitized_warning_for_unsafe_path(
    settings,
    tmp_path: Path,
) -> None:
    settings.logs_base_dir = str(tmp_path)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    outside = tmp_path / "secret-report.txt"
    outside.write_text("do not leak", encoding="utf-8")
    evaluation = EvaluationResult(
        summary="ok",
        artifacts=(
            EvaluationArtifact(
                key="unsafe_path",
                kind="log",
                mime_type="text/plain",
                path="../secret-report.txt",
                summary="Unsafe path should degrade to metadata only.",
                visibility="agent_visible",
            ),
        ),
    )

    result = write_job_artifacts(
        JobArtifactWriteRequest(
            job_id=uuid.uuid4(),
            run_token=uuid.uuid4(),
            plan=_plan_response(),
            coding=_coding_response(),
            evaluation=evaluation,
            base_commit_hash="base",
            candidate_commit_hash="candidate",
            commit_message="message",
            worktree=worktree,
            settings=settings,
        )
    )

    assert len(result.evaluation_artifacts) == 1
    assert result.evaluation_artifacts[0].storage_path is None
    assert result.validation_warnings[0].code == "path_escape"
    payload = json.loads(Path(result.fixed.evaluation_json_path or "").read_text(encoding="utf-8"))
    dumped = json.dumps(payload)
    assert "../secret-report.txt" not in dumped
    assert str(outside) not in dumped
    assert payload["artifact_validation_warnings"][0] == result.validation_warnings[0].as_dict()


def _plan_response() -> PlanningAgentResponse:
    return PlanningAgentResponse(
        plan=PlanDocument(summary="plan", markdown="## Summary\n- plan\n"),
        raw_output="plan raw",
        prompt="plan prompt",
        command=("codex", "exec"),
        stderr="",
        attempts=1,
        duration_seconds=1.5,
    )


def _coding_response() -> CodingAgentResponse:
    return CodingAgentResponse(
        report=ExecutionReport(summary="coding", markdown="## Summary\n- coding\n"),
        raw_output="coding raw",
        prompt="coding prompt",
        command=("codex", "exec"),
        stderr="",
        attempts=1,
        duration_seconds=1.0,
    )
