from __future__ import annotations

import json
import hashlib
import os
import uuid
from pathlib import Path

import pytest

from loreley.core.worker.artifacts import (
    FailureJobArtifactWriteRequest,
    JobArtifactWriteRequest,
    JobArtifactWriteResult,
    write_failure_job_artifacts,
    write_job_artifacts,
)
from loreley.core.usage import LLMUsageEventPayload
from loreley.core.worker.coding import CodingAgentResponse, ExecutionReport
from loreley.core.worker.evaluator import (
    EvaluationArtifact,
    EvaluationDiagnostic,
    EvaluationFailureResult,
    EvaluationMetric,
    EvaluationOutcome,
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
        working_directory="/tmp/loreley-job-worktree",
        usage_events=(
            _usage_event(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                cost_usd="0.001",
                cost_source="estimated",
            ),
            _usage_event(
                input_tokens=50,
                output_tokens=10,
                total_tokens=60,
                cost_source="unpriced",
            ),
        ),
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
        working_directory="/tmp/loreley-job-worktree",
        usage_events=(
            _usage_event(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_source="unavailable",
            ),
        ),
    )
    evaluation = EvaluationResult(
        summary="ok",
        candidate_identity="binary-sha256:abc",
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
    assert evaluation_payload["candidate_identity"] == "binary-sha256:abc"
    assert planning_payload["worker"]["pid"] == os.getpid()
    assert coding_payload["worker"]["pid"] == os.getpid()
    assert evaluation_payload["worker"]["pid"] == os.getpid()
    assert planning_payload["backend"]["working_directory"] == "/tmp/loreley-job-worktree"
    assert coding_payload["backend"]["working_directory"] == "/tmp/loreley-job-worktree"
    assert str(run_token) in paths["planning_plan_json_path"]
    assert planning_payload["backend"]["usage_summary"] == {
        "event_count": 2,
        "input_tokens": 150,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 30,
        "reasoning_output_tokens": 0,
        "total_tokens": 180,
        "cost_usd": "0.001",
        "unpriced_count": 1,
        "unavailable_count": 0,
    }
    assert coding_payload["backend"]["usage_summary"]["unavailable_count"] == 1


@pytest.mark.usefixtures("settings")
def test_write_failure_job_artifacts_includes_usage_summary(settings, tmp_path: Path) -> None:
    settings.logs_base_dir = str(tmp_path)
    run_token = uuid.uuid4()
    plan = _plan_response()
    plan.usage_events = (
        _usage_event(
            input_tokens=7,
            output_tokens=3,
            total_tokens=10,
            cost_usd="0.0002",
            cost_source="provider_reported",
        ),
    )
    coding = _coding_response()
    coding.usage_events = (
        _usage_event(
            input_tokens=11,
            output_tokens=5,
            total_tokens=16,
            cost_source="unpriced",
        ),
    )
    outcome = EvaluationOutcome(
        evaluator_name="pytest",
        candidate_commit_hash="candidate",
        outcome_kind="candidate_failed",
        failure=EvaluationFailureResult(
            failure_stage="evaluation",
            failure_kind="test_failed",
            safe_failure_summary="tests failed",
        ),
    )

    result = write_failure_job_artifacts(
        FailureJobArtifactWriteRequest(
            job_id=uuid.uuid4(),
            run_token=run_token,
            base_commit_hash="base",
            candidate_commit_hash="candidate",
            message="failed",
            outcome=outcome,
            plan=plan,
            coding=coding,
            settings=settings,
        )
    )

    paths = result.fixed.as_dict()
    planning_payload = json.loads(Path(paths["planning_plan_json_path"]).read_text(encoding="utf-8"))
    coding_payload = json.loads(Path(paths["coding_execution_json_path"]).read_text(encoding="utf-8"))

    assert planning_payload["backend"]["usage_summary"]["total_tokens"] == 10
    assert planning_payload["backend"]["usage_summary"]["cost_usd"] == "0.0002"
    assert coding_payload["backend"]["usage_summary"]["total_tokens"] == 16
    assert coding_payload["backend"]["usage_summary"]["unpriced_count"] == 1


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


def test_write_job_artifacts_omits_agent_files_when_stages_were_skipped(
    settings,
    tmp_path: Path,
) -> None:
    settings.logs_base_dir = str(tmp_path)

    result = write_job_artifacts(
        JobArtifactWriteRequest(
            job_id=uuid.uuid4(),
            run_token=uuid.uuid4(),
            plan=None,
            coding=None,
            evaluation=EvaluationResult(summary="supplied candidate passed"),
            base_commit_hash="base",
            candidate_commit_hash="candidate",
            commit_message="Evaluate supplied candidate",
            worktree=tmp_path,
            settings=settings,
        )
    )

    paths = result.fixed.as_dict()
    assert "planning_prompt_path" not in paths
    assert "planning_raw_output_path" not in paths
    assert "planning_plan_json_path" not in paths
    assert "coding_prompt_path" not in paths
    assert "coding_raw_output_path" not in paths
    assert "coding_execution_json_path" not in paths
    assert Path(paths["evaluation_json_path"]).exists()


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


def _usage_event(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cost_source: str,
    cost_usd: str | None = None,
) -> LLMUsageEventPayload:
    return LLMUsageEventPayload(
        source="codex_cli",
        phase="planning",
        provider="openai",
        model="gpt-test",
        api_surface="codex_exec",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        cost_source=cost_source,
    )
