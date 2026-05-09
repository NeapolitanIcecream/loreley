from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.exc import IntegrityError

import loreley.api.routers.agent as agent_router
import loreley.api.services.agent as agent_service
import loreley.api.services.evidence as evidence_service
from loreley.api.agent_errors import (
    AgentAPIError,
    agent_api_error_handler,
    agent_validation_exception_handler,
)
from loreley.api.routers.agent import router as agent_api_router
from loreley.api.schemas.agent import AgentActionRequest
from loreley.db.models import AgentAction, EvolutionJob, JobStatus
from tests.support import TestSettings


def _client() -> TestClient:
    app = FastAPI()
    app.add_exception_handler(AgentAPIError, agent_api_error_handler)
    app.add_exception_handler(RequestValidationError, agent_validation_exception_handler)
    app.include_router(agent_api_router, prefix="/api/v1")
    return TestClient(app)


def _settings(token: str | None = None) -> TestSettings:
    return TestSettings(LORELEY_AGENT_API_TOKEN=token)


def _capabilities_payload(*, token_configured: bool = False) -> dict[str, object]:
    return {
        "schema_version": "agent-rest-control-facade.v1",
        "database_schema_version": 12,
        "auth": {"configured": token_configured, "optional_when_unset": True},
        "read_resources": [{"resource": "operator_status", "path": "/api/v1/agent/status"}],
        "actions": [
            {
                "action_type": "retry_job",
                "risk": "medium",
                "dry_run_supported": True,
                "reason_expected": True,
                "idempotency_key_expected": True,
                "required_params": ["job_id"],
                "expected_state_fields": ["status"],
            }
        ],
        "error_shape": {"error_code": "string"},
    }


def _status_payload() -> dict[str, object]:
    now = datetime(2026, 5, 9, tzinfo=timezone.utc)
    return {
        "campaign_program": {
            "current_file": {
                "found": True,
                "source_path": "loreley.program.md",
                "hash": "a" * 64,
                "normalized_hash": "a" * 64,
                "title": "Campaign",
                "recognized_sections": [],
                "parse_warnings": [],
                "sections": {},
            },
            "scheduler": {
                "active_hash": "a" * 64,
                "active_source": "database:evolution_jobs",
                "persisted_hash": "a" * 64,
                "persisted_source": "database:evolution_jobs",
                "current_hash": "a" * 64,
                "current_matches_active": True,
                "change_policy": "locked",
            },
        },
        "baseline": None,
        "repair_pool": {
            "total_failed_candidates": 1,
            "active_repair_jobs": 0,
            "by_repair_state": {"eligible": 1},
            "by_lifecycle_status": {"active": 1},
            "by_failure_kind": {"test_failed": 1},
        },
        "job_health": {
            "jobs": {"unfinished": 0, "pending_ingestion": 0},
            "job_leases": {
                "running": 0,
                "recovery_exhausted_failed": 2,
            },
            "by_status": {"failed": 2},
            "by_job_kind": {"evolution": 2},
        },
        "generated_at": now,
    }


def test_agent_capabilities_status_and_next_actions_serialize(monkeypatch) -> None:
    settings = _settings()
    settings.failed_candidate_repair_enabled = True
    settings.failed_candidate_repair_max_active_jobs = 1
    settings.failed_candidate_repair_max_jobs_per_tick = 1
    settings.failed_candidate_repair_max_tokens = 1
    monkeypatch.setattr(agent_router, "get_settings", lambda: settings)
    monkeypatch.setattr(agent_service, "get_settings", lambda: settings)
    monkeypatch.setattr(agent_service, "operator_status", _status_payload)

    client = _client()

    capabilities = client.get("/api/v1/agent/capabilities")
    status = client.get("/api/v1/agent/status")
    next_actions = client.get("/api/v1/agent/next-actions")

    assert capabilities.status_code == 200
    assert capabilities.json()["schema_version"] == "agent-rest-control-facade.v1"
    assert status.status_code == 200
    assert status.json()["health"] == "actionable"
    action_types = [item["action_type"] for item in next_actions.json()]
    assert action_types == [
        "retry_failed_stale_jobs",
        "baseline_ensure",
        "repair_schedule_one",
    ]
    assert status.json()["safe_next_actions"] == next_actions.json()


def test_agent_auth_unset_token_allows_local_requests(monkeypatch) -> None:
    monkeypatch.setattr(agent_router, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        agent_router,
        "agent_capabilities",
        lambda: _capabilities_payload(token_configured=False),
    )

    response = _client().get("/api/v1/agent/capabilities")

    assert response.status_code == 200


def test_agent_auth_configured_token_rejects_missing_and_wrong_credentials(monkeypatch) -> None:
    monkeypatch.setattr(agent_router, "get_settings", lambda: _settings("secret"))
    monkeypatch.setattr(
        agent_router,
        "agent_capabilities",
        lambda: _capabilities_payload(token_configured=True),
    )
    client = _client()

    missing = client.get("/api/v1/agent/capabilities")
    wrong = client.get(
        "/api/v1/agent/capabilities",
        headers={"Authorization": "Bearer wrong"},
    )
    lowercase = client.get(
        "/api/v1/agent/capabilities",
        headers={"Authorization": "bearer secret"},
    )
    correct = client.get(
        "/api/v1/agent/capabilities",
        headers={"Authorization": "Bearer secret"},
    )

    assert missing.status_code == 401
    assert missing.json()["error_code"] == "unauthorized"
    assert wrong.status_code == 403
    assert wrong.json()["error_code"] == "forbidden"
    assert lowercase.status_code == 200
    assert correct.status_code == 200


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def first(self):
        return self.rows[0] if self.rows else None

    def __iter__(self):
        return iter(self.rows)


class _ExecRows:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def scalars(self) -> _ScalarRows:
        return _ScalarRows(self.rows)


class _ActionSession:
    def __init__(self) -> None:
        self.records: list[AgentAction] = []
        self.jobs: dict[UUID, object] = {}

    def execute(self, _stmt):
        return _ExecRows(self.records)

    def add(self, row: AgentAction) -> None:
        self.records.append(row)

    def flush(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def get(self, model, key):
        if model is EvolutionJob:
            return self.jobs.get(key)
        if model is AgentAction:
            for record in self.records:
                if record.id == key:
                    return record
        return None


def _install_action_session(monkeypatch, session: _ActionSession) -> None:
    @contextmanager
    def _scope():
        yield session

    monkeypatch.setattr(agent_service, "session_scope", _scope)


def test_agent_action_dry_run_does_not_call_write_service(monkeypatch) -> None:
    session = _ActionSession()
    _install_action_session(monkeypatch, session)
    monkeypatch.setattr(agent_service, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        agent_service,
        "schedule_one_repair",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run must not execute writes")),
    )

    payload = agent_service.run_agent_action(
        AgentActionRequest(action_type="repair_schedule_one", dry_run=True),
        actor="test-agent",
    )

    assert payload["status"] == "succeeded"
    assert payload["dry_run"] is True
    assert payload["result"]["validated"] is True
    assert len(session.records) == 1


def test_agent_action_execute_calls_write_service_and_writes_action_record(monkeypatch) -> None:
    session = _ActionSession()
    _install_action_session(monkeypatch, session)
    monkeypatch.setattr(agent_service, "get_settings", lambda: _settings())
    calls: list[str] = []

    def _schedule_one_repair():
        calls.append("schedule_one_repair")
        return {
            "scheduled": False,
            "job_id": None,
            "repair_source_candidate_id": None,
            "base_commit_hash": None,
            "message": "No eligible repair candidate was scheduled.",
        }

    monkeypatch.setattr(agent_service, "schedule_one_repair", _schedule_one_repair)

    payload = agent_service.run_agent_action(
        AgentActionRequest(
            action_type="repair_schedule_one",
            dry_run=False,
            reason="agent triage",
        ),
        actor="test-agent",
    )

    assert calls == ["schedule_one_repair"]
    assert payload["status"] == "succeeded"
    assert payload["result"]["scheduled"] is False
    assert session.records[0].status == "succeeded"
    assert session.records[0].actor == "test-agent"


def test_agent_action_idempotency_replay_does_not_call_write_service_twice(monkeypatch) -> None:
    session = _ActionSession()
    _install_action_session(monkeypatch, session)
    monkeypatch.setattr(agent_service, "get_settings", lambda: _settings())
    calls: list[str] = []

    def _schedule_one_repair():
        calls.append("schedule_one_repair")
        return {
            "scheduled": False,
            "job_id": None,
            "repair_source_candidate_id": None,
            "base_commit_hash": None,
            "message": "No eligible repair candidate was scheduled.",
        }

    monkeypatch.setattr(agent_service, "schedule_one_repair", _schedule_one_repair)
    request = AgentActionRequest(
        action_type="repair_schedule_one",
        dry_run=False,
        idempotency_key="agent-key-1",
        reason="agent triage",
    )

    first = agent_service.run_agent_action(request, actor="test-agent")
    second = agent_service.run_agent_action(request, actor="test-agent")

    assert calls == ["schedule_one_repair"]
    assert second["action_id"] == first["action_id"]
    assert len(session.records) == 1


def test_agent_action_idempotency_reuse_with_different_payload_conflicts(monkeypatch) -> None:
    session = _ActionSession()
    _install_action_session(monkeypatch, session)
    monkeypatch.setattr(agent_service, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        agent_service,
        "schedule_one_repair",
        lambda: (_ for _ in ()).throw(AssertionError("conflict must not execute")),
    )

    first = AgentActionRequest(
        action_type="repair_schedule_one",
        dry_run=True,
        idempotency_key="agent-key-1",
        reason="same operation",
    )
    second = AgentActionRequest(
        action_type="repair_schedule_one",
        dry_run=False,
        idempotency_key="agent-key-1",
        reason="different operation",
    )

    agent_service.run_agent_action(first, actor="test-agent")
    with pytest.raises(AgentAPIError) as exc_info:
        agent_service.run_agent_action(second, actor="test-agent")

    assert exc_info.value.status_code == 409
    assert exc_info.value.error_code == "idempotency_conflict"
    assert len(session.records) == 1


def test_agent_action_idempotency_insert_race_replays_existing_record(monkeypatch) -> None:
    class _RacingInsertSession(_ActionSession):
        rolled_back = False

        def flush(self) -> None:
            raise IntegrityError("insert agent action", {}, Exception("unique"))

        def rollback(self) -> None:
            self.rolled_back = True

    existing = AgentAction(
        id=uuid4(),
        idempotency_key="agent-race-key",
        actor="first-agent",
        action_type="repair_schedule_one",
        status="succeeded",
        dry_run=False,
        request_payload={
            "action_type": "repair_schedule_one",
            "dry_run": False,
            "idempotency_key": "agent-race-key",
            "reason": None,
            "expected_state": {},
            "params": {},
        },
        expected_state={},
        result_payload={"preconditions": [], "result": {"scheduled": False}},
        created_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )
    first_session = _RacingInsertSession()
    second_session = _ActionSession()
    second_session.records.append(existing)
    sessions = iter([first_session, second_session])

    @contextmanager
    def _scope():
        yield next(sessions)

    monkeypatch.setattr(agent_service, "session_scope", _scope)
    monkeypatch.setattr(
        agent_service,
        "schedule_one_repair",
        lambda: (_ for _ in ()).throw(AssertionError("idempotency replay must not execute")),
    )

    payload = agent_service.run_agent_action(
        AgentActionRequest(
            action_type="repair_schedule_one",
            dry_run=False,
            idempotency_key="agent-race-key",
        ),
        actor="second-agent",
    )

    assert first_session.rolled_back is True
    assert payload["action_id"] == existing.id
    assert payload["result"] == {"scheduled": False}


def test_agent_action_invalid_action_type_is_rejected_before_audit_insert(monkeypatch) -> None:
    session = _ActionSession()
    _install_action_session(monkeypatch, session)

    with pytest.raises(AgentAPIError) as exc_info:
        agent_service.run_agent_action(
            AgentActionRequest(action_type="x" * 65, dry_run=True),
            actor="test-agent",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "invalid_action_type"
    assert session.records == []


def test_agent_action_oversized_idempotency_key_is_rejected_before_audit_insert(
    monkeypatch,
) -> None:
    session = _ActionSession()
    _install_action_session(monkeypatch, session)

    with pytest.raises(AgentAPIError) as exc_info:
        agent_service.run_agent_action(
            AgentActionRequest(
                action_type="repair_schedule_one",
                dry_run=True,
                idempotency_key="k" * 257,
            ),
            actor="test-agent",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "invalid_request"
    assert session.records == []


def test_agent_action_retry_failed_stale_requires_boolean_all_param(monkeypatch) -> None:
    session = _ActionSession()
    _install_action_session(monkeypatch, session)
    monkeypatch.setattr(
        agent_service,
        "load_failed_stale_retry_rows",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("invalid params")),
    )

    with pytest.raises(AgentAPIError) as exc_info:
        agent_service.run_agent_action(
            AgentActionRequest(
                action_type="retry_failed_stale_jobs",
                dry_run=True,
                params={"all": "false"},
            ),
            actor="test-agent",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "invalid_request"
    assert len(session.records) == 1
    assert session.records[0].status == "failed"
    assert session.records[0].error_code == "invalid_request"


def test_agent_action_expected_state_mismatch_returns_structured_error_and_audit(
    monkeypatch,
    captured_logs,
) -> None:
    settings = _settings()
    job_id = uuid4()
    session = _ActionSession()
    session.jobs[job_id] = SimpleNamespace(
        id=job_id,
        status=JobStatus.FAILED,
        lease_expires_at=None,
        run_token=None,
        worker_id=None,
        heartbeat_at=None,
        recovery_count=1,
    )
    _install_action_session(monkeypatch, session)
    monkeypatch.setattr(agent_router, "get_settings", lambda: settings)

    response = _client().post(
        "/api/v1/agent/actions",
        json={
            "action_type": "retry_job",
            "dry_run": False,
            "reason": "retry failed job",
            "params": {"job_id": str(job_id)},
            "expected_state": {"status": "running"},
        },
    )

    assert response.status_code == 409
    assert response.json()["error_code"] == "precondition_failed"
    assert response.json()["resource"] == {"type": "job", "id": str(job_id)}
    assert len(session.records) == 1
    assert session.records[0].status == "failed"
    assert session.records[0].error_code == "precondition_failed"
    assert any(
        record["module"] == "api.agent"
        and record["message"] == "Agent action failed"
        and record["extra"].get("error_code") == "precondition_failed"
        for record in captured_logs
    )


def _artifact_row(
    *,
    key: str,
    visibility: str,
    job_id: UUID | None = None,
    commit_hash: str = "c" * 40,
    summary: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        job_id=job_id or uuid4(),
        commit_hash=commit_hash,
        key=key,
        kind="benchmark_json",
        mime_type="application/json",
        label=None,
        summary=summary if summary is not None else f"{key} summary",
        visibility=visibility,
        agent_projection="summary",
        storage_path=f"/tmp/{key}.json",
        size_bytes=12,
        sha256="a" * 64,
        diagnostics=[],
        created_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )


def test_agent_job_feedback_excludes_non_agent_visible_evidence_from_payload(monkeypatch) -> None:
    """Regression: agent job feedback exposed human-only artifact metadata."""

    job_id = uuid4()
    visible = _artifact_row(
        key="agent-visible",
        visibility="agent_visible",
        job_id=job_id,
        summary="agent-safe evaluator summary",
    )
    human_only = _artifact_row(
        key="human-only",
        visibility="human_only",
        job_id=job_id,
        summary="human only secret summary",
    )
    hidden = _artifact_row(
        key="hidden",
        visibility="hidden",
        job_id=job_id,
        summary="hidden secret summary",
    )

    def _list_evaluation_artifacts_for_job(**kwargs):
        assert kwargs == {"job_id": job_id, "visibility": "agent_visible"}
        return [visible, human_only, hidden]

    monkeypatch.setattr(agent_router, "get_settings", lambda: _settings())
    monkeypatch.setattr(evidence_service, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        agent_router,
        "list_evaluation_artifacts_for_job",
        _list_evaluation_artifacts_for_job,
    )

    response = _client().get(f"/api/v1/agent/jobs/{job_id}/feedback")

    assert response.status_code == 200
    payload = response.json()
    artifact_keys = [item["key"] for item in payload["artifacts"]]
    download_urls = [item["download_url"] for item in payload["artifacts"]]
    feedback_text = payload["feedback"]["text"]
    assert artifact_keys == ["agent-visible"]
    assert download_urls == [f"/api/v1/jobs/{job_id}/evaluation-artifacts/agent-visible"]
    assert "agent-safe evaluator summary" in feedback_text
    assert "human-only" not in artifact_keys
    assert "hidden" not in artifact_keys
    assert "human-only" not in "\n".join(download_urls)
    assert "hidden" not in "\n".join(download_urls)
    assert "human only secret summary" not in feedback_text
    assert "hidden secret summary" not in feedback_text


def test_agent_commit_feedback_excludes_non_agent_visible_evidence_from_payload(monkeypatch) -> None:
    """Regression: agent commit feedback exposed human-only artifact metadata."""

    job_id = uuid4()
    commit_hash = "d" * 40
    visible = _artifact_row(
        key="commit-agent-visible",
        visibility="agent_visible",
        job_id=job_id,
        commit_hash=commit_hash,
        summary="commit agent-safe evaluator summary",
    )
    human_only = _artifact_row(
        key="commit-human-only",
        visibility="human_only",
        job_id=job_id,
        commit_hash=commit_hash,
        summary="commit human only secret summary",
    )
    hidden = _artifact_row(
        key="commit-hidden",
        visibility="hidden",
        job_id=job_id,
        commit_hash=commit_hash,
        summary="commit hidden secret summary",
    )

    def _list_evaluation_artifacts_for_commit(**kwargs):
        assert kwargs == {
            "commit_hash": commit_hash,
            "visibility": "agent_visible",
        }
        return [visible, human_only, hidden]

    monkeypatch.setattr(agent_router, "get_settings", lambda: _settings())
    monkeypatch.setattr(evidence_service, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        agent_router,
        "list_evaluation_artifacts_for_commit",
        _list_evaluation_artifacts_for_commit,
    )

    response = _client().get(f"/api/v1/agent/commits/{commit_hash}/feedback")

    assert response.status_code == 200
    payload = response.json()
    artifact_keys = [item["key"] for item in payload["artifacts"]]
    download_urls = [item["download_url"] for item in payload["artifacts"]]
    feedback_text = payload["feedback"]["text"]
    assert artifact_keys == ["commit-agent-visible"]
    assert download_urls == [
        f"/api/v1/jobs/{job_id}/evaluation-artifacts/commit-agent-visible"
    ]
    assert "commit agent-safe evaluator summary" in feedback_text
    assert "commit-human-only" not in artifact_keys
    assert "commit-hidden" not in artifact_keys
    assert "commit-human-only" not in "\n".join(download_urls)
    assert "commit-hidden" not in "\n".join(download_urls)
    assert "commit human only secret summary" not in feedback_text
    assert "commit hidden secret summary" not in feedback_text
