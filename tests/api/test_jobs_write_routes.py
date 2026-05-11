from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

import loreley.api.auth as api_auth
from loreley.api.auth import require_write_auth
import loreley.api.routers.jobs as jobs_router
from loreley.api.routers.jobs import router as jobs_api_router
from loreley.api.services.jobs import JobNotFoundError, JobRetryConflictError, JobRetryValidationError
from tests.support import TestSettings


def _client(*, authenticated: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(jobs_api_router, prefix="/api/v1")
    if authenticated:
        app.dependency_overrides[require_write_auth] = lambda: "test-operator"
    return TestClient(app)


def test_retry_job_route_requeues_retryable_job(monkeypatch) -> None:
    job_id = uuid4()

    def _retry_job_by_id(**kwargs):
        assert kwargs["job_id"] == job_id
        return {
            "job_id": str(job_id),
            "previous_status": "failed",
            "new_status": "pending",
            "recovery_count_reset_from": 4,
            "reason": "operator retry",
        }

    monkeypatch.setattr(jobs_router, "retry_job_by_id", _retry_job_by_id)

    response = _client().post(f"/api/v1/jobs/{job_id}/retry", json={"reason": "operator retry"})

    assert response.status_code == 200
    assert response.json()["new_status"] == "pending"


def test_retry_job_route_requires_write_auth_configuration(monkeypatch) -> None:
    """Regression: direct job retry writes must not run without a write token."""

    job_id = uuid4()
    monkeypatch.setattr(api_auth, "get_settings", lambda: TestSettings(LORELEY_API_WRITE_TOKEN=None))
    monkeypatch.setattr(
        jobs_router,
        "retry_job_by_id",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unauthenticated write executed")),
    )

    response = _client(authenticated=False).post(f"/api/v1/jobs/{job_id}/retry", json={})

    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == "write_auth_not_configured"


def test_retry_job_route_accepts_configured_write_token(monkeypatch) -> None:
    job_id = uuid4()
    monkeypatch.setattr(
        api_auth,
        "get_settings",
        lambda: TestSettings(LORELEY_API_WRITE_TOKEN="secret"),
    )

    def _retry_job_by_id(**kwargs):
        assert kwargs["job_id"] == job_id
        return {
            "job_id": str(job_id),
            "previous_status": "failed",
            "new_status": "pending",
            "recovery_count_reset_from": 1,
            "reason": "operator retry",
        }

    monkeypatch.setattr(jobs_router, "retry_job_by_id", _retry_job_by_id)

    response = _client(authenticated=False).post(
        f"/api/v1/jobs/{job_id}/retry",
        json={},
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.json()["new_status"] == "pending"


def test_retry_job_route_rejects_active_running_job(monkeypatch) -> None:
    job_id = uuid4()

    def _retry_job_by_id(**_kwargs):
        raise JobRetryConflictError("Only failed or stuck RUNNING jobs can be retried.")

    monkeypatch.setattr(jobs_router, "retry_job_by_id", _retry_job_by_id)

    response = _client().post(f"/api/v1/jobs/{job_id}/retry", json={})

    assert response.status_code == 409
    assert "Only failed or stuck" in response.json()["detail"]


def test_retry_job_route_returns_404_for_missing_job(monkeypatch) -> None:
    job_id = uuid4()

    def _retry_job_by_id(**_kwargs):
        raise JobNotFoundError("Job not found.")

    monkeypatch.setattr(jobs_router, "retry_job_by_id", _retry_job_by_id)

    response = _client().post(f"/api/v1/jobs/{job_id}/retry", json={})

    assert response.status_code == 404


def test_retry_failed_stale_route_accepts_limit(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _retry_failed_stale_jobs(**kwargs):
        captured.update(kwargs)
        return {
            "filters": {"failed_stale": True, "all": False, "limit": 2},
            "count": 1,
            "retried_jobs": [
                {
                    "job_id": str(uuid4()),
                    "previous_status": "failed",
                    "new_status": "pending",
                    "recovery_count_reset_from": 4,
                    "reason": "operator retry",
                }
            ],
        }

    monkeypatch.setattr(jobs_router, "retry_failed_stale_jobs", _retry_failed_stale_jobs)

    response = _client().post("/api/v1/jobs/retry-failed-stale", json={"limit": 2})

    assert response.status_code == 200
    assert captured["retry_all"] is False
    assert captured["limit"] == 2
    assert response.json()["count"] == 1


def test_retry_failed_stale_route_validates_all_or_limit(monkeypatch) -> None:
    def _retry_failed_stale_jobs(**_kwargs):
        raise JobRetryValidationError("Use either all=true or limit.")

    monkeypatch.setattr(jobs_router, "retry_failed_stale_jobs", _retry_failed_stale_jobs)

    response = _client().post("/api/v1/jobs/retry-failed-stale", json={})

    assert response.status_code == 400
    assert "all=true or limit" in response.json()["detail"]
