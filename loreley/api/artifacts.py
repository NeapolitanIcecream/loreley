"""Shared artifact helpers for the UI API.

This module centralizes artifact key metadata so that routers and Streamlit pages
stay in sync when new artifacts are added.
"""

from __future__ import annotations

from typing import Any, Final
from uuid import UUID

ARTIFACT_SPECS: Final[tuple[tuple[str, str], ...]] = (
    ("planning_prompt", "Planning prompt"),
    ("planning_raw_output", "Planning raw output"),
    ("planning_plan_json", "Planning plan JSON"),
    ("coding_prompt", "Coding prompt"),
    ("coding_raw_output", "Coding raw output"),
    ("coding_execution_json", "Coding execution JSON"),
    ("evaluation_json", "Evaluation JSON"),
    ("evaluation_logs", "Evaluation logs"),
)

ARTIFACT_KEYS: Final[tuple[str, ...]] = tuple(key for key, _ in ARTIFACT_SPECS)


def artifact_path_column(artifact_key: str) -> str:
    """Return the DB attribute/column name holding the artifact path."""

    return f"{artifact_key}_path"


def artifact_url_field(artifact_key: str) -> str:
    """Return the response field name used for the artifact download URL."""

    return f"{artifact_key}_url"


def artifact_filename(artifact_key: str) -> str:
    """Return a user-friendly filename for downloads."""

    if artifact_key.endswith("_json"):
        return f"{artifact_key[:-5]}.json"
    return f"{artifact_key}.txt"


def artifact_media_type(artifact_key: str) -> str:
    """Return a MIME type for the artifact download endpoint."""

    if artifact_key.endswith("_json"):
        return "application/json"
    return "text/plain"


def artifact_download_path(*, job_id: UUID, artifact_key: str) -> str:
    """Return the API path for downloading a single artifact."""

    return f"/api/v1/jobs/{job_id}/artifacts/{artifact_key}"


def build_artifact_urls(*, job_id: UUID, row: Any | None) -> dict[str, str]:
    """Build the artifact URL payload for schema models.

    The returned dict is keyed by `*_url` schema fields and only includes
    artifacts that exist on the row.
    """

    if row is None:
        return {}
    payload: dict[str, str] = {}
    for artifact_key, _label in ARTIFACT_SPECS:
        path_attr = artifact_path_column(artifact_key)
        if getattr(row, path_attr, None):
            payload[artifact_url_field(artifact_key)] = artifact_download_path(job_id=job_id, artifact_key=artifact_key)
    return payload

