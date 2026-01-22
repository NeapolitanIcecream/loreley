"""Instance metadata access for the UI API."""

from __future__ import annotations

from loreley.db.base import session_scope
from loreley.db.models import InstanceMetadata


def get_instance_metadata() -> InstanceMetadata:
    """Return the single instance metadata row or raise."""

    with session_scope() as session:
        meta = session.get(InstanceMetadata, 1)
        if meta is None:
            raise RuntimeError("Instance metadata is missing.")
        return meta

