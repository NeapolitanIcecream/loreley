from __future__ import annotations

import uuid

from loreley.naming import DEFAULT_TASKS_QUEUE_PREFIX, tasks_queue_name


def test_tasks_queue_name_uses_prefix_and_uuid_hex() -> None:
    exp = uuid.UUID("12345678-1234-5678-1234-567812345678")
    queue = tasks_queue_name(exp)
    assert queue == f"{DEFAULT_TASKS_QUEUE_PREFIX}.{exp.hex}"

