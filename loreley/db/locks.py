from __future__ import annotations

"""Postgres advisory locks used for process-level coordination.

We use session-level advisory locks to enforce "single scheduler per experiment".

Important:
- Postgres exposes `pg_try_advisory_lock(bigint)` and `pg_try_advisory_lock(int, int)`.
- The 2-arg variant uses 32-bit integers, which would require narrowing a UUID.
- We therefore map the full UUID to a single signed 64-bit key and use the bigint variant.
"""

from dataclasses import dataclass
import hashlib
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine


@dataclass(frozen=True, slots=True)
class AdvisoryLock:
    """An acquired advisory lock held by a dedicated DB connection."""

    connection: Connection
    key: int


def uuid_to_pg_bigint_lock_key(experiment_id: UUID) -> int:
    """Map a UUID to a signed bigint suitable for `pg_try_advisory_lock(bigint)`."""

    if not isinstance(experiment_id, UUID):
        raise TypeError("experiment_id must be a UUID.")
    digest = hashlib.sha256(experiment_id.bytes).digest()
    # Signed 64-bit integer range: [-2^63, 2^63-1]
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


def try_acquire_pg_advisory_lock(*, engine: Engine, key: int) -> AdvisoryLock | None:
    """Try to acquire a session-level advisory lock.

    Returns an `AdvisoryLock` (holding a dedicated Connection) on success, else None.
    """

    conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        locked = conn.execute(
            text("SELECT pg_try_advisory_lock(:key)"),
            {"key": int(key)},
        ).scalar_one()
        if not locked:
            conn.close()
            return None
        return AdvisoryLock(connection=conn, key=int(key))
    except Exception:
        conn.close()
        raise


def release_pg_advisory_lock(lock: AdvisoryLock) -> None:
    """Release the advisory lock and close the underlying connection."""

    try:
        lock.connection.execute(
            text("SELECT pg_advisory_unlock(:key)"),
            {"key": int(lock.key)},
        )
    finally:
        lock.connection.close()


