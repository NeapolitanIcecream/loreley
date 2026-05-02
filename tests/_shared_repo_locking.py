from __future__ import annotations

from pathlib import Path
import time
from typing import Any


def hold_repo_lock(lock_path: str, hold_seconds: float, ready_queue: Any) -> None:
    import fcntl

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        ready_queue.put("locked")
        time.sleep(hold_seconds)
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
