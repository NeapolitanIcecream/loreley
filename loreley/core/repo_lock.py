from __future__ import annotations

"""Cross-process file locks for shared git repositories."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["file_lock", "repo_lock", "resolve_repo_lock_path"]


def resolve_repo_lock_path(repo_root: Path | str) -> Path:
    """Return the lock path used to coordinate mutations for repo_root."""

    resolved = Path(repo_root).expanduser().resolve()
    return resolved.parent / f".{resolved.name}.lock"


@contextmanager
def file_lock(lock_path: Path | str) -> Iterator[None]:
    """Acquire an advisory cross-process file lock."""

    path = Path(lock_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows fallback
            import msvcrt

            fh.seek(0)
            if fh.tell() == 0:
                fh.write("\n")
                fh.flush()
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover - Windows fallback
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
        fh.close()


@contextmanager
def repo_lock(repo_root: Path | str) -> Iterator[None]:
    """Acquire the shared mutation lock for repo_root."""

    with file_lock(resolve_repo_lock_path(repo_root)):
        yield
