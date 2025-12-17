"""Log file browsing helpers for the UI API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loreley.api.deps import resolve_logs_root
from loreley.config import Settings


@dataclass(frozen=True, slots=True)
class LogFileInfo:
    name: str
    size_bytes: int
    modified_at: datetime


def _role_dir(settings: Settings, role: str) -> Path:
    root = resolve_logs_root(settings)
    role_name = role.strip()
    if role_name not in {"scheduler", "worker", "ui_api"}:
        raise ValueError("Invalid role; expected scheduler|worker|ui_api.")
    return (root / role_name).resolve()


def list_log_files(settings: Settings, *, role: str) -> list[LogFileInfo]:
    """List log files for a role (newest first)."""

    directory = _role_dir(settings, role)
    if not directory.exists():
        return []

    items: list[LogFileInfo] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        stat = path.stat()
        items.append(
            LogFileInfo(
                name=path.name,
                size_bytes=int(stat.st_size),
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    items.sort(key=lambda x: x.modified_at, reverse=True)
    return items


def tail_log_file(settings: Settings, *, role: str, filename: str, lines: int = 200) -> str:
    """Return the last N lines of a log file."""

    directory = _role_dir(settings, role)
    safe_name = Path(filename).name
    path = (directory / safe_name).resolve()
    if directory not in path.parents and path != directory:
        raise ValueError("Invalid filename.")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(safe_name)

    target_lines = max(1, min(int(lines), 5000))

    # Simple tail implementation: read from end in fixed-size chunks.
    chunk_size = 64 * 1024
    buffer = b""
    with path.open("rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        offset = 0
        while file_size - offset > 0 and buffer.count(b"\n") <= target_lines:
            offset = min(file_size, offset + chunk_size)
            f.seek(file_size - offset)
            buffer = f.read(offset) + buffer
            if offset >= file_size:
                break

    text = buffer.decode("utf-8", errors="replace")
    parts = text.splitlines()
    return "\n".join(parts[-target_lines:])


