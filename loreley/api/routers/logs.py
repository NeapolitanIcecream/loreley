"""Log browsing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from loreley.api.schemas.logs import LogFileOut, LogTailOut
from loreley.api.services.logs import list_log_files, tail_log_file
from loreley.config import get_settings

router = APIRouter()


@router.get("/logs", response_model=list[LogFileOut])
def get_logs(
    role: str = Query(description="One of: scheduler, worker, ui_api"),
) -> list[LogFileOut]:
    settings = get_settings()
    try:
        files = list_log_files(settings, role=role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [
        LogFileOut(
            name=f.name,
            size_bytes=f.size_bytes,
            modified_at=f.modified_at,
        )
        for f in files
    ]


@router.get("/logs/tail", response_model=LogTailOut)
def tail_log(
    role: str = Query(description="One of: scheduler, worker, ui_api"),
    filename: str = Query(description="Log filename from /logs listing."),
    lines: int = Query(default=200, ge=1, le=5000),
) -> LogTailOut:
    settings = get_settings()
    try:
        content = tail_log_file(settings, role=role, filename=filename, lines=lines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found.")
    return LogTailOut(name=filename, lines=int(lines), content=content)


