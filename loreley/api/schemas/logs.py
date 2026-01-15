"""Log browsing schemas."""

from __future__ import annotations

from datetime import datetime

from loreley.api.schemas import OrmOutModel


class LogFileOut(OrmOutModel):
    name: str
    size_bytes: int
    modified_at: datetime


class LogTailOut(OrmOutModel):
    name: str
    lines: int
    content: str


