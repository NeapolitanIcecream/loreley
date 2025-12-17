"""Log browsing schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class LogFileOut(BaseModel):
    name: str
    size_bytes: int
    modified_at: datetime


class LogTailOut(BaseModel):
    name: str
    lines: int
    content: str


