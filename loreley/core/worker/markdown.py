"""Markdown parsing helpers for the evolution worker."""

from __future__ import annotations

import re

_SUMMARY_HEADING = re.compile(r"^##\s+summary\s*$", re.I)
_ANY_H2_HEADING = re.compile(r"^##\s+.+$")
_BULLET = re.compile(r"^\s*(?:[-*]|•)\s+(?P<item>.+\S)\s*$")


def extract_markdown_summary(markdown: str) -> str:
    """Extract a short summary line from a Markdown document (best-effort).

    Strategy:
    - Prefer the "## Summary" section (up to 3 non-empty lines, bullet-stripped when possible).
    - Fall back to the first non-heading, non-empty line of the document.
    """

    text = (markdown or "").strip()
    if not text:
        return ""

    in_summary = False
    collected: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SUMMARY_HEADING.match(stripped):
            in_summary = True
            continue
        if in_summary and _ANY_H2_HEADING.match(stripped):
            break
        if in_summary:
            match = _BULLET.match(stripped)
            collected.append(match.group("item").strip() if match else stripped)
            if len(collected) >= 3:
                break

    if collected:
        return " / ".join(collected[:3]).strip()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        match = _BULLET.match(stripped)
        return match.group("item").strip() if match else stripped

    return ""


__all__ = ["extract_markdown_summary"]

