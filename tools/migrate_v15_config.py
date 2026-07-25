"""Convert pre-v15 scalar MAP-Elites environment settings once.

Usage:
    uv run python tools/migrate_v15_config.py .env --in-place
    uv run python tools/migrate_v15_config.py old.env --output migrated.env
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

_ASSIGNMENT = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)"
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*=\s*(?P<value>.*?)(?P<newline>\r?\n)?$"
)

_REMOVED_KEYS = {
    "MAPELITES_DEFAULT_ISLAND_ID",
    "MAPELITES_FITNESS_METRIC",
    "MAPELITES_FITNESS_HIGHER_IS_BETTER",
    "MAPELITES_FITNESS_FLOOR",
    "MAPELITES_ARCHIVE_EPSILON",
    "MAPELITES_ARCHIVE_LEARNING_RATE",
    "MAPELITES_ARCHIVE_THRESHOLD_MIN",
    "MAPELITES_ARCHIVE_QD_SCORE_OFFSET",
}


def migrate_env_text(source: str) -> str:
    """Return an idempotent v15 environment-file conversion."""

    lines = source.splitlines(keepends=True)
    assignments = _parse_assignments(lines)
    replacements = _build_replacements(assignments)
    return "".join(_migrate_line(line, replacements) for line in lines)


def _parse_assignments(lines: Sequence[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line in lines:
        match = _ASSIGNMENT.match(line)
        if match:
            assignments[match.group("key")] = _unquote(match.group("value").strip())
    return assignments


def _build_replacements(assignments: dict[str, str]) -> dict[str, str]:
    replacements: dict[str, str] = {}
    islands = _legacy_islands_replacement(assignments)
    if islands is not None:
        replacements["MAPELITES_DEFAULT_ISLAND_ID"] = islands
    objectives = _legacy_objectives_replacement(assignments)
    if objectives is not None:
        anchor = (
            "MAPELITES_FITNESS_METRIC"
            if "MAPELITES_FITNESS_METRIC" in assignments
            else "MAPELITES_FITNESS_HIGHER_IS_BETTER"
        )
        replacements[anchor] = objectives
    epsilon = _legacy_epsilon_replacement(assignments)
    if epsilon is not None:
        replacements["MAPELITES_ARCHIVE_EPSILON"] = epsilon
    return replacements


def _legacy_islands_replacement(assignments: dict[str, str]) -> str | None:
    if "MAPELITES_ISLANDS" in assignments:
        return None
    island_id = assignments.get("MAPELITES_DEFAULT_ISLAND_ID")
    if island_id is None:
        return None
    payload = json.dumps([island_id], ensure_ascii=True, separators=(",", ":"))
    return f"MAPELITES_ISLANDS={payload}"


def _legacy_objectives_replacement(assignments: dict[str, str]) -> str | None:
    if "MAPELITES_OBJECTIVES" in assignments:
        return None
    metric_name = assignments.get("MAPELITES_FITNESS_METRIC")
    direction = assignments.get("MAPELITES_FITNESS_HIGHER_IS_BETTER")
    if metric_name is None and direction is None:
        return None
    metric_name = (metric_name or "").strip() or "composite_score"
    higher = _parse_legacy_bool(
        direction or "true"
    )
    payload = json.dumps(
        [
            {
                "name": metric_name,
                "direction": "max" if higher else "min",
            }
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"MAPELITES_OBJECTIVES={payload}"


def _legacy_epsilon_replacement(assignments: dict[str, str]) -> str | None:
    if "MAPELITES_PARETO_EPSILON" in assignments:
        return None
    epsilon = assignments.get("MAPELITES_ARCHIVE_EPSILON")
    if epsilon is None:
        return None
    return f"MAPELITES_PARETO_EPSILON={epsilon}"


def _migrate_line(line: str, replacements: dict[str, str]) -> str:
    match = _ASSIGNMENT.match(line)
    if not match:
        return line
    key = match.group("key")
    replacement = replacements.get(key)
    if replacement is not None:
        newline = match.group("newline") or ""
        return f"{match.group('prefix')}{replacement}{newline}"
    if key in _REMOVED_KEYS:
        return ""
    return line


def _unquote(value: str) -> str:
    value = _strip_inline_comment(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if (
            quote is None
            and character == "#"
            and (index == 0 or value[index - 1].isspace())
        ):
            return value[:index].rstrip()
        quote, escaped = _advance_quote_state(
            character,
            quote=quote,
            escaped=escaped,
        )
    return value


def _advance_quote_state(
    character: str,
    *,
    quote: str | None,
    escaped: bool,
) -> tuple[str | None, bool]:
    if escaped:
        return quote, False
    if character == "\\" and quote == '"':
        return quote, True
    if quote is not None:
        return (None if character == quote else quote), False
    return (character if character in {"'", '"'} else None), False


def _parse_legacy_bool(value: str) -> bool:
    normalized = _unquote(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(
        "MAPELITES_FITNESS_HIGHER_IS_BETTER must be true, false, 1, or 0."
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing environment file.")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--in-place", action="store_true")
    destination.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_path = args.source.expanduser().resolve()
    target_path = (
        source_path
        if args.in_place
        else args.output.expanduser().resolve()
    )
    migrated = migrate_env_text(source_path.read_text(encoding="utf-8"))
    target_path.write_text(migrated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
