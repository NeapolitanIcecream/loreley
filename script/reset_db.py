"""Dangerous helper to reset the Loreley database schema.

This project intentionally does not ship migrations. For prototype workflows,
the fastest path is to drop all tables and recreate the schema from ORM models.

Usage:
    uv run python script/reset_db.py --yes
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger
from rich.console import Console

from loreley.db.base import reset_database_schema

console = Console()
log = logger.bind(module="script.reset_db")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drop and recreate all Loreley DB tables.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that you want to irreversibly drop all tables.",
    )
    args = parser.parse_args(argv)

    if not args.yes:
        console.print("[bold red]Refusing to reset DB without --yes[/]")
        console.print("This will drop ALL tables and recreate them from ORM models.")
        return 2

    reset_database_schema(include_console_log=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


