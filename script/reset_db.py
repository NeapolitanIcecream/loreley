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

from loreley.db.base import Base, engine

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

    # Import models so all tables are registered on Base.metadata.
    import loreley.db.models  # noqa: F401  # pylint: disable=unused-import

    dsn = getattr(engine, "url", None)
    console.print(f"[yellow]Resetting database schema[/] url={dsn}")
    log.warning("Resetting database schema (drop_all + create_all) url={}", dsn)

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    console.print("[bold green]Database schema reset complete[/]")
    log.info("Database schema reset complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


