from __future__ import annotations

import sys
"""Compatibility wrapper for the unified `loreley doctor` command.

This repository historically exposed `script/doctor.py`. The preferred entrypoint
is now the package CLI:

    uv run loreley doctor --role all

This wrapper remains so existing docs and scripts keep working.
"""

from loreley.cli import main as loreley_main


def main(argv: list[str] | None = None) -> int:
    """Run `loreley doctor` with forwarded argv."""
    forwarded = list(argv) if argv is not None else sys.argv[1:]
    return int(loreley_main(["doctor", *forwarded]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


