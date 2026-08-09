from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def installed_version() -> str:
    """Return the installed Loreley distribution version."""

    try:
        return version("loreley")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = installed_version()

__all__ = ["__version__", "installed_version"]
