from __future__ import annotations

from importlib.metadata import version

from loreley.api import __version__ as api_version
from loreley.ui import __version__ as ui_version


def test_optional_packages_expose_distribution_version() -> None:
    expected = version("loreley")

    assert api_version == expected
    assert ui_version == expected
