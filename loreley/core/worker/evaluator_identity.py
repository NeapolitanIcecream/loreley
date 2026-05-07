from __future__ import annotations

"""Evaluator identity helpers used before an evaluation run exists."""

import hashlib
from importlib import util
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import sys
from typing import Any, Sequence

from loreley.core.contracts import clamp_text, normalize_single_line


def evaluator_identity_version(
    *,
    plugin_ref: str | None,
    explicit_version: str | None = None,
    python_paths: Sequence[str] = (),
) -> str | None:
    """Resolve the best pre-run evaluator version/fingerprint for baseline keys."""

    explicit = _bounded_identity(explicit_version)
    if explicit:
        return explicit

    normalized_ref = normalize_single_line(str(plugin_ref or ""))
    if not normalized_ref:
        return None
    module_name, _ = _split_reference(normalized_ref)
    _prepare_pythonpath(python_paths)
    distribution_version = _distribution_version(module_name)
    if distribution_version:
        return distribution_version
    return _source_fingerprint(module_name)


def _bounded_identity(value: Any) -> str | None:
    if value is None:
        return None
    return clamp_text(normalize_single_line(str(value)), 128) or None


def _split_reference(ref: str) -> tuple[str, str]:
    if ":" in ref:
        module_name, attr_name = ref.split(":", 1)
        return module_name, attr_name
    module_name, _, attr_name = ref.rpartition(".")
    if not module_name or not attr_name:
        return ref, ""
    return module_name, attr_name


def _prepare_pythonpath(python_paths: Sequence[str]) -> None:
    for entry in python_paths:
        entry_str = str(Path(entry).expanduser().resolve())
        if entry_str not in sys.path:
            sys.path.insert(0, entry_str)


def _distribution_version(module_name: str) -> str | None:
    parts = module_name.split(".")
    for index in range(len(parts), 0, -1):
        package_name = ".".join(parts[:index])
        try:
            version = package_version(package_name)
        except PackageNotFoundError:
            continue
        except Exception:
            return None
        bounded = _bounded_identity(version)
        if bounded:
            return bounded
    return None


def _source_fingerprint(module_name: str) -> str | None:
    try:
        spec = util.find_spec(module_name)
    except Exception:
        return None
    origin = getattr(spec, "origin", None)
    if not origin or origin in {"built-in", "frozen"}:
        return None
    path = Path(origin)
    if not path.is_file():
        return None
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
    return f"source-sha256:{digest}"
