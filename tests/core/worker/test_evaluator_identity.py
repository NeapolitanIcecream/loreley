from __future__ import annotations

import sys

from loreley.core.worker.evaluator_identity import evaluator_identity_version


def test_evaluator_identity_pythonpath_lookup_is_scoped(tmp_path) -> None:
    """Regression: identity lookup must not leave evaluator plugin paths in sys.path."""

    (tmp_path / "plugin_eval.py").write_text("def evaluate(context):\n    return None\n", encoding="utf-8")
    original_path = list(sys.path)
    path_entry = str(tmp_path.resolve())

    version = evaluator_identity_version(
        plugin_ref="plugin_eval:evaluate",
        python_paths=[path_entry],
    )

    assert version is not None
    assert version.startswith("source-sha256:")
    assert path_entry not in sys.path
    assert sys.path == original_path
