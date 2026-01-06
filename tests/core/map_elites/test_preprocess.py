from __future__ import annotations

from pathlib import Path

import pytest

from loreley.config import Settings
from loreley.core.map_elites.preprocess import (
    CodePreprocessor,
)


def make_preprocessor(tmp_path: Path, *, settings: Settings | None = None) -> CodePreprocessor:
    test_settings = settings or Settings(mapelites_code_embedding_dimensions=8)
    return CodePreprocessor(repo_root=tmp_path, settings=test_settings, commit_hash=None)


def test_prepare_excluded_globs_normalises_patterns(tmp_path: Path, settings: Settings) -> None:
    preprocessor = make_preprocessor(tmp_path, settings=settings)
    patterns = ["./foo/*.py", "foo/bar", "", "   ", r"pkg\module"]

    globs = preprocessor._prepare_excluded_globs(patterns)  # type: ignore[attr-defined]

    assert set(globs) == {
        "foo/*.py",
        "**/foo/*.py",
        "foo/bar",
        "**/foo/bar",
        "pkg/module",
        "**/pkg/module",
    }


def test_is_code_file_and_is_excluded(tmp_path: Path, settings: Settings) -> None:
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = ["Special"]
    settings.mapelites_preprocess_excluded_globs = ["tests/**", "build/*"]

    preprocessor = make_preprocessor(tmp_path, settings=settings)

    assert preprocessor.is_code_file(Path("src/main.py"))
    assert not preprocessor.is_code_file(Path("src/readme.txt"))
    assert preprocessor.is_code_file(Path("Special"))

    assert preprocessor.is_excluded(Path("tests/test_file.py"))
    assert preprocessor.is_excluded(Path("build/output.py"))
    assert not preprocessor.is_excluded(Path("src/loreley.py"))


def test_cleanup_text_strips_comments_tabs_and_excess_blank_lines(
    tmp_path: Path,
    settings: Settings,
) -> None:
    settings.mapelites_preprocess_max_blank_lines = 1
    settings.mapelites_preprocess_tab_width = 4
    settings.mapelites_preprocess_strip_comments = True
    settings.mapelites_preprocess_strip_block_comments = True

    preprocessor = make_preprocessor(tmp_path, settings=settings)

    raw = (
        "line1\n"
        "\n"
        "/* block comment\n"
        "   continues */\n"
        "\n"
        "# single line comment\n"
        "\tindented = 1\n"
        "\n"
        "\n"
        "last = 2\n"
    )

    cleaned = preprocessor.cleanup_text(raw)

    assert "/*" not in cleaned and "*/" not in cleaned
    assert "# single line comment" not in cleaned
    assert "\t" not in cleaned

    blank_streak = 0
    for line in cleaned.splitlines():
        if not line.strip():
            blank_streak += 1
            assert blank_streak <= settings.mapelites_preprocess_max_blank_lines
        else:
            blank_streak = 0


