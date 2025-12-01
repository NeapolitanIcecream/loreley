from __future__ import annotations

from pathlib import Path

import pytest

from loreley.config import Settings
from loreley.core.map_elites.preprocess import (
    ChangedFile,
    CodePreprocessor,
    preprocess_changed_files,
)


def make_preprocessor(tmp_path: Path, *, settings: Settings | None = None) -> CodePreprocessor:
    test_settings = settings or Settings()
    return CodePreprocessor(repo_root=tmp_path, settings=test_settings, treeish=None)


def test_changed_file_normalises_path() -> None:
    cf_str = ChangedFile(path="foo/bar.py")
    assert isinstance(cf_str.path, Path)
    assert str(cf_str.path).endswith("foo/bar.py")

    raw_path = Path("baz/qux.py")
    cf_path = ChangedFile(path=raw_path)
    assert isinstance(cf_path.path, Path)
    assert cf_path.path == raw_path


def test_coerce_changed_file_variants(tmp_path: Path, settings: Settings) -> None:
    preprocessor = make_preprocessor(tmp_path, settings=settings)

    original = ChangedFile(path="file.py", change_count=5)
    assert preprocessor._coerce_changed_file(original) is original  # type: ignore[attr-defined]

    from_str = preprocessor._coerce_changed_file("foo.py")  # type: ignore[attr-defined]
    assert isinstance(from_str, ChangedFile)
    assert from_str.path == Path("foo.py")
    assert from_str.change_count == 0

    as_tuple = preprocessor._coerce_changed_file(("bar.py", "10"))  # type: ignore[attr-defined]
    assert isinstance(as_tuple, ChangedFile)
    assert as_tuple.change_count == 10

    invalid_delta = preprocessor._coerce_changed_file(("baz.py", "not-a-number"))  # type: ignore[attr-defined]
    assert isinstance(invalid_delta, ChangedFile)
    assert invalid_delta.change_count == 0

    mapping = preprocessor._coerce_changed_file(  # type: ignore[attr-defined]
        {
            "file": "mapped.py",
            "change_count": "7",
            "content": 123,  # Non-string content should be discarded
        }
    )
    assert isinstance(mapping, ChangedFile)
    assert mapping.path == Path("mapped.py")
    assert mapping.change_count == 7
    assert mapping.content is None


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

    assert preprocessor._is_code_file(Path("src/main.py"))  # type: ignore[attr-defined]
    assert not preprocessor._is_code_file(Path("src/readme.txt"))  # type: ignore[attr-defined]
    assert preprocessor._is_code_file(Path("Special"))  # type: ignore[attr-defined]

    assert preprocessor._is_excluded(Path("tests/test_file.py"))  # type: ignore[attr-defined]
    assert preprocessor._is_excluded(Path("build/output.py"))  # type: ignore[attr-defined]
    assert not preprocessor._is_excluded(Path("src/loreley.py"))  # type: ignore[attr-defined]


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

    cleaned = preprocessor._cleanup_text(raw)  # type: ignore[attr-defined]

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


def test_run_and_wrapper_preprocess_changed_files(
    tmp_path: Path,
    settings: Settings,
) -> None:
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_max_files = 2
    settings.mapelites_preprocess_max_file_size_kb = 1

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    code_file = src_dir / "main.py"
    code_file.write_text("print('hello')\n", encoding="utf-8")

    text_file = src_dir / "notes.txt"
    text_file.write_text("not code\n", encoding="utf-8")

    big_file = src_dir / "big.py"
    big_file.write_text("x" * (settings.mapelites_preprocess_max_file_size_kb * 1024 + 10), encoding="utf-8")

    changed_files = [
        {"path": code_file, "change_count": 10},
        {"path": text_file, "change_count": 20},
        {"path": big_file, "change_count": 30},
    ]

    preprocessor = make_preprocessor(tmp_path, settings=settings)
    result = preprocessor.run(changed_files)

    assert len(result) == 1
    preprocessed = result[0]
    assert preprocessed.path == Path("src/main.py")
    assert "print('hello')" in preprocessed.content

    wrapped = preprocess_changed_files(
        changed_files,
        repo_root=tmp_path,
        settings=settings,
        treeish=None,
        repo=None,
    )
    assert len(wrapped) == 1
    assert wrapped[0].path == Path("src/main.py")


