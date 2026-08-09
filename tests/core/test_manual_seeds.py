from __future__ import annotations

from pathlib import Path

import pytest

from loreley.core.manual_seeds import ManualSeedError, load_manual_seed_manifest


def _write_manifest(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_manual_seed_manifest_is_strict_and_preserves_order(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "seeds.yaml",
        """
schema_version: 1
seeds:
  - key: allocation
    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    remote_ref: refs/heads/loreley-seeds/allocation
    summary: Reduce allocations on the hot path.
    island_id: main
    tags: [allocation, hot-path]
  - key: branch-layout
    commit: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    remote_ref: refs/heads/loreley-seeds/branch-layout
    summary: Reorder a common branch.
""".lstrip(),
    )

    manifest = load_manual_seed_manifest(path)

    assert [seed.key for seed in manifest.seeds] == ["allocation", "branch-layout"]
    assert manifest.seeds[0].remote_ref == "refs/heads/loreley-seeds/allocation"
    assert len(manifest.sha256) == 64
    assert manifest.source_name == "seeds.yaml"


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("", "remote_ref.*required"),
        ("    remote_ref: local-only\n", "full, fetchable Git ref"),
        ("    remote_ref: refs/heads/bad..ref\n", "full, fetchable Git ref"),
    ],
)
def test_manual_seed_manifest_requires_safe_remote_ref(
    tmp_path: Path,
    fragment: str,
    message: str,
) -> None:
    path = _write_manifest(
        tmp_path / "bad.yaml",
        (
            "schema_version: 1\n"
            "seeds:\n"
            "  - key: seed\n"
            "    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
            f"{fragment}"
            "    summary: One seed.\n"
        ),
    )

    with pytest.raises(ManualSeedError, match=message):
        load_manual_seed_manifest(path)


def test_manual_seed_manifest_rejects_duplicate_keys_and_oversized_goal(
    tmp_path: Path,
) -> None:
    duplicated = _write_manifest(
        tmp_path / "duplicate.yaml",
        """
schema_version: 1
seeds:
  - key: same
    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    remote_ref: refs/heads/loreley-seeds/a
    summary: First.
  - key: same
    commit: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    remote_ref: refs/heads/loreley-seeds/b
    summary: Second.
""".lstrip(),
    )
    with pytest.raises(ManualSeedError, match="keys must be unique"):
        load_manual_seed_manifest(duplicated)

    oversized = _write_manifest(
        tmp_path / "oversized.yaml",
        (
            "schema_version: 1\n"
            "seeds:\n"
            "  - key: seed\n"
            "    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
            "    remote_ref: refs/heads/loreley-seeds/a\n"
            "    summary: First.\n"
            f"    goal: {'x' * 513}\n"
        ),
    )
    with pytest.raises(ManualSeedError, match="exceeds 512"):
        load_manual_seed_manifest(oversized)
