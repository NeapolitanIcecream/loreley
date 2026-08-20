"""Prepare the fixed Zstandard root used by the method-efficacy experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


UPSTREAM_COMMIT = "82d322c4973d9e2968d94047a40892bc6d9a9bdf"

PROGRAM_TEXT = """# Zstandard single-thread performance campaign

## Goal
Improve single-thread Zstandard compression and decompression throughput at
levels 1, 3, and 5 while preserving format compatibility, public behavior,
compressed size, memory use, and all upstream tests.

## Primary metric
name: throughput_geomean
direction: higher_is_better
unit: root_ratio

## Correctness gates
- `make check` passes before a candidate is scored.
- Candidate-compressed data decodes with both candidate and root binaries.
- Root-compressed data decodes with the candidate binary.
- Round trips reproduce every input byte.
- Compressed size does not regress by more than 0.1% in any scored cell.
- Peak RSS does not exceed root by more than 16 MiB.
- No corpus, filename, digest, benchmark, compiler, host, or split special case.

## Editable scope
- `lib/*.c`
- `lib/*.h`
- `lib/**/*.c`
- `lib/**/*.h`

## Protected scope
- `programs/**`
- `tests/**`
- `build/**`
- `contrib/**`
- `examples/**`
- `doc/**`
- `educational_decoder/**`
- `zlibWrapper/**`
- `.github/**`
- `Makefile`
- `lib/Makefile`
- `lib/libzstd.mk`
- `loreley.program.md`
- `.loreleyignore`

## Evaluation budget
- Training evaluation uses a frozen mixed-content corpus at levels 1, 3, and 5.
- Root and candidate use the same frozen Linux ARM64 release toolchain.
- Performance measurements are single-threaded and pinned to an assigned CPU.
- Evaluation runs without network access.

## Complexity policy
- Prefer small causal hot-path changes over broad rewrites.
- Preserve portable C and existing architecture dispatch.
- Do not add dependencies, generated code, global mutable state, or persistent caches.

## Failure policy
- Scope, build, test, round-trip, cross-decode, size, memory, timeout, parser,
  measurement-precision, or non-finite metric failures reject the candidate.

## Logging policy
- Report compression and decompression speedups, compressed-size ratio, peak
  RSS, measurement precision, gate status, and bounded diagnostics.
"""

IGNORE_TEXT = """# Experiment-only repository-state exclusions
.git/
programs/
tests/
build/
contrib/
examples/
doc/
educational_decoder/
zlibWrapper/
.github/
*.o
*.a
*.dylib
*.so
"""

FIXED_GIT_ENV = {
    "GIT_AUTHOR_NAME": "Loreley Experiment",
    "GIT_AUTHOR_EMAIL": "experiment@example.invalid",
    "GIT_COMMITTER_NAME": "Loreley Experiment",
    "GIT_COMMITTER_EMAIL": "experiment@example.invalid",
    "GIT_AUTHOR_DATE": "2026-08-11T00:00:00Z",
    "GIT_COMMITTER_DATE": "2026-08-11T00:00:00Z",
}


class ZstdTargetError(RuntimeError):
    """Raised when the frozen Zstandard target cannot be prepared safely."""


def atomic_write_json(path: Path, payload: Any) -> str:
    """Write canonical JSON without depending on the private run harness."""

    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(
    repository: Path,
    *arguments: str,
    environment: Mapping[str, str] | None = None,
) -> str:
    env = dict(os.environ)
    if environment:
        env.update({str(key): str(value) for key, value in environment.items()})
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
        env=env,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "")[-4_000:]
        raise ZstdTargetError(
            f"git {' '.join(arguments)} failed in {repository}: {detail}"
        )
    return completed.stdout.strip()


def verify_upstream(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    if not (repository / ".git").is_dir():
        raise ZstdTargetError(f"Not a Git checkout: {repository}")
    observed = _git(repository, "rev-parse", "HEAD")
    if observed != UPSTREAM_COMMIT:
        raise ZstdTargetError(
            f"Expected upstream {UPSTREAM_COMMIT}, observed {observed}"
        )
    if _git(repository, "status", "--porcelain"):
        raise ZstdTargetError(f"Upstream checkout is dirty: {repository}")
    return {
        "upstream_commit": observed,
        "upstream_tree": _git(repository, "rev-parse", "HEAD^{tree}"),
    }


def _verify_root(destination: Path) -> dict[str, Any]:
    destination = destination.resolve()
    if not (destination / ".git").is_dir():
        raise ZstdTargetError(f"Root template is not a Git checkout: {destination}")
    if _git(destination, "status", "--porcelain"):
        raise ZstdTargetError(f"Root template is dirty: {destination}")
    parent = _git(destination, "rev-parse", "HEAD^")
    if parent != UPSTREAM_COMMIT:
        raise ZstdTargetError(f"Root parent drifted: {parent}")
    changed = _git(destination, "diff", "--name-only", "HEAD^", "HEAD").splitlines()
    if changed != [".loreleyignore", "loreley.program.md"]:
        raise ZstdTargetError(f"Unexpected experiment-root changes: {changed}")
    if (destination / "loreley.program.md").read_text() != PROGRAM_TEXT:
        raise ZstdTargetError("Frozen Zstandard program text drifted")
    if (destination / ".loreleyignore").read_text() != IGNORE_TEXT:
        raise ZstdTargetError("Frozen Zstandard ignore text drifted")
    return {
        "upstream_commit": parent,
        "root_commit": _git(destination, "rev-parse", "HEAD"),
        "root_tree": _git(destination, "rev-parse", "HEAD^{tree}"),
        "program_sha256": _sha256(destination / "loreley.program.md"),
        "ignore_sha256": _sha256(destination / ".loreleyignore"),
        "changed_paths": changed,
    }


def prepare_root_template(*, upstream: Path, destination: Path) -> dict[str, Any]:
    """Create or verify the one-commit experiment root."""

    upstream = upstream.resolve(strict=True)
    destination = destination.resolve()
    verify_upstream(upstream)
    if destination.exists():
        return _verify_root(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["git", "clone", "--no-hardlinks", str(upstream), str(destination)],
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if completed.returncode:
        raise ZstdTargetError(
            "Cannot clone Zstandard root template: "
            + (completed.stderr or completed.stdout or "")[-4_000:]
        )
    try:
        _git(destination, "checkout", "--detach", UPSTREAM_COMMIT)
        _git(destination, "switch", "-c", "method-efficacy-root")
        (destination / "loreley.program.md").write_text(
            PROGRAM_TEXT, encoding="utf-8"
        )
        (destination / ".loreleyignore").write_text(
            IGNORE_TEXT, encoding="utf-8"
        )
        _git(destination, "add", "loreley.program.md", ".loreleyignore")
        _git(
            destination,
            "commit",
            "-m",
            "chore: add frozen Loreley method-efficacy contract",
            environment=FIXED_GIT_ENV,
        )
        return _verify_root(destination)
    except Exception:
        # A partially constructed target must never be accepted on resume.
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_root_template(
        upstream=args.upstream,
        destination=args.destination,
    )
    atomic_write_json(args.manifest, {"schema_version": 1, **result})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "IGNORE_TEXT",
    "PROGRAM_TEXT",
    "UPSTREAM_COMMIT",
    "ZstdTargetError",
    "prepare_root_template",
    "verify_upstream",
]


if __name__ == "__main__":
    raise SystemExit(main())
