from __future__ import annotations

"""Path-level editable/protected scope gate for campaign programs."""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import fnmatch
import os
import re
import subprocess
from typing import Any, Sequence

from loguru import logger

from loreley.core.campaign_program import (
    CampaignProgramSnapshot,
    unsafe_scope_pattern_reason,
)
from loreley.core.contracts import clamp_text, normalize_single_line

log = logger.bind(module="worker.scope_gate")

_IGNORED_PREFIXES: tuple[str, ...] = (
    ".git/",
    "logs/",
    ".loreley/",
    ".loreley-artifacts/",
)
_IGNORED_EXACT: set[str] = {".git", "logs", ".loreley", ".loreley-artifacts"}


@dataclass(frozen=True, slots=True)
class ScopeViolation:
    """One path or pattern violation from the campaign scope gate."""

    code: str
    path: str | None = None
    pattern: str | None = None
    reason: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.pattern is not None:
            payload["pattern"] = self.pattern
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


@dataclass(frozen=True, slots=True)
class ScopeGateResult:
    """Machine-readable result from validating a modified worktree."""

    checked_paths: tuple[str, ...]
    violations: tuple[ScopeViolation, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checked_paths": list(self.checked_paths),
            "violations": [violation.as_dict() for violation in self.violations],
        }

    def summary(self) -> str:
        if self.passed:
            return "Campaign scope gate passed."
        return f"Campaign scope gate rejected {len(self.violations)} violation(s)."


def validate_campaign_scope(
    *,
    worktree: Path,
    program: CampaignProgramSnapshot | None,
    git_bin: str = "git",
) -> ScopeGateResult:
    """Validate changed tracked and untracked paths against campaign scope rules."""

    if program is None:
        return ScopeGateResult(checked_paths=())

    editable_patterns = tuple(program.editable_scope or ())
    protected_patterns = tuple(program.protected_scope or ())
    violations: list[ScopeViolation] = []
    violations.extend(_pattern_violations(editable_patterns, code="invalid_editable_scope_pattern"))
    violations.extend(_pattern_violations(protected_patterns, code="invalid_protected_scope_pattern"))

    paths = _changed_paths(worktree=worktree, git_bin=git_bin)
    checked_paths: list[str] = []
    for raw_path in paths:
        path = _normalize_repo_path(raw_path)
        if path is None or _ignored_path(raw_path):
            continue
        checked_paths.append(path)
        unsafe_path_reason = _unsafe_repo_path_reason(path)
        if unsafe_path_reason is not None:
            violations.append(
                ScopeViolation(
                    code="unsafe_changed_path",
                    path=path,
                    reason=unsafe_path_reason,
                    detail="Changed path is not a safe repo-relative POSIX path.",
                )
            )
            continue
        symlink_reason = _unsafe_symlink_reason(worktree=worktree, repo_path=path)
        if symlink_reason is not None:
            violations.append(
                ScopeViolation(
                    code="unsafe_symlink_target",
                    path=path,
                    reason=symlink_reason,
                    detail="Changed symlink target escapes the repository or uses an unsafe target.",
                )
            )
            continue
        protected_match = _first_matching_pattern(path, protected_patterns)
        if protected_match is not None:
            violations.append(
                ScopeViolation(
                    code="protected_scope_modified",
                    path=path,
                    pattern=protected_match,
                    detail="Protected scope wins over editable scope.",
                )
            )
            continue
        if editable_patterns and _first_matching_pattern(path, editable_patterns) is None:
            violations.append(
                ScopeViolation(
                    code="outside_editable_scope",
                    path=path,
                    detail="Non-empty editable scope allows only matching paths.",
                )
            )

    result = ScopeGateResult(
        checked_paths=tuple(dict.fromkeys(checked_paths)),
        violations=tuple(violations),
    )
    if result.violations:
        log.warning(
            "Campaign scope gate failed checked_paths={} violations={}",
            len(result.checked_paths),
            len(result.violations),
        )
    else:
        log.info("Campaign scope gate passed checked_paths={}", len(result.checked_paths))
    return result


def _changed_paths(*, worktree: Path, git_bin: str) -> tuple[str, ...]:
    target = Path(worktree).expanduser().resolve()
    try:
        result = subprocess.run(
            [
                git_bin or "git",
                "-C",
                str(target),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to inspect worktree changes for scope gate: {stderr or exc}") from exc
    return _parse_porcelain_z(result.stdout.decode("utf-8", errors="surrogateescape"))


def _parse_porcelain_z(output: str) -> tuple[str, ...]:
    tokens = [token for token in output.split("\0") if token]
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        entry = tokens[index]
        if len(entry) < 4:
            index += 1
            continue
        status = entry[:2]
        path = entry[3:]
        if path:
            paths.append(path)
        index += 1
        if ("R" in status or "C" in status) and index < len(tokens):
            old_path = tokens[index]
            if old_path:
                paths.append(old_path)
            index += 1
    return tuple(dict.fromkeys(paths))


def _pattern_violations(patterns: Sequence[str], *, code: str) -> list[ScopeViolation]:
    violations: list[ScopeViolation] = []
    for raw in patterns:
        pattern = normalize_single_line(raw)
        reason = unsafe_scope_pattern_reason(pattern)
        if reason is None:
            continue
        violations.append(
            ScopeViolation(
                code=code,
                pattern=clamp_text(pattern, 200),
                reason=reason,
                detail="Campaign scope pattern must be repo-relative POSIX and must not traverse.",
            )
        )
    return violations


def _normalize_repo_path(raw_path: str) -> str | None:
    path = raw_path.strip()
    if not path:
        return None
    path = path.replace("\\", "/")
    return path.strip("/")


def _unsafe_repo_path_reason(path: str) -> str | None:
    if not path:
        return "empty"
    if "\x00" in path:
        return "nul_byte"
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        return "absolute_path"
    parts = PurePosixPath(path).parts
    if any(part == ".." for part in parts):
        return "path_traversal"
    if any(part == ".git" for part in parts):
        return "git_internal"
    return None


def _ignored_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if normalized is None:
        return True
    if normalized in _IGNORED_EXACT:
        return True
    return any(normalized.startswith(prefix) for prefix in _IGNORED_PREFIXES)


def _unsafe_symlink_reason(*, worktree: Path, repo_path: str) -> str | None:
    candidate = Path(worktree) / repo_path
    if not candidate.is_symlink():
        return None
    try:
        raw_target = os.readlink(candidate)
    except OSError:
        return "unreadable_symlink"
    if not raw_target:
        return "empty_symlink"
    target = Path(raw_target)
    if target.is_absolute():
        return "absolute_symlink"
    try:
        resolved = candidate.resolve(strict=False)
        root = Path(worktree).expanduser().resolve()
        resolved.relative_to(root)
    except ValueError:
        return "symlink_escape"
    except OSError:
        return "unresolved_symlink"
    return None


def _first_matching_pattern(path: str, patterns: Sequence[str]) -> str | None:
    for raw in patterns:
        pattern = normalize_single_line(raw)
        if unsafe_scope_pattern_reason(pattern) is not None:
            continue
        if _pattern_matches(path, pattern):
            return pattern
    return None


def _pattern_matches(path: str, pattern: str) -> bool:
    normalized_path = path.strip("/")
    normalized_pattern = pattern.strip("/")
    if not normalized_pattern:
        return False
    if normalized_path == normalized_pattern:
        return True
    if not _has_glob(normalized_pattern) and normalized_path.startswith(f"{normalized_pattern}/"):
        return True
    if pattern.endswith("/") and normalized_path.startswith(f"{normalized_pattern}/"):
        return True
    if fnmatch.fnmatchcase(normalized_path, normalized_pattern):
        return True
    if "/" not in normalized_pattern and fnmatch.fnmatchcase(PurePosixPath(normalized_path).name, normalized_pattern):
        return True
    return False


def _has_glob(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")
