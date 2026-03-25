from __future__ import annotations

import builtins
from pathlib import Path, PosixPath
from types import SimpleNamespace
import sys

import loreley.core.repo_lock as repo_lock_module


class _FakeWindowsAppendFile:
    _next_fd = 100

    def __init__(self, state: dict[str, object]) -> None:
        self._state = state
        self._position = 0
        self._fd = _FakeWindowsAppendFile._next_fd
        _FakeWindowsAppendFile._next_fd += 1
        handles = self._state["handles"]
        assert isinstance(handles, dict)
        handles[self._fd] = self

    def fileno(self) -> int:
        return self._fd

    def seek(self, offset: int, whence: int = 0) -> int:
        content = self._content
        if whence == 0:
            self._position = offset
        elif whence == 1:
            self._position += offset
        elif whence == 2:
            self._position = len(content) + offset
        else:  # pragma: no cover - defensive
            raise ValueError(f"unexpected whence={whence}")
        return self._position

    def tell(self) -> int:
        return self._position

    def write(self, text: str) -> int:
        # Simulate append mode: writes go to EOF regardless of current seek.
        self._content = self._content + text
        self._position = len(self._content)
        return len(text)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        handles = self._state["handles"]
        assert isinstance(handles, dict)
        handles.pop(self._fd, None)

    @property
    def _content(self) -> str:
        value = self._state["content"]
        assert isinstance(value, str)
        return value

    @_content.setter
    def _content(self, value: str) -> None:
        self._state["content"] = value


def test_file_lock_windows_fallback_locks_fixed_sentinel_byte(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Regression: Windows fallback must coordinate on one byte instead of per-open EOF."""

    state: dict[str, object] = {"content": "", "handles": {}}
    calls: list[tuple[int, int, int]] = []

    def _fake_open(path: object, mode: str, encoding: str | None = None) -> _FakeWindowsAppendFile:
        assert str(path) == str(tmp_path / ".worker.lock")
        assert mode == "a+"
        assert encoding == "utf-8"
        return _FakeWindowsAppendFile(state)

    def _fake_locking(fd: int, mode: int, size: int) -> None:
        handles = state["handles"]
        assert isinstance(handles, dict)
        handle = handles[fd]
        assert isinstance(handle, _FakeWindowsAppendFile)
        calls.append((mode, handle.tell(), size))

    fake_msvcrt = SimpleNamespace(
        LK_LOCK=1,
        LK_UNLCK=2,
        locking=_fake_locking,
    )

    monkeypatch.setattr(repo_lock_module, "Path", PosixPath)
    monkeypatch.setattr(repo_lock_module.os, "name", "nt", raising=False)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(builtins, "open", _fake_open)

    lock_path = tmp_path / ".worker.lock"
    with repo_lock_module.file_lock(lock_path):
        pass
    with repo_lock_module.file_lock(lock_path):
        pass

    assert calls == [
        (fake_msvcrt.LK_LOCK, 0, 1),
        (fake_msvcrt.LK_UNLCK, 0, 1),
        (fake_msvcrt.LK_LOCK, 0, 1),
        (fake_msvcrt.LK_UNLCK, 0, 1),
    ]
    assert state["content"] == "\0"
