from __future__ import annotations

from pathlib import Path

from loreley.api.services import logs as logs_service


def test_tail_log_file_reads_large_sparse_newline_files_once_per_chunk(
    monkeypatch,
    settings,
    tmp_path: Path,
) -> None:
    """Regression: tailing large logs should not reread the same bytes repeatedly."""

    logs_root = tmp_path / "logs"
    scheduler_dir = logs_root / "scheduler"
    scheduler_dir.mkdir(parents=True)
    target = scheduler_dir / "scheduler.log"

    line1 = "A" * 70000
    line2 = "B" * 70000
    line3 = "C" * 70000
    target.write_text(f"{line1}\n{line2}\n{line3}\n", encoding="utf-8")

    monkeypatch.setattr(logs_service, "resolve_logs_root", lambda _settings: logs_root)

    original_open = Path.open
    bytes_read = 0

    class _TrackingFile:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __enter__(self):
            self._wrapped.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._wrapped.__exit__(exc_type, exc, tb)

        def seek(self, *args, **kwargs):
            return self._wrapped.seek(*args, **kwargs)

        def tell(self):
            return self._wrapped.tell()

        def read(self, *args, **kwargs):
            nonlocal bytes_read
            data = self._wrapped.read(*args, **kwargs)
            bytes_read += len(data)
            return data

    def _patched_open(path: Path, *args, **kwargs):
        wrapped = original_open(path, *args, **kwargs)
        if path == target:
            return _TrackingFile(wrapped)
        return wrapped

    monkeypatch.setattr(Path, "open", _patched_open)

    text = logs_service.tail_log_file(
        settings,
        role="scheduler",
        filename=target.name,
        lines=2,
    )

    assert text == f"{line2}\n{line3}"
    assert bytes_read <= target.stat().st_size + (64 * 1024)
