from __future__ import annotations

import signal

from script.run_ui import _coerce_exit_code


def test_coerce_exit_code_passes_through_normal_codes() -> None:
    assert _coerce_exit_code(0, stop_requested=False) == 0
    assert _coerce_exit_code(1, stop_requested=False) == 1


def test_coerce_exit_code_graceful_stop_requested() -> None:
    assert _coerce_exit_code(0, stop_requested=True) == 0
    assert _coerce_exit_code(1, stop_requested=True) == 0


def test_coerce_exit_code_signal_termination() -> None:
    assert _coerce_exit_code(-signal.SIGINT, stop_requested=False) == 0
    assert _coerce_exit_code(-signal.SIGTERM, stop_requested=False) == 0
    assert _coerce_exit_code(-signal.SIGABRT, stop_requested=False) == 128 + signal.SIGABRT


