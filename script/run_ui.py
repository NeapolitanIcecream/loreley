from __future__ import annotations

"""Entry script for running the Loreley Streamlit UI.

Usage (with uv):

    uv run python script/run_ui.py
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Sequence

from rich.console import Console

console = Console()

_POLL_INTERVAL_SECONDS = 0.2
_STOP_TIMEOUT_SECONDS = 5.0


def _coerce_exit_code(returncode: int, *, stop_requested: bool) -> int:
    """Normalize subprocess return codes for a friendly CLI experience.

    Notes:
    - When the user requests a stop (Ctrl+C / SIGTERM), we treat it as a graceful
      shutdown and exit with 0.
    - When a subprocess is terminated by a signal, Python reports a negative
      returncode (e.g. -2 for SIGINT). Convert it to the conventional
      128+signum form to avoid shell-specific modulo behavior.
    """

    if stop_requested:
        return 0

    if returncode == 0:
        return 0

    if returncode < 0:
        signum = -returncode
        if signum in (signal.SIGINT, signal.SIGTERM):
            # Treat common termination signals as a graceful stop.
            return 0
        return 128 + signum

    return int(returncode)


def _send_signal(proc: subprocess.Popen, signum: int) -> None:
    """Best-effort signal propagation to the Streamlit process (and its children)."""

    if proc.poll() is not None:
        return

    try:
        if os.name == "posix":
            # We start Streamlit in a new session so proc.pid is also the PGID.
            os.killpg(int(proc.pid), int(signum))
        else:  # pragma: no cover - Windows behavior
            proc.send_signal(signum)
    except ProcessLookupError:
        return
    except Exception:
        # Best-effort only. The process may have already exited.
        return


def _stop_proc(proc: subprocess.Popen, *, first_signal: int) -> None:
    """Attempt a graceful shutdown, escalating to terminate/kill if needed."""

    _send_signal(proc, int(first_signal))
    try:
        proc.wait(timeout=_STOP_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        console.log("[yellow]UI did not stop after signal; terminating...[/]")

    # Escalate to SIGTERM / terminate.
    try:
        if os.name == "posix":
            _send_signal(proc, signal.SIGTERM)
        else:  # pragma: no cover - Windows behavior
            proc.terminate()
    except Exception:
        pass

    try:
        proc.wait(timeout=_STOP_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        console.log("[bold red]UI did not terminate; killing...[/]")

    # Final escalation: SIGKILL / kill.
    try:
        if os.name == "posix" and hasattr(signal, "SIGKILL"):
            _send_signal(proc, signal.SIGKILL)
        else:  # pragma: no cover - Windows behavior
            proc.kill()
    except Exception:
        pass

    try:
        proc.wait(timeout=_STOP_TIMEOUT_SECONDS)
    except Exception:
        pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Loreley Streamlit UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("LORELEY_UI_API_BASE_URL", "http://127.0.0.1:8000"),
        help="Base URL of the Loreley UI API.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Streamlit bind host.")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit bind port.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a browser.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)

    env = dict(os.environ)
    env["LORELEY_UI_API_BASE_URL"] = str(args.api_base_url)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "loreley/ui/app.py",
        "--server.address",
        str(args.host),
        "--server.port",
        str(int(args.port)),
    ]
    if args.headless:
        cmd += ["--server.headless", "true"]

    console.log(
        "[bold green]Loreley UI online[/] "
        "host={} port={} api_base_url={}".format(args.host, args.port, args.api_base_url)
    )

    stop_requested = False
    stop_signal = signal.SIGTERM
    proc: subprocess.Popen | None = None

    def _handle_sigterm(signum: int, _frame: object) -> None:
        nonlocal stop_requested, stop_signal, proc
        stop_requested = True
        stop_signal = int(signum)
        if proc is not None:
            _send_signal(proc, stop_signal)

    # Allow graceful shutdown when this wrapper is terminated (e.g. Docker/K8s).
    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        popen_kwargs: dict[str, object] = {"env": env}
        if os.name == "posix":
            # Isolate Streamlit into its own process group so the parent can
            # forward Ctrl+C without both processes handling SIGINT.
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]
    except FileNotFoundError as exc:  # pragma: no cover
        console.log(
            "[bold red]Failed to start Streamlit[/] "
            "Install with `uv sync --extra ui` and retry. "
            f"reason={exc}"
        )
        return 1
    except KeyboardInterrupt:
        # Defensive: KeyboardInterrupt could surface while creating the subprocess.
        console.log("[yellow]Keyboard interrupt received[/]; exiting...")
        return 0

    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                return _coerce_exit_code(int(rc), stop_requested=stop_requested)

            if stop_requested:
                name = {signal.SIGTERM: "SIGTERM"}.get(stop_signal, str(stop_signal))
                console.log(f"[yellow]Stop signal received ({name})[/]; stopping UI...")
                _stop_proc(proc, first_signal=stop_signal)
                return 0

            time.sleep(_POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        console.log("[yellow]Keyboard interrupt received[/]; stopping UI...")
        _stop_proc(proc, first_signal=signal.SIGINT)
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


