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
from typing import Sequence

from rich.console import Console

console = Console()


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

    try:
        proc = subprocess.Popen(cmd, env=env)
    except FileNotFoundError as exc:  # pragma: no cover
        console.log(
            "[bold red]Failed to start Streamlit[/] "
            "Install with `uv sync --extra ui` and retry. "
            f"reason={exc}"
        )
        return 1
    except KeyboardInterrupt:
        # Defensive: KeyboardInterrupt could surface while creating the subprocess.
        console.log("[yellow]Keyboard interrupt received[/]; stopping UI...")
        return 130

    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        # Graceful shutdown: forward SIGINT to the Streamlit process and wait.
        # This avoids printing an uncaught KeyboardInterrupt traceback in callers.
        console.log("[yellow]Keyboard interrupt received[/]; stopping UI...")
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            # Best-effort only. The process may have already exited.
            pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            console.log("[yellow]UI did not stop after SIGINT; terminating...[/]")
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                console.log("[bold red]UI did not terminate; killing...[/]")
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


