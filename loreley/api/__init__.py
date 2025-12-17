"""Read-only API used by the Streamlit UI.

This package is intentionally optional: it is only imported when the UI API is
started (e.g. via ``script/run_api.py``) and should not impact core execution
paths like scheduler/worker.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"


