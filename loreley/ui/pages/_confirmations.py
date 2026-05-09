"""Streamlit operator write confirmation helpers."""

from __future__ import annotations


def operator_confirmation_key(base_key: str) -> str:
    """Return the current checkbox key for an operator write confirmation."""

    import streamlit as st

    nonce = int(st.session_state.get(_nonce_key(base_key), 0) or 0)
    return f"{base_key}_{nonce}"


def expire_operator_confirmation(base_key: str) -> None:
    """Rotate a confirmation key after a successful operator write."""

    import streamlit as st

    state_key = _nonce_key(base_key)
    st.session_state[state_key] = int(st.session_state.get(state_key, 0) or 0) + 1


def _nonce_key(base_key: str) -> str:
    return f"{base_key}_nonce"
