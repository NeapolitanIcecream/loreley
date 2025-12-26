from __future__ import annotations

from loreley.core.worker.output_sanitizer import sanitize_json_payload


def test_sanitize_json_payload_strips_surrounding_fence() -> None:
    payload = """```json
{"a": 1, "b": "x"}
```"""
    assert sanitize_json_payload(payload) == '{"a": 1, "b": "x"}'


def test_sanitize_json_payload_strips_surrounding_fence_with_whitespace() -> None:
    payload = """

    ```json
    {"a": 1}
    ```

    """
    assert sanitize_json_payload(payload) == '{"a": 1}'


def test_sanitize_json_payload_extracts_first_fenced_block() -> None:
    payload = """Sure, here is the JSON:

```json
{"a": 1}
```

Thanks!"""
    assert sanitize_json_payload(payload) == '{"a": 1}'


def test_sanitize_json_payload_returns_original_when_no_fence() -> None:
    payload = '{"a": 1}'
    assert sanitize_json_payload(payload) == payload


