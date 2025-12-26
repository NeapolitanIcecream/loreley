"""Best-effort sanitizers for LLM backend outputs.

The worker relies on agent backends to return structured payloads (most often JSON).
In practice, some models occasionally wrap the payload in Markdown code fences, e.g.:

```json
{ ... }
```

Downstream parsers (e.g. Pydantic's ``model_validate_json``) expect raw JSON only.
This module provides small, conservative helpers to strip the most common wrappers.
"""

from __future__ import annotations


def sanitize_json_payload(payload: str) -> str:
    """Return a best-effort cleaned JSON payload.

    This helper intentionally does not attempt to "repair" invalid JSON. It only removes
    common formatting wrappers that should not be present in the first place, such as
    Markdown code fences (```json ... ```).
    """

    text = (payload or "").strip()
    if not text:
        return ""

    cleaned = _strip_surrounding_code_fence(text)
    if cleaned != text:
        return cleaned

    # Fallback: extract the first fenced block if the model included extra prose, e.g.
    # "Here is the JSON:\n```json\n{...}\n```".
    cleaned = _extract_first_fenced_block(text)
    return cleaned


def _strip_surrounding_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    if not lines[0].strip().startswith("```"):
        return text
    if not lines[-1].strip().startswith("```"):
        return text
    inner = "\n".join(lines[1:-1]).strip()
    return inner or ""


def _extract_first_fenced_block(text: str) -> str:
    lines = text.splitlines()
    for start_idx, line in enumerate(lines):
        if not line.strip().startswith("```"):
            continue
        for end_idx in range(start_idx + 1, len(lines)):
            if lines[end_idx].strip() == "```":
                inner = "\n".join(lines[start_idx + 1 : end_idx]).strip()
                return inner or text
        return text
    return text


