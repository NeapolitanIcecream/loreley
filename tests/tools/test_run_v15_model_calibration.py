from __future__ import annotations

import pytest

from tools.run_v15_model_calibration import attributed_quota, extract_python_source


def test_extract_python_source_accepts_fenced_full_file() -> None:
    response = """\
Here is the file:
```python
def pack_circles(n: int):
    return []
```
"""

    assert extract_python_source(response) == (
        "def pack_circles(n: int):\n    return []\n"
    )


def test_extract_python_source_rejects_non_code_answer() -> None:
    with pytest.raises(ValueError, match="does not define"):
        extract_python_source("I would use a grid.")


def test_attributed_quota_uses_proxy_model_completion_and_group_ratios() -> None:
    quota = attributed_quota(
        input_tokens=1000,
        output_tokens=200,
        model_ratio=1.25,
        completion_ratio=6,
        group_multiplier=1.2,
    )

    assert quota == pytest.approx(3300.0)
