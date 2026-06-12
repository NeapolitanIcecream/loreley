from __future__ import annotations

import ast
from pathlib import Path


def test_streamlit_navigation_hides_repair_pool_entry() -> None:
    source = Path("loreley/ui/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    page_titles: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "Page"
            and isinstance(func.value, ast.Name)
            and func.value.id == "st"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "title" and isinstance(keyword.value, ast.Constant):
                page_titles.append(str(keyword.value.value))

    assert "Repair Pool" not in page_titles
