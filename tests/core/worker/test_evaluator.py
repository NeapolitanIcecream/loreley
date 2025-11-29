from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from loreley.config import Settings
from loreley.core.worker.evaluator import (
    EvaluationContext,
    EvaluationError,
    EvaluationMetric,
    EvaluationResult,
    Evaluator,
)


def test_evaluation_metric_as_dict_serialises_fields() -> None:
    metric = EvaluationMetric(
        name="accuracy",
        value=0.9,
        unit="%",
        higher_is_better=False,
        details={"threshold": 0.5},
    )
    data = metric.as_dict()
    assert data["name"] == "accuracy"
    assert isinstance(data["value"], float)
    assert data["unit"] == "%"
    assert data["higher_is_better"] is False
    assert data["details"] == {"threshold": 0.5}

    empty_details = EvaluationMetric(name="loss", value=1.0)
    data2 = empty_details.as_dict()
    assert data2["details"] == {}


def test_evaluation_result_requires_summary() -> None:
    with pytest.raises(ValueError):
        EvaluationResult(summary="  ")


def test_split_reference_parses_dotted_and_colon_notation() -> None:
    module, attr = Evaluator._split_reference("pkg.mod:func")  # type: ignore[attr-defined]
    assert module == "pkg.mod"
    assert attr == "func"

    module2, attr2 = Evaluator._split_reference("pkg.mod.func")  # type: ignore[attr-defined]
    assert module2 == "pkg.mod"
    assert attr2 == "func"

    with pytest.raises(EvaluationError):
        Evaluator._split_reference("invalid")  # type: ignore[attr-defined]


def test_resolve_callable_supports_class_instance_and_function() -> None:
    class Plugin:
        def evaluate(self, context: EvaluationContext) -> str:  # pragma: no cover - trivial
            return f"ok:{context.worktree}"

    instance = Plugin()

    fn_from_class = Evaluator._resolve_callable(Plugin)  # type: ignore[attr-defined]
    fn_from_instance = Evaluator._resolve_callable(instance)  # type: ignore[attr-defined]

    dummy_context = EvaluationContext(worktree=Path("."))

    assert callable(fn_from_class)
    assert callable(fn_from_instance)
    assert isinstance(fn_from_class(dummy_context), str)
    assert isinstance(fn_from_instance(dummy_context), str)

    def plugin_fn(context: EvaluationContext) -> str:  # pragma: no cover - trivial
        return f"fn:{context.worktree}"

    fn = Evaluator._resolve_callable(plugin_fn)  # type: ignore[attr-defined]
    assert fn is plugin_fn

    with pytest.raises(EvaluationError):
        Evaluator._resolve_callable(object())  # type: ignore[attr-defined]


def test_coerce_result_from_mapping_and_truncates_metrics(settings: Settings) -> None:
    settings.worker_evaluator_max_metrics = 2
    evaluator = Evaluator(settings=settings)

    payload: Mapping[str, Any] = {
        "summary": "ok",
        "metrics": [
            {"name": "m1", "value": 1},
            {"name": "m2", "value": 2},
            {"name": "m3", "value": 3},
        ],
        "tests_executed": ["pytest -q"],
        "logs": ["log1", "log2"],
        "extra": {"key": "value"},
    }
    result = evaluator._coerce_result(payload)  # type: ignore[attr-defined]
    assert isinstance(result, EvaluationResult)
    assert result.summary == "ok"
    assert len(result.metrics) == 2
    assert result.tests_executed == ("pytest -q",)
    assert result.logs == ("log1", "log2")
    assert result.extra == {"key": "value"}

    direct = EvaluationResult(summary="s", metrics=(EvaluationMetric(name="m", value=1.0),))
    again = evaluator._coerce_result(direct)  # type: ignore[attr-defined]
    assert again is direct


def test_coerce_metrics_and_normalise_sequence(settings: Settings) -> None:
    evaluator = Evaluator(settings=settings)

    metric = EvaluationMetric(name="acc", value=1.0)
    metrics = evaluator._coerce_metrics(metric)  # type: ignore[attr-defined]
    assert metrics == (metric,)

    metrics2 = evaluator._coerce_metrics({"name": "loss", "value": 0.5})  # type: ignore[attr-defined]
    assert len(metrics2) == 1
    assert metrics2[0].name == "loss"

    with pytest.raises(EvaluationError):
        evaluator._coerce_metrics([{"name": "bad", "value": True}])  # type: ignore[attr-defined]

    seq = Evaluator._normalise_sequence("  x  ", "label")  # type: ignore[attr-defined]
    assert seq == ("x",)
    seq2 = Evaluator._normalise_sequence(["a", " ", None, "b"], "label")  # type: ignore[attr-defined]
    assert seq2 == ("a", "b")
    with pytest.raises(EvaluationError):
        Evaluator._normalise_sequence(42, "label")  # type: ignore[attr-defined]


def test_coerce_extra_and_validate_context(tmp_path: Path, settings: Settings) -> None:
    mapping = {"a": 1}
    assert Evaluator._coerce_extra(mapping) == mapping  # type: ignore[attr-defined]
    assert Evaluator._coerce_extra(None) == {}  # type: ignore[attr-defined]

    with pytest.raises(EvaluationError):
        Evaluator._coerce_extra(["not-mapping"])  # type: ignore[attr-defined]

    ctx_valid = EvaluationContext(worktree=tmp_path)
    Evaluator._validate_context(ctx_valid)  # type: ignore[attr-defined]

    ctx_missing = EvaluationContext(worktree=tmp_path / "missing")
    with pytest.raises(EvaluationError):
        Evaluator._validate_context(ctx_missing)  # type: ignore[attr-defined]

    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    ctx_file = EvaluationContext(worktree=file_path)
    with pytest.raises(EvaluationError):
        Evaluator._validate_context(ctx_file)  # type: ignore[attr-defined]


