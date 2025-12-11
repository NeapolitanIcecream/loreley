from __future__ import annotations

from math import hypot, isfinite, pi
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple, TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from loreley.core.worker.evaluator import EvaluationContext


log = logger.bind(module="examples.circle_packing_env.evaluate")
console = Console()


UNIT_SQUARE_MIN = 0.0
UNIT_SQUARE_MAX = 1.0
EPS = 1e-9

# Default number of circles expected from the candidate solution.
DEFAULT_NUM_CIRCLES = 26


def _load_pack_circles(worktree: Path) -> Any:
    """
    Dynamically load the pack_circles function from the candidate worktree.

    The expected location is:
        <worktree>/solution.py
    """
    import importlib.util
    import sys

    worktree_path = Path(worktree).expanduser().resolve()
    solution_path = worktree_path / "solution.py"
    if not solution_path.is_file():
        raise FileNotFoundError(
            f"Could not find solution.py at {solution_path!s}. "
            "Ensure your candidate worktree is a circle-packing repository with solution.py at its root.",
        )

    spec = importlib.util.spec_from_file_location(
        "circle_packing_solution",
        solution_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {solution_path!s}.")

    module = importlib.util.module_from_spec(spec)
    # Ensure the module is registered so decorators (e.g., dataclass) can resolve __module__.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[assignment]
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    if not hasattr(module, "pack_circles"):
        raise AttributeError(
            f"Module {solution_path!s} does not define a 'pack_circles' function.",
        )
    pack_circles = getattr(module, "pack_circles")
    if not callable(pack_circles):
        raise TypeError("'pack_circles' must be callable.")
    return pack_circles


def _coerce_circle(entry: Any, index: int) -> Tuple[float, float, float]:
    """Coerce a single entry into an (x, y, r) triple of finite floats."""
    try:
        x, y, r = entry  # type: ignore[misc]
    except Exception as exc:
        raise TypeError(
            f"Circle #{index} is not an iterable of length 3: {entry!r}",
        ) from exc

    try:
        xf = float(x)
        yf = float(y)
        rf = float(r)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Circle #{index} contains non-numeric values: {entry!r}",
        ) from exc

    if not (isfinite(xf) and isfinite(yf) and isfinite(rf)):
        raise ValueError(f"Circle #{index} contains non-finite values: {entry!r}")
    if rf <= 0.0:
        raise ValueError(f"Circle #{index} has non-positive radius {rf!r}.")
    return xf, yf, rf


def _validate_bounds(circles: Sequence[Tuple[float, float, float]]) -> None:
    """Ensure all circles lie entirely within the unit square."""
    for idx, (x, y, r) in enumerate(circles):
        if x - r < UNIT_SQUARE_MIN - EPS or x + r > UNIT_SQUARE_MAX + EPS:
            raise ValueError(
                f"Circle #{idx} with centre x={x:.6f} and radius r={r:.6f} "
                "does not fit horizontally inside the unit square.",
            )
        if y - r < UNIT_SQUARE_MIN - EPS or y + r > UNIT_SQUARE_MAX + EPS:
            raise ValueError(
                f"Circle #{idx} with centre y={y:.6f} and radius r={r:.6f} "
                "does not fit vertically inside the unit square.",
            )


def _validate_no_overlap(circles: Sequence[Tuple[float, float, float]]) -> None:
    """Ensure no two circles overlap (touching is allowed)."""
    n = len(circles)
    for i in range(n):
        x1, y1, r1 = circles[i]
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]
            dx = x2 - x1
            dy = y2 - y1
            centre_distance = hypot(dx, dy)
            min_distance = r1 + r2
            if centre_distance + EPS < min_distance:
                raise ValueError(
                    f"Circles #{i} and #{j} overlap: "
                    f"distance={centre_distance:.6f}, required>={min_distance:.6f}.",
                )


def _compute_metrics(
    circles: Sequence[Tuple[float, float, float]],
) -> Mapping[str, Any]:
    """Compute objective and auxiliary metrics from the circles."""
    total_area = sum(pi * (r**2) for _, _, r in circles)
    density = float(total_area)  # container area is 1.0
    sum_radii = sum(r for _, _, r in circles)
    num_circles = len(circles)
    return {
        "sum_radii": sum_radii,
        "packing_density": density,
        "num_circles": num_circles,
    }


def plugin(context: "EvaluationContext") -> Mapping[str, Any]:
    """
    Evaluation plugin for the circle-packing example.

    It loads the candidate's `pack_circles()` implementation from the worktree,
    validates the returned configuration, and reports packing metrics.
    """
    worktree = Path(context.worktree).expanduser().resolve()
    log.info("Starting circle-packing evaluation for worktree={}", worktree)

    pack_circles = _load_pack_circles(worktree)
    raw_circles = list(pack_circles(DEFAULT_NUM_CIRCLES))
    log.info("pack_circles() returned {} circle entries", len(raw_circles))

    coerced: list[Tuple[float, float, float]] = []
    for idx, entry in enumerate(raw_circles):
        coerced.append(_coerce_circle(entry, idx))

    _validate_bounds(coerced)
    _validate_no_overlap(coerced)

    metrics_values = _compute_metrics(coerced)
    sum_radii = metrics_values["sum_radii"]
    density = metrics_values["packing_density"]
    num_circles = metrics_values["num_circles"]

    if num_circles != DEFAULT_NUM_CIRCLES:
        raise ValueError(
            f"Expected exactly {DEFAULT_NUM_CIRCLES} circles, "
            f"but got {num_circles}.",
        )

    summary = (
        "Circle packing in unit square: "
        f"sum_radii={sum_radii:.6f}, circles={num_circles}, density={density:.6f}"
    )
    log.info(summary)

    metrics: list[Mapping[str, Any]] = [
        {
            "name": "sum_radii",
            "value": sum_radii,
            "unit": "radius_sum",
            "higher_is_better": True,
            "details": {
                "num_circles": num_circles,
                "packing_density": density,
            },
        },
        {
            "name": "packing_density",
            "value": density,
            "unit": "fraction",
            "higher_is_better": True,
        },
        {
            "name": "num_circles",
            "value": num_circles,
            "unit": "count",
            "higher_is_better": True,
        },
    ]
    tests_executed: Iterable[str] = [
        "type_check",
        "radius_positive_check",
        "bounds_check",
        "overlap_check",
    ]

    logs: Iterable[str] = [
        f"Loaded solution from worktree: {worktree}",
        f"Number of circles: {num_circles}",
        f"Sum of radii: {sum_radii:.6f}",
        f"Packing density: {density:.6f}",
    ]

    return {
        "summary": summary,
        "metrics": list(metrics),
        "tests_executed": list(tests_executed),
        "logs": list(logs),
        "extra": {
            "circles": coerced,
        },
    }


def _print_result_human_readable(result: Mapping[str, Any]) -> None:
    """Pretty-print evaluation results using Rich."""
    summary = result.get("summary", "")
    metrics = result.get("metrics", []) or []
    tests = result.get("tests_executed", []) or []

    console.rule("[bold cyan]Circle Packing Evaluation[/bold cyan]")
    if summary:
        console.print(f"[bold white]{summary}[/bold white]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Unit")
    table.add_column("Higher is Better")

    for entry in metrics:
        if hasattr(entry, "name") and hasattr(entry, "value"):
            name = getattr(entry, "name")
            value = getattr(entry, "value")
            unit = getattr(entry, "unit", "") or ""
            hib = getattr(entry, "higher_is_better", True)
        else:
            name = str(entry.get("name", ""))
            value = entry.get("value", "")
            unit = entry.get("unit", "") or ""
            hib = bool(entry.get("higher_is_better", True))
        table.add_row(str(name), f"{value}", str(unit), "yes" if hib else "no")

    console.print(table)

    if tests:
        console.print("\n[bold green]Tests executed:[/bold green]")
        for t in tests:
            console.print(f" - {t}")


def _main() -> None:
    """
    Local entrypoint to evaluate the current solution without running the worker.

    By default this treats the companion `examples/circle-packing` repository as the
    candidate worktree.
    """
    import sys

    here = Path(__file__).resolve()
    env_root = here.parent
    project_root = env_root.parents[1]

    # Ensure that the Loreley project root (which contains the `loreley` package)
    # is on sys.path so that `from loreley.core.worker.evaluator import EvaluationContext`
    # works even when this file is executed directly via
    # `python examples/circle_packing_env/evaluate.py`.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from loreley.core.worker.evaluator import EvaluationContext

    # Default local worktree is the circle-packing example repository.
    worktree_root = env_root.parent / "circle-packing"

    context = EvaluationContext(worktree=worktree_root)
    result = plugin(context)
    _print_result_human_readable(result)


if __name__ == "__main__":
    _main()


