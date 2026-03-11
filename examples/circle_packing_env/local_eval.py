from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


Circle = tuple[float, float, float]
DEFAULT_SAMPLE_NS: tuple[int, ...] = (1, 2, 5, 26)
DEFAULT_RUNS: int = 50
TOLERANCE: float = 1e-10


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, float(percentile))) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stats(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(item) for item in values]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "mean": _mean(numeric),
        "p50": _percentile(numeric, 0.50),
        "p90": _percentile(numeric, 0.90),
        "p99": _percentile(numeric, 0.99),
        "min": min(numeric),
        "max": max(numeric),
    }


def _load_solution_module(repo_root: Path) -> Any:
    solution_path = Path(repo_root).expanduser().resolve() / "solution.py"
    if not solution_path.is_file():
        raise FileNotFoundError(f"Could not find solution.py at {solution_path}.")
    spec = importlib.util.spec_from_file_location("circle_packing_solution", solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import solution module from {solution_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _coerce_circles(raw: Iterable[Any], *, n: int) -> list[Circle]:
    circles = [(float(x), float(y), float(r)) for x, y, r in raw]
    if len(circles) != n:
        raise AssertionError(f"Expected {n} circles, got {len(circles)}")
    return circles


def _validate(circles: Sequence[Circle], *, n: int, tolerance: float = TOLERANCE) -> None:
    if len(circles) != n:
        raise AssertionError(f"Expected {n} circles, got {len(circles)}")
    for index, (x, y, r) in enumerate(circles):
        if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(r):
            raise AssertionError(f"Circle #{index} contains non-finite values.")
        if r <= 0.0:
            raise AssertionError(f"Circle #{index} has non-positive radius {r!r}.")
        if x < r - tolerance or x > 1.0 - r + tolerance:
            raise AssertionError(f"Circle #{index} violates x-boundary constraints.")
        if y < r - tolerance or y > 1.0 - r + tolerance:
            raise AssertionError(f"Circle #{index} violates y-boundary constraints.")
    for i in range(n - 1):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            gap = math.hypot(xj - xi, yj - yi) - (ri + rj)
            if gap < -tolerance:
                raise AssertionError(f"Circles {i} and {j} overlap by {-gap:.3e}.")


def _circle_metrics(circles: Sequence[Circle]) -> dict[str, float]:
    min_gap = float("inf")
    min_boundary_margin = float("inf")
    total_area = 0.0
    sum_radii = 0.0

    for x, y, r in circles:
        sum_radii += r
        total_area += math.pi * (r**2)
        min_boundary_margin = min(min_boundary_margin, x - r, y - r, 1.0 - (x + r), 1.0 - (y + r))

    for i in range(len(circles) - 1):
        xi, yi, ri = circles[i]
        for j in range(i + 1, len(circles)):
            xj, yj, rj = circles[j]
            min_gap = min(min_gap, math.hypot(xj - xi, yj - yi) - (ri + rj))

    if math.isinf(min_gap):
        min_gap = 0.0
    if math.isinf(min_boundary_margin):
        min_boundary_margin = 0.0

    return {
        "sum_radii": float(sum_radii),
        "packing_density": float(total_area),
        "min_gap": float(min_gap),
        "min_boundary_margin": float(min_boundary_margin),
    }


def evaluate_repo(
    *,
    repo_root: Path,
    runs: int = DEFAULT_RUNS,
    target_n: int = 26,
    sample_ns: Sequence[int] = DEFAULT_SAMPLE_NS,
) -> dict[str, Any]:
    module = _load_solution_module(repo_root)
    pack_circles = getattr(module, "pack_circles", None)
    if not callable(pack_circles):
        raise RuntimeError(f"{repo_root / 'solution.py'} does not define callable pack_circles.")

    samples: dict[str, dict[str, float]] = {}
    for n in sample_ns:
        circles = _coerce_circles(pack_circles(n), n=n)
        _validate(circles, n=n)
        samples[str(n)] = _circle_metrics(circles)

    repeated_times_ms: list[float] = []
    repeated_sums: list[float] = []
    repeated_densities: list[float] = []
    repeated_gaps: list[float] = []
    repeated_boundary: list[float] = []
    baseline_output: list[Circle] | None = None
    deterministic = True

    for _ in range(int(runs)):
        started = time.perf_counter()
        circles = _coerce_circles(pack_circles(target_n), n=target_n)
        repeated_times_ms.append((time.perf_counter() - started) * 1000.0)
        _validate(circles, n=target_n)
        if baseline_output is None:
            baseline_output = circles
        elif circles != baseline_output:
            deterministic = False
        metrics = _circle_metrics(circles)
        repeated_sums.append(metrics["sum_radii"])
        repeated_densities.append(metrics["packing_density"])
        repeated_gaps.append(metrics["min_gap"])
        repeated_boundary.append(metrics["min_boundary_margin"])

    return {
        "repo_root": str(Path(repo_root).expanduser().resolve()),
        "target_n": int(target_n),
        "runs": int(runs),
        "samples": samples,
        "target_metrics": samples[str(target_n)],
        "repeated_runs": {
            "deterministic": deterministic,
            "time_ms": _stats(repeated_times_ms),
            "sum_radii": _stats(repeated_sums),
            "packing_density": _stats(repeated_densities),
            "min_gap": _stats(repeated_gaps),
            "min_boundary_margin": _stats(repeated_boundary),
        },
    }


def _render_human_report(payload: dict[str, Any]) -> str:
    lines = [
        f"repo_root={payload['repo_root']}",
        f"target_n={payload['target_n']} runs={payload['runs']}",
        "",
        "single-run samples:",
    ]
    for n, metrics in payload["samples"].items():
        lines.append(
            "  n={} sum_radii={:.12f} density={:.12f} min_gap={:.3e} boundary={:.3e}".format(
                n,
                metrics["sum_radii"],
                metrics["packing_density"],
                metrics["min_gap"],
                metrics["min_boundary_margin"],
            )
        )

    repeated = payload["repeated_runs"]
    time_ms = repeated["time_ms"]
    lines.extend(
        [
            "",
            "repeated target run:",
            "  deterministic={}".format(repeated["deterministic"]),
            "  time_ms mean={:.3f} p50={:.3f} p90={:.3f} p99={:.3f}".format(
                time_ms["mean"],
                time_ms["p50"],
                time_ms["p90"],
                time_ms["p99"],
            ),
            "  sum_radii mean={:.12f} p50={:.12f}".format(
                repeated["sum_radii"]["mean"],
                repeated["sum_radii"]["p50"],
            ),
            "  packing_density mean={:.12f} p50={:.12f}".format(
                repeated["packing_density"]["mean"],
                repeated["packing_density"]["p50"],
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local evaluator for circle-packing commits.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "circle-packing",
        help="Repository root containing solution.py.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Repeated runs for determinism/runtime measurement.",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=26,
        help="Target circle count for repeated measurement.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    payload = evaluate_repo(
        repo_root=args.repo_root,
        runs=args.runs,
        target_n=args.target_n,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_render_human_report(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
