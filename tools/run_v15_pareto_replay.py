#!/usr/bin/env python3
"""Replay measured circle-packing candidates through scalar and Pareto policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loreley.core.map_elites.pareto_archive import ParetoCandidate, ParetoGridArchive

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_SAMPLES = 31
DEFAULT_ITERATIONS = 1_000

_BENCHMARK_RUNNER = r"""
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path

source_path = Path("/candidate/solution.py")
spec = importlib.util.spec_from_file_location("candidate_solution", source_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

def validate(circles, n):
    if len(circles) != n:
        raise ValueError(f"expected {n} circles, got {len(circles)}")
    parsed = [tuple(float(value) for value in circle) for circle in circles]
    for index, (x, y, radius) in enumerate(parsed):
        if not all(math.isfinite(value) for value in (x, y, radius)):
            raise ValueError(f"circle {index} is not finite")
        if radius <= 0.0:
            raise ValueError(f"circle {index} has non-positive radius")
        if x - radius < -1e-9 or y - radius < -1e-9:
            raise ValueError(f"circle {index} crosses lower boundary")
        if x + radius > 1.0 + 1e-9 or y + radius > 1.0 + 1e-9:
            raise ValueError(f"circle {index} crosses upper boundary")
    for left in range(len(parsed)):
        x1, y1, r1 = parsed[left]
        for right in range(left + 1, len(parsed)):
            x2, y2, r2 = parsed[right]
            if math.hypot(x1 - x2, y1 - y2) + 1e-9 < r1 + r2:
                raise ValueError(f"circles {left} and {right} overlap")
    return parsed

n = 26
samples = int(__import__("os").environ["BENCHMARK_SAMPLES"])
iterations = int(__import__("os").environ["BENCHMARK_ITERATIONS"])
warmup = max(10, iterations // 10)

first = validate(list(module.pack_circles(n)), n)
for _ in range(warmup):
    list(module.pack_circles(n))

samples_ms_per_call = []
for _ in range(samples):
    started = time.perf_counter_ns()
    for _ in range(iterations):
        list(module.pack_circles(n))
    elapsed = time.perf_counter_ns() - started
    samples_ms_per_call.append(elapsed / iterations / 1_000_000)

second = validate(list(module.pack_circles(n)), n)
if first != second:
    raise ValueError("candidate is not deterministic")

print(json.dumps({
    "quality_sum_radii": math.fsum(circle[2] for circle in first),
    "samples_ms_per_call": samples_ms_per_call,
    "sample_count": samples,
    "iterations_per_sample": iterations,
    "deterministic": True,
}, sort_keys=True))
"""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sample.")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_median_ci(
    values: list[float],
    *,
    resamples: int = 4_000,
    seed: int = 15,
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)
    ]
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def summarize_runtime(values: list[float]) -> dict[str, float | int]:
    low, high = _bootstrap_median_ci(values)
    return {
        "sample_count": len(values),
        "mean_ms": statistics.fmean(values),
        "p05_ms": _percentile(values, 0.05),
        "p50_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "median_bootstrap_95ci_low_ms": low,
        "median_bootstrap_95ci_high_ms": high,
    }


def _find_result(payload: Any, model: str) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(results, list):
        raise ValueError("Kilo artifact must contain a result list.")
    for result in results:
        if result.get("model") == model:
            return result
    raise ValueError(f"Kilo artifact does not contain model {model!r}.")


def load_candidates(
    *,
    baseline_path: Path,
    mini_artifact: Path,
    contender_artifact: Path,
) -> dict[str, dict[str, str]]:
    baseline_source = baseline_path.read_text(encoding="utf-8")
    mini_payload = json.loads(mini_artifact.read_text(encoding="utf-8"))
    contender_payload = json.loads(contender_artifact.read_text(encoding="utf-8"))
    mini = _find_result(mini_payload, "gpt-5.4-mini")
    deepseek = _find_result(contender_payload, "deepseek-v4-flash")
    candidates = {
        "baseline": {
            "source": baseline_source,
            "model": "repository-baseline",
        },
        "deepseek": {
            "source": str(deepseek["source"]),
            "model": "deepseek-v4-flash",
        },
        "mini": {
            "source": str(mini["source"]),
            "model": "gpt-5.4-mini",
        },
    }
    for candidate in candidates.values():
        candidate["source_sha256"] = _sha256(candidate["source"])
    return candidates


def benchmark_source(
    source: str,
    *,
    image: str,
    samples: int,
    iterations: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="loreley-v15-pareto-") as raw_dir:
        workdir = Path(raw_dir)
        candidate_dir = workdir / "candidate"
        candidate_dir.mkdir()
        (candidate_dir / "solution.py").write_text(source, encoding="utf-8")
        runner = workdir / "runner.py"
        runner.write_text(_BENCHMARK_RUNNER, encoding="utf-8")
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cpus",
            "1",
            "--memory",
            "256m",
            "--pids-limit",
            "128",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=32m",
            "-e",
            f"BENCHMARK_SAMPLES={samples}",
            "-e",
            f"BENCHMARK_ITERATIONS={iterations}",
            "-v",
            f"{candidate_dir}:/candidate:ro",
            "-v",
            f"{runner}:/runner.py:ro",
            image,
            "python",
            "/runner.py",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Candidate benchmark failed "
                f"(exit={completed.returncode}): {completed.stderr[-2_000:]}"
            )
        result = json.loads(completed.stdout)
        values = [float(value) for value in result.pop("samples_ms_per_call")]
        result["runtime"] = summarize_runtime(values)
        result["samples_ms_per_call"] = values
        return result


def replay_policies(
    measurements: dict[str, dict[str, float]],
) -> dict[str, Any]:
    def candidate(name: str) -> ParetoCandidate:
        measured = measurements[name]
        quality = float(measured["quality_sum_radii"])
        runtime = float(measured["runtime_p50_ms"])
        return ParetoCandidate(
            commit_hash=name,
            objective_values=(quality, runtime),
            objective_scores=(quality, -runtime),
            measures=(0.5,),
            timestamp=0.0,
        )

    archive = ParetoGridArchive(
        dims=(1,),
        ranges=((0.0, 1.0),),
        objective_count=2,
        max_front_size=8,
        epsilon=1e-12,
    )
    pair = (candidate("baseline"), candidate("deepseek"))
    archive.add_many(pair)
    pair_front = [item.commit_hash for item in archive.records()]
    scalar_primary = max(
        pair,
        key=lambda item: (item.objective_scores[0], item.commit_hash),
    ).commit_hash
    archive.add(candidate("mini"))
    final_front = [item.commit_hash for item in archive.records()]
    return {
        "controlled_measure": [0.5],
        "pair": ["baseline", "deepseek"],
        "pair_pareto_front": pair_front,
        "pair_scalar_primary_retained": [scalar_primary],
        "pair_tradeoff_observed": set(pair_front) == {"baseline", "deepseek"},
        "dominator_added": "mini",
        "front_after_dominator": final_front,
        "dominator_collapsed_front": final_front == ["mini"],
    }


def _docker_image_id(image: str) -> str | None:
    completed = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("examples/circle-packing/solution.py"),
    )
    parser.add_argument(
        "--mini-artifact",
        type=Path,
        default=Path(
            "docs/research/artifacts/2026-07-26-v15-validation/kilo-bakeoff.json"
        ),
    )
    parser.add_argument(
        "--contender-artifact",
        type=Path,
        default=Path(
            "docs/research/artifacts/2026-07-26-v15-validation/"
            "kilo-bakeoff-contenders.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/research/artifacts/2026-07-26-v15-validation/pareto-replay.json"
        ),
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples < 5 or args.iterations < 1:
        raise SystemExit("--samples must be >= 5 and --iterations must be positive.")
    candidates = load_candidates(
        baseline_path=args.baseline,
        mini_artifact=args.mini_artifact,
        contender_artifact=args.contender_artifact,
    )
    measured: dict[str, dict[str, Any]] = {}
    for name, candidate in candidates.items():
        benchmark = benchmark_source(
            candidate["source"],
            image=args.image,
            samples=args.samples,
            iterations=args.iterations,
        )
        measured[name] = {
            "model": candidate["model"],
            "source_sha256": candidate["source_sha256"],
            "quality_sum_radii": benchmark["quality_sum_radii"],
            "runtime_p50_ms": benchmark["runtime"]["p50_ms"],
            "runtime": benchmark["runtime"],
            "samples_ms_per_call": benchmark["samples_ms_per_call"],
            "deterministic": benchmark["deterministic"],
        }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": {
            "task": "26-circle packing",
            "container_image": args.image,
            "container_image_id": _docker_image_id(args.image),
            "network": "none",
            "cpus": 1,
            "memory": "256m",
            "warmup_calls": max(10, args.iterations // 10),
            "sample_count": args.samples,
            "iterations_per_sample": args.iterations,
            "bootstrap_resamples": 4_000,
            "bootstrap_seed": 15,
            "objective_scores": ["sum_radii", "-runtime_p50_ms"],
        },
        "candidates": measured,
        "replay": replay_policies(measured),
        "api_cost_usd": 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["replay"], indent=2, sort_keys=True))
    print(f"artifact={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
