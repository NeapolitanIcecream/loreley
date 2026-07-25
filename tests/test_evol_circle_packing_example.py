from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from loreley.core.worker.evaluator import EvaluationContext


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SCRIPT = ROOT / "examples" / "evol_circle_packing.py"
LOCAL_EVAL_SCRIPT = ROOT / "examples" / "circle_packing_env" / "local_eval.py"
EVALUATE_SCRIPT = ROOT / "examples" / "circle_packing_env" / "evaluate.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_build_env_overrides_accepts_max_total_jobs_override() -> None:
    module = _load_module("test_evol_circle_packing_override", EXAMPLE_SCRIPT)

    env = module._build_env_overrides(  # noqa: SLF001 - spec-level assertion
        phase="main",
        max_total_jobs_override=96,
    )

    assert env["SCHEDULER_MAX_TOTAL_JOBS"] == "96"
    assert "250 ms" in env["WORKER_EVOLUTION_GLOBAL_GOAL"]
    assert json.loads(env["MAPELITES_OBJECTIVES"]) == [
        {"name": "sum_radii", "direction": "max"},
        {"name": "runtime_p50_ms", "direction": "min"},
    ]
    assert json.loads(env["MAPELITES_ISLANDS"]) == [
        "circle_packing_alpha",
        "circle_packing_beta",
    ]


def test_run_workers_delegates_to_product_worker_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("test_evol_circle_packing_pool", EXAMPLE_SCRIPT)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        module,
        "_run_worker",
        lambda **kwargs: captured.update(kwargs) or 7,
    )

    assert module._run_workers(  # noqa: SLF001 - spec-level assertion
        phase="smoke",
        count=4,
        log_level="INFO",
        no_preflight=True,
        preflight_timeout_seconds=1.5,
    ) == 7
    assert captured == {
        "phase": "smoke",
        "processes": 4,
        "log_level": "INFO",
        "no_preflight": True,
        "preflight_timeout_seconds": 1.5,
    }


def test_main_dispatches_workers_with_parsed_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("test_evol_circle_packing_main_dispatch", EXAMPLE_SCRIPT)
    captured: dict[str, Any] = {}

    def fake_run_workers(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 42

    monkeypatch.setattr(module, "_run_workers", fake_run_workers)

    exit_code = module.main(
        [
            "workers",
            "--phase",
            "main",
            "--count",
            "2",
            "--no-preflight",
            "--preflight-timeout-seconds",
            "1.5",
            "--log-level",
            "DEBUG",
        ]
    )

    assert exit_code == 42
    assert captured == {
        "phase": "main",
        "count": 2,
        "log_level": "DEBUG",
        "no_preflight": True,
        "preflight_timeout_seconds": 1.5,
    }


def test_build_report_payload_aggregates_worker_and_objective_stats() -> None:
    module = _load_module("test_evol_circle_packing_report", EXAMPLE_SCRIPT)

    jobs = [
        {
            "job_id": "job-1",
            "status": "succeeded",
            "is_seed_job": True,
            "worker_instance_id": "worker-01",
            "result_commit_hash": "a",
            "sum_radii": 0.8,
            "packing_density": 0.4,
            "total_duration_seconds": 600.0,
            "planning_duration_seconds": 10.0,
            "coding_duration_seconds": 500.0,
            "evaluator_duration_seconds": 1.5,
            "planning_attempts": 1,
            "coding_attempts": 1,
            "_created_ts": 10.0,
            "_started_ts": 20.0,
            "_completed_ts": 620.0,
            "completed_at": "2026-03-11T00:10:20+00:00",
            "last_error": None,
        },
        {
            "job_id": "job-2",
            "status": "failed",
            "is_seed_job": False,
            "worker_instance_id": "worker-02",
            "result_commit_hash": None,
            "sum_radii": None,
            "packing_density": None,
            "total_duration_seconds": 300.0,
            "planning_duration_seconds": 8.0,
            "coding_duration_seconds": None,
            "evaluator_duration_seconds": None,
            "planning_attempts": 1,
            "coding_attempts": None,
            "_created_ts": 30.0,
            "_started_ts": 40.0,
            "_completed_ts": 340.0,
            "completed_at": None,
            "last_error": "coding failed",
        },
        {
            "job_id": "job-3",
            "status": "succeeded",
            "is_seed_job": False,
            "worker_instance_id": "worker-01",
            "result_commit_hash": "b",
            "sum_radii": 1.1,
            "packing_density": 0.5,
            "total_duration_seconds": 660.0,
            "planning_duration_seconds": 12.0,
            "coding_duration_seconds": 540.0,
            "evaluator_duration_seconds": 2.0,
            "planning_attempts": 1,
            "coding_attempts": 2,
            "_created_ts": 50.0,
            "_started_ts": 60.0,
            "_completed_ts": 720.0,
            "completed_at": "2026-03-11T00:12:00+00:00",
            "last_error": None,
        },
    ]
    archive_cells = [
        {
            "island_id": "circle_packing_alpha",
            "cell_index": 0,
            "commit_hash": "b",
            "objective_values": [1.1, 100.0],
            "timestamp": 1.0,
        },
    ]
    references = [
        {
            "label": "root",
            "commit_hash": "root",
            "target_metrics": {"sum_radii": 0.25, "packing_density": 0.02},
            "repeated_runs": {"time_ms": {"p50": 0.5}, "deterministic": True},
        }
    ]

    report = module._build_report_payload(  # noqa: SLF001 - spec-level assertion
        phase="smoke",
        experiment_id="circle-packing-codex-gpt54-smoke-4w",
        jobs=jobs,
        archive_cells=archive_cells,
        references=references,
    )

    assert report["jobs"]["total"] == 3
    assert report["jobs"]["succeeded"] == 2
    assert report["jobs"]["failed"] == 1
    assert report["best"]["commit_hash"] == "b"
    assert report["archive"]["occupied_cells"] == 1
    assert report["archive"]["retained_elites"] == 1
    assert report["archive"]["best_primary_value"] == pytest.approx(1.1)
    assert report["first_above_baseline"]["job_id"] == "job-1"
    assert report["worker_throughput"][0]["worker_instance_id"] == "worker-01"
    assert report["worker_throughput"][0]["jobs_succeeded"] == 2


def test_build_report_payload_tracks_runtime_and_skips_unassigned_workers() -> None:
    module = _load_module("test_evol_circle_packing_report_runtime", EXAMPLE_SCRIPT)

    jobs = [
        {
            "job_id": "job-1",
            "status": "queued",
            "is_seed_job": True,
            "worker_instance_id": None,
            "result_commit_hash": None,
            "sum_radii": None,
            "packing_density": None,
            "runtime_p50_ms": None,
            "total_duration_seconds": None,
            "planning_duration_seconds": None,
            "coding_duration_seconds": None,
            "evaluator_duration_seconds": None,
            "planning_attempts": None,
            "coding_attempts": None,
            "_created_ts": 10.0,
            "_started_ts": None,
            "_completed_ts": None,
            "completed_at": None,
            "last_error": None,
        },
        {
            "job_id": "job-2",
            "status": "succeeded",
            "is_seed_job": True,
            "worker_instance_id": "worker-02",
            "result_commit_hash": "a",
            "sum_radii": 2.0,
            "packing_density": 0.5,
            "runtime_p50_ms": 120.0,
            "total_duration_seconds": 300.0,
            "planning_duration_seconds": 40.0,
            "coding_duration_seconds": 240.0,
            "evaluator_duration_seconds": 1.0,
            "planning_attempts": 1,
            "coding_attempts": 1,
            "_created_ts": 20.0,
            "_started_ts": 21.0,
            "_completed_ts": 321.0,
            "completed_at": "2026-03-11T00:05:21+00:00",
            "last_error": None,
        },
    ]

    report = module._build_report_payload(  # noqa: SLF001 - spec-level assertion
        phase="main",
        experiment_id="circle-packing-codex-gpt54-main-4w",
        jobs=jobs,
        archive_cells=[],
        references=[],
    )

    assert "runtime_p50_ms" in report["timing"]
    assert report["timing"]["runtime_p50_ms"]["p50"] == pytest.approx(120.0)
    assert [item["worker_instance_id"] for item in report["worker_throughput"]] == ["worker-02"]
    assert report["best"]["runtime_p50_ms"] == pytest.approx(120.0)


def test_run_report_records_missing_optional_historical_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("test_evol_circle_packing_report_missing_historical", EXAMPLE_SCRIPT)
    refs_root = tmp_path / "refs"
    output_dir = tmp_path / "reports"
    jobs = [
        {
            "job_id": "job-1",
            "status": "succeeded",
            "is_seed_job": False,
            "worker_instance_id": "worker-01",
            "result_commit_hash": "current",
            "sum_radii": 1.1,
            "packing_density": 0.5,
            "runtime_p50_ms": 90.0,
            "total_duration_seconds": 300.0,
            "planning_duration_seconds": 10.0,
            "coding_duration_seconds": 250.0,
            "evaluator_duration_seconds": 1.0,
            "planning_attempts": 1,
            "coding_attempts": 1,
            "_created_ts": 10.0,
            "_started_ts": 11.0,
            "_completed_ts": 311.0,
            "completed_at": "2026-03-11T00:05:11+00:00",
            "last_error": None,
        }
    ]

    class FakeLocalEval:
        @staticmethod
        def evaluate_repo(*, repo_root: Path, runs: int, target_n: int) -> dict[str, Any]:
            del repo_root
            assert runs == 3
            assert target_n == 26
            return {
                "target_metrics": {"sum_radii": 0.7, "packing_density": 0.3},
                "repeated_runs": {"time_ms": {"p50": 1.5}, "deterministic": True},
            }

    def fake_materialize(commit_hash: str) -> Path:
        if commit_hash == "missing":
            raise RuntimeError(
                "Could not materialize solution.py: fatal: bad revision "
                "/Users/chenmohan/private/repo"
            )
        path = refs_root / commit_hash
        path.mkdir(parents=True)
        (path / "solution.py").write_text("def pack_circles(n=26): return []\n", encoding="utf-8")
        return path

    monkeypatch.setattr(module, "_apply_base_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_ensure_repo_on_sys_path", lambda: None)
    monkeypatch.setattr(module, "_load_experiment_jobs", lambda: (jobs, []))
    monkeypatch.setattr(module, "_load_local_eval_module", lambda: FakeLocalEval)
    monkeypatch.setattr(module, "_materialize_solution_for_commit", fake_materialize)
    monkeypatch.setattr(module, "MAPELITES_EXPERIMENT_ROOT_COMMIT", "root")
    monkeypatch.setattr(module, "HISTORICAL_BEST_COMMIT", "missing")

    exit_code = module._run_report(phase="smoke", runs=3, output_dir=output_dir)  # noqa: SLF001

    payload = json.loads((output_dir / "smoke-report.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "smoke-report.md").read_text(encoding="utf-8")
    refs = {item["label"]: item for item in payload["references"]}

    assert exit_code == 0
    assert refs["root"]["status"] == "available"
    assert refs["current_best"]["status"] == "available"
    assert refs["historical_best"]["status"] == "missing"
    assert refs["historical_best"]["error"]
    assert refs["historical_best"]["target_metrics"] is None
    assert refs["historical_best"]["repeated_runs"] is None
    assert "/Users/" not in refs["historical_best"]["error"]
    assert "/Users/" not in markdown
    assert "historical_best" in markdown
    assert "missing" in markdown


def test_collect_reference_stats_keeps_current_best_reference_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("test_evol_circle_packing_report_current_required", EXAMPLE_SCRIPT)

    class FakeLocalEval:
        @staticmethod
        def evaluate_repo(*, repo_root: Path, runs: int, target_n: int) -> dict[str, Any]:
            del repo_root, runs, target_n
            return {
                "target_metrics": {"sum_radii": 0.5, "packing_density": 0.2},
                "repeated_runs": {"time_ms": {"p50": 1.0}, "deterministic": True},
            }

    def fake_materialize(commit_hash: str) -> Path:
        if commit_hash == "current":
            raise RuntimeError("current best missing")
        path = tmp_path / commit_hash
        path.mkdir()
        (path / "solution.py").write_text("def pack_circles(n=26): return []\n", encoding="utf-8")
        return path

    monkeypatch.setattr(module, "_load_local_eval_module", lambda: FakeLocalEval)
    monkeypatch.setattr(module, "_materialize_solution_for_commit", fake_materialize)
    monkeypatch.setattr(module, "MAPELITES_EXPERIMENT_ROOT_COMMIT", "root")
    monkeypatch.setattr(module, "HISTORICAL_BEST_COMMIT", "historical")

    with pytest.raises(RuntimeError, match="current best missing"):
        module._collect_reference_stats(best_commit_hash="current", runs=3)  # noqa: SLF001


def test_evaluate_main_expansion_recommends_96_when_thresholds_met() -> None:
    module = _load_module("test_evol_circle_packing_expansion_ok", EXAMPLE_SCRIPT)
    jobs = []
    for idx in range(24):
        jobs.append(
            {
                "job_id": f"job-{idx}",
                "status": "succeeded",
                "completed_at": f"2026-03-11T00:{idx:02d}:00+00:00",
                "_completed_ts": float(idx + 1),
                "_created_ts": float(idx),
                "total_duration_seconds": 12.0 * 60.0,
            }
        )

    check = module._evaluate_main_expansion(  # noqa: SLF001 - spec-level assertion
        jobs=jobs,
        wall_clock_hours_remaining=6.0,
    )

    assert check.eligible is True
    assert check.recommended_max_total_jobs == 96
    assert check.completed_jobs_considered == 24
    assert check.failure_rate == 0.0


def test_evaluate_main_expansion_stays_at_64_when_thresholds_fail() -> None:
    module = _load_module("test_evol_circle_packing_expansion_no", EXAMPLE_SCRIPT)
    jobs = []
    for idx in range(24):
        jobs.append(
            {
                "job_id": f"job-{idx}",
                "status": "failed" if idx < 5 else "succeeded",
                "completed_at": f"2026-03-11T00:{idx:02d}:00+00:00",
                "_completed_ts": float(idx + 1),
                "_created_ts": float(idx),
                "total_duration_seconds": 16.0 * 60.0,
            }
        )

    check = module._evaluate_main_expansion(  # noqa: SLF001 - spec-level assertion
        jobs=jobs,
        wall_clock_hours_remaining=4.5,
    )

    assert check.eligible is False
    assert check.recommended_max_total_jobs == 64
    assert any("15 minutes" in reason for reason in check.reasons)
    assert any("15%" in reason for reason in check.reasons)
    assert any("under 5 hours" in reason for reason in check.reasons)


def test_local_eval_reports_baseline_metrics(tmp_path: Path) -> None:
    module = _load_module("test_circle_packing_local_eval", LOCAL_EVAL_SCRIPT)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "solution.py").write_text(
        "\n".join(
            [
                "from typing import Iterable, Tuple",
                "",
                "def pack_circles(n: int = 26) -> Iterable[Tuple[float, float, float]]:",
                "    if n <= 0:",
                "        raise ValueError('n must be positive')",
                "    if n == 1:",
                "        return [(0.5, 0.5, 0.5)]",
                "    r = 1.0 / (4.0 * n)",
                "    step = 1.0 - 2.0 * r",
                "    circles = []",
                "    for i in range(n):",
                "        t = i / (n - 1)",
                "        circles.append((r + t * step, r + t * step, r))",
                "    return circles",
            ]
        ),
        encoding="utf-8",
    )

    payload = module.evaluate_repo(repo_root=repo_root, runs=3, target_n=26)

    assert payload["repeated_runs"]["deterministic"] is True
    assert payload["target_metrics"]["sum_radii"] == pytest.approx(0.25)
    assert payload["target_metrics"]["packing_density"] > 0.0


def test_local_eval_computes_target_metrics_when_target_n_not_in_sample_ns(tmp_path: Path) -> None:
    module = _load_module("test_circle_packing_local_eval_custom_target", LOCAL_EVAL_SCRIPT)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "solution.py").write_text(
        "\n".join(
            [
                "from typing import Iterable, Tuple",
                "",
                "def pack_circles(n: int = 26) -> Iterable[Tuple[float, float, float]]:",
                "    if n <= 0:",
                "        raise ValueError('n must be positive')",
                "    if n == 1:",
                "        return [(0.5, 0.5, 0.5)]",
                "    r = 1.0 / (4.0 * n)",
                "    step = 1.0 - 2.0 * r",
                "    circles = []",
                "    for i in range(n):",
                "        t = i / (n - 1)",
                "        circles.append((r + t * step, r + t * step, r))",
                "    return circles",
            ]
        ),
        encoding="utf-8",
    )

    payload = module.evaluate_repo(
        repo_root=repo_root,
        runs=3,
        target_n=10,
        sample_ns=(1, 2, 5),
    )

    assert set(payload["samples"]) == {"1", "2", "5", "10"}
    assert payload["target_metrics"] == payload["samples"]["10"]
    assert payload["target_metrics"]["sum_radii"] == pytest.approx(0.25)


def test_load_local_eval_module_falls_back_to_main_repo_copy(tmp_path: Path) -> None:
    module = _load_module("test_circle_packing_local_eval_loader", EXAMPLE_SCRIPT)
    module.REPO_ROOT = tmp_path / "missing-circle-packing"
    module.EVAL_ENV_ROOT = ROOT / "examples" / "circle_packing_env"

    local_eval = module._load_local_eval_module()  # noqa: SLF001 - spec-level assertion

    assert hasattr(local_eval, "evaluate_repo")


def test_circle_packing_evaluator_emits_runtime_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("test_circle_packing_evaluate_runtime", EVALUATE_SCRIPT)
    monkeypatch.setenv("CIRCLE_PACKING_RUNTIME_RUNS", "3")
    monkeypatch.setenv("CIRCLE_PACKING_RUNTIME_BUDGET_MS", "250")

    worktree = tmp_path / "repo"
    worktree.mkdir()
    (worktree / "solution.py").write_text(
        "\n".join(
            [
                "from typing import Iterable, Tuple",
                "",
                "def pack_circles(n: int = 26) -> Iterable[Tuple[float, float, float]]:",
                "    if n <= 0:",
                "        raise ValueError('n must be positive')",
                "    if n == 1:",
                "        return [(0.5, 0.5, 0.5)]",
                "    r = 1.0 / (4.0 * n)",
                "    step = 1.0 - 2.0 * r",
                "    circles = []",
                "    for i in range(n):",
                "        t = i / (n - 1)",
                "        circles.append((r + t * step, r + t * step, r))",
                "    return circles",
            ]
        ),
        encoding="utf-8",
    )

    result = module.plugin(EvaluationContext(worktree=worktree))

    metrics = {entry["name"]: entry["value"] for entry in result["metrics"]}
    assert metrics["runtime_p50_ms"] >= 0.0
    assert result["extra"]["runtime_budget_ms"] == pytest.approx(250.0)
    assert result["extra"]["runtime_runs"] == 3
    assert result["extra"]["runtime_deterministic"] is True


def test_circle_packing_evaluator_rejects_runtime_budget_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module("test_circle_packing_evaluate_slow", EVALUATE_SCRIPT)
    monkeypatch.setenv("CIRCLE_PACKING_RUNTIME_RUNS", "3")
    monkeypatch.setenv("CIRCLE_PACKING_RUNTIME_BUDGET_MS", "250")

    worktree = tmp_path / "repo"
    worktree.mkdir()
    (worktree / "solution.py").write_text(
        "\n".join(
            [
                "import time",
                "from typing import Iterable, Tuple",
                "",
                "def pack_circles(n: int = 26) -> Iterable[Tuple[float, float, float]]:",
                "    time.sleep(0.3)",
                "    if n <= 0:",
                "        raise ValueError('n must be positive')",
                "    if n == 1:",
                "        return [(0.5, 0.5, 0.5)]",
                "    r = 1.0 / (4.0 * n)",
                "    step = 1.0 - 2.0 * r",
                "    circles = []",
                "    for i in range(n):",
                "        t = i / (n - 1)",
                "        circles.append((r + t * step, r + t * step, r))",
                "    return circles",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"violates runtime budget: budget=250.*observed p50="):
        module.plugin(EvaluationContext(worktree=worktree))
