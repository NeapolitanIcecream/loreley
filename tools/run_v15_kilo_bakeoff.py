"""Compare Kilo coding-agent runs across models on one circle-packing task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from loreley.config import Settings
from loreley.core.usage import LLMUsageEventPayload
from loreley.core.worker.agent.backends.kilocode_cli import (
    KilocodeCliBackend,
    _build_kilocode_openai_env,
    _kilocode_backend_model,
)
from loreley.core.worker.agent.contracts import AgentTask
from tools.run_v15_model_calibration import (
    DEFAULT_EVALUATOR,
    DEFAULT_LEDGER,
    DEFAULT_MODELS,
    _evaluate_candidate,
    _ledger_spend,
    _load_json,
    _pricing,
    _quota_snapshot,
    _record_ledger_entry,
    _record_unattributed_ledger_entry,
    _utc_now,
    _write_json,
    attributed_quota,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = PROJECT_ROOT / "examples" / "circle-packing"
DEFAULT_OUTPUT = DEFAULT_LEDGER.with_name("kilo-bakeoff.json")


def optional_quota_snapshot(
    client: httpx.Client,
    *,
    base_url: str,
) -> dict[str, Any]:
    """Sample shared proxy quota without making it an experiment dependency."""

    try:
        return _quota_snapshot(client, base_url=base_url)
    except (httpx.HTTPError, ValueError) as exc:
        status_code = (
            exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        )
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "status_code": status_code,
        }


def usage_counts(events: Sequence[LLMUsageEventPayload]) -> dict[str, int]:
    """Aggregate provider-neutral usage returned by the product Kilo backend."""

    return {
        "input_tokens": sum(max(0, int(event.input_tokens)) for event in events),
        "cached_input_tokens": sum(
            max(0, int(event.cached_input_tokens)) for event in events
        ),
        "output_tokens": sum(max(0, int(event.output_tokens)) for event in events),
        "reasoning_output_tokens": sum(
            max(0, int(event.reasoning_output_tokens)) for event in events
        ),
        "total_tokens": sum(max(0, int(event.total_tokens)) for event in events),
    }


def conservative_input_tokens(usage: Mapping[str, int]) -> int:
    """Price cache reads as ordinary input when the proxy discount is unknown."""

    return max(0, int(usage["input_tokens"])) + max(
        0,
        int(usage["cached_input_tokens"]),
    )


def _prompt() -> str:
    return """\
Act as the coding agent for this repository. Inspect solution.py and
scripts/local_eval.py. Improve pack_circles so the evaluator returns exactly n
valid, deterministic, non-overlapping circles inside the unit square for every
positive n, with high sum_radii at n=26 and p50 runtime comfortably below
250 ms. Use only the standard library, no randomness, network, filesystem
access from solution.py, subprocesses, or import-time side effects.

Modify only solution.py. Run `python scripts/local_eval.py --runs 30` and repair
any failure before finishing. Do not merely describe a patch: edit and verify
the worktree.
"""


def _settings(*, api_key: str, base_url: str, model: str) -> Settings:
    return Settings(
        WORKER_KILOCODE_MODEL=f"openai/{model}",
        WORKER_KILOCODE_JSON_OUTPUT=True,
        WORKER_KILOCODE_PROVIDER_CONFIG_MODE="config",
        WORKER_KILOCODE_OPENAI_API_SPEC="chat_completions",
        WORKER_KILOCODE_OPENAI_BASE_URL=base_url,
        WORKER_KILOCODE_OPENAI_API_KEY=api_key,
        WORKER_KILOCODE_OPENAI_MODEL=model,
        LLM_USAGE_TRACKING_ENABLED=True,
    )


def _backend(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
) -> KilocodeCliBackend:
    settings = _settings(api_key=api_key, base_url=base_url, model=model)
    extra_env = _build_kilocode_openai_env(settings)
    return KilocodeCliBackend(
        bin="kilo",
        model=_kilocode_backend_model(settings, extra_env),
        timeout_seconds=int(timeout_seconds),
        json_output=True,
        extra_env=extra_env,
        settings=settings,
        usage_tracking_enabled=True,
    )


def _clone_repository(source: Path, destination: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            str(source.resolve()),
            str(destination.resolve()),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "git clone failed")[-2000:]
        )
    commit = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    return (commit.stdout or "").strip()


def _git_capture(worktree: Path) -> tuple[str, str]:
    status = subprocess.run(
        ["git", "-C", str(worktree), "status", "--short"],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    diff = subprocess.run(
        ["git", "-C", str(worktree), "diff", "--", "solution.py"],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    return (status.stdout or "").strip(), (diff.stdout or "").strip()


def _event_payload(event: LLMUsageEventPayload) -> dict[str, Any]:
    return {
        "source": event.source,
        "phase": event.phase,
        "provider": event.provider,
        "model": event.model,
        "api_surface": event.api_surface,
        "input_tokens": event.input_tokens,
        "cached_input_tokens": event.cached_input_tokens,
        "output_tokens": event.output_tokens,
        "reasoning_output_tokens": event.reasoning_output_tokens,
        "total_tokens": event.total_tokens,
        "cost_usd": str(event.cost_usd) if event.cost_usd is not None else None,
        "cost_source": event.cost_source,
        "pricing_version": event.pricing_version,
        "external_usage_id": event.external_usage_id,
    }


def _run_agent(
    *,
    backend: KilocodeCliBackend,
    worktree: Path,
) -> tuple[dict[str, Any], tuple[LLMUsageEventPayload, ...]]:
    job_id = uuid4()
    run_token = uuid4()
    started = time.perf_counter()
    task = AgentTask(
        name="v15-kilo-bakeoff",
        prompt=_prompt(),
        job_id=job_id,
        run_token=run_token,
        phase="coding",
        attempt=1,
    )
    try:
        invocation = backend.run(task, working_dir=worktree)
    except RuntimeError as exc:
        events = tuple(getattr(exc, "usage_events", ()) or ())
        return (
            {
                "succeeded": False,
                "error": str(exc)[:2000],
                "duration_seconds": time.perf_counter() - started,
            },
            events,
        )
    return (
        {
            "succeeded": True,
            "error": None,
            "duration_seconds": invocation.duration_seconds,
            "working_directory": invocation.working_directory,
            "stdout_sha256": hashlib.sha256(
                invocation.stdout.encode("utf-8")
            ).hexdigest(),
            "stdout_tail": invocation.stdout[-4000:],
            "stderr_tail": invocation.stderr[-2000:],
        },
        invocation.usage_events,
    )


def _model_result(
    *,
    candidate: Mapping[str, Any],
    usage_events: Sequence[LLMUsageEventPayload],
    accounting: Mapping[str, Any],
) -> dict[str, Any]:
    agent = candidate["agent"]
    evaluation = candidate["evaluation"]
    source = str(candidate["source"])
    diff = str(candidate["diff"])
    pricing = accounting["pricing"]
    group_multiplier = float(accounting["group_multiplier"])
    quota_before = accounting["quota_before"]
    quota_after = accounting["quota_after"]
    usage = usage_counts(usage_events)
    quota = attributed_quota(
        input_tokens=conservative_input_tokens(usage),
        output_tokens=usage["output_tokens"],
        model_ratio=float(pricing["model_ratio"]),
        completion_ratio=float(pricing["completion_ratio"]),
        group_multiplier=group_multiplier,
    )
    return {
        "model": candidate["model"],
        "source_commit": candidate["source_commit"],
        "agent": dict(agent),
        "usage": usage,
        "usage_events": [_event_payload(event) for event in usage_events],
        "pricing": {
            **pricing,
            "cached_input_priced_as_full_input_upper_bound": True,
            "group_multiplier_upper_bound": group_multiplier,
        },
        "attributed_quota_upper_bound": quota if usage["total_tokens"] else None,
        "attributed_cost_usd_upper_bound": (
            quota / 500000.0 if usage["total_tokens"] else None
        ),
        "global_quota_before": dict(quota_before),
        "global_quota_after": dict(quota_after),
        "global_quota_delta_contaminated": _quota_delta(
            quota_before,
            quota_after,
        ),
        "git_status": candidate["status"],
        "diff": diff,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "evaluation": dict(evaluation),
        "candidate_succeeded": bool(
            agent.get("succeeded") and diff and evaluation.get("valid")
        ),
    }


def _quota_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> int | None:
    if "total_used" not in before or "total_used" not in after:
        return None
    return int(after["total_used"]) - int(before["total_used"])


def _document(
    *,
    args: argparse.Namespace,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "repository": str(Path(args.repository).resolve()),
        "prompt_sha256": hashlib.sha256(_prompt().encode("utf-8")).hexdigest(),
        "models": list(args.models),
        "results": list(results),
    }


def run_bakeoff(args: argparse.Namespace) -> dict[str, Any]:
    api_key = str(os.getenv("LLM_API_KEY") or "").strip()
    base_url = str(os.getenv("LLM_BASE_URL") or "").strip()
    if not api_key or not base_url:
        raise RuntimeError("LLM_API_KEY and LLM_BASE_URL are required")

    ledger_path = Path(args.ledger).resolve()
    ledger = _load_json(ledger_path)
    results: list[dict[str, Any]] = []
    headers = {"Authorization": f"Bearer {api_key}"}

    with httpx.Client(headers=headers, timeout=30.0) as client:
        pricing_by_model = _pricing(client, base_url=base_url)
        for model in args.models:
            if _ledger_spend(ledger) >= float(ledger["operational_stop_usd"]):
                raise RuntimeError("operational API spend stop reached")
            pricing = pricing_by_model.get(model)
            if pricing is None:
                raise ValueError(f"proxy pricing is unavailable for model {model}")
            quota_before = optional_quota_snapshot(client, base_url=base_url)
            with tempfile.TemporaryDirectory(
                prefix=f"loreley-v15-kilo-{model.replace('/', '-')}-"
            ) as raw_dir:
                worktree = Path(raw_dir) / "candidate"
                source_commit = _clone_repository(
                    Path(args.repository),
                    worktree,
                )
                backend = _backend(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    timeout_seconds=int(args.agent_timeout),
                )
                agent, events = _run_agent(backend=backend, worktree=worktree)
                status, diff = _git_capture(worktree)
                source = (worktree / "solution.py").read_text(encoding="utf-8")
                evaluation = _evaluate_candidate(
                    source,
                    evaluator_path=Path(args.evaluator).resolve(),
                    runs=int(args.runs),
                    timeout_seconds=float(args.evaluation_timeout),
                )
            quota_after = optional_quota_snapshot(client, base_url=base_url)
            result = _model_result(
                candidate={
                    "model": model,
                    "source_commit": source_commit,
                    "agent": agent,
                    "status": status,
                    "diff": diff,
                    "source": source,
                    "evaluation": evaluation,
                },
                usage_events=events,
                accounting={
                    "pricing": pricing,
                    "group_multiplier": float(args.group_multiplier),
                    "quota_before": quota_before,
                    "quota_after": quota_after,
                },
            )
            results.append(result)
            if result["attributed_quota_upper_bound"] is None:
                _record_unattributed_ledger_entry(
                    ledger,
                    model=model,
                    evidence=f"{Path(args.output).name} Kilo usage unavailable",
                )
            else:
                _record_ledger_entry(
                    ledger,
                    model=model,
                    usage=result["usage"],
                    quota=float(result["attributed_quota_upper_bound"]),
                    evidence=f"{Path(args.output).name} source={result['source_sha256']}",
                    valid=bool(result["candidate_succeeded"]),
                )
                ledger["entries"][-1]["action"] = (
                    "Kilo circle-packing coding-agent bake-off"
                )
            _write_json(ledger_path, ledger)
            _write_json(
                Path(args.output).resolve(),
                _document(args=args, results=results),
            )

    return _document(args=args, results=results)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--agent-timeout", type=int, default=600)
    parser.add_argument("--evaluation-timeout", type=float, default=30.0)
    parser.add_argument("--group-multiplier", type=float, default=1.2)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_bakeoff(args)
    _write_json(Path(args.output).resolve(), result)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "models": len(result["results"]),
                "agent_succeeded": sum(
                    1 for item in result["results"] if item["agent"]["succeeded"]
                ),
                "valid": sum(
                    1 for item in result["results"] if item["candidate_succeeded"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
