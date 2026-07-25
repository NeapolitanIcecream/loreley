"""Run a bounded one-shot model calibration for the v15 validation sprint.

The script sends the same circle-packing task to each OpenAI-compatible model,
evaluates returned ``solution.py`` files in a network-disabled Docker container,
and updates the sprint resource ledger from observed response usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_MODELS = (
    "deepseek-v4-flash",
    "gpt-5.4-mini",
    "gpt-5.4",
    "claude-sonnet-4-6",
)
DEFAULT_LEDGER = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "research"
    / "artifacts"
    / "2026-07-26-v15-validation"
    / "resource-ledger.json"
)
DEFAULT_OUTPUT = DEFAULT_LEDGER.with_name("model-calibration.json")
DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1] / "examples" / "circle-packing" / "solution.py"
)
DEFAULT_EVALUATOR = DEFAULT_SOURCE.parent / "scripts" / "local_eval.py"
PYTHON_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_source(text: str) -> str:
    """Extract the returned Python file and require the public task function."""

    raw = str(text or "").strip()
    fenced = [
        block.strip()
        for block in PYTHON_FENCE.findall(raw)
        if "def pack_circles" in block
    ]
    source = fenced[0] if fenced else raw
    if "def pack_circles" not in source:
        raise ValueError("model response does not define pack_circles")
    return source.rstrip() + "\n"


def attributed_quota(
    *,
    input_tokens: int,
    output_tokens: int,
    model_ratio: float,
    completion_ratio: float,
    group_multiplier: float,
) -> float:
    """Return conservative New API quota from observed response tokens."""

    weighted_tokens = max(0, int(input_tokens)) + (
        max(0, int(output_tokens)) * max(0.0, float(completion_ratio))
    )
    return (
        weighted_tokens
        * max(0.0, float(model_ratio))
        * max(0.0, float(group_multiplier))
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _proxy_root(base_url: str) -> str:
    normalized = str(base_url or "").rstrip("/")
    return normalized[:-3] if normalized.endswith("/v1") else normalized


def _chat_endpoint(base_url: str) -> str:
    return f"{str(base_url or '').rstrip('/')}/chat/completions"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for item in content:
        if isinstance(item, Mapping):
            value = item.get("text")
            if value is not None:
                parts.append(str(value))
        elif item is not None:
            parts.append(str(item))
    return "".join(parts)


def _response_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("model response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("model response choice is invalid")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("model response has no message")
    return _content_text(message.get("content"))


def _usage(payload: Mapping[str, Any]) -> dict[str, int]:
    usage = payload.get("usage")
    values = usage if isinstance(usage, Mapping) else {}
    input_tokens = int(values.get("prompt_tokens") or values.get("input_tokens") or 0)
    output_tokens = int(
        values.get("completion_tokens") or values.get("output_tokens") or 0
    )
    total_tokens = int(values.get("total_tokens") or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _quota_snapshot(client: httpx.Client, *, base_url: str) -> dict[str, Any]:
    response = client.get(f"{_proxy_root(base_url)}/api/usage/token/")
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("proxy token-usage response has no data object")
    return {
        "total_granted": int(data.get("total_granted") or 0),
        "total_used": int(data.get("total_used") or 0),
        "total_available": int(data.get("total_available") or 0),
        "unlimited_quota": bool(data.get("unlimited_quota")),
    }


def _pricing(client: httpx.Client, *, base_url: str) -> dict[str, dict[str, float]]:
    response = client.get(f"{_proxy_root(base_url)}/api/pricing")
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError("proxy pricing response has no data list")
    pricing: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("model_name") or "")
        if not name:
            continue
        pricing[name] = {
            "model_ratio": float(row.get("model_ratio") or 0.0),
            "completion_ratio": float(row.get("completion_ratio") or 0.0),
        }
    return pricing


def _prompt(source: str) -> str:
    return f"""\
Return only the complete revised solution.py, without Markdown commentary.

Task: improve this deterministic unit-square circle packing implementation.
The evaluator calls pack_circles(n) for n=1,2,5,26 and requires exactly n
positive, finite, non-overlapping circles fully inside [0,1]^2. For n=26,
maximize the sum of radii while keeping a repeated-call p50 comfortably below
250 ms on CPU. Use only the Python standard library. Do not use randomness,
filesystem access, network access, subprocesses, or import-time side effects.
The implementation must work for every positive integer n and raise ValueError
for n <= 0.

Current solution.py:

{source}
"""


def _request_model(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[dict[str, Any], float, str]:
    started = time.perf_counter()
    response = client.post(
        _chat_endpoint(base_url),
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a numerical Python coding agent. Follow the "
                        "requested output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(max_tokens),
        },
    )
    duration = time.perf_counter() - started
    request_id = str(
        response.headers.get("x-oneapi-request-id")
        or response.headers.get("x-request-id")
        or ""
    )
    response.raise_for_status()
    return response.json(), duration, request_id


def _evaluate_candidate(
    source: str,
    *,
    evaluator_path: Path,
    runs: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="loreley-v15-calibration-") as raw_dir:
        candidate_dir = Path(raw_dir)
        (candidate_dir / "solution.py").write_text(source, encoding="utf-8")
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "64",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--volume",
            f"{candidate_dir.resolve()}:/candidate:ro",
            "--volume",
            f"{evaluator_path.resolve()}:/runner/local_eval.py:ro",
            "python:3.12-slim",
            "python",
            "/runner/local_eval.py",
            "--repo-root",
            "/candidate",
            "--runs",
            str(int(runs)),
            "--json",
        ]
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "valid": False,
                "error": f"evaluation timed out after {timeout_seconds:.1f}s",
                "stdout": str(exc.stdout or "")[-2000:],
                "stderr": str(exc.stderr or "")[-2000:],
            }
    if result.returncode != 0:
        return {
            "valid": False,
            "error": f"evaluation exited with code {result.returncode}",
            "stdout": (result.stdout or "")[-2000:],
            "stderr": (result.stderr or "")[-2000:],
        }
    payload = json.loads(result.stdout)
    deterministic = bool(payload.get("repeated_runs", {}).get("deterministic"))
    return {
        "valid": deterministic,
        "error": None if deterministic else "candidate is nondeterministic",
        "metrics": payload,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ledger_spend(ledger: Mapping[str, Any]) -> float:
    totals = ledger.get("totals")
    if not isinstance(totals, list):
        return 0.0
    for item in totals:
        if isinstance(item, Mapping) and item.get("unit") == "USD":
            return float(item.get("amount") or 0.0)
    return 0.0


def _record_ledger_entry(
    ledger: dict[str, Any],
    *,
    model: str,
    usage: Mapping[str, int],
    quota: float,
    evidence: str,
    valid: bool,
) -> None:
    cost_usd = float(quota) / float(ledger["quota_points_per_usd"])
    ledger.setdefault("entries", []).append(
        {
            "time": _utc_now(),
            "action": "one-shot circle-packing model calibration",
            "resource": f"LLM proxy model={model}",
            "amount": cost_usd,
            "unit": "USD",
            "status": "observed_tokens_conservative_proxy_price",
            "input_tokens": int(usage["input_tokens"]),
            "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
            "output_tokens": int(usage["output_tokens"]),
            "quota_points": quota,
            "evidence": evidence,
            "decision_changed": bool(valid),
        }
    )
    for total in ledger.get("totals", []):
        if total.get("unit") == "USD":
            total["amount"] = round(float(total.get("amount") or 0.0) + cost_usd, 8)
        elif total.get("unit") == "quota_points":
            total["amount"] = round(float(total.get("amount") or 0.0) + quota, 3)


def _record_unattributed_ledger_entry(
    ledger: dict[str, Any],
    *,
    model: str,
    evidence: str,
) -> None:
    ledger.setdefault("entries", []).append(
        {
            "time": _utc_now(),
            "action": "one-shot circle-packing model calibration",
            "resource": f"LLM proxy model={model}",
            "amount": None,
            "unit": "USD",
            "status": "unattributed_provider_failure",
            "evidence": evidence,
            "decision_changed": True,
        }
    )


def _result_document(
    *,
    args: argparse.Namespace,
    prompt: str,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "source": str(Path(args.source).resolve()),
        "source_commit": str(args.source_commit),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "evaluator": str(Path(args.evaluator).resolve()),
        "models": list(args.models),
        "results": list(results),
    }


def _calibration_result(
    *,
    observation: Mapping[str, Any],
    payload: Mapping[str, Any],
    accounting: Mapping[str, Any],
) -> dict[str, Any]:
    source = str(observation["source"])
    pricing = accounting["pricing"]
    group_multiplier = float(accounting["group_multiplier"])
    quota_before = accounting["quota_before"]
    quota_after = accounting["quota_after"]
    usage = _usage(payload)
    quota = attributed_quota(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        model_ratio=float(pricing["model_ratio"]),
        completion_ratio=float(pricing["completion_ratio"]),
        group_multiplier=group_multiplier,
    )
    return {
        "model": observation["model"],
        "request_id": observation["request_id"],
        "duration_seconds": observation["duration_seconds"],
        "usage": usage,
        "pricing": {
            **pricing,
            "group_multiplier_upper_bound": group_multiplier,
        },
        "attributed_quota_upper_bound": quota,
        "attributed_cost_usd_upper_bound": quota / 500000.0,
        "global_quota_before": dict(quota_before),
        "global_quota_after": dict(quota_after),
        "global_quota_delta_contaminated": (
            int(quota_after["total_used"]) - int(quota_before["total_used"])
        ),
        "response_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source": source,
        "evaluation": dict(observation["evaluation"]),
    }


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    api_key = str(os.getenv("LLM_API_KEY") or "").strip()
    base_url = str(os.getenv("LLM_BASE_URL") or "").strip()
    if not api_key or not base_url:
        raise RuntimeError("LLM_API_KEY and LLM_BASE_URL are required")

    source = Path(args.source).resolve().read_text(encoding="utf-8")
    prompt = _prompt(source)
    ledger_path = Path(args.ledger).resolve()
    ledger = _load_json(ledger_path)
    results: list[dict[str, Any]] = []

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(headers=headers, timeout=float(args.request_timeout)) as client:
        pricing_by_model = _pricing(client, base_url=base_url)
        for model in args.models:
            if _ledger_spend(ledger) >= float(ledger["operational_stop_usd"]):
                raise RuntimeError("operational API spend stop reached")
            pricing = pricing_by_model.get(model)
            if pricing is None:
                raise ValueError(f"proxy pricing is unavailable for model {model}")
            quota_before = _quota_snapshot(client, base_url=base_url)
            try:
                payload, duration, request_id = _request_model(
                    client,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    max_tokens=int(args.max_tokens),
                )
            except (httpx.HTTPError, ValueError) as exc:
                quota_after = _quota_snapshot(client, base_url=base_url)
                results.append(
                    {
                        "model": model,
                        "error": str(exc)[:2000],
                        "global_quota_before": quota_before,
                        "global_quota_after": quota_after,
                        "global_quota_delta_contaminated": (
                            int(quota_after["total_used"])
                            - int(quota_before["total_used"])
                        ),
                    }
                )
                _record_unattributed_ledger_entry(
                    ledger,
                    model=model,
                    evidence=f"{Path(args.output).name} provider request failed",
                )
                _write_json(ledger_path, ledger)
                _write_json(
                    Path(args.output).resolve(),
                    _result_document(args=args, prompt=prompt, results=results),
                )
                continue
            response_text = _response_content(payload)
            try:
                candidate_source = extract_python_source(response_text)
                evaluation = _evaluate_candidate(
                    candidate_source,
                    evaluator_path=Path(args.evaluator).resolve(),
                    runs=int(args.runs),
                    timeout_seconds=float(args.evaluation_timeout),
                )
            except (ValueError, json.JSONDecodeError) as exc:
                candidate_source = response_text.rstrip() + "\n"
                evaluation = {"valid": False, "error": str(exc)}
            quota_after = _quota_snapshot(client, base_url=base_url)
            result = _calibration_result(
                observation={
                    "model": model,
                    "duration_seconds": duration,
                    "request_id": request_id,
                    "source": candidate_source,
                    "evaluation": evaluation,
                },
                payload=payload,
                accounting={
                    "pricing": pricing,
                    "group_multiplier": float(args.group_multiplier),
                    "quota_before": quota_before,
                    "quota_after": quota_after,
                },
            )
            results.append(result)
            _record_ledger_entry(
                ledger,
                model=model,
                usage=result["usage"],
                quota=float(result["attributed_quota_upper_bound"]),
                evidence=f"{Path(args.output).name} request_id={request_id}",
                valid=bool(evaluation.get("valid")),
            )
            _write_json(ledger_path, ledger)
            _write_json(
                Path(args.output).resolve(),
                _result_document(args=args, prompt=prompt, results=results),
            )

    return _result_document(args=args, prompt=prompt, results=results)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-commit", default="6dab191")
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--evaluation-timeout", type=float, default=30.0)
    parser.add_argument("--group-multiplier", type=float, default=1.2)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_calibration(args)
    _write_json(Path(args.output).resolve(), result)
    valid = sum(
        1 for item in result["results"] if item.get("evaluation", {}).get("valid")
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "models": len(result["results"]),
                "valid": valid,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
