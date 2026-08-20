#!/usr/bin/env python3
"""Rebuild the Python-campaign generation-cost audit from proxy events.

Run from the repository root:

    python paper/evidence/generate_python_generation_cost_audit.py --write

Without ``--write``, the script verifies that the checked-in JSON is current.
Costs are summed with ``Decimal`` from the event field
``calculated_cost_usd``.  The proxy records contain no provider-billed cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT = REPOSITORY / "paper/evidence/python_generation_cost_audit.json"

CAMPAIGNS = {
    "markdown-it-py": {
        "experiment": "mdit-v18-deepseek-seeded-qd",
        "events_root": Path(
            "output/markdown-it-manual-seed-deepseek-20260802-v18/campaign/jobs"
        ),
        "legacy_report": Path(
            "output/markdown-it-manual-seed-deepseek-20260802-v18/report/final-report.json"
        ),
        "expected": {
            "event_files": 56,
            "records": 3792,
            "completed": 3792,
            "observed_tokens": 215_349_501,
            "calculated_cost_usd": Decimal("2.0833022056"),
        },
    },
    "python-pathspec": {
        "experiment": "pathspec-deepseek-20260803-v1",
        "events_root": Path(
            "output/pathspec-deepseek-20260803-v1/campaign/jobs"
        ),
        "legacy_report": Path(
            "output/pathspec-deepseek-20260803-v1/report/final-report.json"
        ),
        "expected": {
            "event_files": 58,
            "records": 3977,
            "completed": 3855,
            "observed_tokens": 241_634_477,
            "calculated_cost_usd": Decimal("2.485592144"),
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY).as_posix()


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _read_events(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: event is not an object")
        records.append(value)
    return records


def _campaign_record(name: str, config: dict[str, Any]) -> dict[str, Any]:
    events_root = REPOSITORY / config["events_root"]
    event_paths = sorted(events_root.glob("*/proxy-events.jsonl"))
    if len(event_paths) != config["expected"]["event_files"]:
        raise ValueError(f"{name}: unexpected proxy-event file count")

    status_counts: Counter[str] = Counter()
    pricing_counts: Counter[str] = Counter()
    provider_claim_status_counts: Counter[str] = Counter()
    total_records = 0
    total_tokens = 0
    total_cost = Decimal("0")
    provider_cost_rows = 0
    files: list[dict[str, Any]] = []

    for path in event_paths:
        records = _read_events(path)
        file_cost = Decimal("0")
        file_tokens = 0
        file_statuses: Counter[str] = Counter()
        for record in records:
            status = str(record.get("status"))
            status_counts[status] += 1
            file_statuses[status] += 1
            pricing = record.get("pricing_version")
            pricing_counts["null" if pricing is None else str(pricing)] += 1
            claim_status = record.get("provider_cost_claim_status")
            provider_claim_status_counts[
                "null" if claim_status is None else str(claim_status)
            ] += 1
            if record.get("provider_cost_usd") is not None:
                provider_cost_rows += 1
            observed_tokens = record.get("observed_tokens")
            if observed_tokens is not None:
                file_tokens += int(observed_tokens)
            calculated = record.get("calculated_cost_usd")
            if calculated is not None:
                value = Decimal(str(calculated))
                file_cost += value
                if Decimal(str(record.get("cost_usd"))) != value:
                    raise ValueError(f"{path}: cost_usd differs from calculated_cost_usd")

        total_records += len(records)
        total_tokens += file_tokens
        total_cost += file_cost
        files.append(
            {
                "path": _relative(path),
                "sha256": _sha256(path),
                "records": len(records),
                "status_counts": dict(sorted(file_statuses.items())),
                "observed_tokens": file_tokens,
                "calculated_cost_usd": _decimal_text(file_cost),
            }
        )

    expected = config["expected"]
    observed = {
        "event_files": len(event_paths),
        "records": total_records,
        "completed": status_counts["completed"],
        "observed_tokens": total_tokens,
        "calculated_cost_usd": total_cost,
    }
    for key, expected_value in expected.items():
        if observed[key] != expected_value:
            raise ValueError(
                f"{name}: {key}={observed[key]!r}, expected {expected_value!r}"
            )
    if provider_cost_rows:
        raise ValueError(f"{name}: unexpectedly found provider-billed cost rows")

    legacy_path = REPOSITORY / config["legacy_report"]
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy_value = legacy.get("observed_cost_usd")

    return {
        "experiment": config["experiment"],
        "canonical_source": "campaign/jobs/*/proxy-events.jsonl",
        "aggregate": {
            "event_files": len(event_paths),
            "records": total_records,
            "status_counts": dict(sorted(status_counts.items())),
            "observed_tokens": total_tokens,
            "calculated_cost_usd": _decimal_text(total_cost),
            "pricing_version_counts": dict(sorted(pricing_counts.items())),
            "provider_cost_claim_status_counts": dict(
                sorted(provider_claim_status_counts.items())
            ),
            "provider_cost_rows": provider_cost_rows,
        },
        "legacy_report": {
            "path": _relative(legacy_path),
            "sha256": _sha256(legacy_path),
            "observed_cost_usd": legacy_value,
            "canonical_for_paper_cost": False,
        },
        "files": files,
    }


def build() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cost_semantics": {
            "paper_value": (
                "Sum of proxy calculated_cost_usd fields under the recorded "
                "pricing version; an estimate, not a provider invoice."
            ),
            "provider_billed_cost_available": False,
            "pathspec_legacy_field": (
                "The legacy final-report observed_cost_usd includes resource-ledger "
                "fallback reservations and is not an observed bill or the paper value."
            ),
        },
        "campaigns": {
            name: _campaign_record(name, config)
            for name, config in CAMPAIGNS.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write", action="store_true", help="replace the checked-in audit JSON"
    )
    args = parser.parse_args()

    rendered = json.dumps(build(), indent=2, sort_keys=False) + "\n"
    if args.write:
        OUTPUT.write_text(rendered, encoding="utf-8")
        return 0

    if not OUTPUT.is_file():
        raise SystemExit(f"missing {OUTPUT}; rerun with --write")
    if OUTPUT.read_text(encoding="utf-8") != rendered:
        raise SystemExit(f"{OUTPUT} is stale; rerun with --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
