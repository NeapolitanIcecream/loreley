#!/usr/bin/env python3
"""Build the sanitized, machine-readable record for the formal Zstandard study.

The source files are frozen experiment artifacts copied from the private run
host.  This script deliberately excludes filesystem paths, prompts, candidate
source, and hidden corpus contents while retaining enough structure to replay
finalist selection, endpoint statistics, and the parent/inspiration audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SOURCE_NAMES = (
    "manifest.json",
    "zstd-top10-premeasurement-amendment.json",
    "training-finalists.json",
    "validation-winners.json",
    "holdout-results.json",
    "stepping-stone-lineages.json",
    "top5-sensitivity-evaluations.json",
)

EXPECTED_SOURCE_SHA256 = {
    "manifest.json": "4373301d52b784ceacabdad8059b07216404181b27d807f3dbbde03b8e66e507",
    "zstd-top10-premeasurement-amendment.json": "7ab1f58620958c77aa389189dede89d708cca077fc149389d6aface80e588e3d",
    "training-finalists.json": "bd45538dc77e5366ac1468759fdfaf5f7712cdf10ff80132b81229b768f1840f",
    "validation-winners.json": "17ebb091eb6b1bde6cdb2cde61e49bb2379bf790a9da5254d308f9d10d6a7968",
    "holdout-results.json": "fafe019287b9ddfd27eedb52c068977f09e6e984fe195ae1f148b9f0bd89baa8",
    "stepping-stone-lineages.json": "8e3b87ffba50f908af64dbf363e55ccbc8b2b62b6205f305da3d43de8657b12c",
    "top5-sensitivity-evaluations.json": "6580e2f58a3b2ab6b86ca589b9c6eb176dc22020a8eda1ac2bbe30e349ab31a3",
}

METRIC_NAMES = (
    "compression_geomean",
    "compression_lower_95",
    "decompression_geomean",
    "decompression_lower_95",
    "throughput_geomean",
    "worst_cell_speedup",
    "max_compressed_size_ratio",
    "peak_rss_delta_mib",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), Mapping) else {}
    return {
        "identity": str(raw["identity"]),
        "commit": str(raw["commit"]),
        "is_root": bool(raw.get("is_root")),
        "passed": raw.get("passed") is True,
        "diff_lines": int(raw.get("diff_lines") or 0),
        "metrics": {
            name: float(metrics[name])
            for name in METRIC_NAMES
            if metrics.get(name) is not None
        },
    }


def validation_eligible(raw: Mapping[str, Any]) -> tuple[bool, float]:
    if raw.get("is_root"):
        return True, 1.0
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), Mapping) else {}
    score = float(metrics.get("compression_lower_95") or 0.0)
    eligible = bool(
        raw.get("passed") is True
        and score > 1.0
        and float(metrics.get("decompression_geomean") or 0.0) >= 0.995
        and float(metrics.get("worst_cell_speedup") or 0.0) >= 0.98
    )
    return eligible, score


def select_validation_winner(
    identities: Sequence[str], evaluations: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    eligible: list[tuple[float, int, int, str, str]] = []
    for identity in identities:
        row = evaluations[identity]
        passed, score = validation_eligible(row)
        if passed:
            eligible.append(
                (
                    -score,
                    0 if row.get("is_root") else 1,
                    int(row.get("diff_lines") or 0),
                    str(row.get("commit") or ""),
                    identity,
                )
            )
    if not eligible:
        return None
    identity = min(eligible)[-1]
    passed, score = validation_eligible(evaluations[identity])
    assert passed
    return {
        "identity": identity,
        "commit": str(evaluations[identity]["commit"]),
        "validation_selection_score": score,
        "is_root": bool(evaluations[identity].get("is_root")),
    }


def unique_slot_rows(raw_slots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, raw in enumerate(raw_slots, start=1):
        identity = str(raw["evaluator_identity"])
        if identity in seen:
            continue
        seen.add(identity)
        result.append(
            {
                "training_rank": rank,
                "identity": identity,
                "commit": str(raw["commit"]),
                "logical_job": int(raw.get("logical_job") or 0),
                "training_score": float(raw["training_score"]),
                "is_root": bool(raw.get("is_root")),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("zstd_formal_records.json"),
    )
    arguments = parser.parse_args()
    source_dir = arguments.source_dir.resolve(strict=True)
    paths = {name: (source_dir / name).resolve(strict=True) for name in SOURCE_NAMES}
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_SOURCE_SHA256[name], "observed": observed}
        for name, observed in observed_hashes.items()
        if observed != EXPECTED_SOURCE_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(
            "frozen formal-study source hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    manifest = read_json(paths["manifest.json"])
    amendment = read_json(paths["zstd-top10-premeasurement-amendment.json"])
    finalists = read_json(paths["training-finalists.json"])
    validation = read_json(paths["validation-winners.json"])
    holdout = read_json(paths["holdout-results.json"])
    lineages = read_json(paths["stepping-stone-lineages.json"])
    top5_measurements = read_json(paths["top5-sensitivity-evaluations.json"])

    target = dict(manifest["targets"]["zstandard"])
    validation_evaluations = {
        identity: evaluation_summary(raw)
        for identity, raw in validation["evaluations"]["zstandard"].items()
    }
    official_holdout_evaluations = {
        identity: evaluation_summary(raw)
        for identity, raw in holdout["evaluations"]["zstandard"].items()
    }
    extra_holdout_evaluations = {
        identity: evaluation_summary(raw)
        for identity, raw in top5_measurements[
            "new_fixed_candidate_evaluations"
        ].items()
    }
    all_holdout_evaluations = official_holdout_evaluations | extra_holdout_evaluations

    validation_index = {
        (int(row["block"]), str(row["arm"]), int(row["checkpoint"])): row
        for row in validation["selections"]
        if row["target"] == "zstandard"
    }
    holdout_index = {
        (int(row["block"]), str(row["arm"]), int(row["checkpoint"])): row
        for row in holdout["selections"]
        if row["target"] == "zstandard"
    }

    selection_groups: list[dict[str, Any]] = []
    top5_endpoint: list[dict[str, Any]] = []
    for raw in finalists["selections"]:
        if raw["target"] != "zstandard":
            continue
        key = (int(raw["block"]), str(raw["arm"]), int(raw["checkpoint"]))
        finalist_rows = unique_slot_rows(raw["slots"])
        identities = [row["identity"] for row in finalist_rows]
        top10_winner = validation_index[key]
        selection_groups.append(
            {
                "block": key[0],
                "arm": key[1],
                "checkpoint": key[2],
                "finalists": finalist_rows,
                "top10_winner_identity": str(top10_winner["winner_identity"]),
                "top10_winner_commit": str(top10_winner["winner_commit"]),
                "top10_validation_selection_score": float(
                    top10_winner["validation_selection_score"]
                ),
            }
        )
        if key[2] != 48:
            continue
        original = select_validation_winner(identities[:5], validation_evaluations)
        if original is None:
            top5_endpoint.append(
                {
                    "block": key[0],
                    "arm": key[1],
                    "winner_identity": "root-fallback",
                    "winner_commit": str(target["root_commit"]),
                    "no_top5_finalist_passed_validation": True,
                    "holdout_passed": True,
                    "compression_geomean": 1.0,
                    "decompression_geomean": 1.0,
                    "throughput_geomean": 1.0,
                }
            )
            continue
        measured = all_holdout_evaluations[original["identity"]]
        metrics = measured["metrics"]
        conservative = (
            float(metrics["compression_geomean"])
            if measured["passed"]
            else 1.0
        )
        top5_endpoint.append(
            {
                "block": key[0],
                "arm": key[1],
                "winner_identity": original["identity"],
                "winner_commit": original["commit"],
                "no_top5_finalist_passed_validation": False,
                "holdout_passed": measured["passed"],
                "compression_geomean": conservative,
                "decompression_geomean": float(metrics.get("decompression_geomean", 0)),
                "throughput_geomean": float(metrics.get("throughput_geomean", 0)),
            }
        )

    if len(selection_groups) != 126 or len(top5_endpoint) != 21:
        raise RuntimeError("formal Zstandard selection record is incomplete")

    endpoint_records: list[dict[str, Any]] = []
    for key, row in sorted(holdout_index.items()):
        if key[2] != 48:
            continue
        measured = official_holdout_evaluations[str(row["winner_identity"])]
        endpoint_records.append(
            {
                "block": key[0],
                "arm": key[1],
                "winner_identity": str(row["winner_identity"]),
                "winner_commit": str(row["winner_commit"]),
                "holdout_passed": bool(row["holdout_passed"]),
                "conservative_compression_geomean": float(
                    row["conservative_performance_ratio"]
                ),
                "metrics": measured["metrics"],
            }
        )

    sanitized_lineages = []
    for raw in lineages["lineages"]:
        nodes = []
        for node in raw["nodes"]:
            clean = dict(node)
            legacy = clean.pop("sequential_champion_would_discard", None)
            clean["one_incumbent_rule_same_qd_stream_would_not_retain"] = bool(
                legacy
                if legacy is not None
                else clean.get("was_nonchampion_at_admission")
            )
            nodes.append(clean)
        sanitized_lineages.append(
            {
                "block": int(raw["block"]),
                "winner_commit": str(raw["winner_commit"]),
                "winner_is_root": bool(raw["winner_is_root"]),
                "nodes": nodes,
                "edges": list(raw["edges"]),
            }
        )

    payload = {
        "schema_version": 3,
        "target": "Zstandard",
        "record_scope": (
            "Sanitized frozen summaries sufficient to replay finalist selection, "
            "endpoint statistics, and parent/inspiration dependency counts; no "
            "candidate source, prompts, private paths, or hidden corpus contents."
        ),
        "source_sha256": {
            name: digest
            for name, digest in observed_hashes.items()
            if name != "zstd-top10-premeasurement-amendment.json"
        },
        "protocol": {
            "formal_manifest_sha256": sha256_file(paths["manifest.json"]),
            "loreley_git_head": str(manifest["loreley_source"]["git_head"]),
            "loreley_source_fingerprint": str(manifest["loreley_source"]["sha256"]),
            "root_commit": str(target["root_commit"]),
            "blocks": int(target["blocks"]),
            "jobs_per_arm": int(target["arms"]["qd"]["jobs"]),
            "checkpoints": list(map(int, target["checkpoints"])),
            "initialization": "root-only",
            "warmup_jobs_counted_in_budget": int(target["seed_population_per_island"]),
            "training_score_metric": str(target["training_score_metric"]),
            "primary_finalist_width": int(amendment["amended_width"]),
            "sensitivity_finalist_width": int(amendment["previous_width"]),
            "arm_order_by_block": list(target["arm_order_by_block"]),
            "within_block_launch_delays_seconds": [0, 120, 240],
            "sampler_seeds": {
                arm: int(raw["sampler_seed"])
                for arm, raw in target["arms"].items()
            },
            "sampler_seed_derivation": {
                "block_base_seed": "arm_seed + 100000 * (block - 1)",
                "qd_campaign_seed": "block_base_seed",
                "one_job_control_seed": "block_base_seed + (job_ordinal - 1)",
            },
            "native_generation_concurrency": {
                "qd": 4,
                "sequential-champion": 1,
                "independent-root": 4,
            },
            "qd_online_configuration": {
                "objectives": [
                    "compression_lower_95",
                    "decompression_lower_95",
                    "worst_cell_speedup",
                ],
                "embedding_dims": 1536,
                "embedding_model": "text-embedding-3-small",
                "repository_embedding_aggregation": "uniform mean of eligible file embeddings",
                "pca_dims": 3,
                "pca_whiten": True,
                "pca_random_state": 0,
                "feature_clip": True,
                "feature_clip_standard_deviations": 3.0,
                "coordinate_range": [0.0, 1.0],
                "pca_min_fit_samples": 4,
                "pca_history_size": 4096,
                "cells_per_dimension": 4,
                "grid_cells": 64,
                "pareto_capacity_per_cell": 8,
                "objective_epsilon": 0.003,
                "pca_refit_interval": 4,
                "pca_refit_alignment": True,
                "archive_rebuild_source": "records retained immediately before refit",
                "inspirations_per_job": 2,
                "sampler_neighbor_radius": 1,
                "sampler_neighbor_max_radius": 3,
                "sampler_fallback_sample_size": 8,
                "sampler_max_resample_attempts": 32,
                "sampler_recipe_cooldown_jobs": 64,
                "scheduling_batch_size": 4,
                "same_batch_snapshot": True,
                "islands": 1,
                "migration_interval_jobs": 0,
            },
        },
        "validation_evaluations": dict(sorted(validation_evaluations.items())),
        "selection_groups": sorted(
            selection_groups,
            key=lambda row: (row["checkpoint"], row["block"], row["arm"]),
        ),
        "holdout_evaluations": dict(sorted(official_holdout_evaluations.items())),
        "holdout_selections": sorted(
            [
                {
                    key: row[key]
                    for key in (
                        "block",
                        "arm",
                        "checkpoint",
                        "winner_identity",
                        "winner_commit",
                        "holdout_passed",
                        "heldout_performance_ratio",
                        "conservative_performance_ratio",
                    )
                }
                for row in holdout["selections"]
                if row["target"] == "zstandard"
            ],
            key=lambda row: (row["checkpoint"], row["block"], row["arm"]),
        ),
        "endpoint_records": endpoint_records,
        "top5_posthoc_sensitivity": {
            "status": "post-hoc fixed-candidate sensitivity; primary analysis remains amended Top-10",
            "new_holdout_candidate_count": len(extra_holdout_evaluations),
            "new_holdout_evaluations": dict(sorted(extra_holdout_evaluations.items())),
            "endpoint_records": sorted(
                top5_endpoint, key=lambda row: (row["block"], row["arm"])
            ),
        },
        "lineage_dependency_records": sorted(
            sanitized_lineages, key=lambda row: row["block"]
        ),
    }

    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "selection_groups": len(selection_groups),
                "endpoint_records": len(endpoint_records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
