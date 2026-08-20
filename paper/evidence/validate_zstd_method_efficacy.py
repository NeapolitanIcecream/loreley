#!/usr/bin/env python3
"""Recompute the paper's matched Zstandard statistics and mechanism counts."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any


SOURCE = Path(__file__).with_name("zstd_method_efficacy.json")
FORMAL_RECORDS = Path(__file__).with_name("zstd_formal_records.json")
FORMAL_TREATMENT = Path(__file__).with_name("zstd_formal_treatment.json")
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_811


def close(observed: float, expected: float, *, tolerance: float = 1e-6) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{observed!r} != {expected!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bca_mean(values: list[float], *, seed: int) -> tuple[float, float, float]:
    observed = statistics.fmean(values)
    rng = random.Random(seed)
    boot = [
        statistics.fmean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    less = sum(value < observed for value in boot)
    equal = sum(value == observed for value in boot)
    proportion = (less + 0.5 * equal) / BOOTSTRAP_SAMPLES
    epsilon = 0.5 / BOOTSTRAP_SAMPLES
    normal = NormalDist()
    z0 = normal.inv_cdf(min(1.0 - epsilon, max(epsilon, proportion)))
    jack = [
        statistics.fmean(values[:index] + values[index + 1 :])
        for index in range(len(values))
    ]
    jack_mean = statistics.fmean(jack)
    deviations = [jack_mean - value for value in jack]
    denominator = 6.0 * sum(value * value for value in deviations) ** 1.5
    acceleration = (
        sum(value**3 for value in deviations) / denominator if denominator else 0.0
    )

    def adjusted(probability: float) -> float:
        z_value = normal.inv_cdf(probability)
        divisor = 1.0 - acceleration * (z0 + z_value)
        if divisor == 0:
            return probability
        return normal.cdf(z0 + (z0 + z_value) / divisor)

    return observed, quantile(boot, adjusted(0.025)), quantile(boot, adjusted(0.975))


def exact_sign_flip(values: list[float]) -> float:
    observed = abs(statistics.fmean(values))
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(
            statistics.fmean(sign * value for sign, value in zip(signs, values))
        )
        if statistic + 1e-15 >= observed:
            exceed += 1
    return exceed / (1 << len(values))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    result: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        result[name] = running
    return result


def contrast(
    values: dict[tuple[int, str], float], treatment: str, control: str, seed: int
) -> dict[str, Any]:
    differences = [
        math.log(values[(block, treatment)]) - math.log(values[(block, control)])
        for block in range(1, 8)
    ]
    point, lower, upper = bca_mean(differences, seed=seed)
    return {
        "ratio": math.exp(point),
        "percent": 100.0 * math.expm1(point),
        "log_bca": [lower, upper],
        "percent_bca": [100.0 * math.expm1(lower), 100.0 * math.expm1(upper)],
        "p": exact_sign_flip(differences),
        "wins": sum(value > 0 for value in differences),
        "loo_percent": [
            100.0
            * math.expm1(
                statistics.fmean(differences[:index] + differences[index + 1 :])
            )
            for index in range(7)
        ],
    }


def validation_eligible(row: dict[str, Any]) -> tuple[bool, float]:
    """Return eligibility and the score used by the frozen validation selector."""
    if row.get("is_root"):
        return True, 1.0
    metrics = row.get("metrics", {})
    score = float(metrics.get("compression_lower_95", 0.0))
    eligible = bool(
        row.get("passed") is True
        and score > 1.0
        and float(metrics.get("decompression_geomean", 0.0)) >= 0.995
        and float(metrics.get("worst_cell_speedup", 0.0)) >= 0.98
    )
    return eligible, score


def select_validation_winner(
    identities: list[str], evaluations: dict[str, Any]
) -> tuple[str, float]:
    eligible: list[tuple[float, int, int, str, str]] = []
    for identity in identities:
        row = evaluations[identity]
        passed, score = validation_eligible(row)
        if passed:
            eligible.append(
                (
                    -score,
                    0 if row.get("is_root") else 1,
                    int(row.get("diff_lines", 0)),
                    str(row.get("commit", "")),
                    identity,
                )
            )
    if not eligible:
        raise AssertionError("validation group has no eligible candidate")
    identity = min(eligible)[-1]
    _, score = validation_eligible(evaluations[identity])
    return identity, score


def validate_selection_and_endpoints(
    evidence: dict[str, Any], records: dict[str, Any]
) -> None:
    evaluations = records["validation_evaluations"]
    selection_groups = {
        (int(row["block"]), str(row["arm"]), int(row["checkpoint"])): row
        for row in records["selection_groups"]
    }
    holdout_selections = {
        (int(row["block"]), str(row["arm"]), int(row["checkpoint"])): row
        for row in records["holdout_selections"]
    }
    assert len(selection_groups) == len(holdout_selections) == 126
    assert selection_groups.keys() == holdout_selections.keys()

    for key, group in selection_groups.items():
        identities = [str(row["identity"]) for row in group["finalists"]]
        assert identities and len(identities) <= 10
        assert len(identities) == len(set(identities))
        assert all(identity in evaluations for identity in identities)
        winner, score = select_validation_winner(identities, evaluations)
        assert winner == str(group["top10_winner_identity"])
        assert str(evaluations[winner]["commit"]) == str(
            group["top10_winner_commit"]
        )
        close(score, float(group["top10_validation_selection_score"]), tolerance=1e-12)

        selected = holdout_selections[key]
        assert str(selected["winner_identity"]) == winner
        assert str(selected["winner_commit"]) == str(evaluations[winner]["commit"])
        measured = records["holdout_evaluations"][winner]
        assert bool(selected["holdout_passed"]) is bool(measured["passed"])
        measured_compression = float(measured["metrics"]["compression_geomean"])
        close(float(selected["heldout_performance_ratio"]), measured_compression)
        close(float(selected["conservative_performance_ratio"]), measured_compression)

    endpoints = {
        (int(row["block"]), str(row["arm"])): row
        for row in records["endpoint_records"]
    }
    assert len(endpoints) == 21
    for (block, arm), endpoint in endpoints.items():
        selected = holdout_selections[(block, arm, 48)]
        assert str(endpoint["winner_identity"]) == str(selected["winner_identity"])
        assert str(endpoint["winner_commit"]) == str(selected["winner_commit"])
        assert bool(endpoint["holdout_passed"]) is bool(selected["holdout_passed"])
        close(
            float(endpoint["conservative_compression_geomean"]),
            float(selected["conservative_performance_ratio"]),
        )
        measured = records["holdout_evaluations"][str(endpoint["winner_identity"])]
        for name, value in endpoint["metrics"].items():
            close(float(value), float(measured["metrics"][name]), tolerance=1e-12)

    arm_names = {
        "independent": "independent-root",
        "qd": "qd",
        "sequential": "sequential-champion",
    }
    useful_threshold = float(evidence["final_checkpoint"]["useful_threshold"])
    summaries = {int(row["jobs"]): row for row in evidence["checkpoint_summaries"]}
    for checkpoint in records["protocol"]["checkpoints"]:
        for field, arm in arm_names.items():
            values = [
                float(holdout_selections[(block, arm, checkpoint)]["conservative_performance_ratio"])
                for block in range(1, 8)
            ]
            close(
                statistics.median(values),
                float(summaries[checkpoint][field]["median_ratio"]),
                tolerance=1e-6,
            )
            assert sum(value >= useful_threshold for value in values) == int(
                summaries[checkpoint][field]["useful_blocks"]
            )

    final_blocks = {
        int(row["block"]): row for row in evidence["final_checkpoint"]["blocks"]
    }
    for block, summary in final_blocks.items():
        for field, arm in arm_names.items():
            close(
                float(summary[field]),
                float(holdout_selections[(block, arm, 48)]["conservative_performance_ratio"]),
                tolerance=1e-6,
            )


def validate_primary(evidence: dict[str, Any], records: dict[str, Any]) -> None:
    final = evidence["final_checkpoint"]
    blocks = final["blocks"]
    fields = {
        "Independent Root": "independent",
        "Loreley QD": "qd",
        "Sequential Champion": "sequential",
    }
    useful_threshold = float(final["useful_threshold"])
    for arm, field in fields.items():
        values = [float(row[field]) for row in blocks]
        recorded = final["arm_summaries"][arm]
        close(statistics.fmean(values), float(recorded["mean_ratio"]))
        close(statistics.median(values), float(recorded["median_ratio"]))
        close(min(values), float(recorded["min_ratio"]))
        close(max(values), float(recorded["max_ratio"]))
        assert sum(value >= useful_threshold for value in values) == int(
            recorded["useful_blocks"]
        )

    # Recompute inferential quantities from the full-precision public endpoint
    # records.  The compact arm-summary table above is intentionally rounded for
    # readability and is checked separately.
    values = {
        (int(row["block"]), str(row["arm"])): float(
            row["metrics"]["compression_geomean"]
        )
        for row in records["endpoint_records"]
    }
    computed = {
        "Sequential Champion": contrast(
            values, "qd", "sequential-champion", BOOTSTRAP_SEED
        ),
        "Independent Root": contrast(
            values, "qd", "independent-root", BOOTSTRAP_SEED + 1
        ),
    }
    adjusted = holm({name: row["p"] for name, row in computed.items()})
    for recorded in evidence["primary_contrasts"]:
        row = computed[recorded["control"]]
        close(row["ratio"], float(recorded["qd_over_control_ratio"]))
        close(row["percent"], float(recorded["percent_effect"]), tolerance=5e-5)
        for observed, expected in zip(row["log_bca"], recorded["mean_log_bca_95"]):
            close(observed, float(expected), tolerance=1e-6)
        close(row["p"], float(recorded["exact_p"]))
        close(adjusted[recorded["control"]], float(recorded["holm_p"]))
        assert row["wins"] == int(recorded["qd_block_wins"])

    sensitivity = evidence["secondary_analyses"][
        "leave_one_block_out_percent_effect_range"
    ]
    for name, row in computed.items():
        key = f"QD vs {name}"
        close(min(row["loo_percent"]), float(sensitivity[key][0]), tolerance=1e-6)
        close(max(row["loo_percent"]), float(sensitivity[key][1]), tolerance=1e-6)

    exploratory = contrast(
        values, "sequential-champion", "independent-root", BOOTSTRAP_SEED + 2
    )
    recorded_exploratory = evidence["secondary_analyses"][
        "exploratory_sequential_vs_independent_compression"
    ]
    close(exploratory["percent"], recorded_exploratory["percent_effect"], tolerance=1e-6)
    close(exploratory["p"], recorded_exploratory["exact_p_unadjusted"])


def validate_secondary_endpoints(
    evidence: dict[str, Any], records: dict[str, Any]
) -> None:
    values: dict[str, dict[tuple[int, str], float]] = {
        "combined": {},
        "decompression": {},
    }
    metric_names = {
        "combined": "throughput_geomean",
        "decompression": "decompression_geomean",
    }
    for row in records["endpoint_records"]:
        key = (int(row["block"]), str(row["arm"]))
        for endpoint, metric in metric_names.items():
            values[endpoint][key] = float(row["metrics"][metric])
    assert all(len(endpoint_values) == 21 for endpoint_values in values.values())

    presentation = {
        int(row["block"]): row for row in evidence["secondary_endpoint_blocks"]
    }
    arm_names = {
        "independent": "independent-root",
        "qd": "qd",
        "sequential": "sequential-champion",
    }
    for block, row in presentation.items():
        for endpoint in values:
            for field, arm in arm_names.items():
                close(
                    float(row[endpoint][field]),
                    values[endpoint][(block, arm)],
                    tolerance=5.1e-7,
                )

    analysis_keys = {
        "combined": "combined_throughput",
        "decompression": "decompression_throughput",
    }
    for endpoint, analysis_key in analysis_keys.items():
        recorded = {
            row["control"]: row
            for row in evidence["secondary_analyses"][analysis_key]
        }
        for offset, (control_name, control_arm) in enumerate(
            (
                ("Sequential Champion", "sequential-champion"),
                ("Independent Root", "independent-root"),
            )
        ):
            computed = contrast(
                values[endpoint], "qd", control_arm, BOOTSTRAP_SEED + offset
            )
            target = recorded[control_name]
            close(computed["percent"], target["percent_effect"], tolerance=1e-6)
            for observed, expected in zip(
                computed["percent_bca"], target["percent_bca_95"]
            ):
                close(observed, expected, tolerance=1e-6)
            close(computed["p"], target["exact_p_unadjusted"])
            assert computed["wins"] == target["qd_block_wins"]


def validate_top5(evidence: dict[str, Any], records: dict[str, Any]) -> None:
    rows = records["top5_posthoc_sensitivity"]["endpoint_records"]
    assert len(rows) == 21
    assert sum(row["no_top5_finalist_passed_validation"] for row in rows) == 2
    assert records["top5_posthoc_sensitivity"]["new_holdout_candidate_count"] == 5
    values = {
        (int(row["block"]), str(row["arm"])): float(row["compression_geomean"])
        for row in rows
    }
    computed = {
        "Sequential Champion": contrast(
            values, "qd", "sequential-champion", BOOTSTRAP_SEED
        ),
        "Independent Root": contrast(
            values, "qd", "independent-root", BOOTSTRAP_SEED + 1
        ),
    }
    adjusted = holm({name: row["p"] for name, row in computed.items()})
    summary = evidence["secondary_analyses"]["original_top5_width"]
    for target in summary["contrasts"]:
        row = computed[target["control"]]
        close(row["percent"], target["percent_effect"], tolerance=1e-6)
        for observed, expected in zip(row["percent_bca"], target["percent_bca_95"]):
            close(observed, expected, tolerance=1e-6)
        close(row["p"], target["exact_p"])
        close(adjusted[target["control"]], target["holm_p"])
        assert row["wins"] == target["qd_block_wins"]

    amended = {
        (int(row["block"]), str(row["arm"])): str(row["winner_commit"])
        for row in records["endpoint_records"]
    }
    changed = sum(
        str(row["winner_commit"]) != amended[(int(row["block"]), str(row["arm"]))]
        for row in rows
    )
    assert changed == summary["endpoint_outputs_changed"] == 8


def validate_mechanism(evidence: dict[str, Any], records: dict[str, Any]) -> None:
    winner_parent_count = 0
    winner_combined_count = 0
    node_categories: Counter[str] = Counter()
    total_nodes = 0
    delayed_lags: list[int] = []
    delayed_exceeded = 0
    delayed_lineages = 0
    delayed_lineages_exceeded = 0
    for lineage in records["lineage_dependency_records"]:
        nodes = {row["commit"]: row for row in lineage["nodes"]}
        assert all(
            bool(node["one_incumbent_rule_same_qd_stream_would_not_retain"])
            is bool(node["was_nonchampion_at_admission"])
            for node in nodes.values()
        )
        total_nodes += len(nodes)
        incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
        outgoing: dict[str, set[str]] = defaultdict(set)
        for edge in lineage["edges"]:
            incoming[edge["to"]].append((edge["from"], edge["kind"]))
            outgoing[edge["from"]].add(edge["kind"])

        def ancestors(kinds: set[str]) -> set[str]:
            result: set[str] = set()
            stack = [lineage["winner_commit"]]
            while stack:
                child = stack.pop()
                for source, kind in incoming.get(child, []):
                    if kind in kinds and source not in result:
                        result.add(source)
                        stack.append(source)
            return result

        parent = ancestors({"primary_parent"})
        combined = ancestors({"primary_parent", "inspiration"})
        has_nonincumbent_parent = any(
            commit in nodes and nodes[commit]["was_nonchampion_at_admission"]
            for commit in parent
        )
        winner_parent_count += has_nonincumbent_parent
        winner_combined_count += any(
            commit in nodes and nodes[commit]["was_nonchampion_at_admission"]
            for commit in combined
        )
        if has_nonincumbent_parent:
            delayed_lineages += 1
            winner = nodes[lineage["winner_commit"]]
            lineage_exceeded = False
            for commit in parent:
                node = nodes.get(commit)
                if not node or not node["was_nonchampion_at_admission"]:
                    continue
                delayed_lags.append(int(winner["logical_job"]) - int(node["logical_job"]))
                if float(winner["training_score"]) > float(
                    node["training_champion_score_at_admission"]
                ):
                    delayed_exceeded += 1
                    lineage_exceeded = True
            delayed_lineages_exceeded += lineage_exceeded
        for commit, node in nodes.items():
            if not node["was_nonchampion_at_admission"]:
                continue
            kinds = outgoing.get(commit, set())
            if kinds == {"primary_parent"}:
                node_categories["parent_only"] += 1
            elif kinds == {"inspiration"}:
                node_categories["inspiration_only"] += 1
            elif kinds == {"primary_parent", "inspiration"}:
                node_categories["both"] += 1

    mechanism = evidence["qd_mechanism"]
    assert total_nodes == mechanism["winner_lineage_nodes"] == 67
    assert winner_parent_count == mechanism[
        "final_winners_with_retained_nonchampion_primary_parent_ancestor"
    ] == 4
    assert winner_combined_count == mechanism[
        "final_winner_parent_or_inspiration_graphs_with_retained_nonchampion"
    ] == 6
    recorded = mechanism["nonchampion_node_use_by_edge_kind"]
    for name in ("parent_only", "inspiration_only", "both"):
        assert node_categories[name] == recorded[name]
    assert sum(node_categories.values()) == mechanism["nonchampion_nodes_later_used"] == 49
    delayed = mechanism["delayed_primary_branch_analysis"]
    assert delayed_lineages == delayed["primary_lineages_with_nonincumbent_ancestor"] == 4
    assert len(delayed_lags) == delayed["nonincumbent_primary_ancestors"] == 15
    assert [min(delayed_lags), max(delayed_lags)] == delayed[
        "logical_job_ordinal_gap_range"
    ] == [3, 42]
    assert delayed_exceeded == delayed[
        "ancestors_whose_final_descendant_exceeded_admission_incumbent"
    ] == 7
    assert delayed_lineages_exceeded == delayed[
        "lineages_with_at_least_one_such_ancestor"
    ] == 3


def validate_treatment(evidence: dict[str, Any], records: dict[str, Any]) -> None:
    treatment = json.loads(FORMAL_TREATMENT.read_text(encoding="utf-8"))
    assert treatment["schema_version"] == 1
    assert treatment["source"]["loreley_git_head"] == records["protocol"][
        "loreley_git_head"
    ]
    assert treatment["source"]["loreley_source_fingerprint_sha256"] == records[
        "protocol"
    ]["loreley_source_fingerprint"]
    assert treatment["target"]["experiment_root_commit"] == records["protocol"][
        "root_commit"
    ]
    assert treatment["online_qd"]["warmup"]["jobs"] == records["protocol"][
        "warmup_jobs_counted_in_budget"
    ]
    assert treatment["online_qd"]["base_sampler_seed"] == records["protocol"][
        "sampler_seeds"
    ]["qd"]
    assert treatment["online_qd"]["seed_derivation"] == records["protocol"][
        "sampler_seed_derivation"
    ]
    assert evidence["protocol"]["sampler_seed_derivation"] == records["protocol"][
        "sampler_seed_derivation"
    ]
    treatment_cfg = treatment["online_qd"]
    record_cfg = records["protocol"]["qd_online_configuration"]
    expected = {
        "pca_whiten": treatment_cfg["descriptor"]["pca_whiten"],
        "pca_random_state": treatment_cfg["descriptor"]["pca_random_state"],
        "feature_clip": treatment_cfg["descriptor"]["feature_clip"],
        "feature_clip_standard_deviations": treatment_cfg["descriptor"][
            "clip_radius_standard_deviations"
        ],
        "pca_history_size": treatment_cfg["descriptor"]["history_size"],
        "sampler_neighbor_radius": treatment_cfg["inspiration_sampler"][
            "initial_neighbor_radius"
        ],
        "sampler_neighbor_max_radius": treatment_cfg["inspiration_sampler"][
            "maximum_neighbor_radius"
        ],
        "sampler_fallback_sample_size": treatment_cfg["inspiration_sampler"][
            "fallback_sample_size"
        ],
        "sampler_max_resample_attempts": treatment_cfg["inspiration_sampler"][
            "maximum_resample_attempts"
        ],
        "sampler_recipe_cooldown_jobs": treatment_cfg["inspiration_sampler"][
            "recipe_cooldown_jobs"
        ],
        "scheduling_batch_size": treatment_cfg["scheduling_batch_size"],
        "same_batch_snapshot": True,
    }
    for key, value in expected.items():
        assert record_cfg[key] == value
    assert sha256_file(FORMAL_TREATMENT) == evidence["source_artifact_sha256"][
        "formal_treatment"
    ]


def main() -> None:
    evidence = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = json.loads(FORMAL_RECORDS.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == records["schema_version"] == 3
    protocol = evidence["protocol"]
    assert protocol["blocks"] == 7
    assert protocol["jobs_per_arm_per_block"] == 48
    assert protocol["total_candidate_jobs"] == 7 * 3 * 48 == 1008
    assert len(records["selection_groups"]) == 126
    assert len(records["validation_evaluations"]) == 439
    assert len(records["holdout_evaluations"]) == 56

    # Most source artifacts remain private and are represented by recorded
    # digests only.  The sanitized formal record is included, so its byte hash
    # can be verified directly.  Cross-check matching recorded digests without
    # claiming byte verification of absent private files.
    for value in evidence["source_artifact_sha256"].values():
        assert len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )
    assert sha256_file(FORMAL_RECORDS) == evidence["source_artifact_sha256"][
        "sanitized_formal_records"
    ]
    source_links = {
        "formal_manifest": "manifest.json",
        "training_finalists": "training-finalists.json",
        "validation_winners": "validation-winners.json",
        "holdout_results": "holdout-results.json",
        "stepping_stone_lineages": "stepping-stone-lineages.json",
        "top5_posthoc_sensitivity": "top5-sensitivity-evaluations.json",
    }
    for summary_name, record_name in source_links.items():
        assert evidence["source_artifact_sha256"][summary_name] == records[
            "source_sha256"
        ][record_name]

    validate_selection_and_endpoints(evidence, records)
    validate_primary(evidence, records)
    validate_secondary_endpoints(evidence, records)
    validate_top5(evidence, records)
    validate_mechanism(evidence, records)
    validate_treatment(evidence, records)

    resources = evidence["resources"]
    fields = ("Independent Root", "Loreley QD", "Sequential Champion")
    assert sum(int(resources[arm]["jobs"]) for arm in fields) == 1008
    assert 1008 - sum(int(resources[arm]["unsuccessful"]) for arm in fields) == 948
    attributable = sum(
        float(resources[arm]["generation_usd"])
        + float(resources[arm]["embedding_usd"])
        for arm in fields
    )
    close(attributable, float(resources["total_attributable_usd"]), tolerance=5e-7)

    print(
        "Zstandard evidence: 126 validation selections, holdout mappings, "
        "checkpoint summaries, full-precision primary/secondary statistics, "
        "Top-5 sensitivity, public lineage and delayed-branch counts, "
        "formal treatment, included-record hash, "
        "recorded external-digest consistency, and resources verified"
    )


if __name__ == "__main__":
    main()
