#!/usr/bin/env python3
"""Generate the dynamics figures used by the four-page technical report."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QUALITY_TIME = ROOT / "technical_report" / "evidence" / "zstd_quality_time.json"
QUALITY_TIME_SUMMARY = (
    ROOT / "technical_report" / "evidence" / "zstd_quality_time_summary.json"
)
METHOD_EVIDENCE = ROOT / "paper" / "evidence" / "zstd_method_efficacy.json"
PATHSPEC_REPORT = (
    ROOT / "docs" / "research" / "2026-08-03-pathspec-deepseek-case-study.md"
)
OUT_DIR = Path(__file__).resolve().parent
PDF_METADATA = {"CreationDate": None}

COLORS = {
    "qd": "#2563EB",
    "sequential-champion": "#D97706",
    "independent-root": "#64748B",
}
LABELS = {
    "zh": {
        "qd": "Loreley QD",
        "sequential-champion": "顺序冠军",
        "independent-root": "从根独立",
    },
    "en": {
        "qd": "Loreley QD",
        "sequential-champion": "Sequential champion",
        "independent-root": "Independent root",
    },
}
TEXT = {
    "zh": {
        "threshold_title": "A  首次达到 +0.50%",
        "search_time": "搜索时间 (小时)",
        "blocks_reaching": "达到阈值的区组数",
        "block": "区组",
        "deadline_gain": "QD 相对顺序搜索的提升 (%)",
        "deadline_title": "B  QD 完成时的质量差",
        "iqr": "四分位区间",
        "median": "中位数",
        "candidate_jobs": "候选任务数",
        "heldout_gain": "留出集压缩吞吐提升 (%)",
        "retained_jobs": "保留 20 个任务",
        "lineage_title": "B  python-pathspec 谱系",
        "job_index": "任务序号",
        "training_gain": "训练吞吐提升 (%)",
    },
    "en": {
        "threshold_title": "A  Time to +0.50%",
        "search_time": "Search time (h)",
        "blocks_reaching": "Blocks reaching target",
        "block": "Block",
        "deadline_gain": "QD gain over sequential (%)",
        "deadline_title": "B  Quality at QD completion",
        "iqr": "IQR",
        "median": "Median",
        "candidate_jobs": "Candidate jobs",
        "heldout_gain": "Held-out throughput gain (%)",
        "retained_jobs": "Retained for 20 jobs",
        "lineage_title": "B  python-pathspec lineage",
        "job_index": "Job index",
        "training_gain": "Training throughput gain (%)",
    },
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def gain_percent(ratio: float) -> float:
    return 100.0 * (ratio - 1.0)


def style(language: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": (
                ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
                if language == "zh"
                else ["DejaVu Sans"]
            ),
            "font.size": 8.4,
            "axes.titlesize": 9.7,
            "axes.labelsize": 8.8,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.7,
            "axes.unicode_minus": False,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def indexed_rows(records: list[dict]) -> dict[tuple[int, str, int], dict]:
    return {
        (int(row["block"]), str(row["arm"]), int(row["checkpoint"])): row
        for row in records
    }


def first_threshold_times(
    index: dict[tuple[int, str, int], dict],
    *,
    arm: str,
    checkpoints: list[int],
    blocks: int,
    threshold: float,
) -> tuple[list[float], list[int]]:
    reached: list[float] = []
    censored: list[int] = []
    for block in range(1, blocks + 1):
        event = None
        for checkpoint in checkpoints:
            row = index[(block, arm, checkpoint)]
            if float(row["heldout_performance_ratio"]) >= threshold:
                event = float(row["elapsed_seconds"]) / 3600.0
                break
        if event is None:
            censored.append(block)
        else:
            reached.append(event)
    return sorted(reached), censored


def ecdf_step(
    event_times: list[float], *, x_end: float
) -> tuple[list[float], list[float]]:
    x = [0.0, *event_times, x_end]
    y = [0.0, *range(1, len(event_times) + 1)]
    y.append(len(event_times))
    return x, y


def make_quality_time_figure(language: str) -> None:
    labels = LABELS[language]
    text = TEXT[language]
    joined = load_json(QUALITY_TIME)
    summary = load_json(QUALITY_TIME_SUMMARY)
    records = joined["quality_time_records"]
    checkpoints = [int(v) for v in joined["timing"]["checkpoints"]]
    blocks = int(joined["timing"]["blocks"])
    index = indexed_rows(records)

    fig = plt.figure(figsize=(7.35, 2.72))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.20, 1.0], wspace=0.30)
    ax_threshold = fig.add_subplot(grid[0, 0])
    ax_deadline = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.84)

    threshold = 1.005
    x_end = 16.1
    for arm, linestyle in [
        ("independent-root", ":"),
        ("qd", "-"),
        ("sequential-champion", "-"),
    ]:
        events, censored = first_threshold_times(
            index,
            arm=arm,
            checkpoints=checkpoints,
            blocks=blocks,
            threshold=threshold,
        )
        x, y = ecdf_step(events, x_end=x_end)
        linewidth = 2.1 if arm == "qd" else 1.55
        alpha = 1.0 if arm != "independent-root" else 0.72
        ax_threshold.step(
            x,
            y,
            where="post",
            color=COLORS[arm],
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=f"{labels[arm]} ({len(events)}/7)",
        )
        if censored:
            final_rows = [index[(block, arm, checkpoints[-1])] for block in censored]
            censor_x = [float(row["elapsed_seconds"]) / 3600.0 for row in final_rows]
            censor_y = [len(events)] * len(censor_x)
            ax_threshold.scatter(
                censor_x,
                censor_y,
                marker="|",
                s=50,
                color=COLORS[arm],
                linewidth=1.4,
                zorder=5,
            )

    ax_threshold.set_title(text["threshold_title"], loc="left", fontweight="bold")
    ax_threshold.set_xlabel(text["search_time"])
    ax_threshold.set_ylabel(text["blocks_reaching"])
    ax_threshold.set_xlim(0, x_end)
    ax_threshold.set_ylim(0, 7.15)
    ax_threshold.set_yticks(range(8))
    ax_threshold.grid(color="#E2E8F0", linewidth=0.6)
    ax_threshold.spines[["top", "right"]].set_visible(False)
    ax_threshold.legend(loc="lower right", frameon=False, handlelength=2.1)

    deadline = summary["posthoc_qd_completion_deadline_comparison"]["rows"]
    deadline = sorted(deadline, key=lambda row: int(row["block"]), reverse=True)
    margins = np.array([float(row["qd_over_sequential_percent"]) for row in deadline])
    blocks_desc = [int(row["block"]) for row in deadline]
    positions = np.arange(len(deadline))
    ax_deadline.barh(
        positions,
        margins,
        height=0.58,
        color="#93C5FD",
        edgecolor=COLORS["qd"],
        linewidth=0.8,
    )
    ax_deadline.set_yticks(positions)
    ax_deadline.set_yticklabels([f"{text['block']} {block}" for block in blocks_desc])
    ax_deadline.set_xlim(0, 1.34)
    ax_deadline.set_xlabel(text["deadline_gain"])
    ax_deadline.set_title(text["deadline_title"], loc="left", fontweight="bold")
    ax_deadline.grid(axis="x", color="#E2E8F0", linewidth=0.6)
    ax_deadline.spines[["top", "right", "left"]].set_visible(False)
    ax_deadline.tick_params(axis="y", length=0)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"quality_time{'_en' if language == 'en' else ''}.{suffix}",
            bbox_inches="tight",
            metadata=PDF_METADATA if suffix == "pdf" else None,
        )
    plt.close(fig)


def load_pathspec_lineage() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read the generation/job/training rows from the checked case-study report."""
    text = PATHSPEC_REPORT.read_text(encoding="utf-8")
    rows = re.findall(
        r"\|\s*([0-4])\s*\|\s*(\d+)\s*\|\s*([0-9.]+)x\s*\|",
        text,
    )
    if len(rows) != 5:
        raise ValueError(f"Expected five Pathspec lineage rows, found {len(rows)}")
    generations = [f"G{generation}" for generation, _, _ in rows]
    jobs = np.array([int(job) for _, job, _ in rows])
    gains = np.array([100.0 * (float(ratio) - 1.0) for _, _, ratio in rows])
    return jobs, gains, generations


def make_job_dynamics_figure(language: str) -> None:
    text = TEXT[language]
    joined = load_json(QUALITY_TIME)
    records = joined["quality_time_records"]
    checkpoints = [int(v) for v in joined["timing"]["checkpoints"]]
    qd_rows: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        if row["arm"] == "qd":
            qd_rows[int(row["block"])].append(row)
    for rows in qd_rows.values():
        rows.sort(key=lambda row: int(row["checkpoint"]))

    fig = plt.figure(figsize=(7.35, 2.72))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.33)
    ax = fig.add_subplot(grid[0, 0])
    lineage_ax = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.84)

    per_block = []
    for block in range(1, 8):
        values = np.array(
            [
                gain_percent(float(row["heldout_performance_ratio"]))
                for row in qd_rows[block]
            ]
        )
        per_block.append(values)
    stacked = np.vstack(per_block)
    median_values = np.median(stacked, axis=0)
    lower = np.quantile(stacked, 0.25, axis=0)
    upper = np.quantile(stacked, 0.75, axis=0)
    ax.fill_between(
        checkpoints,
        lower,
        upper,
        color="#BFDBFE",
        alpha=0.65,
        linewidth=0,
        label=text["iqr"],
    )
    ax.plot(
        checkpoints,
        median_values,
        color="#1D4ED8",
        linewidth=2.1,
        marker="o",
        markersize=4.0,
        label=text["median"],
        zorder=4,
    )
    ax.axhline(0.5, color="#94A3B8", linewidth=0.8, linestyle=":")
    ax.text(
        48.5,
        0.515,
        "+0.50%",
        color="#64748B",
        fontsize=7.0,
        va="bottom",
        ha="right",
    )
    ax.set_title("A  Zstandard QD", loc="left", fontweight="bold")
    ax.set_xlabel(text["candidate_jobs"])
    ax.set_ylabel(text["heldout_gain"])
    ax.set_xticks(checkpoints)
    ax.set_xlim(7, 49)
    ax.set_ylim(0.0, 1.35)
    ax.grid(color="#E2E8F0", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    jobs, lineage_gains, generations = load_pathspec_lineage()
    lineage_ax.axvspan(18, 38, color="#FFF7ED", alpha=0.9, linewidth=0)
    lineage_ax.plot(
        jobs,
        lineage_gains,
        color="#7C3AED",
        linewidth=2.0,
        marker="o",
        markersize=5.2,
    )
    for job, value, generation in zip(jobs, lineage_gains, generations, strict=True):
        lineage_ax.text(
            job,
            value + 1.25,
            generation,
            ha="center",
            va="bottom",
            fontsize=7.1,
            color="#5B21B6",
            fontweight="bold",
        )
    lineage_ax.text(
        28,
        27.7,
        text["retained_jobs"],
        ha="center",
        va="top",
        fontsize=7.2,
        color="#B45309",
    )
    lineage_ax.set_title(text["lineage_title"], loc="left", fontweight="bold")
    lineage_ax.set_xlabel(text["job_index"])
    lineage_ax.set_ylabel(text["training_gain"])
    lineage_ax.set_xticks(jobs)
    lineage_ax.set_xlim(4, 40)
    lineage_ax.set_ylim(-3.0, 29.0)
    lineage_ax.grid(color="#E2E8F0", linewidth=0.6)
    lineage_ax.spines[["top", "right"]].set_visible(False)

    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"job_dynamics{'_en' if language == 'en' else ''}.{suffix}",
            bbox_inches="tight",
            metadata=PDF_METADATA if suffix == "pdf" else None,
        )
    plt.close(fig)


def validate_expected_metrics() -> None:
    summary = load_json(QUALITY_TIME_SUMMARY)
    ratios = summary["final_sequential_over_qd_time_ratio"]
    assert abs(float(ratios["geometric_mean"]) - 2.7797608563) < 1e-9
    assert abs(float(ratios["median"]) - 3.1219086734) < 1e-9
    deadline = summary["posthoc_qd_completion_deadline_comparison"]
    assert int(deadline["qd_higher_quality_blocks"]) == 7
    method = load_json(METHOD_EVIDENCE)
    assert (
        int(method["final_checkpoint"]["arm_summaries"]["Loreley QD"]["useful_blocks"])
        == 6
    )
    assert (
        int(
            method["qd_mechanism"][
                "final_winners_with_retained_nonchampion_primary_parent_ancestor"
            ]
        )
        == 4
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    args = parser.parse_args()
    validate_expected_metrics()
    style(args.language)
    make_quality_time_figure(args.language)
    make_job_dynamics_figure(args.language)


if __name__ == "__main__":
    main()
