#!/usr/bin/env python3
"""Generate paper figures from checked-in Zstandard evidence JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "zstandard-gpt-v19-top10-validation-supplement.json"
METHOD_SOURCE = ROOT / "paper" / "evidence" / "zstd_method_efficacy.json"
OUT_DIR = Path(__file__).resolve().parent
PDF_METADATA = {"CreationDate": None}


def gain(ratio: float) -> float:
    return 100.0 * (ratio - 1.0)


def load_rows() -> tuple[list[dict], list[dict]]:
    evidence = json.loads(SOURCE.read_text(encoding="utf-8"))
    validation = sorted(evidence["top10"]["results"], key=lambda row: row["training_rank"])
    holdout = sorted(
        evidence["fixed_top10_holdout_comparison"]["results"],
        key=lambda row: row["training_rank"],
    )
    return validation, holdout


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def candidate_color(rank: int) -> str:
    if rank == 3:
        return "#d97706"  # preregistered winner
    if rank == 10:
        return "#2563eb"  # expanded-validation winner
    return "#64748b"


def make_top10_figure() -> None:
    validation, holdout = load_rows()
    ranks = np.array([row["training_rank"] for row in validation])

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.19, top=0.83, wspace=0.24)

    # The log-t interval is symmetric in log space, so the upper endpoint is
    # point**2 / lower when the frozen summary stores only point and lower.
    ax = axes[0]
    validation_points = np.array([gain(row["validation_compression_geomean"]) for row in validation])
    validation_lowers = np.array([gain(row["validation_compression_lower_95"]) for row in validation])
    validation_uppers = np.array(
        [
            gain(
                row["validation_compression_geomean"] ** 2
                / row["validation_compression_lower_95"]
            )
            for row in validation
        ]
    )
    for rank, point, lower, upper in zip(
        ranks,
        validation_points,
        validation_lowers,
        validation_uppers,
        strict=True,
    ):
        color = candidate_color(int(rank))
        ax.errorbar(
            rank,
            point,
            yerr=np.array([[point - lower], [upper - point]]),
            fmt="o",
            markersize=4.3,
            color=color,
            ecolor=color,
            elinewidth=1.05,
            capsize=2.3,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
    ax.axhline(0.0, color="#94a3b8", linewidth=0.7)
    ax.set_title("Expanded validation (selection set)", loc="left", fontweight="bold")
    ax.set_ylabel("Compression-throughput gain (%)")
    ax.text(
        0.02,
        0.04,
        "point estimate and two-sided 95% interval",
        transform=ax.transAxes,
        color="#475569",
        fontsize=7.2,
    )
    ax.annotate(
        "selected rank 10: +1.234%\n95% CI [+1.156, +1.312%]\nnot selection-adjusted",
        xy=(10, validation_points[-1]),
        xytext=(5.6, 1.46),
        arrowprops={"arrowstyle": "-", "color": "#2563eb", "lw": 0.8},
        color="#1d4ed8",
        fontsize=7.4,
        ha="left",
    )

    # Original holdout: full two-sided intervals are available for every finalist.
    ax = axes[1]
    holdout_points = np.array([gain(row["compression_geomean"]) for row in holdout])
    holdout_lowers = np.array([gain(row["compression_lower_95"]) for row in holdout])
    holdout_uppers = np.array([gain(row["compression_upper_95"]) for row in holdout])
    for rank, point, lower, upper in zip(
        ranks, holdout_points, holdout_lowers, holdout_uppers, strict=True
    ):
        color = candidate_color(int(rank))
        ax.errorbar(
            rank,
            point,
            yerr=np.array([[point - lower], [upper - point]]),
            fmt="o",
            markersize=4.3,
            color=color,
            ecolor=color,
            elinewidth=1.05,
            capsize=2.3,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
    ax.axhline(0.0, color="#94a3b8", linewidth=0.7)
    ax.set_title("Original holdout (post-selection Top 10)", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.035,
        "reported rank 10: +1.173%\n95% CI [+1.102, +1.245%]",
        transform=ax.transAxes,
        color="#475569",
        fontsize=7.2,
        va="bottom",
    )
    ax.text(
        0.02,
        0.90,
        "point max: rank 2 (+1.239%)\nlower-bound max: rank 7 (+1.228%)",
        transform=ax.transAxes,
        color="#475569",
        fontsize=7.2,
        va="top",
    )

    for ax in axes:
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(ranks)
        ax.set_xlabel("Training rank")
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0.0, 1.85)

    legend_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", color="#d97706", label="registered winner (rank 3)"),
        mpl.lines.Line2D([], [], marker="o", linestyle="", color="#2563eb", label="reported validation winner (rank 10)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.99),
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=1.0,
    )

    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"zstd_top10.{suffix}",
            bbox_inches="tight",
            metadata=PDF_METADATA if suffix == "pdf" else None,
        )
    plt.close(fig)


def make_method_efficacy_figure() -> None:
    evidence = json.loads(METHOD_SOURCE.read_text(encoding="utf-8"))
    checkpoints = evidence["checkpoint_summaries"]
    blocks = evidence["final_checkpoint"]["blocks"]
    contrasts = {
        row["control"]: row for row in evidence["primary_contrasts"]
    }

    jobs = np.array([row["jobs"] for row in checkpoints])
    policies = [
        ("independent", "Independent Root", "#64748b", "s"),
        ("qd", "Loreley QD", "#2563eb", "o"),
        ("sequential", "Sequential Champion", "#d97706", "^"),
    ]

    fig = plt.figure(figsize=(6.5, 5.15))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.88, 1.0])
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.84,
        hspace=0.70,
        wspace=0.40,
    )

    ax = axes[0]
    for key, label, color, marker in policies:
        values = np.array([gain(row[key]["median_ratio"]) for row in checkpoints])
        ax.plot(
            jobs,
            values,
            color=color,
            marker=marker,
            markersize=4.2,
            linewidth=1.25,
            label=label,
        )
    ax.axhline(0.5, color="#94a3b8", linewidth=0.75, linestyle="--")
    ax.text(8.2, 0.515, "useful threshold", color="#64748b", fontsize=8.8)
    ax.set_title("Validation-selected holdout winner", loc="left", fontweight="bold")
    ax.set_xlabel("Candidate jobs per block and policy")
    ax.set_ylabel("Median compression gain (%)")
    ax.set_xticks(jobs)
    ax.set_ylim(0.0, 0.94)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        fontsize=9,
    )

    scatter_specs = [
        ("independent", "Independent Root"),
        ("sequential", "Sequential Champion"),
    ]
    for ax, (control_key, control_name) in zip(
        axes[1:], scatter_specs, strict=True
    ):
        qd_values = np.array([gain(row["qd"]) for row in blocks])
        control_values = np.array([gain(row[control_key]) for row in blocks])
        for row, x_value, y_value in zip(blocks, control_values, qd_values, strict=True):
            ax.scatter(
                x_value,
                y_value,
                s=27,
                color="#2563eb",
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            annotation_offsets = {
                1: (3, 2),
                2: (3, 2),
                3: (4, 4),
                4: (3, 3),
                5: (4, 3),
                6: (4, 12),
                7: (8, -15),
            }
            if control_key == "sequential" and int(row["block"]) == 2:
                annotation_offsets[2] = (4, 12)
            ax.annotate(
                str(row["block"]),
                (x_value, y_value),
                xytext=annotation_offsets[int(row["block"])],
                textcoords="offset points",
                fontsize=8.5,
                color="#334155",
            )
        limit = 1.95
        ax.plot([0, limit], [0, limit], color="#94a3b8", linewidth=0.8, zorder=1)
        ax.fill_between(
            [0, limit],
            [0, limit],
            [limit, limit],
            color="#2563eb",
            alpha=0.035,
            linewidth=0,
        )
        contrast = contrasts[control_name]
        short_name = "Independent" if control_key == "independent" else "Sequential"
        ax.set_title(
            f"QD vs. {short_name}\n"
            f"QD wins {contrast['qd_block_wins']}/7; effect {contrast['percent_effect']:+.3f}%",
            loc="left",
            fontweight="bold",
            pad=8,
        )
        ax.set_xlabel(f"{control_name} gain (%)")
        ax.set_ylabel("Loreley QD gain (%)")
        ax.set_xlim(0.0, limit)
        ax.set_ylim(0.0, limit)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes:
        ax.grid(color="#e2e8f0", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"zstd_method_efficacy.{suffix}",
            bbox_inches="tight",
            metadata=PDF_METADATA if suffix == "pdf" else None,
        )
    plt.close(fig)


def main() -> None:
    style()
    make_top10_figure()
    make_method_efficacy_figure()


if __name__ == "__main__":
    main()
