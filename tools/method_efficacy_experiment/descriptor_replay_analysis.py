"""Run a no-model-call final-basis archive sensitivity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row

from loreley.core.map_elites.pareto_archive import ParetoCandidate, ParetoGridArchive
from loreley.core.map_elites.snapshot import deserialize_projection


class DescriptorReplayError(RuntimeError):
    """Raised when a dump cannot support an exact final-basis replay."""


OBJECTIVES = (
    "compression_lower_95",
    "decompression_lower_95",
    "worst_cell_speedup",
)


def _project_measure(
    *, projection: Any, vector: Sequence[float], dimensions: int, clip_radius: float
) -> tuple[float, ...]:
    projected = np.asarray(projection.transform_array(vector), dtype=np.float64)
    fitted = np.zeros(dimensions, dtype=np.float64)
    count = min(dimensions, projected.size)
    fitted[:count] = projected[:count]
    clipped = np.clip(fitted, -clip_radius, clip_radius)
    normalized = (clipped + clip_radius) / (2 * clip_radius)
    return tuple(float(value) for value in normalized)


def _archive(
    *,
    dimensions: int,
    cells_per_dimension: int,
    front_size: int,
    epsilon: float,
) -> ParetoGridArchive:
    return ParetoGridArchive(
        dims=tuple(cells_per_dimension for _ in range(dimensions)),
        ranges=tuple((0.0, 1.0) for _ in range(dimensions)),
        objective_count=len(OBJECTIVES),
        max_front_size=front_size,
        epsilon=epsilon,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _history_rows(connection: psycopg.Connection[Any]) -> list[Mapping[str, Any]]:
    query = """
        SELECT
            h.commit_hash,
            h.vector,
            h.last_seen_at,
            j.sampling_ordinal,
            j.ingestion_status,
            j.ingestion_cell_index,
            MAX(m.value) FILTER (WHERE m.name = 'compression_lower_95')
                AS compression_lower_95,
            MAX(m.value) FILTER (WHERE m.name = 'decompression_lower_95')
                AS decompression_lower_95,
            MAX(m.value) FILTER (WHERE m.name = 'worst_cell_speedup')
                AS worst_cell_speedup
        FROM map_elites_pca_history AS h
        JOIN commit_cards AS cc ON cc.commit_hash = h.commit_hash
        JOIN evolution_jobs AS j ON j.id = cc.job_id
        JOIN metrics AS m ON m.commit_card_id = cc.id
        GROUP BY
            h.commit_hash,
            h.vector,
            h.last_seen_at,
            j.sampling_ordinal,
            j.ingestion_status,
            j.ingestion_cell_index
        ORDER BY
            j.sampling_ordinal NULLS FIRST,
            h.last_seen_at,
            h.commit_hash
    """
    with connection.cursor() as cursor:
        cursor.execute(query)
        return list(cursor.fetchall())


def _actual_archive(connection: psycopg.Connection[Any]) -> list[Mapping[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT commit_hash, cell_index, objective_values, measures, "timestamp"
            FROM map_elites_archive_cells
            ORDER BY cell_index, commit_hash
            """
        )
        return list(cursor.fetchall())


def _state(connection: psycopg.Connection[Any]) -> Mapping[str, Any]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT island_id, snapshot FROM map_elites_states ORDER BY island_id"
        )
        rows = list(cursor.fetchall())
    if len(rows) != 1 or not isinstance(rows[0].get("snapshot"), Mapping):
        raise DescriptorReplayError("Expected exactly one MAP-Elites state snapshot")
    return rows[0]


def _candidate(row: Mapping[str, Any], *, measures: Sequence[float]) -> ParetoCandidate:
    objectives = tuple(float(row[name]) for name in OBJECTIVES)
    if not all(math.isfinite(value) for value in objectives):
        raise DescriptorReplayError("History row has a missing or non-finite objective")
    return ParetoCandidate(
        commit_hash=str(row["commit_hash"]),
        objective_values=objectives,
        objective_scores=objectives,
        measures=tuple(float(value) for value in measures),
        timestamp=float(row["last_seen_at"]),
    )


def analyze_database(
    *,
    database: str,
    block: int,
    host: str = "127.0.0.1",
    port: int = 55432,
    user: str = "loreley",
    dimensions: int = 3,
    cells_per_dimension: int = 4,
    front_size: int = 8,
    epsilon: float = 0.003,
    clip_radius: float = 3.0,
    winner_commit: str | None = None,
) -> dict[str, Any]:
    password = os.environ.get("PGPASSWORD")
    if not password:
        raise DescriptorReplayError("PGPASSWORD is required")
    with psycopg.connect(
        dbname=database,
        host=host,
        port=port,
        user=user,
        password=password,
        row_factory=dict_row,
    ) as connection:
        state = _state(connection)
        history = _history_rows(connection)
        actual = _actual_archive(connection)
    snapshot = state["snapshot"]
    projection = deserialize_projection(snapshot.get("projection"))
    if projection is None:
        raise DescriptorReplayError("Final PCA projection is missing")
    if projection.dimensions < dimensions:
        raise DescriptorReplayError("Final PCA projection has too few dimensions")
    measures_by_commit = {
        str(row["commit_hash"]): _project_measure(
            projection=projection,
            vector=row["vector"],
            dimensions=dimensions,
            clip_radius=clip_radius,
        )
        for row in history
    }
    history_by_commit = {str(row["commit_hash"]): row for row in history}
    if len(history_by_commit) != len(history):
        raise DescriptorReplayError("PCA history has duplicate commit hashes")
    indexer = _archive(
        dimensions=dimensions,
        cells_per_dimension=cells_per_dimension,
        front_size=front_size,
        epsilon=epsilon,
    )
    final_cell_by_commit = {
        commit: int(indexer.index_of([measures])[0])
        for commit, measures in measures_by_commit.items()
    }
    actual_commits = {str(row["commit_hash"]) for row in actual}
    if not actual_commits <= history_by_commit.keys():
        raise DescriptorReplayError("Final archive commit is missing from PCA history")
    maximum_measure_error = 0.0
    cell_mismatches = 0
    for row in actual:
        commit = str(row["commit_hash"])
        stored = np.asarray(row["measures"], dtype=np.float64)
        replayed = np.asarray(measures_by_commit[commit], dtype=np.float64)
        maximum_measure_error = max(
            maximum_measure_error, float(np.max(np.abs(stored - replayed)))
        )
        cell_mismatches += int(int(row["cell_index"]) != final_cell_by_commit[commit])
    retained_replay = _archive(
        dimensions=dimensions,
        cells_per_dimension=cells_per_dimension,
        front_size=front_size,
        epsilon=epsilon,
    )
    retained_candidates = [
        ParetoCandidate(
            commit_hash=str(row["commit_hash"]),
            objective_values=tuple(float(value) for value in row["objective_values"]),
            objective_scores=tuple(float(value) for value in row["objective_values"]),
            measures=measures_by_commit[str(row["commit_hash"])],
            timestamp=float(row["timestamp"]),
        )
        for row in actual
    ]
    retained_replay.add_many(retained_candidates)
    retained_replay_commits = {
        candidate.commit_hash for candidate in retained_replay.records()
    }
    full_replay = _archive(
        dimensions=dimensions,
        cells_per_dimension=cells_per_dimension,
        front_size=front_size,
        epsilon=epsilon,
    )
    full_replay.add_many(
        [
            _candidate(row, measures=measures_by_commit[str(row["commit_hash"])])
            for row in history
        ]
    )
    full_records = full_replay.records()
    full_commits = {candidate.commit_hash for candidate in full_records}
    admitted = [row for row in history if row["ingestion_cell_index"] is not None]
    changed_admission_cells = sum(
        int(row["ingestion_cell_index"])
        != final_cell_by_commit[str(row["commit_hash"])]
        for row in admitted
    )
    skipped_commits = {
        str(row["commit_hash"])
        for row in history
        if str(row["ingestion_status"]) == "skipped"
    }
    if cell_mismatches or maximum_measure_error > 1e-10:
        raise DescriptorReplayError(
            "Final projection does not reproduce stored archive cells/measures "
            f"(cells={cell_mismatches}, max_error={maximum_measure_error})"
        )
    if retained_replay_commits != actual_commits:
        raise DescriptorReplayError("Retained-only replay does not reproduce archive")
    return {
        "schema_version": 1,
        "kind": "offline_final_basis_archive_sensitivity",
        "block": block,
        "inferential_status": (
            "offline same-candidate sensitivity audit; not a causal descriptor ablation"
        ),
        "configuration": {
            "dimensions": dimensions,
            "cells_per_dimension": cells_per_dimension,
            "front_size": front_size,
            "epsilon": epsilon,
            "clip_radius": clip_radius,
            "projection_epoch": int(projection.epoch),
            "projection_sample_count": int(projection.sample_count),
        },
        "counts": {
            "history_candidates": len(history),
            "admitted_at_observation": len(admitted),
            "actual_final_entries": len(actual_commits),
            "actual_final_occupied_cells": len(
                {int(row["cell_index"]) for row in actual}
            ),
            "final_basis_history_occupied_cells": len(
                set(final_cell_by_commit.values())
            ),
            "full_history_replay_entries": len(full_commits),
            "full_history_replay_occupied_cells": int(full_replay.stats.num_occupied),
            "actual_and_full_history_overlap": len(actual_commits & full_commits),
            "actual_only_entries": len(actual_commits - full_commits),
            "full_history_only_entries": len(full_commits - actual_commits),
            "initially_skipped_retained_by_full_history": len(
                skipped_commits & full_commits
            ),
            "admission_to_final_cell_changes": changed_admission_cells,
        },
        "rates": {
            "admission_to_final_cell_change_fraction": (
                changed_admission_cells / len(admitted) if admitted else 0.0
            ),
            "actual_archive_overlap_with_full_history_fraction": (
                len(actual_commits & full_commits) / len(actual_commits)
                if actual_commits
                else 0.0
            ),
        },
        "validation_selected_winner": (
            {
                "in_pca_history": winner_commit in history_by_commit,
                "in_actual_final_archive": winner_commit in actual_commits,
                "in_full_history_replay": winner_commit in full_commits,
            }
            if winner_commit
            else None
        ),
        "digests": {
            "projection": _digest(snapshot["projection"]),
            "history_commit_set": _digest(sorted(history_by_commit)),
            "actual_archive_commit_set": _digest(sorted(actual_commits)),
            "full_history_archive_commit_set": _digest(sorted(full_commits)),
        },
        "checks": {
            "actual_cells_reproduced": cell_mismatches == 0,
            "actual_measures_reproduced": maximum_measure_error <= 1e-10,
            "retained_only_archive_reproduced": retained_replay_commits
            == actual_commits,
            "maximum_measure_absolute_error": maximum_measure_error,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55432)
    parser.add_argument("--user", default="loreley")
    parser.add_argument("--winner-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_database(
        database=args.database,
        block=args.block,
        host=args.host,
        port=args.port,
        user=args.user,
        winner_commit=args.winner_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
