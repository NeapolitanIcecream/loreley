"""LCA-aware trajectory rollups for inspiration commits.

Loreley evolves whole repositories, where a git commit represents the full repo state.
When using one repo state as the baseline and another as an inspiration, the planning
agent should see the *unique evolution path* between the two states rather than only
the inspiration tip commit metadata.

This module builds baseline-aligned trajectory rollups by:
- computing the lowest common ancestor (LCA) on the CommitCard parent chain,
- extracting the unique path `LCA(base,insp) -> insp`,
- compressing older steps with root-aligned (index-aligned) cached chunk summaries.
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass
from typing import Any, Sequence

from loguru import logger
from openai import OpenAI, OpenAIError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from loreley.config import Settings, get_settings
from loreley.db.models import CommitCard, CommitChunkSummary

log = logger.bind(module="worker.trajectory")

__all__ = [
    "ChunkSummaryError",
    "TrajectoryError",
    "TrajectoryRollup",
    "build_inspiration_trajectory_rollup",
    "find_lca",
    "get_full_chunk_pairs_from_tip",
    "get_or_build_chunk_summary",
    "walk_unique_path",
]


class TrajectoryError(RuntimeError):
    """Raised when trajectory extraction fails unexpectedly."""


class ChunkSummaryError(RuntimeError):
    """Raised when the chunk summarizer cannot produce a cacheable summary."""


@dataclass(frozen=True, slots=True)
class TrajectoryRollup:
    """Bounded, baseline-aligned trajectory representation for planning prompts."""

    lines: tuple[str, ...]
    meta: dict[str, Any]


def find_lca(
    base_commit_hash: str,
    inspiration_commit_hash: str,
    *,
    session: Session,
    max_depth: int = 4096,
) -> str | None:
    """Return the LCA hash on the CommitCard parent chain.

    Notes:
    - The chain uses `CommitCard.parent_commit_hash` (single-parent evolution chain),
      not the full git DAG.
    - The returned hash may not have a CommitCard row (e.g., upstream git history
      not registered in the DB) but can still serve as a stopping point.
    """

    base_commit_hash = (base_commit_hash or "").strip()
    inspiration_commit_hash = (inspiration_commit_hash or "").strip()
    if not base_commit_hash or not inspiration_commit_hash:
        return None

    ancestors: set[str] = set()
    current: str | None = base_commit_hash
    depth = 0
    while current and current not in ancestors and depth < max_depth:
        ancestors.add(current)
        card = session.get(CommitCard, current)
        parent = (getattr(card, "parent_commit_hash", None) or "").strip() if card else ""
        current = parent or None
        depth += 1

    current = inspiration_commit_hash
    depth = 0
    visited: set[str] = set()
    while current and current not in visited and depth < max_depth:
        if current in ancestors:
            return current
        visited.add(current)
        card = session.get(CommitCard, current)
        parent = (getattr(card, "parent_commit_hash", None) or "").strip() if card else ""
        current = parent or None
        depth += 1
    return None


def walk_unique_path(
    lca_commit_hash: str,
    inspiration_commit_hash: str,
    *,
    session: Session,
    limit: int = 4096,
) -> list[CommitCard]:
    """Return CommitCards on the path `lca -> ... -> inspiration` (excluding lca).

    The returned list is ordered from oldest to newest (towards the inspiration tip).
    """

    lca_commit_hash = (lca_commit_hash or "").strip()
    inspiration_commit_hash = (inspiration_commit_hash or "").strip()
    if not lca_commit_hash or not inspiration_commit_hash:
        return []

    cards_tip_to_root: list[CommitCard] = []
    current: str | None = inspiration_commit_hash
    visited: set[str] = set()
    steps = 0
    while current and current not in visited and steps < limit:
        if current == lca_commit_hash:
            break
        visited.add(current)
        card = session.get(CommitCard, current)
        if card is None:
            break
        cards_tip_to_root.append(card)
        parent = (getattr(card, "parent_commit_hash", None) or "").strip()
        current = parent or None
        steps += 1

    cards_tip_to_root.reverse()
    return cards_tip_to_root


@dataclass(frozen=True, slots=True)
class _AnchorIndex:
    """Root-aligned depth index for a single CommitCard parent chain."""

    anchor_commit_hash: str
    depth_by_hash: dict[str, int]
    hash_by_depth: dict[int, str]
    reached_requested_anchor: bool


def _build_anchor_index(
    tip_commit_hash: str,
    *,
    session: Session,
    anchor_commit_hash: str | None,
    max_depth: int = 4096,
) -> _AnchorIndex:
    """Build a depth index anchored at a requested root (or the chain root fallback)."""

    tip_commit_hash = (tip_commit_hash or "").strip()
    requested_anchor = (anchor_commit_hash or "").strip() or None
    if not tip_commit_hash:
        return _AnchorIndex(
            anchor_commit_hash="",
            depth_by_hash={},
            hash_by_depth={},
            reached_requested_anchor=False,
        )

    lineage_tip_to_root: list[str] = []
    current: str | None = tip_commit_hash
    visited: set[str] = set()
    reached = False
    depth = 0
    while current and current not in visited and depth < max_depth:
        visited.add(current)
        lineage_tip_to_root.append(current)
        if requested_anchor and current == requested_anchor:
            reached = True
            break
        card = session.get(CommitCard, current)
        if card is None:
            break
        parent = (getattr(card, "parent_commit_hash", None) or "").strip()
        if not parent:
            break
        current = parent
        depth += 1

    if not lineage_tip_to_root:
        return _AnchorIndex(
            anchor_commit_hash="",
            depth_by_hash={},
            hash_by_depth={},
            reached_requested_anchor=False,
        )

    resolved_anchor = requested_anchor if reached and requested_anchor else lineage_tip_to_root[-1]

    path_root_to_tip = list(reversed(lineage_tip_to_root))
    depth_by_hash: dict[str, int] = {}
    hash_by_depth: dict[int, str] = {}
    for idx, commit_hash in enumerate(path_root_to_tip):
        depth_by_hash[commit_hash] = idx
        hash_by_depth[idx] = commit_hash

    return _AnchorIndex(
        anchor_commit_hash=resolved_anchor,
        depth_by_hash=depth_by_hash,
        hash_by_depth=hash_by_depth,
        reached_requested_anchor=reached,
    )


def get_full_chunk_pairs_from_tip(
    inspiration_commit_hash: str,
    *,
    block_size: int,
    session: Session,
    anchor_commit_hash: str | None = None,
    max_pairs: int | None = None,
    max_depth: int = 4096,
) -> list[tuple[str, str]]:
    """Return root-aligned full chunk pairs (start_hash, end_hash) for an inspiration tip.

    Chunk boundaries are aligned to the anchor depth index:
    - `anchor` has depth 0
    - a full chunk ends at depth `k * block_size` (k>=1)
    - each pair represents exactly `block_size` edges on the CommitCard parent chain

    The index is anchored at `anchor_commit_hash` when it is reachable from the tip;
    otherwise the function falls back to the earliest known CommitCard ancestor in
    the chain.
    """

    inspiration_commit_hash = (inspiration_commit_hash or "").strip()
    block_size = int(block_size)
    if not inspiration_commit_hash or block_size <= 0:
        return []

    index = _build_anchor_index(
        inspiration_commit_hash,
        session=session,
        anchor_commit_hash=anchor_commit_hash,
        max_depth=max_depth,
    )
    tip_depth = index.depth_by_hash.get(inspiration_commit_hash)
    if tip_depth is None:
        return []

    full_chunks = tip_depth // block_size
    pairs: list[tuple[str, str]] = []
    for k in range(full_chunks, 0, -1):
        start_depth = (k - 1) * block_size
        end_depth = k * block_size
        start_hash = index.hash_by_depth.get(start_depth, "")
        end_hash = index.hash_by_depth.get(end_depth, "")
        if start_hash and end_hash:
            pairs.append((start_hash, end_hash))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def get_or_build_chunk_summary(
    start_commit_hash: str,
    end_commit_hash: str,
    block_size: int,
    *,
    session: Session,
    settings: Settings | None = None,
    client: OpenAI | None = None,
) -> str:
    """Return cached chunk summary or build it via LLM and persist on success."""

    settings = settings or get_settings()
    start_commit_hash = (start_commit_hash or "").strip()
    end_commit_hash = (end_commit_hash or "").strip()
    block_size = int(block_size)
    if not start_commit_hash or not end_commit_hash or block_size <= 0:
        return ""

    model = (settings.worker_planning_trajectory_summary_model or "").strip() or settings.worker_evolution_commit_model
    prompt_signature = (settings.worker_planning_trajectory_summary_prompt_signature or "v1").strip() or "v1"

    existing = session.execute(
        select(CommitChunkSummary).where(
            CommitChunkSummary.start_commit_hash == start_commit_hash,
            CommitChunkSummary.end_commit_hash == end_commit_hash,
            CommitChunkSummary.block_size == block_size,
            CommitChunkSummary.model == model,
            CommitChunkSummary.prompt_signature == prompt_signature,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return (existing.summary or "").strip()

    step_cards = _collect_chunk_cards(
        start_commit_hash=start_commit_hash,
        end_commit_hash=end_commit_hash,
        step_count=block_size,
        session=session,
    )
    step_lines = [_format_step(card) for card in step_cards]
    fallback = _fallback_chunk_summary(step_lines, max_chars=settings.worker_planning_trajectory_summary_max_chars)

    summarizer = _ChunkSummarizer(settings=settings, client=client)
    try:
        summary = summarizer.summarize_chunk(step_lines)
    except ChunkSummaryError as exc:
        log.warning(
            "Chunk summarizer failed for start={} end={} block={}: {}",
            start_commit_hash[:12],
            end_commit_hash[:12],
            block_size,
            exc,
        )
        return fallback

    cleaned = _clamp_text(summary, settings.worker_planning_trajectory_summary_max_chars)
    if not cleaned:
        return fallback

    row = CommitChunkSummary(
        start_commit_hash=start_commit_hash,
        end_commit_hash=end_commit_hash,
        block_size=block_size,
        model=model,
        prompt_signature=prompt_signature,
        step_count=block_size,
        summary=cleaned,
    )
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError:
        # Another worker may have inserted the same cache row concurrently.
        existing = session.execute(
            select(CommitChunkSummary).where(
                CommitChunkSummary.start_commit_hash == start_commit_hash,
                CommitChunkSummary.end_commit_hash == end_commit_hash,
                CommitChunkSummary.block_size == block_size,
                CommitChunkSummary.model == model,
                CommitChunkSummary.prompt_signature == prompt_signature,
            )
        ).scalar_one_or_none()
        return (existing.summary or "").strip() if existing else cleaned
    return cleaned


def build_inspiration_trajectory_rollup(
    base_commit_hash: str,
    inspiration_commit_hash: str,
    *,
    session: Session,
    settings: Settings | None = None,
    client: OpenAI | None = None,
) -> TrajectoryRollup:
    """Build a bounded trajectory rollup for the planning prompt."""

    settings = settings or get_settings()
    base_commit_hash = (base_commit_hash or "").strip()
    inspiration_commit_hash = (inspiration_commit_hash or "").strip()
    meta: dict[str, Any] = {
        "base_commit_hash": base_commit_hash,
        "inspiration_commit_hash": inspiration_commit_hash,
        "lca_commit_hash": None,
        "unique_steps_count": 0,
        "omitted_steps": 0,
    }

    if not base_commit_hash or not inspiration_commit_hash:
        return TrajectoryRollup(lines=(), meta=meta)

    lca = find_lca(
        base_commit_hash=base_commit_hash,
        inspiration_commit_hash=inspiration_commit_hash,
        session=session,
    )
    meta["lca_commit_hash"] = lca
    if not lca:
        return TrajectoryRollup(
            lines=(
                "  - Trajectory unavailable: missing parent-chain overlap for base/inspiration.",
            ),
            meta=meta,
        )

    steps = walk_unique_path(
        lca_commit_hash=lca,
        inspiration_commit_hash=inspiration_commit_hash,
        session=session,
    )
    unique_steps_count = len(steps)
    meta["unique_steps_count"] = unique_steps_count
    if unique_steps_count == 0:
        return TrajectoryRollup(
            lines=(
                f"  - unique_steps_count: 0 (inspiration is identical to or an ancestor of base; lca={lca[:12]})",
            ),
            meta=meta,
        )

    block_size = max(1, int(settings.worker_planning_trajectory_block_size))
    max_chunks = max(0, int(settings.worker_planning_trajectory_max_chunks))
    max_raw_steps = max(0, int(settings.worker_planning_trajectory_max_raw_steps))

    anchor_candidate = (settings.mapelites_experiment_root_commit or "").strip() or None
    anchor_index = _build_anchor_index(
        inspiration_commit_hash,
        session=session,
        anchor_commit_hash=anchor_candidate,
    )
    meta["anchor_commit_hash"] = anchor_index.anchor_commit_hash
    meta["anchor_mode"] = (
        "experiment_root"
        if anchor_candidate and anchor_index.reached_requested_anchor
        else "chain_root"
    )

    depths_raw = [
        anchor_index.depth_by_hash.get((getattr(card, "commit_hash", None) or "").strip())
        for card in steps
    ]
    if not depths_raw or any(depth is None for depth in depths_raw):
        # Always include the freshest raw steps; reuse max_raw_steps knob as K (bounded).
        recent_raw_count = min(max_raw_steps, unique_steps_count) if max_raw_steps > 0 else 0
        recent_start_index = unique_steps_count - recent_raw_count
        recent_raw = list(steps[recent_start_index:]) if recent_raw_count else []
        included_indices: set[int] = set(range(recent_start_index, unique_steps_count))

        # If the depth index is incomplete, fall back to a pure raw-tail rendering.
        omitted = max(0, unique_steps_count - len(included_indices))
        meta["omitted_steps"] = omitted
        lines = [f"  - unique_steps_count: {unique_steps_count} (lca={lca[:12]})"]
        if recent_raw:
            lines.append(f"  - Recent unique steps (raw, last {len(recent_raw)}):")
            for card in recent_raw:
                lines.append(f"    - {_format_step(card)}")
        if omitted:
            lines.append(f"  - Omitted {omitted} older unique step(s).")
        return TrajectoryRollup(lines=tuple(lines), meta=meta)

    depths: list[int] = [depth for depth in depths_raw if depth is not None]
    if len(depths) != unique_steps_count:
        recent_raw_count = min(max_raw_steps, unique_steps_count) if max_raw_steps > 0 else 0
        recent_start_index = unique_steps_count - recent_raw_count
        recent_raw = list(steps[recent_start_index:]) if recent_raw_count else []
        included_indices = set(range(recent_start_index, unique_steps_count))

        omitted = max(0, unique_steps_count - len(included_indices))
        meta["omitted_steps"] = omitted
        lines = [f"  - unique_steps_count: {unique_steps_count} (lca={lca[:12]})"]
        if recent_raw:
            lines.append(f"  - Recent unique steps (raw, last {len(recent_raw)}):")
            for card in recent_raw:
                lines.append(f"    - {_format_step(card)}")
        if omitted:
            lines.append(f"  - Omitted {omitted} older unique step(s).")
        return TrajectoryRollup(lines=tuple(lines), meta=meta)

    # Recent raw steps: align the start to the nearest root-aligned chunk boundary
    # so we do not create a "gap" when MAX_RAW_STEPS < block_size.
    base_recent_raw_count = min(max_raw_steps, unique_steps_count) if max_raw_steps > 0 else 0
    recent_start_index = unique_steps_count - base_recent_raw_count
    if base_recent_raw_count > 0:
        depth_at_recent_start = depths[recent_start_index]
        offset_within_chunk = (depth_at_recent_start - 1) % block_size
        recent_start_index = max(0, recent_start_index - offset_within_chunk)
    recent_raw = list(steps[recent_start_index:]) if base_recent_raw_count > 0 else []
    included_indices = set(range(recent_start_index, unique_steps_count))

    first_depth = depths[0]
    partial_len = 0
    if first_depth % block_size != 1:
        partial_len = min(
            block_size - ((first_depth - 1) % block_size),
            unique_steps_count,
        )
    earliest_raw_end = min(partial_len, max_raw_steps, recent_start_index) if max_raw_steps > 0 else 0
    if earliest_raw_end == 0 and max_raw_steps > 0 and recent_start_index > 0:
        # Even when the unique path starts on a chunk boundary, show 1-2 earliest
        # raw steps to make the divergence from the base more legible.
        earliest_raw_end = min(2, max_raw_steps, recent_start_index)
    earliest_raw = list(steps[:earliest_raw_end]) if earliest_raw_end else []

    # Identify root-aligned full chunks within the unique path.
    full_chunks: list[tuple[int, int, str, str]] = []
    idx = 0
    while idx < unique_steps_count:
        depth = depths[idx]
        chunk_index = (depth - 1) // block_size
        end = idx + 1
        while end < unique_steps_count:
            next_depth = depths[end]
            if (next_depth - 1) // block_size != chunk_index:
                break
            end += 1
        group_size = end - idx
        last_depth = depths[end - 1]
        is_full = group_size == block_size and last_depth % block_size == 0
        if is_full:
            start_hash = (getattr(steps[idx], "parent_commit_hash", None) or "").strip()
            end_hash = (getattr(steps[end - 1], "commit_hash", None) or "").strip()
            if start_hash and end_hash:
                full_chunks.append((idx, end, start_hash, end_hash))
        idx = end

    # Eligible chunks must not overlap with the recent raw tail.
    eligible = [chunk for chunk in full_chunks if chunk[1] <= recent_start_index]

    # Select chunks with a bias towards baseline context: always include the oldest,
    # then fill remaining slots from the newest side.
    selected: list[tuple[int, int, str, str]] = []
    if max_chunks > 0 and eligible:
        selected.append(eligible[0])
        if max_chunks > 1:
            for chunk in reversed(eligible[1:]):
                if len(selected) >= max_chunks:
                    break
                selected.append(chunk)

    chunk_summaries: list[tuple[int, int, str, str, str]] = []
    for start_i, end_i, start_hash, end_hash in selected:
        summary = get_or_build_chunk_summary(
            start_commit_hash=start_hash,
            end_commit_hash=end_hash,
            block_size=block_size,
            session=session,
            settings=settings,
            client=client,
        )
        if summary:
            chunk_summaries.append((start_i, end_i, start_hash, end_hash, summary))

    # Compute omission count as steps not covered by either raw items or chunk summaries.
    covered = set(included_indices)
    covered.update(range(0, earliest_raw_end))
    for start_i, end_i, _start_hash, _end_hash, _summary in chunk_summaries:
        covered.update(range(start_i, end_i))

    omitted = max(0, unique_steps_count - len(covered))
    meta["omitted_steps"] = omitted

    lines: list[str] = []
    anchor_short = (anchor_index.anchor_commit_hash or "")[:12]
    anchor_note = f", anchor={anchor_short}" if anchor_short else ""
    lines.append(f"  - unique_steps_count: {unique_steps_count} (lca={lca[:12]}{anchor_note})")

    if earliest_raw:
        lines.append(f"  - Earliest unique steps (raw, up to {max_raw_steps}):")
        for card in earliest_raw:
            lines.append(f"    - {_format_step(card)}")

    if chunk_summaries:
        lines.append(
            f"  - Older unique steps (cached chunks, root-aligned, {block_size} steps each):"
        )
        for _start_i, _end_i, start_hash, end_hash, summary in chunk_summaries:
            lines.append(
                "    - "
                f"[{start_hash[:12]}..{end_hash[:12]}] "
                f"{_clamp_text(summary, settings.worker_planning_trajectory_summary_max_chars)}"
            )

    if recent_raw:
        lines.append(f"  - Recent unique steps (raw, last {len(recent_raw)}):")
        for card in recent_raw:
            lines.append(f"    - {_format_step(card)}")

    if omitted:
        lines.append(f"  - Omitted {omitted} older unique step(s).")

    return TrajectoryRollup(lines=tuple(lines), meta=meta)


# Internal helpers -------------------------------------------------------------


def _ancestor_hash(
    commit_hash: str,
    *,
    steps: int,
    session: Session,
    max_depth: int = 4096,
) -> str | None:
    """Return the ancestor hash after following `steps` parents from `commit_hash`."""

    commit_hash = (commit_hash or "").strip()
    steps = int(steps)
    if not commit_hash or steps <= 0:
        return commit_hash or None

    current: str | None = commit_hash
    walked = 0
    visited: set[str] = set()
    while current and walked < steps and walked < max_depth and current not in visited:
        visited.add(current)
        card = session.get(CommitCard, current)
        if card is None:
            return None
        parent = (getattr(card, "parent_commit_hash", None) or "").strip()
        if not parent:
            return None
        current = parent
        walked += 1
    return current


def _collect_chunk_cards(
    *,
    start_commit_hash: str,
    end_commit_hash: str,
    step_count: int,
    session: Session,
) -> list[CommitCard]:
    """Collect CommitCards representing `step_count` edges from start->...->end."""

    start_commit_hash = (start_commit_hash or "").strip()
    end_commit_hash = (end_commit_hash or "").strip()
    step_count = int(step_count)
    if not start_commit_hash or not end_commit_hash or step_count <= 0:
        return []

    cards_tip_to_root: list[CommitCard] = []
    current: str | None = end_commit_hash
    for _ in range(step_count):
        if not current:
            return []
        card = session.get(CommitCard, current)
        if card is None:
            return []
        cards_tip_to_root.append(card)
        parent = (getattr(card, "parent_commit_hash", None) or "").strip()
        if not parent:
            return []
        current = parent

    # After walking `step_count` parents from end, we must land on start.
    if (current or "").strip() != start_commit_hash:
        return []

    cards_tip_to_root.reverse()
    return cards_tip_to_root


def _format_step(card: CommitCard) -> str:
    """Format a single step summary line for prompt inclusion."""

    commit_hash = (getattr(card, "commit_hash", None) or "").strip()
    summary = (getattr(card, "change_summary", None) or "").strip()
    summary = summary or "N/A"
    prefix = commit_hash[:12] if commit_hash else "unknown"
    return _clamp_text(f"{prefix}: {summary}", 240)


def _fallback_chunk_summary(step_lines: Sequence[str], *, max_chars: int) -> str:
    """Deterministic fallback summary when the LLM call fails."""

    if not step_lines:
        return ""
    joined = " | ".join(_clamp_text(line, 160) for line in step_lines[:8])
    return _clamp_text(joined, max_chars)


def _clamp_text(text: str, limit: int) -> str:
    """Clamp text to a maximum number of characters."""

    limit = max(1, int(limit))
    snippet = (text or "").strip()
    if len(snippet) <= limit:
        return snippet
    return f"{snippet[: limit - 1].rstrip()}…"


class _ChunkSummarizer:
    """LLM helper for chunk summaries with retry and budget controls."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: OpenAI | None = None,
    ) -> None:
        self.settings = settings
        if client is not None:
            self._client = client
        else:
            client_kwargs: dict[str, object] = {}
            if self.settings.openai_api_key:
                client_kwargs["api_key"] = self.settings.openai_api_key
            if self.settings.openai_base_url:
                client_kwargs["base_url"] = self.settings.openai_base_url
            self._client = (
                OpenAI(**client_kwargs)  # type: ignore[call-arg]
                if client_kwargs
                else OpenAI()
            )
        self._model = (
            (self.settings.worker_planning_trajectory_summary_model or "").strip()
            or self.settings.worker_evolution_commit_model
        )
        self._temperature = float(self.settings.worker_planning_trajectory_summary_temperature)
        self._max_tokens = max(32, int(self.settings.worker_planning_trajectory_summary_max_output_tokens))
        self._max_retries = max(1, int(self.settings.worker_planning_trajectory_summary_max_retries))
        self._retry_backoff = max(
            0.0,
            float(self.settings.worker_planning_trajectory_summary_retry_backoff_seconds),
        )
        self._max_chars = max(64, int(self.settings.worker_planning_trajectory_summary_max_chars))
        self._api_spec = self.settings.openai_api_spec

    def summarize_chunk(self, step_lines: Sequence[str]) -> str:
        """Summarize a fixed-size list of step summaries into a compact description."""

        if not step_lines:
            raise ChunkSummaryError("Empty chunk input.")

        prompt = self._build_prompt(step_lines)
        attempt = 0
        while attempt < self._max_retries:
            attempt += 1
            try:
                instructions = (
                    "Summarize the evolution trajectory described by the provided step summaries.\n"
                    f"- Stay under {self._max_chars} characters.\n"
                    "- Be concrete and faithful to the provided text; do not infer missing details.\n"
                    "- Output plain text only (no markdown fences)."
                )
                if self._api_spec == "responses":
                    response = self._client.responses.create(
                        model=self._model,
                        input=prompt,
                        temperature=self._temperature,
                        max_output_tokens=self._max_tokens,
                        instructions=instructions,
                    )
                    text = (response.output_text or "").strip()
                else:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
                    text = _extract_chat_completion_text(response).strip()

                if not text:
                    raise ChunkSummaryError("Chunk summarizer returned empty output.")
                return _clamp_text(" ".join(text.split()), self._max_chars)
            except (OpenAIError, ChunkSummaryError) as exc:
                if attempt >= self._max_retries:
                    raise ChunkSummaryError(
                        f"Chunk summarizer failed after {attempt} attempt(s): {exc}",
                    ) from exc
                delay = self._retry_backoff * attempt
                log.warning(
                    "Chunk summarizer attempt {} failed: {}. Retrying in {:.1f}s",
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise ChunkSummaryError("Chunk summarizer exhausted retries without success.")

    @staticmethod
    def _build_prompt(step_lines: Sequence[str]) -> str:
        bullet_block = "\n".join(f"- {line}" for line in step_lines)
        prompt = f"""
You are summarizing a fixed-size block of repository evolution steps.
Each item is a short description of the change from a commit's parent to that commit.

Step summaries (oldest -> newest):
{bullet_block}

Return a compact summary describing the overall trajectory across these steps.
"""
        return textwrap.dedent(prompt).strip()


def _extract_chat_completion_text(response: Any) -> str:
    """Extract assistant text content from a chat completion response."""

    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if text:
                parts.append(str(text))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content or "")


