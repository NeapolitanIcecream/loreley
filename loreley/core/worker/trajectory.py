"""LCA-aware trajectory rollups for inspiration commits.

Loreley evolves whole repositories, where a git commit represents the full repo state.
When using one repo state as the baseline and another as an inspiration, the planning
agent should see the *unique evolution path* between the two states rather than only
the inspiration tip commit metadata.

This module builds baseline-aligned trajectory rollups by:
- computing the lowest common ancestor (LCA) on the CommitCard parent chain,
- extracting the unique path `LCA(base,insp) -> insp`,
- compressing older steps with tip-aligned cached chunk summaries.
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


def get_full_chunk_pairs_from_tip(
    inspiration_commit_hash: str,
    *,
    block_size: int,
    session: Session,
    max_pairs: int | None = None,
    max_depth: int = 4096,
) -> list[tuple[str, str]]:
    """Return tip-aligned full chunk pairs (start_hash, end_hash) for an inspiration tip.

    Each pair represents exactly `block_size` edges on the CommitCard parent chain:
    `start_hash -> ... -> end_hash` where `end_hash` is the chunk tip.
    """

    inspiration_commit_hash = (inspiration_commit_hash or "").strip()
    block_size = int(block_size)
    if not inspiration_commit_hash or block_size <= 0:
        return []

    pairs: list[tuple[str, str]] = []
    end_hash: str | None = inspiration_commit_hash
    visited: set[str] = set()

    while end_hash and end_hash not in visited and len(pairs) < (max_pairs or 10**9):
        visited.add(end_hash)
        start_hash = _ancestor_hash(end_hash, steps=block_size, session=session, max_depth=max_depth)
        if not start_hash:
            break
        pairs.append((start_hash, end_hash))
        end_hash = start_hash
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

    # Tip-aligned blocks over the unique path (oldest->newest commits).
    blocks = _blocks_from_tip(steps, block_size=block_size)

    # Always include the freshest raw steps; reuse max_raw_steps knob as K (bounded).
    # (This keeps the prompt size predictable without adding another knob.)
    recent_raw_count = min(max_raw_steps, unique_steps_count) if max_raw_steps > 0 else 0
    recent_start_index = unique_steps_count - recent_raw_count
    recent_raw = list(steps[recent_start_index:]) if recent_raw_count else []
    included_indices: set[int] = set(range(recent_start_index, unique_steps_count))

    # Oldest partial block (near LCA) is never cached; render raw (bounded).
    partial_len = blocks[-1].size if blocks and blocks[-1].size < block_size else 0
    earliest_raw_end = min(partial_len, max_raw_steps, recent_start_index) if max_raw_steps > 0 else 0
    earliest_raw = list(steps[:earliest_raw_end]) if earliest_raw_end else []

    # Select cached full chunks that do not overlap with the recent raw tail.
    chunk_summaries: list[tuple[str, str, str]] = []
    for block in blocks:
        if block.size != block_size:
            continue
        # Skip any block that overlaps with the recent raw tail indices.
        if block.end_index > recent_start_index:
            continue
        if len(chunk_summaries) >= max_chunks:
            break
        start_hash = (getattr(block.items[0], "parent_commit_hash", None) or "").strip()
        if not start_hash:
            continue
        summary = get_or_build_chunk_summary(
            start_commit_hash=start_hash,
            end_commit_hash=block.end_hash,
            block_size=block_size,
            session=session,
            settings=settings,
            client=client,
        )
        if summary:
            chunk_summaries.append((start_hash, block.end_hash, summary))

    # Compute omission count as steps not covered by either raw items or chunk summaries.
    covered = set(included_indices)
    covered.update(range(0, earliest_raw_end))

    # Exact coverage accounting based on block index ranges.
    for block in blocks:
        if block.size != block_size:
            continue
        # Find the matching included chunk by end hash.
        if any(end == block.end_hash for _s, end, _t in chunk_summaries):
            covered.update(range(block.start_index, block.end_index))

    omitted = max(0, unique_steps_count - len(covered))
    meta["omitted_steps"] = omitted

    lines: list[str] = []
    lines.append(f"  - unique_steps_count: {unique_steps_count} (lca={lca[:12]})")

    if earliest_raw:
        lines.append(f"  - Earliest unique steps (raw, up to {max_raw_steps}):")
        for card in earliest_raw:
            lines.append(f"    - {_format_step(card)}")

    if chunk_summaries:
        lines.append(
            f"  - Older unique steps (cached chunks, {block_size} steps each):"
        )
        for start_hash, end_hash, summary in chunk_summaries:
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


@dataclass(frozen=True, slots=True)
class _Block:
    start_index: int
    end_index: int
    end_hash: str
    items: tuple[CommitCard, ...]

    @property
    def size(self) -> int:
        return self.end_index - self.start_index


def _blocks_from_tip(steps: Sequence[CommitCard], *, block_size: int) -> list[_Block]:
    """Partition steps into tip-aligned blocks (newest-first blocks list)."""

    block_size = max(1, int(block_size))
    n = len(steps)
    blocks: list[_Block] = []
    end_index = n
    while end_index > 0:
        start_index = max(0, end_index - block_size)
        block_items = tuple(steps[start_index:end_index])
        end_hash = (getattr(block_items[-1], "commit_hash", "") or "").strip() if block_items else ""
        blocks.append(
            _Block(
                start_index=start_index,
                end_index=end_index,
                end_hash=end_hash,
                items=block_items,
            )
        )
        end_index = start_index
    return blocks  # newest-first


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


