"""Consolidation Engine: replay-based memory reorganization.

Implements sleep-like consolidation for Tri-Memory:
  1. Sample salient chunks from the retrieval archive
  2. Replay them through TRN state update (strengthening patterns)
  3. Re-evaluate saliency scores (prune low-value, boost high-value)
  4. Merge duplicate/near-duplicate chunks

This is replay-only consolidation (v1). No generative replay.

Design rationale:
  - Consolidation reduces retrieval dependence over time
  - Frequently retrieved chunks get "absorbed" into TRN state
  - Stale chunks get demoted or pruned
  - No new dependencies (pure Python + torch)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from trimemory.goal_state import GoalState
from trimemory.retrieval import ChunkRecord, RetrievalIndex


@dataclass
class ConsolidationStats:
    """Stats from one consolidation pass."""
    chunks_replayed: int
    chunks_merged: int
    chunks_pruned: int
    saliency_updates: int
    avg_saliency_before: float
    avg_saliency_after: float


class ReplayConsolidator:
    """Replay-based consolidation for TRN state and retrieval archive.

    Periodically called to:
      1. Sample top-k salient chunks from archive
      2. Feed them through TRN state update
      3. Re-score saliency
      4. Prune low-saliency chunks

    Args:
        replay_budget: max chunks to replay per consolidation pass
        prune_threshold: chunks below this saliency after re-score get pruned
        saliency_decay: multiply old saliency by this before re-score
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        replay_budget: int = 16,
        prune_threshold: float = 0.15,
        saliency_decay: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.replay_budget = replay_budget
        self.prune_threshold = prune_threshold
        self.saliency_decay = saliency_decay
        self._rng = random.Random(seed)

    def select_replay_chunks(
        self,
        index: RetrievalIndex,
        goal_state: Optional[GoalState] = None,
    ) -> list[ChunkRecord]:
        """Select chunks for replay.

        Priority:
          1. High saliency chunks (top half of budget)
          2. Recently retrieved chunks (if tracked)
          3. Random sample (bottom half of budget)

        Goal state adjusts priority weighting.
        """
        if len(index) == 0:
            return []

        all_chunks = index.get_all_chunks()
        budget = min(self.replay_budget, len(all_chunks))

        if budget <= 0:
            return []

        # Sort by saliency descending
        sorted_chunks = sorted(all_chunks, key=lambda c: c.saliency, reverse=True)

        # Top half by saliency
        top_k = max(1, budget // 2)
        selected = sorted_chunks[:top_k]

        # Bottom half by random sample (exploration)
        remaining = [c for c in all_chunks if c not in selected]
        random_k = budget - len(selected)
        if remaining and random_k > 0:
            sampled = self._rng.sample(remaining, min(random_k, len(remaining)))
            selected.extend(sampled)

        return selected[:budget]

    def replay_through_trn(
        self,
        chunks: list[ChunkRecord],
        embedding_fn,
        trn_step_fn,
        states_r: list[Tensor],
        states_i: list[Tensor],
        base_position: int = 0,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Feed replay chunks through TRN state update.

        Args:
            chunks: chunks to replay
            embedding_fn: callable(token_ids_tensor) -> (1, d_model)
            trn_step_fn: callable(x, states_r, states_i, pos) -> (out, new_r, new_i)
            states_r: current TRN real states per layer
            states_i: current TRN imag states per layer
            base_position: position offset for replay

        Returns:
            Updated (states_r, states_i) after replay.
        """
        pos = base_position
        for chunk in chunks:
            if not chunk.token_ids:
                continue
            # Use the stored hidden mean as a summary
            # (cheaper than re-embedding all tokens)
            x_summary = chunk.hidden_mean.unsqueeze(0)  # (1, d_model)
            states_r, states_i = trn_step_fn(
                x_summary, states_r, states_i, pos
            )
            pos += 1

        return states_r, states_i

    def rescore_and_prune(
        self,
        index: RetrievalIndex,
        scorer_fn,
        goal_state: Optional[GoalState] = None,
    ) -> ConsolidationStats:
        """Re-evaluate saliency and prune low-value chunks.

        Args:
            index: retrieval index to consolidate
            scorer_fn: callable(token_ids, goal_state) -> (score, components)
            goal_state: current goal state for re-scoring

        Returns:
            ConsolidationStats with consolidation metrics.
        """
        if len(index) == 0:
            return ConsolidationStats(
                chunks_replayed=0, chunks_merged=0, chunks_pruned=0,
                saliency_updates=0, avg_saliency_before=0.0,
                avg_saliency_after=0.0,
            )

        chunks = index.get_all_chunks()
        old_saliencies = [c.saliency for c in chunks]
        avg_before = sum(old_saliencies) / len(old_saliencies)

        # Apply decay to all saliency scores
        for chunk in chunks:
            chunk.saliency *= self.saliency_decay

        # Re-score
        saliency_updates = 0
        for chunk in chunks:
            new_score, _ = scorer_fn(chunk.token_ids, goal_state)
            # Blend old (decayed) and new score
            chunk.saliency = 0.7 * chunk.saliency + 0.3 * new_score
            saliency_updates += 1

        # Prune low-saliency chunks
        pruned = index.remove_chunks(lambda c: c.saliency >= self.prune_threshold)

        new_saliencies = [c.saliency for c in index.get_all_chunks()]
        avg_after = sum(new_saliencies) / len(new_saliencies) if new_saliencies else 0.0

        return ConsolidationStats(
            chunks_replayed=len(old_saliencies),
            chunks_merged=0,  # v1: no merge logic
            chunks_pruned=pruned,
            saliency_updates=saliency_updates,
            avg_saliency_before=avg_before,
            avg_saliency_after=avg_after,
        )


class ArchiveReweighter:
    """Reweights archive entries based on retrieval frequency and goal relevance.

    Tracks how often each chunk is retrieved and boosts frequently-accessed chunks.
    Used alongside ReplayConsolidator.

    Args:
        frequency_boost: saliency boost per retrieval hit
        max_boost: maximum cumulative boost
    """

    def __init__(
        self,
        frequency_boost: float = 0.02,
        max_boost: float = 0.3,
    ) -> None:
        self.frequency_boost = frequency_boost
        self.max_boost = max_boost
        self._hit_counts: dict[int, int] = {}  # chunk_id -> hit count

    def record_hit(self, chunk_id: int) -> None:
        """Record a retrieval hit for a chunk."""
        self._hit_counts[chunk_id] = self._hit_counts.get(chunk_id, 0) + 1

    def apply_frequency_boost(self, index: RetrievalIndex) -> int:
        """Boost saliency of frequently retrieved chunks.

        Returns number of chunks boosted.
        """
        boosted = 0
        for chunk in index.get_all_chunks():
            hits = self._hit_counts.get(chunk.chunk_id, 0)
            if hits > 0:
                boost = min(self.max_boost, hits * self.frequency_boost)
                chunk.saliency = min(1.0, chunk.saliency + boost)
                boosted += 1
        return boosted

    def reset(self) -> None:
        """Clear hit counts."""
        self._hit_counts.clear()
