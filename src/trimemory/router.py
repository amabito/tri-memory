"""Goal-Aware Memory Router for Tri-Memory.

Enhanced router that adjusts KV/TRN/Retrieval gate weights
based on both content features and goal state.

Rule-based v2: extends the original RuleBasedMemoryRouter with:
  - Goal state bias (from GoalAwareScorer)
  - Goal change detection (temporarily boosts retrieval)
  - Anomaly pressure (boosts TRN for pattern detection)
  - Urgency (boosts KV for exact recent recall)

All routing decisions are logged with reasons for explainability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from trimemory.goal_state import GoalAwareScorer, GoalState


@dataclass
class RouterDecision:
    """Explainable routing decision."""
    g_kv: float
    g_trn: float
    g_ret: float
    reason: str


class RuleBasedMemoryRouter:
    """Rule-based router for KV / TRN / Retrieval gate weights.

    Features used:
      - position (within KV window?)
      - numeric density (exact lookup signal)
      - entity density (named entity signal)
      - tool query (tool output boundary)

    Returns (g_kv, g_trn, g_ret) gate weights summing to 1.0.
    """

    def __init__(
        self,
        kv_window_size: int = 64,
        retrieval_entity_threshold: float = 0.3,
        retrieval_numeric_threshold: float = 0.3,
    ) -> None:
        self.kv_window_size = kv_window_size
        self.retrieval_entity_threshold = retrieval_entity_threshold
        self.retrieval_numeric_threshold = retrieval_numeric_threshold

    def route(
        self,
        position: int,
        query_token_ids: list[int],
        vocab_size: int = 256,
        is_tool_query: bool = False,
        has_retrieval_chunks: bool = False,
    ) -> RouterDecision:
        """Decide gate weights based on current context features."""
        n = len(query_token_ids) if query_token_ids else 1

        if position < self.kv_window_size:
            return RouterDecision(
                g_kv=0.8, g_trn=0.15, g_ret=0.05,
                reason="within_kv_window"
            )

        high_range = sum(1 for t in query_token_ids if t >= vocab_size * 3 // 4)
        numeric_density = high_range / n

        entity_range = sum(1 for t in query_token_ids if t >= vocab_size // 2)
        entity_density = entity_range / n

        if is_tool_query:
            g_kv, g_trn, g_ret = 0.2, 0.1, 0.7
            reason = "tool_query"
        elif numeric_density > self.retrieval_numeric_threshold:
            g_kv, g_trn, g_ret = 0.3, 0.2, 0.5
            reason = "numeric_lookup"
        elif entity_density > self.retrieval_entity_threshold:
            g_kv, g_trn, g_ret = 0.3, 0.2, 0.5
            reason = "entity_lookup"
        else:
            distance_factor = min(1.0, (position - self.kv_window_size) / 500.0)
            g_kv = 0.5 - 0.2 * distance_factor
            g_trn = 0.3 + 0.2 * distance_factor
            g_ret = 0.2
            reason = f"distance_blend(d={position})"

        if not has_retrieval_chunks and g_ret > 0.05:
            redistribute = g_ret - 0.05
            g_kv += redistribute * 0.4
            g_trn += redistribute * 0.6
            g_ret = 0.05
            reason += "+no_ret_chunks"

        total = g_kv + g_trn + g_ret
        g_kv /= total
        g_trn /= total
        g_ret /= total

        return RouterDecision(g_kv=g_kv, g_trn=g_trn, g_ret=g_ret, reason=reason)


class GoalAwareRouter:
    """Goal-conditioned rule-based router for KV / TRN / Retrieval.

    Features used:
      - position (within KV window?)
      - numeric density (exact lookup signal)
      - entity density (named entity signal)
      - tool query (tool output boundary)
      - goal state (bias from GoalAwareScorer)

    Returns (g_kv, g_trn, g_ret) gate weights summing to 1.0.
    """

    def __init__(
        self,
        kv_window_size: int = 64,
        retrieval_entity_threshold: float = 0.3,
        retrieval_numeric_threshold: float = 0.3,
    ) -> None:
        self.kv_window_size = kv_window_size
        self.retrieval_entity_threshold = retrieval_entity_threshold
        self.retrieval_numeric_threshold = retrieval_numeric_threshold
        self._scorer = GoalAwareScorer()

    def route(
        self,
        position: int,
        query_token_ids: list[int],
        vocab_size: int = 256,
        is_tool_query: bool = False,
        has_retrieval_chunks: bool = False,
        goal_state: Optional[GoalState] = None,
    ) -> RouterDecision:
        """Decide gate weights based on context features and goal state."""

        n = len(query_token_ids) if query_token_ids else 1

        # Position-based: early positions are entirely KV
        if position < self.kv_window_size:
            return RouterDecision(
                g_kv=0.8, g_trn=0.15, g_ret=0.05,
                reason="within_kv_window"
            )

        # Numeric density
        high_range = sum(1 for t in query_token_ids if t >= vocab_size * 3 // 4)
        numeric_density = high_range / n

        # Entity density
        entity_range = sum(1 for t in query_token_ids if t >= vocab_size // 2)
        entity_density = entity_range / n

        # Base routing decision
        if is_tool_query:
            g_kv, g_trn, g_ret = 0.2, 0.1, 0.7
            reason = "tool_query"
        elif numeric_density > self.retrieval_numeric_threshold:
            g_kv, g_trn, g_ret = 0.3, 0.2, 0.5
            reason = "numeric_lookup"
        elif entity_density > self.retrieval_entity_threshold:
            g_kv, g_trn, g_ret = 0.3, 0.2, 0.5
            reason = "entity_lookup"
        else:
            distance_factor = min(1.0, (position - self.kv_window_size) / 500.0)
            g_kv = 0.5 - 0.2 * distance_factor
            g_trn = 0.3 + 0.2 * distance_factor
            g_ret = 0.2
            reason = f"distance_blend(d={position})"

        # Apply goal state bias
        if goal_state is not None:
            bias = self._scorer.router_bias(goal_state)
            g_kv += bias["g_kv_bias"]
            g_trn += bias["g_trn_bias"]
            g_ret += bias["g_ret_bias"]
            # Clamp to positive before normalization
            g_kv = max(0.01, g_kv)
            g_trn = max(0.01, g_trn)
            g_ret = max(0.01, g_ret)
            if any(v != 0.0 for v in bias.values()):
                reason += "+goal_bias"

        # If no retrieval chunks available, redistribute to KV+TRN
        if not has_retrieval_chunks and g_ret > 0.05:
            redistribute = g_ret - 0.05
            g_kv += redistribute * 0.4
            g_trn += redistribute * 0.6
            g_ret = 0.05
            reason += "+no_ret_chunks"

        # Normalize to sum=1
        total = g_kv + g_trn + g_ret
        g_kv /= total
        g_trn /= total
        g_ret /= total

        return RouterDecision(g_kv=g_kv, g_trn=g_trn, g_ret=g_ret, reason=reason)
