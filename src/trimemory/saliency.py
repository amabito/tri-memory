"""Goal-Aware Saliency Archiver for Tri-Memory.

Enhanced saliency scoring that incorporates goal state signals.

Score components:
  a * surprisal_proxy       -- high token variance / rare tokens
  b * goal_relevance        -- goal state adjustment
  c * entity_or_number      -- digits / hex / IDs / named entities
  d * tool_boundary         -- chunk at tool output boundary
  e * anomaly_score         -- anomaly event indicator
  f * unresolved_goal       -- pending goals boost archival
  g * reward_penalty        -- failure events get archived

All component weights are configurable. Default threshold = 0.3.
"""
from __future__ import annotations

from typing import Optional

import torch

from trimemory.goal_state import GoalAwareScorer, GoalState


class SaliencyArchiver:
    """Rule-based saliency scorer for evicted chunks.

    Score components:
      a * number_score      -- contains digits / hex / IDs
      b * entity_score      -- uppercase sequences (proper nouns)
      c * tool_boundary     -- chunk at tool output boundary
      d * high_token_var    -- high variance in token IDs (diverse content)
      e * rare_token_score  -- contains rare tokens (high ID)

    Chunks above threshold are archived to RetrievalIndex.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        vocab_size: int = 256,
        w_number: float = 0.3,
        w_entity: float = 0.2,
        w_tool: float = 0.25,
        w_variance: float = 0.15,
        w_rare: float = 0.1,
    ) -> None:
        self.threshold = threshold
        self.vocab_size = vocab_size
        self.w_number = w_number
        self.w_entity = w_entity
        self.w_tool = w_tool
        self.w_variance = w_variance
        self.w_rare = w_rare

    def score(
        self,
        token_ids: list[int],
        is_tool_boundary: bool = False,
        loss_values: Optional[list[float]] = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute saliency score for a chunk of token IDs."""
        n = len(token_ids) if token_ids else 1

        high_range_count = sum(1 for t in token_ids if t >= self.vocab_size * 3 // 4)
        number_score = high_range_count / n

        max_consecutive_high = 0
        current_run = 0
        for t in token_ids:
            if t >= self.vocab_size // 2:
                current_run += 1
                max_consecutive_high = max(max_consecutive_high, current_run)
            else:
                current_run = 0
        entity_score = min(1.0, max_consecutive_high / 4.0)

        tool_score = 1.0 if is_tool_boundary else 0.0

        if n > 1:
            t_tensor = torch.tensor(token_ids, dtype=torch.float32)
            variance_score = min(1.0, t_tensor.std(correction=0).item() / (self.vocab_size / 4))
        else:
            variance_score = 0.0

        rare_threshold = self.vocab_size * 7 // 8
        rare_count = sum(1 for t in token_ids if t >= rare_threshold)
        rare_score = min(1.0, rare_count / max(n * 0.1, 1))

        total = (
            self.w_number * number_score
            + self.w_entity * entity_score
            + self.w_tool * tool_score
            + self.w_variance * variance_score
            + self.w_rare * rare_score
        )

        components = {
            "number": number_score,
            "entity": entity_score,
            "tool": tool_score,
            "variance": variance_score,
            "rare": rare_score,
            "total": total,
        }
        return total, components

    def should_archive(self, score: float) -> bool:
        return score >= self.threshold


class GoalAwareSaliencyArchiver(SaliencyArchiver):
    """Saliency scorer enhanced with goal state signals.

    Extends SaliencyArchiver with:
      - goal_relevance from GoalAwareScorer
      - anomaly_score from GoalState
      - unresolved_goal_score from GoalState
      - reward_penalty from GoalState

    Args:
        threshold: minimum score for archival
        vocab_size: vocabulary size for token-based heuristics
        w_number: weight for number/ID detection
        w_entity: weight for entity detection
        w_tool: weight for tool boundary
        w_variance: weight for token diversity
        w_rare: weight for rare tokens
        w_goal: weight for goal relevance
        w_anomaly: weight for anomaly signal
        w_unresolved: weight for unresolved goals
    """

    def __init__(
        self,
        threshold: float = 0.3,
        vocab_size: int = 256,
        w_number: float = 0.20,
        w_entity: float = 0.15,
        w_tool: float = 0.15,
        w_variance: float = 0.10,
        w_rare: float = 0.05,
        w_goal: float = 0.15,
        w_anomaly: float = 0.10,
        w_unresolved: float = 0.10,
    ) -> None:
        super().__init__(
            threshold=threshold,
            vocab_size=vocab_size,
            w_number=w_number,
            w_entity=w_entity,
            w_tool=w_tool,
            w_variance=w_variance,
            w_rare=w_rare,
        )
        self.w_goal = w_goal
        self.w_anomaly = w_anomaly
        self.w_unresolved = w_unresolved
        self._scorer = GoalAwareScorer()

    def score(  # type: ignore[override]
        self,
        token_ids: list[int],
        is_tool_boundary: bool = False,
        goal_state: Optional[GoalState] = None,
        loss_values: Optional[list[float]] = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute goal-aware saliency score.

        Delegates base component scoring to SaliencyArchiver.score(), then adds
        goal-aware components on top.

        Returns:
            (total_score, component_dict) for logging/explainability.
        """
        # Base components from parent (number, entity, tool, variance, rare)
        base_total, components = super().score(
            token_ids, is_tool_boundary=is_tool_boundary, loss_values=loss_values
        )

        # Goal-aware components
        goal_adjustment = 0.0
        anomaly_score = 0.0
        unresolved_score = 0.0
        if goal_state is not None:
            goal_adjustment = self._scorer.saliency_adjustment(goal_state)
            anomaly_score = goal_state.anomaly_pressure
            unresolved_score = goal_state.unresolved_ratio

        # Rebuild total: parent weights may differ from this class's weights,
        # so we recompute from individual component scores using our own weights.
        total = (
            self.w_number * components["number"]
            + self.w_entity * components["entity"]
            + self.w_tool * components["tool"]
            + self.w_variance * components["variance"]
            + self.w_rare * components["rare"]
            + self.w_goal * goal_adjustment
            + self.w_anomaly * anomaly_score
            + self.w_unresolved * unresolved_score
        )

        components = {
            **components,
            "goal_adjustment": goal_adjustment,
            "anomaly": anomaly_score,
            "unresolved": unresolved_score,
            "total": total,
        }
        return total, components

    def should_archive(  # type: ignore[override]
        self,
        score: float,
        goal_state: Optional[GoalState] = None,
    ) -> bool:
        """Check if chunk should be archived.

        Goal changes temporarily lower the threshold.
        """
        threshold = self.threshold
        if goal_state is not None:
            threshold += self._scorer.retrieval_threshold_adjustment(goal_state)
        threshold = max(0.1, threshold)  # floor always applied
        return score >= threshold
