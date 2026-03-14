"""GoalState: importance weighting for memory prioritization.

Goal-Conditioned Tri-Memory uses a goal/value state vector to:
  - Bias saliency scoring (what gets archived)
  - Bias router decisions (which memory to query)
  - Prioritize consolidation replay (what gets replayed)

The goal state is NOT used for reasoning -- only for memory prioritization.

Update rule (EMA):
  g_t = beta * g_{t-1} + (1 - beta) * f(x_t, context_t)

State dimensions:
  [intent_similarity,   -- cosine similarity to current goal embedding
   urgency,             -- decayed urgency signal
   reward_signal,       -- recent tool success/failure
   anomaly_pressure,    -- anomaly detection pressure
   unresolved_count,    -- normalized unresolved goal count
   user_priority,       -- explicit user priority signal
   recency_bias,        -- decayed recency emphasis
   goal_change_flag]    -- recent goal change indicator

All values in [0, 1] range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


GOAL_DIM = 8

# Named indices for goal state vector
IDX_INTENT = 0
IDX_URGENCY = 1
IDX_REWARD = 2
IDX_ANOMALY = 3
IDX_UNRESOLVED = 4
IDX_USER_PRIORITY = 5
IDX_RECENCY = 6
IDX_GOAL_CHANGE = 7


@dataclass
class GoalEvent:
    """Single event that updates goal state."""
    intent_embedding: Optional[Tensor] = None   # (d,) goal direction
    urgency: float = 0.0                        # [0, 1]
    reward: float = 0.5                         # 0=failure, 0.5=neutral, 1=success
    anomaly: float = 0.0                        # [0, 1]
    unresolved_count: int = 0                   # absolute count
    user_priority: float = 0.5                  # [0, 1]
    is_goal_change: bool = False                # flag for goal shift


class GoalState:
    """Maintains a running goal/value state vector via EMA.

    Not an nn.Module -- no learnable parameters.
    All state manipulation is rule-based.

    Args:
        dim: goal state dimension (default 8)
        beta: EMA decay factor (higher = more inertia)
        max_unresolved: normalization cap for unresolved count
    """

    def __init__(
        self,
        dim: int = GOAL_DIM,
        beta: float = 0.9,
        max_unresolved: int = 10,
    ) -> None:
        self.dim = dim
        self.beta = beta
        self.max_unresolved = max_unresolved
        self.state = torch.zeros(dim, dtype=torch.float32)
        self._goal_embedding: Optional[Tensor] = None
        self._step_count: int = 0

    def update(self, event: GoalEvent) -> Tensor:
        """Update goal state from an event. Returns updated state (dim,)."""
        features = torch.zeros(self.dim, dtype=torch.float32)

        # Intent similarity
        if event.intent_embedding is not None:
            self._goal_embedding = event.intent_embedding.detach().float()
            features[IDX_INTENT] = 1.0  # max when goal just set
        elif self._goal_embedding is not None:
            # Decay intent over time
            features[IDX_INTENT] = max(0.0, self.state[IDX_INTENT].item() * 0.95)

        features[IDX_URGENCY] = max(0.0, min(1.0, event.urgency))
        features[IDX_REWARD] = max(0.0, min(1.0, event.reward))
        features[IDX_ANOMALY] = max(0.0, min(1.0, event.anomaly))
        features[IDX_UNRESOLVED] = min(
            1.0, event.unresolved_count / max(self.max_unresolved, 1)
        )
        features[IDX_USER_PRIORITY] = max(0.0, min(1.0, event.user_priority))
        features[IDX_RECENCY] = 1.0  # always 1 at update time, decays via EMA
        features[IDX_GOAL_CHANGE] = 1.0 if event.is_goal_change else 0.0

        # EMA update
        self.state = self.beta * self.state + (1.0 - self.beta) * features
        self._step_count += 1

        return self.state.clone()

    def reset(self) -> None:
        """Reset goal state to zeros."""
        self.state.zero_()
        self._goal_embedding = None
        self._step_count = 0

    @property
    def urgency(self) -> float:
        return self.state[IDX_URGENCY].item()

    @property
    def anomaly_pressure(self) -> float:
        return self.state[IDX_ANOMALY].item()

    @property
    def reward_signal(self) -> float:
        return self.state[IDX_REWARD].item()

    @property
    def goal_change_recent(self) -> bool:
        return self.state[IDX_GOAL_CHANGE].item() > 0.3

    @property
    def unresolved_ratio(self) -> float:
        return self.state[IDX_UNRESOLVED].item()

    def to_dict(self) -> dict:
        """Serialize for artifacts."""
        return {
            "state": self.state.tolist(),
            "step_count": self._step_count,
            "beta": self.beta,
            "has_goal_embedding": self._goal_embedding is not None,
        }


class GoalAwareScorer:
    """Adjusts saliency and routing based on goal state.

    Provides multipliers and biases, not absolute scores.
    The caller (SaliencyArchiver, Router) uses these to adjust behavior.
    """

    def __init__(
        self,
        anomaly_boost: float = 0.3,
        urgency_boost: float = 0.2,
        goal_change_retrieval_boost: float = 0.3,
        reward_penalty_boost: float = 0.15,
    ) -> None:
        self.anomaly_boost = anomaly_boost
        self.urgency_boost = urgency_boost
        self.goal_change_retrieval_boost = goal_change_retrieval_boost
        self.reward_penalty_boost = reward_penalty_boost

    def saliency_adjustment(self, goal_state: GoalState) -> float:
        """Additional saliency score from goal state. [0, ~1.0]."""
        adj = 0.0
        adj += self.anomaly_boost * goal_state.anomaly_pressure
        adj += self.urgency_boost * goal_state.urgency
        # Low reward (failure) increases saliency (learn from failure)
        reward = goal_state.reward_signal
        if reward < 0.3:
            adj += self.reward_penalty_boost * (1.0 - reward)
        return adj

    def retrieval_threshold_adjustment(self, goal_state: GoalState) -> float:
        """Lower retrieval threshold by this amount when goal changes.

        Returns a negative adjustment (lower threshold = more retrieval).
        """
        if goal_state.goal_change_recent:
            return -self.goal_change_retrieval_boost
        return 0.0

    def consolidation_priority(self, goal_state: GoalState) -> float:
        """Priority multiplier for consolidation replay. [0.5, 2.0]."""
        base = 1.0
        # High anomaly -> more consolidation
        base += 0.5 * goal_state.anomaly_pressure
        # High unresolved -> more consolidation
        base += 0.3 * goal_state.unresolved_ratio
        return max(0.5, min(2.0, base))

    def router_bias(self, goal_state: GoalState) -> dict[str, float]:
        """Additive biases for router gate weights.

        Returns dict with g_kv_bias, g_trn_bias, g_ret_bias.
        These are added before normalization.
        """
        g_kv_bias = 0.0
        g_trn_bias = 0.0
        g_ret_bias = 0.0

        # Goal change -> boost retrieval (need old context)
        if goal_state.goal_change_recent:
            g_ret_bias += 0.15
            g_kv_bias -= 0.05

        # High anomaly -> boost TRN (pattern detection)
        if goal_state.anomaly_pressure > 0.5:
            g_trn_bias += 0.1

        # High urgency -> boost KV (need recent exact)
        if goal_state.urgency > 0.7:
            g_kv_bias += 0.1
            g_trn_bias -= 0.05

        return {
            "g_kv_bias": g_kv_bias,
            "g_trn_bias": g_trn_bias,
            "g_ret_bias": g_ret_bias,
        }
