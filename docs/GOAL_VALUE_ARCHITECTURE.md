# Goal/Value Architecture

## Overview

The Goal/Value system adds importance weighting to the Tri-Memory architecture.
It does NOT perform reasoning -- it only prioritizes what gets remembered,
what gets retrieved, and what gets consolidated.

## GoalState

8-dimensional EMA state vector:

| Index | Dimension | Range | Source |
|-------|-----------|-------|--------|
| 0 | intent_similarity | [0,1] | cosine to current goal embedding |
| 1 | urgency | [0,1] | decayed urgency signal |
| 2 | reward_signal | [0,1] | tool success (1) / failure (0) |
| 3 | anomaly_pressure | [0,1] | anomaly detection signal |
| 4 | unresolved_count | [0,1] | normalized pending goals |
| 5 | user_priority | [0,1] | explicit priority signal |
| 6 | recency_bias | [0,1] | decayed recency emphasis |
| 7 | goal_change_flag | [0,1] | recent goal switch indicator |

Update: `g_t = beta * g_{t-1} + (1 - beta) * features(x_t)`

Default beta = 0.9 (slow adaptation).

## GoalAwareScorer

Converts GoalState into actionable adjustments:

### Saliency adjustment
- High anomaly -> +0.3 saliency
- High urgency -> +0.2 saliency
- Low reward (failure) -> +0.15 saliency

### Retrieval threshold adjustment
- Goal change detected -> threshold lowered by 0.3

### Router bias
- Goal change -> +0.15 retrieval, -0.05 KV
- High anomaly -> +0.1 TRN
- High urgency -> +0.1 KV, -0.05 TRN

### Consolidation priority
- High anomaly -> 1.5x consolidation priority
- High unresolved -> 1.3x consolidation priority

## Integration Points

```
GoalState
  |
  +-> GoalAwareSaliencyArchiver (what gets archived)
  |     score += goal_adjustment + anomaly + unresolved
  |
  +-> GoalAwareRouter (which memory to query)
  |     gates += router_bias(goal_state)
  |
  +-> ReplayConsolidator (what gets replayed)
        priority *= consolidation_priority(goal_state)
```

## Limitations

- No learned parameters in GoalState (rule-based EMA)
- Goal embedding is optional (works without it)
- Not a planning/reasoning system
- Weights are heuristic (not optimized)
- In training mode, goal bias has no effect (only inference-time mechanism)
