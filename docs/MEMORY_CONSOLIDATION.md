# Memory Consolidation Architecture

## Overview

Replay-based consolidation for Tri-Memory, inspired by hippocampal replay.
Periodically re-processes archived chunks to:
1. Strengthen TRN state with frequently-accessed patterns
2. Prune stale/low-value chunks from retrieval archive
3. Re-calibrate saliency scores based on current goal state

## ReplayConsolidator

### Selection strategy
- Top 50% by saliency (exploitation)
- Random 50% from remaining (exploration)
- Budget: 16 chunks per pass (configurable)

### Replay process
```
for chunk in selected_chunks:
    x_summary = chunk.hidden_mean     # (d_model,)
    states_r, states_i = trn_step(x_summary, states_r, states_i, pos)
```

### Re-scoring
```
for chunk in all_chunks:
    chunk.saliency *= saliency_decay   # 0.95 default
    new_score = scorer(chunk.token_ids)
    chunk.saliency = 0.7 * chunk.saliency + 0.3 * new_score

prune chunks where saliency < prune_threshold  # 0.15 default
```

## ArchiveReweighter

Tracks retrieval frequency per chunk.
Frequently retrieved chunks get saliency boost:
- +0.02 per retrieval hit
- Maximum cumulative boost: 0.3

## Expected Effects

| Metric | Expected Change |
|--------|----------------|
| Pattern retention | Improved (TRN absorbs replayed patterns) |
| Retrieval calls | Reduced (TRN handles more, stale pruned) |
| Saliency calibration | Improved (decay + re-score) |
| Memory constancy | Improved (stable over time) |
| Fact recall | Neutral to slight decrease (some pruning) |

## Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| replay_budget | 16 | Chunks per consolidation pass |
| prune_threshold | 0.15 | Min saliency to survive |
| saliency_decay | 0.95 | Per-pass decay multiplier |
| frequency_boost | 0.02 | Boost per retrieval hit |
| max_boost | 0.3 | Maximum cumulative boost |

## Limitations

- v1 is replay-only (no generative replay)
- No learned consolidation policy
- Pruning is irreversible
- Replay uses hidden_mean summary (not full token sequence)
- Effect requires multiple passes and sufficient archive size
- In training mode, consolidation has minimal effect
  (archive is not populated during batch forward)
