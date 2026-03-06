# Tri-Memory LLM Architecture

## Overview

Tri-Memory LLM is a hierarchical memory architecture that combines three memory systems on a Transformer backbone:

- **KV window** (W=64): short-term exact memory via windowed attention
- **TRN state**: long-range compressed pattern/state memory via temporal resonance
- **Retrieval index**: long-range exact sparse memory via archived chunks

## Design Principles

1. TRN is NOT a Transformer replacement
2. TRN is NOT a content-addressable memory
3. Each memory tier handles what it does best
4. Retrieval is NOT always-on (gated by router)
5. KV window is NOT enlarged to compensate (W=64 by default)

## Token Lifecycle

```
Token enters
  |
  v
Transformer forward (embedding + blocks)
  |
  v
KV window updated (FIFO, size W)
  |
  v
Chunk evicted? (every C tokens)
  |-- No --> continue
  |-- Yes -->
      |
      v
    TRN state updated (always, all evicted tokens)
      |
      v
    SaliencyArchiver scores chunk
      |
      v
    Salient? --> RetrievalIndex.add_chunk()
```

## Components

### TriMemoryBlock

Per-layer block with three parallel paths:

```
Input x
  |
  +---> Windowed Attention (KV cache, W tokens) --> attn_out
  |
  +---> TRN Resonance (parallel scan) -----------> trn_out
  |
  +---> Retrieval Context (projected) ------------> ret_out
  |
  v
3-way gate: softmax([g_kv, g_trn, g_ret])
  |
  v
mixed = g_kv * attn_out + g_trn * trn_out + g_ret * ret_out
  |
  v
+ residual + FFN
```

### RuleBasedMemoryRouter (v1)

Decides gate weights based on:
- Position relative to KV window
- Numeric density in query tokens
- Entity density in query tokens
- Tool boundary flags
- Availability of retrieval chunks

Rules:
- Within KV window: heavy KV (0.8)
- Numeric/entity lookup: heavy retrieval (0.5)
- Far from window: increasing TRN weight
- No retrieval chunks: redistribute to KV+TRN

### SaliencyArchiver

Scores evicted chunks for selective archival:

```
s(chunk) = w_number * number_score
         + w_entity * entity_score
         + w_tool * tool_boundary_score
         + w_variance * token_variance_score
         + w_rare * rare_token_score
```

Only chunks above threshold are archived.

### RetrievalIndex

Fixed-capacity chunk store with bag-of-token-id cosine search:
- Max 256 chunks (FIFO eviction of oldest)
- Each chunk stores: token_ids, hidden_mean, metadata
- Search: normalized bag-of-token vector dot product

### StateTokenAdapter

Converts TRN resonance states into pseudo memory tokens:

```
[states_r_layer0, states_i_layer0, ..., states_r_layerN, states_i_layerN]
  --> concat (n_layers * K * 2)
  --> Linear
  --> reshape to (m, d_model)
  --> m state tokens
```

## Memory Budget

| Component | Size | Scaling |
|-----------|------|---------|
| KV window | n_layers * n_heads * W * head_dim * 2 * dtype | O(W) constant |
| TRN state | n_layers * K * 2 * 4 bytes | O(1) constant |
| Retrieval | max_chunks * (d_model*4 + vocab*4 + tokens*4) | O(max_chunks) bounded |

Total memory is bounded regardless of history length.

## Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| W (window) | 64 | Small enough to force TRN/Retrieval usage |
| C (chunk) | 32 | Half window, reasonable eviction granularity |
| top_k | 4 | Retrieval results per query |
| m | 8 | State tokens from TRN adapter |
| max_chunks | 256 | Bounded retrieval index |
| saliency_threshold | 0.3 | Archive ~30-50% of chunks |

## Ablation Structure

All benchmarks compare 4 configurations:

| Config | KV | TRN | Retrieval | Expected Strength |
|--------|-----|-----|-----------|-------------------|
| A | Yes | No | No | Recent exact |
| B | Yes | Yes | No | Recent + pattern |
| C | Yes | No | Yes | Recent + old fact |
| D | Yes | Yes | Yes | All three (composite winner) |

## File Layout

```
src/trn/
  tri_memory.py     -- TriMemoryEngine, Router, Adapter, Archiver
  retrieval.py      -- RetrievalIndex

scripts/
  bench_trimemory_mixed.py         -- Mixed memory benchmark
  bench_trimemory_telemetry.py     -- Agent telemetry benchmark
  bench_trimemory_conversation.py  -- Long conversation benchmark
  run_trimemory_validation.py      -- Full validation + Go/No-Go
```
