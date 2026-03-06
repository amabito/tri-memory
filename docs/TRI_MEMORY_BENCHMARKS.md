# Tri-Memory Benchmark Guide

## Benchmarks

### 1. Mixed Memory Benchmark (`bench_trimemory_mixed.py`)

Tests a single task requiring all three memory types simultaneously.

**Sequence structure:**
```
[OLD_FACT, filler..., PATTERN_BLOCK, filler..., RECENT_VALUES,
 QUERY_OLD_FACT, answer, QUERY_RECENT, answer, QUERY_PATTERN, answer]
```

**Metrics:**
- `recent_exact_acc`: accuracy on recent value query
- `old_fact_acc`: accuracy on old fact query (distance > W)
- `long_pattern_acc`: accuracy on pattern query
- `composite_score`: harmonic mean of all three

### 2. Agent Telemetry Benchmark (`bench_trimemory_telemetry.py`)

Simulates agent monitoring with quantized metrics, regime shifts, incidents.

**Metrics:**
- `trend_direction_acc`: up/down/flat prediction
- `anomaly_detect_acc`: regime shift detection
- `old_incident_lookup_acc`: exact old incident recall
- `recent_state_acc`: current metric value
- Analytical scaling for 1 to 10,000 agents

### 3. Long Conversation Benchmark (`bench_trimemory_conversation.py`)

Consumer-oriented: long dialogue with topic drift and old preferences.

**Metrics:**
- `old_fact_acc`: old user preference recall
- `recent_consistency_acc`: recent dialogue consistency
- `topic_state_acc`: current topic detection
- `composite_score`: harmonic mean

## Running

### Individual benchmarks

```bash
# Mixed memory (default: CPU, 300 steps)
python scripts/bench_trimemory_mixed.py

# With GPU and more steps
python scripts/bench_trimemory_mixed.py --device cuda --steps 500

# Telemetry
python scripts/bench_trimemory_telemetry.py --device cpu --steps 300

# Conversation
python scripts/bench_trimemory_conversation.py --device cpu --steps 300
```

### Full validation (all 3 + Go/No-Go)

```bash
python scripts/run_trimemory_validation.py --device cpu --steps 300
```

## Output

All results are saved to `artifacts/trimemory/{timestamp}/`:

```
artifacts/trimemory/20260306_143000/
  mixed_results.json
  mixed_summary.md
  telemetry_results.json
  telemetry_summary.md
  conversation_results.json
  conversation_summary.md
  gate_result_trimemory.json
  gate_result_trimemory.md
```

## Go/No-Go Criteria

### Tier 1: Must Pass

| Criterion | Condition |
|-----------|-----------|
| mixed_memory_superiority | D composite > max(A, B, C) |
| recent_exact_preservation | D recent >= A recent - 0.05 |
| old_fact_gain | D old_fact >= B old_fact + 0.20 |
| long_pattern_gain | D pattern >= C pattern + 0.10 |
| retrieval_budget | Router is rule-based, not always-retrieve |
| stability | No NaN/Inf during training |

### Tier 2: Known Limitations (expected failures)

| Criterion | Expected |
|-----------|----------|
| exact_long_without_retrieval | Fails (by design) |
| trn_only_old_fact | Fails (TRN is not content-addressable) |
| symbolic_copy_outside_window | Fails (structural limit) |

### Tier 3: Stretch

| Criterion | Target |
|-----------|--------|
| 10k_agent_memory | < 2000 MB |
| vLLM integration | Pass |
| Learned router > rule-based | Outperform |

## Interpreting Results

The key question each benchmark answers:

1. **Mixed**: "Does combining all three memories beat any pair?"
2. **Telemetry**: "Is this practical for agent monitoring at scale?"
3. **Conversation**: "Can this handle long conversations better than KV-only?"

If D (full Tri-Memory) does NOT beat all ablations on composite score,
the architecture needs rethinking. The value proposition is specifically
that no single pair of memory systems covers all three use cases.
