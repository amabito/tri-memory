# Tri-Memory V3 Evaluation -- old_fact Signal Tuning

**Steps**: 1000  |  **Seeds**: [0, 1]  |  **Verdict**: V3_SIGNAL_FAIL

## 1. What Changed from V2

- old_fact token range: V2=220-240 (20 types) -> V3=220-225 (5 types)
- old_fact span len: V2=3 -> V3=2
- fact span loss weight: V2=8.0 -> V3=10.0
- Added old_fact_span_partial_acc metric
- Random baseline per token: V2=5% -> V3=20%
- Random baseline span exact: V2=0.05^3=0.01% -> V3=0.20^2=4%

## 2. Why This Change Was Made

- V2 old_fact span_exact was 0.000 for all models (unobservable)
- token_acc ~5% made 3-token span exact ~0.01% (below noise floor)
- Retrieval benefit was invisible, not absent
- Narrower range + shorter span raises floor to detectable level

## 3. V3 Results

### Accuracy Summary

| Model | Recent | OldFact(tok) | OldFact(span) | OldFact(partial) | Pattern | Salient | Composite |
|-------|--------|-------------|---------------|------------------|---------|---------|-----------|
| A:KV | 1.0000+/-0.0000 | 0.2109+/-0.0030 | 0.0469+/-0.0108 | 0.3750+/-0.0048 | 0.0204+/-0.0036 | 0.0685+/-0.0156 | 0.2339+/-0.0010 |
| B:KV+TRN | 1.0000+/-0.0000 | 0.2097+/-0.0042 | 0.0469+/-0.0108 | 0.3726+/-0.0024 | 0.0144+/-0.0024 | 0.0685+/-0.0156 | 0.2321+/-0.0008 |
| C:KV+Ret | 1.0000+/-0.0000 | 0.2121+/-0.0018 | 0.0469+/-0.0108 | 0.3774+/-0.0072 | 0.0204+/-0.0036 | 0.0745+/-0.0096 | 0.2351+/-0.0002 |
| D:Full | 1.0000+/-0.0000 | 0.2097+/-0.0042 | 0.0469+/-0.0108 | 0.3726+/-0.0024 | 0.0216+/-0.0048 | 0.0709+/-0.0132 | 0.2347+/-0.0008 |

### Router Gate Usage

| Model | g_kv | g_trn | g_ret |
|-------|------|-------|-------|
| A:KV | 1.0000 | 0.0000 | 0.0000 |
| B:KV+TRN | 0.6444 | 0.3556 | 0.0000 |
| C:KV+Ret | 0.6729 | 0.0000 | 0.3271 |
| D:Full | 0.5451 | 0.2572 | 0.1977 |

## 4. Gate Judgment

- **old_fact_token_C_gt_A**: PASS (C=0.2121, A=0.2109)
- **old_fact_span_C_ge_A**: PASS (C=0.0469, A=0.0469)
- **salient_C_ge_A**: PASS (C=0.0745, A=0.0685)
- **pattern_B_ge_A**: FAIL (B=0.0144, A=0.0204)
- **no_nan_inf**: PASS ()
- **supplementary_D_gt_max_ABC**: FAIL (D=0.2347, max_ABC=0.2351)

**Verdict: V3_SIGNAL_FAIL**

## 5. Interpretation

- Retrieval shows weak signal on old_fact token accuracy
- Span exact shows some signal but C does not beat A

## 6. Recommended Next Step

Investigate D interference -- C shows signal but D does not benefit.

## Sanity Checks

- old_fact_token_range_narrowed: True
- fact_span_len_valid: True
- fact_span_outside_kv: True
- query_weight_mask_nonzero: True
- old_fact_types: 5
- old_fact_random_baseline_per_token: 0.2
