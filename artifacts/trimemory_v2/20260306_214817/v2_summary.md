# Tri-Memory V2 Evaluation Results

**Steps**: 1000  |  **Seeds**: [0, 1]  |  **Verdict**: V2_FAIL

## Changes from V1
- Weighted loss (answer=8x, query=4x, fact_span=8x, pattern=6x, salient=8x)
- Multi-token old facts (3-token span)
- Pattern with regime shift (2 phases)
- Salient event recall (rare token burst)
- Composite: 0.20*recent + 0.30*old_fact_span + 0.30*pattern + 0.20*salient

## Accuracy Summary

| Model | Recent | OldFact(span) | OldFact(tok) | Pattern | Salient | Composite |
|-------|--------|---------------|--------------|---------|---------|-----------|
| A:KV | 1.0000+/-0.0000 | 0.0000+/-0.0000 | 0.0493+/-0.0020 | 0.0228+/-0.0036 | 0.0553+/-0.0048 | 0.2179+/-0.0020 |
| B:KV+TRN | 1.0000+/-0.0000 | 0.0000+/-0.0000 | 0.0489+/-0.0032 | 0.0264+/-0.0000 | 0.0553+/-0.0048 | 0.2190+/-0.0010 |
| C:KV+Ret | 1.0000+/-0.0000 | 0.0000+/-0.0000 | 0.0497+/-0.0008 | 0.0252+/-0.0036 | 0.0577+/-0.0072 | 0.2191+/-0.0025 |
| D:Full | 1.0000+/-0.0000 | 0.0000+/-0.0000 | 0.0477+/-0.0020 | 0.0192+/-0.0072 | 0.0553+/-0.0048 | 0.2168+/-0.0012 |

## Router Gate Usage

| Model | g_kv | g_trn | g_ret |
|-------|------|-------|-------|
| A:KV | 1.0000 | 0.0000 | 0.0000 |
| B:KV+TRN | 0.6730 | 0.3270 | 0.0000 |
| C:KV+Ret | 0.6790 | 0.0000 | 0.3210 |
| D:Full | 0.5508 | 0.2620 | 0.1871 |

## Gate Judgment

- **composite_D_gt_max_ABC**: FAIL (D=0.2168, max_ABC=0.2191)
- **pattern_B_gt_A**: PASS (B=0.0264, A=0.0228)
- **old_fact_C_gt_A**: FAIL (C=0.0000, A=0.0000)
- **salient_C_gt_A**: PASS (C=0.0577, A=0.0553)

**Verdict: V2_FAIL**
