# Tri-Memory Internal Validation -- PARTIAL (interrupted)

**Steps**: 3000 | **Seeds**: 0,1 complete, 2 partial | **Date**: 2026-03-06

## Results

```
Seed 0:
A: recent=1.000 old=0.053 pat=0.007 comp=0.418  gate: kv=1.00 trn=0.00 ret=0.00
B: recent=1.000 old=0.053 pat=0.007 comp=0.418  gate: kv=0.71 trn=0.29 ret=0.00
C: recent=1.000 old=0.053 pat=0.007 comp=0.418  gate: kv=0.70 trn=0.00 ret=0.30
D: recent=1.000 old=0.053 pat=0.007 comp=0.418  gate: kv=0.56 trn=0.21 ret=0.23

Seed 1:
A: recent=1.000 old=0.038 pat=0.010 comp=0.414  gate: kv=1.00 trn=0.00 ret=0.00
B: recent=1.000 old=0.046 pat=0.010 comp=0.417  gate: kv=0.57 trn=0.43 ret=0.00
C: recent=1.000 old=0.046 pat=0.012 comp=0.417  gate: kv=0.61 trn=0.00 ret=0.39
D: recent=1.000 old=0.024 pat=0.019 comp=0.413  gate: kv=0.43 trn=0.22 ret=0.34

Seed 2 (A,C only):
A: recent=1.000 old=0.058 pat=0.012 comp=0.421  gate: kv=1.00 trn=0.00 ret=0.00
C: recent=1.000 old=0.055 pat=0.012 comp=0.420  gate: kv=0.65 trn=0.00 ret=0.35
B,D: NOT RUN (interrupted)
```

## Key Observations

1. **recent_exact = 1.000 for all models** -- KV window fully learns recent recall at 3000 steps
2. **old_fact / pattern near random** -- 3-6% accuracy (random baseline ~4-5%)
3. **Gate ratios are healthy** -- D uses all 3 paths (kv=43-56%, trn=21-22%, ret=23-34%)
4. **A = B = C = D on seed 0** -- no differentiation on that seed
5. **Seed 1 shows slight variance** -- B old_fact > A old_fact (0.046 vs 0.038)

## Root Cause: Weak Task Signal

The cross-entropy loss covers all 256 positions equally.
Only 3 positions are query-answer pairs (1.2% of sequence).
The model optimizes filler prediction and ignores the sparse query signal.

## Verdict: INTERNAL_FAIL (partial)

- composite(D) > max(A,B,C): MIXED (seed 0: tie, seed 1: D worst)
- old_fact(C) > old_fact(A): MIXED (seed 0: tie, seed 1: tie, seed 2: C < A)
- pattern(B) > pattern(A): FAIL (seed 0: tie, seed 1: tie)

## Next Step

Task redesign needed before re-running:
- Option 1: Weight query positions 10-50x in loss
- Option 2: Multi-token old facts (3-5 tokens instead of 1)
- Option 3: Increase pattern probe signal (repeat pattern near query)
- Option 4: Reduce sequence length to increase query signal density
