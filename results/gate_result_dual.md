# TRN Go/No-Go Gate Result  [dual]

**Verdict: CONDITIONAL_GO**

## Input File Hashes

- `bench_agent_history.csv`: `44d41117d04abcbf7b91481c7ecd11d806cb1957f400b7414f3707a554373bcc`
- `go_nogo_copy_trn.csv`: `NOT FOUND`
- `go_nogo_copy_tf.csv`: `02bccdef234d189114ed910337195d949bc9c95baba55cc235d52d76febf827a`
- `go_nogo_selcopy_trn.csv`: `NOT FOUND`
- `long_context_scaling.csv`: `d8a0e6e35d8f4c41c5ff5ae89c1e328808a13563e56cd19461abeee41bed12af`
- `bench_needle_haystack.csv`: `babd85b1b895f8f82edcb65d73c5a1f8bde31c89c9303803faeabb355b21d5ee`

## Criteria Summary

| Tier | PASS | FAIL | SKIP | Failing/Skipped |
|------|------|------|------|-----------------|
| T1 | 7 | 0 | 0 | - |
| T2 | 4 | 3 | 1 | selective_copy_pass, dual_nih_long_range, gt_window_recovery, gt_reversal_recovery |
| T3 | 7 | 0 | 0 | - |
