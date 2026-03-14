# TRN Go/No-Go Gate Result  [trn]

**Verdict: GO**

## Input File Hashes

- `bench_agent_history.csv`: `44d41117d04abcbf7b91481c7ecd11d806cb1957f400b7414f3707a554373bcc`
- `bench_stream_tasks.csv`: `bc2caae89e6369f05e14dd4c3bd37035b67a275b3b7f797302794afd313f0542`
- `bench_pattern_memory.csv`: `3c5b443e462901fa75b59f1e780f58650f00621d7b410063f5cf891d40f18849`
- `bench_needle_haystack.csv`: `bb40ff23176e076cbf77f4384a1cea80caf3400a72d49f13e03a11c22e34caa0`
- `go_nogo_copy_trn.csv`: `NOT FOUND`
- `go_nogo_copy_tf.csv`: `02bccdef234d189114ed910337195d949bc9c95baba55cc235d52d76febf827a`
- `go_nogo_selcopy_trn.csv`: `NOT FOUND`
- `w_sweep_comparison.csv`: `e5ce209be39975d0f974335e92ca3010130b6dbf994173c0cc9c3d8e124c119d`

## Criteria Summary

| Tier | PASS | FAIL | SKIP | Failing/Skipped |
|------|------|------|------|-----------------|
| T1 | 8 | 0 | 0 | - |
| T2 | 0 | 0 | 5 | nih_recall_zero, selective_copy_low, gt_beyond_window_chance, gt_reversal_chance, trp_degraded |
| T3 | 7 | 0 | 5 | frequency_drift_parity, amplitude_envelope_parity, running_mean_parity, w_sweep_monotonic, real_use_case_pass |
