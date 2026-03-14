# Public Claims Audit

What we can and cannot claim, with evidence file references.

## Permitted Claims

| Claim | Conditions | Evidence |
|-------|------------|----------|
| TRN generation state is O(1) w.r.t. context length | Always true by construction | `gate_result.md` T1: state_constant PASS |
| TRN state size = `n_layers * K * 2 * 4` bytes | Always true; formula, not measurement | `src/trn/config.py`, analytical |
| trn_100m state = 16 KB | For d=512, L=8, K=256 only | Analytical: `8 * 256 * 2 * 4 = 16384` |
| trn_400m state = 64 KB | For d=1024, L=16, K=512 only | Analytical |
| trn_1b state = 96 KB | For d=2048, L=24, K=512 only | Analytical |
| Decode TPS stays flat as context grows | Measured on CPU at d=128-512 | `results/long_context_scaling.csv` |
| KV cache grows linearly with T | Standard Transformer property | Analytical |
| TRN throughput > TF+KV throughput at long context | Measured: 3.2x at T=1000, 10.8x avg at T>=5000 (d=256, L=8, CPU) | `gate_result.md` T1: speedup>=2x PASS |
| TRN + TF accuracy within 15% on copy task | Measured: 0% degradation (both reach 100%) | `gate_result.md` T1: accuracy_degradation PASS |
| 1,000 agents fit in 16 MB (trn_100m) | Analytical: 1000 * 16 KB | Analytical |
| TRN is numerically stable during generation | Measured: no NaN/Inf | `gate_result.md` T1: numerical_stable PASS |
| Go/No-Go verdict is GO (TRN standalone) | T1: 6/6 PASS | `results/gate_result.json` |
| Go/No-Go verdict is CONDITIONAL_GO (dual) | T1: 7/7 PASS, T2: 3 FAIL (known limitations) | `results/gate_result_dual.json` |
| Selective copy accuracy is 8.8% | Measured at d=128, L=4, 5000 steps | `docs/TRN_ARCHITECTURE_ANALYSIS.md` |
| TRN cannot do content-addressed retrieval | Proven by selective copy + NiH failure | `docs/TRN_ARCHITECTURE_ANALYSIS.md`, `docs/TRN_LIMITATIONS.md` |
| NiH recall = 0.0 at all distances | Measured on bench config | `results/bench_needle_haystack.csv` |
| PPD accuracy = 0.78--1.00 (dual), 1.00 (TRN) | Measured at bench config, 1000 steps | `results/gate_result_dual.md` T1: ppd_window_generalization PASS |
| GT recall = ~0.25 at distance > W | Measured on dual_w64 | `results/gate_result_dual.md` T2: gt_window_recovery |
| TRN is a pattern memory, not content-addressable | Proven by PPD pass + NiH/selective fail | `docs/TRN_LIMITATIONS.md` |

## Conditional Claims (Must State Premises)

| Claim | Required Premises | Evidence |
|-------|-------------------|----------|
| "2,000x memory reduction" | trn_100m, T=1000, 8 KV heads, head_dim=64, fp32 K+V (bf16 halves to 1,000x) | Analytical: `(512 * 1000) / 256 = 2000` |
| "10.8x decode speedup" | d=256, L=8, K=128 bench config vs tf_kv, avg at T>=5000, CPU | `results/bench_agent_history.csv` |
| "3.2x decode speedup at T=1000" | d=256, L=8, K=128 bench config vs tf_kv, CPU | `results/bench_agent_history.csv` |
| "566x speedup at 16K context" | d=256, L=8, CPU, batch=1, TRN vs TF full recompute | `results/long_context_scaling.csv` |
| "TRN solves copy task" | d=128, L=4, 500 steps, period=8 | `results/go_nogo_copy_tf.csv` |
| "PPD gap = 0.22" | dual_w64, bench config, 1000 training steps, tolerance=0.25 | `gate_result_dual.md` T1: ppd_window_generalization |

These numbers change with model config, hardware, and context length.
Always cite the specific config when quoting them.

## Prohibited Claims

| Claim | Why Prohibited |
|-------|----------------|
| "TRN replaces Transformer" | Selective copy failure + NiH = 0.0 proves it cannot |
| "TRN replaces attention" | Same as above; lacks content-gated filtering |
| "174,763x memory reduction" | Unbounded extrapolation; no measurement at that scale |
| "TRN is more accurate than Transformer" | Only true on copy task where both reach 100% |
| "TRN scales to 1B+ parameters" | Not measured; trn_1b config exists but is not trained |
| "TRN is production-ready" | Alpha status; no real workload deployment |
| "Zero memory overhead" | State is small but nonzero (16-96 KB) |
| "Infinite context" | max_seq_len limits prefill; generate() starts from zero state |
| "No quality loss" | NiH = 0.0, selective recall = 8.8% are proven quality losses |
| Any "x" ratio without model/T/head config | Ratio depends on all of these; omitting = misleading |
| "DualMemoryEngine recovers content beyond window" | GT = chance, NiH = 0.0 at distance > W |
| "DualMemoryEngine matches Transformer quality" | Only within KV window; beyond W, content recall fails |

## Prohibited Words

Do not use these in any public-facing document:

- "revolutionary"
- "game-changer"
- "breakthrough"
- "infinite context"
- "zero memory"
- "replaces Transformer"
- "replaces attention"
- "no quality loss"
- "production-ready"

## Claim Template (3 Lines)

Use this exact phrasing when describing TRN externally:

> TRN provides constant-state (8--96 KB) autoregressive generation that does not grow
> with context length, unlike KV caches which scale as O(n). Decode throughput: 3.2x
> over TF+KV at T=1000, 10.8x average at T>=5000 (d=256, L=8, CPU). 1,000 concurrent
> agent states (trn_100m) fit in 16 MB vs ~31 GB for KV (fp32). TRN cannot perform
> content-addressed retrieval; selective copy accuracy is 8.8% (vs Transformer 96.2%),
> Needle-in-Haystack recall is 0.0. It is a memory compression layer, not an attention
> replacement.

## DualMemoryEngine Claim Template (3 Lines)

> DualMemoryEngine combines windowed attention (last W tokens, exact retrieval) with
> TRN resonance state (all tokens, pattern compression). Decode memory is constant at
> W * head_bytes + TRN_state regardless of total context. Within the KV window, quality
> matches full-attention Transformer. Beyond W, TRN state preserves frequency/pattern
> information (PPD = 0.78--1.00) but cannot recall discrete token identity (NiH = 0.0,
> GT = chance). Suitable for streaming and agent memory where distant context tolerates
> lossy compression.

## Evidence Files

| File | Location | Contents |
|------|----------|----------|
| `gate_result.json` | `results/` | TRN standalone verdict + file hashes |
| `gate_result.md` | `results/` | TRN standalone human-readable verdict |
| `gate_result_dual.json` | `results/` | DualMemoryEngine verdict + file hashes |
| `gate_result_dual.md` | `results/` | DualMemoryEngine human-readable verdict |
| `eval_go_no_go.csv` | `results/` | All criteria with values |
| `bench_agent_history.csv` | `results/` | TPS and memory at 1K/5K/10K history |
| `long_context_scaling.csv` | `results/` | Context sweep throughput |
| `bench_needle_haystack.csv` | `results/` | NiH, PPD, GT, TRP results |
| `bench_multi_agent.csv` | `results/` | Agent scaling measurements |
| `env.json` | `artifacts/phase7/{ts}/` | Environment metadata |
| `nvidia_smi.txt` | `artifacts/phase7/{ts}/` | GPU state at measurement |
| `results.json` | `artifacts/phase7/{ts}/` | Raw GPU benchmark data |
| `TRN_ARCHITECTURE_ANALYSIS.md` | `docs/` | Selective copy failure analysis |
| `TRN_LIMITATIONS.md` | `docs/` | Structural limitations and theoretical basis |
| `TRN_PHASE7_VALIDATION_PLAN.md` | `docs/` | Validation criteria definitions |
