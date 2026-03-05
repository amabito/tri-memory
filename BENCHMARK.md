# Benchmark Protocol

Smoke tests run on every push/PR via `.github/workflows/bench_smoke.yml`.

Detailed validation results with tables and analysis: [docs/VALIDATION_RESULTS.md](docs/VALIDATION_RESULTS.md)

## Quick Start

```bash
pip install -e .

# CI smoke test (< 60s)
python scripts/bench_smoke.py
```

## Generation Benchmark

Measures autoregressive generation throughput (tokens/second) and memory for TRN vs Transformer.

```bash
python scripts/bench_generate.py --gen-lens 128,256,512,1024,2048,4096
```

Output: `scripts/results/bench_generate.csv`

Columns: `model, gen_len, batch_size, tps, peak_mb_total, state_mb`

## Training Benchmark

Trains TRN and Transformer on 5 generalization tasks and compares loss curves.

```bash
# All tasks, 5000 steps
python scripts/bench_train.py --tasks all --steps 5000

# Single task
python scripts/bench_train.py --tasks copy --steps 5000
```

Tasks: `copy`, `counting`, `reverse`, `induction`, `assoc_recall`

Output: `scripts/results/{task}_curves.csv`

## Real-Text Language Modeling

Trains on WikiText-2 (HuggingFace), NLTK Gutenberg, or synthetic fallback.

```bash
# Char-level (default)
cd scripts
python train_lm_realdata.py --model trn --size small --steps 2000
python train_lm_realdata.py --model tf  --size small --steps 2000

# BPE token-level
python train_lm_realdata.py --model trn --size small --steps 2000 --tokenizer bpe
python train_lm_realdata.py --model tf  --size small --steps 2000 --tokenizer bpe

# Random-target sanity check
python train_lm_realdata.py --model trn --size small --steps 500 --random-targets

# Debug sample logging
python train_lm_realdata.py --model trn --size small --steps 200 --debug-samples 3
```

Output: `scripts/results/train_lm_realdata_{model}_{size}[_bpe]_curves.csv`

Note: run `train_lm_realdata.py` from the `scripts/` directory to avoid a `profile.py` import conflict at the project root.

## Long-Context Memory Tasks

Four retrieval tasks with context length sweep.

```bash
python scripts/bench_memory_tasks.py --steps 300 \
    --trn-lens 128,256,512,1024,2048,4096 \
    --tf-lens 128,256,512,1024
```

Tasks: `induction_head`, `assoc_recall`, `multi_needle`, `copy_distractors`

Output: `scripts/results/bench_memory_tasks.csv`

## Forward Pass Profiling

Component-level timing breakdown with Chrome traces.

```bash
python scripts/profile_forward.py --d-model 128 --n-layers 4 --seq-len 64
```

Output:
- `scripts/results/profile_forward.csv`
- `scripts/results/profile_breakdown.md`
- `scripts/results/traces/trace_trn.json` (open in `chrome://tracing`)
- `scripts/results/traces/trace_tf.json`

## GPU Scan Verification

Tests correctness and timing of sequential vs parallel resonance scan on GPU.

```bash
python scripts/train_gpu_scan.py
```

Output: `scripts/results/train_gpu_scan.csv`

## Knowledge Distillation

Transfer representations from a pretrained HF Transformer to a TRN student.

```bash
cd scripts

# Quick smoke test (< 5 min)
python distill_lm.py --quick --device cpu

# Full run (GPU)
python distill_lm.py --student-size 100m --teacher gpt2 --steps 100000 --device cuda

# CE-only baseline (no distillation)
python distill_lm.py --kl-weight 0.0 --ce-weight 1.0 --steps 2000
```

Output: `scripts/results/distill_{student}_{teacher}_curves.csv`

Verification suite:

```bash
cd scripts
python verify_overfit_microset.py --steps 2000 --device cpu
python verify_random_targets.py --steps 500 --device cpu
python verify_teacher_ablation.py --steps 2000 --device cpu
cd .. && pytest tests/test_distill_determinism.py -v
```

See [docs/DISTILL.md](docs/DISTILL.md) for full protocol and gate checklist.

## Integrity Audits

Verify that results reflect genuine learning, not artifacts.

```bash
python scripts/audit_dataset_split.py    # train/val overlap check
python scripts/audit_label_shift.py      # single causal shift verification
python scripts/audit_random_targets.py   # random target baseline
python scripts/audit_overfit.py          # overfit capacity check
python scripts/audit_learning_curve.py   # 1000-step gradient statistics
```

## KV Cache vs TRN Generation

```bash
python scripts/bench_kv_vs_trn.py \
    --context-lens 512,1024,2048,4096,8192,16384 \
    --gen-tokens 128 --device cpu
```

Compares three generation modes:
- **TRN**: O(1) constant-state generation via `step_single`
- **TF-KV**: Transformer with KV cache (O(1) per step, O(n*d) state)
- **TF-full**: Transformer without KV cache (O(n) per step, re-computes full forward)

Output: `results/bench_kv_vs_trn.csv`

Columns: `mode, context_len, tps, memory_mb, latency_ms`

### Validated Results (d_model=256, n_layers=8, d_ff=1024, n_osc=128, CPU, seed=42)

| ctx_len | TRN tps | TF+KV tps | TF_full tps | TRN vs KV | TRN vs full |
|---------|---------|-----------|-------------|-----------|-------------|
| 512 | 255.8 | 112.3 | 21.1 | 2.28x | 12.1x |
| 1024 | 228.4 | 54.5 | 8.3 | 4.19x | 27.5x |
| 2048 | 230.1 | 45.1 | 6.7 | 5.10x | 34.3x |
| 4096 | 249.8 | 45.7 | 3.8 | 5.47x | 65.4x |

TRN throughput is constant (~230-256 tps) across all context lengths.
TF+KV throughput degrades due to growing cache (prefill O(n^2), decode O(n) attention per step).
TF_full throughput degrades as O(n^2) per token (no cache, full recompute).

**Note:** TF+KV implements a real KV cache with prefill + single-token decode steps.
The speedup vs KV grows with context length because KV attention cost scales linearly
with cached sequence length, while TRN state update is O(1).

## Agent History Simulation

```bash
python scripts/bench_agent_history.py \
    --checkpoints 1000,2000,5000,10000,20000,50000,100000 \
    --gen-tokens 64 --device cpu
```

Simulates a long-running agent session with growing conversation history.
Measures memory and latency scaling at each checkpoint.

Output: `results/bench_agent_history.csv`

Columns: `history_tokens, trn_tps, trn_state_kb, tf_kv_tps, tf_kv_cache_mb, tf_full_tps`

Key comparison:
- TRN state: constant 8 KB regardless of history length
- TF KV cache: grows linearly (15.6 MB at 1k -> 156.3 MB at 10k tokens)

### Validated Results (d_model=256, n_layers=8, d_ff=1024, n_osc=128, CPU, seed=42)

| history_tokens | TRN tps | TRN state | TF+KV tps | TF KV cache | TF_full tps |
|----------------|---------|-----------|-----------|-------------|-------------|
| 1,000 | 239.7 | 8.0 KB | 73.8 | 15.6 MB | 13.3 |
| 5,000 | 243.9 | 8.0 KB | 35.9 | 78.1 MB | 2.7 |
| 10,000 | 230.9 | 8.0 KB | 15.5 | 156.3 MB | 1.0 |

TRN state is constant at 8 KB (n_layers * n_osc * 2 complex * 4 bytes = 8192 bytes).
TF KV cache grows linearly: n_layers * 2 * n_heads * head_dim * history_tokens * 4 bytes.
At 10k tokens, TRN is 14.9x faster than TF+KV and 231x faster than TF_full.

## Streaming Tasks

```bash
python scripts/bench_stream_tasks.py \
    --tasks timeseries,smoothing,char_lm,running_mean \
    --steps 1000 --device cpu
```

Tasks designed to test TRN's recurrence-based memory advantage:

| Task | seq_len | Description |
|------|---------|-------------|
| `timeseries` | 64 | Next-step prediction of quantized sine + noise |
| `smoothing` | 64 | Moving average prediction (window=4) |
| `char_lm` | 64 | Character-level language modeling on synthetic text |
| `running_mean` | 64 | Predict running mean of input token values |

Output: `results/bench_stream_tasks.csv`

Columns: `task, TRN_final_loss, TF_final_loss, Hybrid_final_loss, best`

### Validated Results (d_model=128, n_layers=4, 300 steps, CPU, seed=42)

| Task | TRN val_loss | TF val_loss | Hybrid val_loss | Best |
|------|-------------|-------------|-----------------|------|
| timeseries | 3.970 | 4.054 | 3.909 | Hybrid |
| smoothing | 3.174 | 3.236 | 3.146 | Hybrid |
| running_mean | 4.174 | 4.172 | 4.172 | Hybrid |

TRN outperforms TF on timeseries (-2.1%) and smoothing (-1.9%). These are tasks
where recurrence-based memory (damped oscillatory integration) provides a structural
advantage over attention. Hybrid wins all tasks, suggesting the combination of
recurrence (for temporal aggregation) and attention (for precise recall) is complementary.
running_mean shows all three models at near-identical loss -- the task may be too simple
to differentiate at 300 steps.

## Hyperparameters

Fixed and identical for TRN and Transformer in all benchmarks:

| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| LR | 3e-4 |
| LR schedule | Cosine with 10% linear warmup |
| Weight decay | 0.1 (weight matrices only) |
| Grad clip | 1.0 |
| Batch size | 32 |
| Dropout | 0.0 |
| Seed | 42 |

## Model Configs

| Param | TRN | Transformer |
|-------|-----|-------------|
| d_model | 128 | 128 |
| n_layers | 4 | 4 |
| d_ff | 512 | 512 |
| n_oscillators | 64 | -- |
| n_heads | -- | 2 |
| ~params | ~0.95M | ~1.05M |

## Acceptance Criteria

| Criterion | Threshold |
|-----------|-----------|
| TRN/TF loss ratio (any task, 5k steps) | <= 1.20 |
| TRN generation speedup (gen_len >= 1024) | > 1.0x |
| TRN state memory growth (512 vs 2048 tokens) | < 1% |
| Determinism (same seed, same losses) | exact match |

## Hybrid Architecture

Evaluates a hybrid model combining TRN recurrence layers with sparse attention heads, motivated by the Go/No-Go finding that TRN fails selective copy (accuracy 0.088 at 5000 steps vs Transformer 0.962).

Architecture under test: TRN layers for context aggregation + small number of attention heads for content-addressed retrieval.

```bash
python scripts/bench_sequence_tasks.py --model hybrid --tasks all --steps 5000
```

Output: `scripts/results/bench_sequence_tasks_hybrid.csv`

Columns: `model, task, step, train_loss, val_loss, accuracy`

*Results pending. See `docs/TRN_ARCHITECTURE_ANALYSIS.md` Section 6.1.*

## Sequence Task Evaluation

Evaluates TRN, Transformer, and Hybrid on five retrieval/memory tasks designed to probe selective information retention.

```bash
python scripts/bench_sequence_tasks.py \
    --models trn,transformer,hybrid \
    --tasks copy,selective_copy,reverse,induction,assoc_recall \
    --steps 5000
```

Tasks:

| Task | seq_len | Description |
|------|---------|-------------|
| `copy` | 64 | Periodic sequence repetition (period=8) |
| `selective_copy` | 128 | Reproduce 8 marked tokens from among noise after separator |
| `reverse` | 32 | Reproduce input sequence in reverse order |
| `induction` | 64 | Complete induction head patterns |
| `assoc_recall` | 64 | Retrieve value for given key from 4 key-value pairs |

Output: `scripts/results/bench_sequence_tasks.csv`

Columns: `model, task, step, train_loss, val_loss, accuracy`

Baseline from Go/No-Go test (5000 steps, CPU, seed=42):

| Model | Task | Final Accuracy |
|-------|------|----------------|
| TRN | copy | 0.000000 |
| TRN | selective_copy | 0.088000 |
| Transformer | selective_copy | 0.962250 |

*Full results pending.*

## Long-Context Scaling

Measures autoregressive generation throughput and memory as context length increases.

```bash
python scripts/bench_long_context_scaling.py \
    --context-lens 512,1024,2048,4096,8192,16384 \
    --gen-tokens 128 --device cpu
```

Output: `results/long_context_scaling.csv`

### Validated Results (d_model=256, n_layers=8, d_ff=1024, n_osc=128, CPU, seed=42)

| ctx_len | TRN tps | TF tps | Speedup | TRN mem (MB) |
|---------|---------|--------|---------|--------------|
| 512 | 255.0 | 27.1 | 9.4x | 0.005 |
| 1024 | 251.1 | 16.5 | 15.2x | 0.005 |
| 2048 | 247.5 | 9.2 | 26.9x | 0.005 |
| 4096 | 255.3 | 3.9 | 64.7x | 0.005 |
| 8192 | 232.7 | 1.2 | 194.2x | 0.007 |
| 16384 | 224.7 | 0.4 | 575.8x | 0.006 |

TRN throughput is constant (~225-255 tps) across all context lengths.
TF throughput degrades as O(n) per token (no KV cache).

### Fairness Validation

Both models benchmarked under identical conditions:

- dtype: float32, device: CPU, vocab: 256, batch_size: 1
- gen_tokens: 128, temperature: 1.0, top_k: 50
- TRN: `model.generate()` using O(1) `step_single` per token
- TF: full forward pass per token (no KV cache -- intentional worst-case)
- Timing: `time.perf_counter()`, 1 warmup + 2 timed runs with stddev
- Memory: `tracemalloc` peak (Python-level, batch_size=1)

Validation script: `scripts/bench_long_context_validation.py`

### Theoretical State Sizes

TRN state per batch element: `128 osc * 2 (real+imag) * 4 bytes = 1024 bytes` (constant)

TF KV cache per batch element (if implemented):
`8 layers * 2 * 4 heads * 64 head_dim * ctx_len * 4 bytes`

| ctx_len | TF KV cache |
|---------|-------------|
| 512 | 8 MB |
| 1024 | 16 MB |
| 4096 | 64 MB |

### Important Caveat

The TF baseline has **no KV cache**, making it O(n) per generation step.
A KV-cached Transformer would be significantly faster (O(1) per step with
O(n*d) memory). The speedup numbers reflect TRN's advantage over the
naive full-recompute baseline, not over optimized Transformer inference.
