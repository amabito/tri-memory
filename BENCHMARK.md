# Benchmark Protocol

All measurements are reproducible from a single command. Artifacts are saved with
environment metadata, random seed, and git hash for independent verification.

## Measurement Conditions (Fixed)

Every benchmark script enforces these defaults unless explicitly overridden:

| Parameter    | Value          | Notes                                         |
|-------------|----------------|-----------------------------------------------|
| seed        | 42             | `seed_everything(42)` before each measurement |
| dtype       | fp32           | `--dtype fp32` (bf16 available via `--dtype bf16`) |
| batch_size  | 1              | single-sequence inference                     |
| temperature | greedy (argmax)| no sampling; `argmax` on logits               |
| warmup      | 3 steps        | discarded before timing (`--warmup-steps 3`)  |
| n_repeats   | 3              | median of 3 timed runs                        |
| device      | cpu or cuda    | `--device cpu` for CI, `--device cuda` for GPU |
| torch_compile | off          | `--torch-compile` to enable (optional)        |

Changing any of these invalidates comparison with published numbers.

## Baseline Definitions

| Mode     | Model Class         | Generation Method | Memory Model |
|----------|---------------------|-------------------|-------------|
| `trn`    | TRNModel            | `model.generate()` using `step_single()` | O(1) resonance state |
| `tf_kv`  | TransformerModel    | Prefill + KV cache decode (`_kv_decode_step`) | O(n) KV cache |
| `tf_full`| TransformerModel    | Full forward pass per step (no cache) | O(1) but O(n^2) compute |
| `dual`   | DualMemoryEngine    | KV window (FIFO) + TRN step_single + mixer | O(W) KV + O(1) TRN |

`tf_kv` is the standard Transformer inference mode.
`tf_full` is included as a worst-case reference (no caching optimization).
`dual` is the DualMemoryEngine with windowed attention + TRN state.

## Model Configurations

| Name       | d_model | n_layers | K (oscillators) | d_ff  | max_seq_len | Params  |
|------------|---------|----------|-----------------|-------|-------------|---------|
| toy        | 128     | 2        | 64              | 512   | 512         | ~0.5M   |
| bench      | 128     | 4        | 64              | 512   | 4,096       | ~1.0M   |
| bench_long | 256     | 8        | 128             | 1,024 | 16,640      | ~2.3M   |
| trn_100m   | 512     | 8        | 256             | 2,048 | 2,048       | ~100M   |
| trn_400m   | 1,024   | 16       | 512             | 4,096 | 4,096       | ~400M   |
| trn_1b     | 2,048   | 24       | 512             | 8,192 | 4,096       | ~1B     |

`bench` is used by sequence/needle-haystack benchmarks. `bench_long` is used by
agent history and long-context scaling benchmarks.

### Memory Formulas

TRN state size = `n_layers * K * 2 * 4` bytes (fp32). Does not depend on context length.

KV cache size = `n_layers * 2 * n_heads * T * head_dim * 4` bytes (fp32 K+V, MHA).
n_heads = d_model / 64, head_dim = 64. Grows linearly with context length T.

DualMemoryEngine state = `n_layers * (K * 2 * 4 + n_heads * W * head_dim * 2 * dtype_bytes)`.
Constant regardless of total context (depends on window size W, not history length T).

## OOM / Timeout Handling

- If a model runs out of memory during measurement, the metric is recorded as `NaN`.
- If a context length exceeds `max_seq_len` for a model, that row is marked `SKIP`.
- OOM events are logged with the exception message but do not abort the full benchmark.
- `NaN` values propagate to CSV and are excluded from aggregate statistics.

## Benchmarks

### 1. GPU Benchmark (Phase 7)

```bash
python scripts/bench_phase7_gpu.py \
    --models trn_100m,tf_kv \
    --context-lens 512,1024,2048,4096,8192 \
    --device cuda \
    --dtype fp32 \
    --warmup-steps 3 \
    --n-repeats 3
```

**Output:** `artifacts/phase7/{timestamp}/`

| File             | Format | Contents                                   |
|------------------|--------|--------------------------------------------|
| `results.json`   | JSON   | Array of per-model per-context measurements |
| `summary.md`     | MD     | Human-readable table                       |
| `env.json`       | JSON   | seed, dtype, torch, CUDA, git hash         |
| `nvidia_smi.txt` | text   | `nvidia-smi` output at measurement time    |

**Metrics per row:**

| Column              | Unit   | Description                              |
|---------------------|--------|------------------------------------------|
| `prefill_latency_ms`| ms     | Forward pass on full context (median)    |
| `decode_tps`        | tok/s  | Tokens per second during generation      |
| `decode_latency_ms` | ms     | Total decode time                        |
| `peak_vram_mb`      | MB     | `torch.cuda.max_memory_allocated`        |
| `state_memory_mb`   | MB     | Analytical: TRN state or KV cache formula|
| `speedup_vs_kv`     | ratio  | `kv_cache_mb / trn_state_mb`             |

**Speedup claims:** The `speedup_vs_kv` column is a memory ratio, not a throughput ratio.
Throughput comparison uses `decode_tps` directly between models at the same context length.

### 2. DualMemoryEngine Benchmark (Phase 8)

```bash
python scripts/bench_vllm_trn.py \
    --models dual_100m_w64,dual_100m_w256,dual_100m_w1024,tf_kv,trn_100m \
    --context-lens 1024,4096,16384,65536,131072 \
    --device cuda
```

**Output:** `artifacts/phase7/{timestamp}/dual/`

Same artifact format as Phase 7. Additional model variants:
- `dual_100m_w64`: DualMemoryEngine with W=64 token KV window
- `dual_100m_w256`: W=256
- `dual_100m_w1024`: W=1024

DualMemoryEngine decode memory is constant at `W * head_bytes + TRN_state` regardless
of total context length T. At T >> W, memory savings approximate those of standalone TRN.

### 3. Information Retention (Needle-in-Haystack)

```bash
python scripts/bench_needle_haystack.py --device cpu
python scripts/bench_needle_haystack.py --device cpu --backend dual_w64
```

**Output:** `results/bench_needle_haystack.csv`

Four sub-tasks:

| Task | What It Tests | TRN Result |
|------|---------------|------------|
| NiH  | Exact token recall at arbitrary distance | 0.0 (fail) |
| TRP  | Linear probe reconstruction of recent tokens | Partial |
| PPD  | Frequency classification from final hidden state | 0.78--1.00 |
| GT   | Goal token tracking over filler tokens | ~0.25 at d > W (chance) |

NiH = 0.0 and GT = chance at distance > W are structural limitations of linear recurrence.
PPD near-parity with Transformer confirms TRN captures frequency/pattern information.
See [docs/TRN_LIMITATIONS.md](docs/TRN_LIMITATIONS.md).

### 4. Selective Recall Mitigation

```bash
python scripts/bench_selective_recall.py --device cpu
```

**Output:** `results/bench_selective_recall.csv`

Strategies evaluated: `vanilla_trn`, `trn_skes` (State-Keyed External Store),
`trn_piss` (Priority Importance State Slots), `tf_kv` (Transformer baseline).

Context: TRN selective copy accuracy is 8.8% (vs Transformer 96.2%).
This benchmark measures whether SKES/PISS augmentation can partially recover recall.

### 5. Multi-Agent Scaling

```bash
python scripts/bench_multi_agent.py \
    --agent-counts 10,100,1000,10000 \
    --device cpu
```

**Output:** `results/bench_multi_agent.csv`

Measures analytical TRN state vs KV cache per agent, `actual_peak_memory_mb` via
`tracemalloc`, and GPU cost estimate (A100/H100 based on VRAM needs).

**Ratio claims:** The "Nx reduction" column shows `kv_total_mb / trn_total_mb` for a
specific model config and history length:

| Config   | T     | Ratio  | Formula: (d_model * T) / K |
|----------|-------|--------|---------------------------|
| trn_100m | 1,000 | 2,000x | (512 * 1000) / 256        |
| trn_100m | 4,096 | 8,192x | (512 * 4096) / 256        |
| trn_400m | 1,000 | 2,000x | (1024 * 1000) / 512       |
| trn_1b   | 1,000 | 4,000x | (2048 * 1000) / 512       |

### 6. Go/No-Go Evaluation

```bash
python scripts/eval_go_no_go.py --device cpu
python scripts/eval_go_no_go.py --device cpu --backend dual
```

**Output:**

| File                         | Format | Contents                            |
|------------------------------|--------|-------------------------------------|
| `results/eval_go_no_go.csv`  | CSV    | All criteria with status/value      |
| `results/gate_result.json`   | JSON   | Verdict + input file SHA-256 hashes |
| `results/gate_result.md`     | MD     | Human-readable verdict summary      |

Backend `dual` produces `gate_result_dual.json` and `gate_result_dual.md`.

**Tier structure:**

- **T1 (Mandatory):** speedup >= 2x, memory reduction >= 10x, accuracy degradation <= 15%,
  state constant, agent scale >= 20x, numerical stable. Dual adds: PPD window generalization.
- **T2 (Quality):** copy task, selective copy, convergence speed, long-context TPS, state size.
  Dual adds: NiH long-range (reference), GT window/reversal recovery (reference).
- **T3 (Performance):** absolute TPS, KV growth, agent scaling, throughput/MB, copy accuracy.

Verdict logic:
- **GO**: T1 all PASS, T1 zero SKIP, T2 zero FAIL
- **CONDITIONAL_GO**: T1 zero FAIL, T1 <= 1 SKIP, T2 <= 2 FAIL
- **NO_GO**: T1 >= 2 FAIL, or (T1 FAIL + SKIP) >= 3

### 7. C Resonance Benchmark (llama.cpp-style)

```bash
python scripts/bench_llamacpp_trn.py --device cpu
```

**Output:** `artifacts/phase7/{timestamp}/llamacpp/`

Validates C implementation of TRN `step_single` against Python reference.
Correctness criterion: max absolute error < 1e-5 over 100 random tokens.
Reports C implementation TPS for single-token inference.

### 8. Agent History Simulation

```bash
python scripts/bench_agent_history.py --checkpoints 1000,5000,10000 --device cpu
```

**Output:** `results/bench_agent_history.csv`

Uses `bench_long` config (d=256, L=8, K=128). Speedup claims from this benchmark
apply to this config, not trn_100m.

### 9. Streaming Tasks

```bash
python scripts/bench_stream_tasks.py --tasks timeseries,smoothing,char_lm,running_mean --device cpu
```

**Output:** `results/bench_stream_tasks.csv`, `results/stream/{task}_curves.csv`

## Hyperparameters (Training Benchmarks)

Identical for TRN and Transformer unless noted:

| Parameter    | Value                     |
|-------------|---------------------------|
| Optimizer   | AdamW (betas=0.9, 0.95)   |
| LR          | 3e-4                      |
| Min LR      | 3e-5                      |
| Schedule    | Cosine with 10% warmup    |
| Weight decay| 0.1 (matrices only)       |
| Grad clip   | 1.0                       |
| Batch size  | 32                        |
| Dropout     | 0.0                       |
| Seed        | 42                        |

## Report Format

All CSV files use the following conventions:

- Header row with lowercase snake_case column names
- Floating point values: 4 decimal places for ratios/accuracy, 2 for latency/memory
- `NaN` for OOM or skipped measurements
- Units specified in column name suffix: `_ms`, `_mb`, `_kb`, `_tps`, `_bytes`

## Reproducing Results

```bash
# Full reproduction from scratch
pip install -e .

# 1. Generate TF reference
python scripts/gen_copy_tf_reference.py --device cpu --seed 42

# 2. Run GPU benchmark
python scripts/bench_phase7_gpu.py --models trn_100m,tf_kv --device cuda

# 3. Run DualMemoryEngine benchmark
python scripts/bench_vllm_trn.py --models dual_100m_w64,tf_kv --device cuda

# 4. Run agent history
python scripts/bench_agent_history.py --device cpu

# 5. Run information retention
python scripts/bench_needle_haystack.py --device cpu

# 6. Run Go/No-Go (both backends)
python scripts/eval_go_no_go.py --device cpu
python scripts/eval_go_no_go.py --device cpu --backend dual

# 7. Verify
cat results/gate_result.md
cat results/gate_result_dual.md
```

Artifacts directory contains `env.json` with the exact git hash, torch version,
and CUDA version used for each run. Compare against your environment before
interpreting differences.
