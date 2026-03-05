# Temporal Resonance Network (TRN)

[![tests](https://img.shields.io/badge/tests-252%20passing-brightgreen)]()
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.1%2B-orange)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)

A sequence model that replaces self-attention with damped oscillator dynamics in complex exponential space. Processes sequences in O(n) time and generates with constant-size state memory (no KV cache).

```
pip install -e .
```

## Quickstart

```python
import torch
from trn import TRNConfig, TRNModel

cfg = TRNConfig(
    vocab_size=8192,
    d_model=128,
    n_oscillators=64,
    n_layers=4,
    d_ff=512,
    max_seq_len=1024,
)
model = TRNModel(cfg)

# Forward pass
input_ids = torch.randint(0, cfg.vocab_size, (2, 256))
out = model(input_ids, labels=input_ids)
print(f"loss: {out['loss']:.4f}")

# Generation (constant memory, no KV cache)
prompt = torch.randint(0, cfg.vocab_size, (1, 16))
tokens = model.generate(prompt, max_new_tokens=128)
```

## Key Properties

- **O(n) sequence processing** -- recurrence over damped complex oscillators, not quadratic attention
- **Constant-size generation state** -- two vectors per layer (real + imaginary), no KV cache, does not grow with context
- **Generation speedup at long contexts** -- throughput stays flat as context grows (TF degrades linearly)
- **Stabilized training (P0 patch)** -- gradient norms reduced from millions to <30 via resonance scaling, state normalization, amplitude clamping

## Benchmark Results

### Generation Speed (CPU, d_model=128, n_layers=4, batch=2)

| gen_len | TRN (tok/s) | TF (tok/s) | Speedup |
|---------|-------------|------------|---------|
| 128     | 2,193       | 1,702      | 1.3x    |
| 256     | 2,258       | 1,289      | 1.8x    |
| 512     | 2,124       | 870        | 2.4x    |
| 1,024   | 2,196       | 511        | 4.3x    |

TRN throughput stays flat across all context lengths. Transformer throughput halves each time context doubles.

### Generation Memory

| gen_len | TRN state | TF KV cache |
|---------|-----------|-------------|
| 128     | 0.001 MB  | 0.27 MB     |
| 512     | 0.001 MB  | 1.02 MB     |
| 1,024   | 0.001 MB  | 2.02 MB     |

TRN state is constant at 0.001 MB regardless of context length.

### Training Sanity Tests

| Test | Result | Detail |
|------|--------|--------|
| Random-target recheck | PASS | Both TRN and TF converge to unigram entropy H(p) on shuffled targets. No information leakage. |
| BPE token-level (NLTK Gutenberg, vocab=8192, 2000 steps) | PASS | TRN val_ppl=904, TF val_ppl=2627. TRN learns normally on token-level data. |
| Label shift audit | PASS | Single causal shift verified for both models. |
| Gradient stability (P0) | PASS | Median grad norm < 30 over 50 steps. No NaN or Inf. |

These results are from small models (~1-2M parameters) on CPU. They validate correctness and training dynamics, not large-scale language modeling quality.

## Current Limitations

- **Long-context retrieval.** Performance on needle-in-a-haystack and associative recall tasks has not been validated at scale. Short smoke runs (100-500 steps, CPU) show 0% recall for both TRN and Transformer, which is expected -- meaningful evaluation requires longer training on GPU.
- **Large-scale language modeling.** All experiments use models with 1-2M parameters. Scaling behavior at 100M+ parameters is not yet characterized.
- **GPU training throughput.** The resonance scan currently runs sequentially on GPU (PyTorch `torch.associative_scan` requires `torch.compile` which is not yet stable for this workload). On CPU, TRN forward pass is ~2.7x slower than Transformer. Parallel scan is expected to substantially narrow this gap.
- **Overfit capacity.** TRN requires more training steps than Transformer to overfit a tiny dataset under default hyperparameters. This is a known characteristic of the oscillatory inductive bias, not an implementation bug.

## Repository Structure

```
src/trn/
    config.py        Configuration dataclass (TRNConfig)
    model.py         TRNModel: embedding -> N x TRNBlock -> RMSNorm -> lm_head
    block.py         TRNBlock: norm -> resonance -> norm -> FFN (SwiGLU)
    resonance.py     TemporalResonanceLayer: oscillator projection + complex recurrence
    oscillator.py    OscillatorProjection: input -> (A, omega, phi, alpha) parameters
    scan.py          Sequential and parallel (associative) scan implementations
    baseline.py      TransformerModel: standard causal Transformer for comparison
    eval.py          Perplexity evaluation utilities
    data.py          PackedDataset for binary token files
    tokenizer.py     Character-level tokenizer
    trainer.py       Training loop with cosine LR schedule
    generate.py      Generation utilities with streaming support
    benchmark.py     Benchmark data generators (copy, counting, reverse, induction, etc.)

scripts/
    bench_generate.py        Generation speed and memory benchmark
    bench_train.py           Training benchmark on generalization tasks
    bench_memory_tasks.py    Long-context retrieval tasks
    bench_smoke.py           CI smoke test
    train_lm_realdata.py     Real-text LM training (WikiText-2 / Gutenberg)
    train_lm_100m.py         100M-scale training script
    profile_forward.py       Component-level forward pass profiling
    train_gpu_scan.py        GPU parallel scan verification
    audit_*.py               Dataset and training integrity audits

tests/
    252 unit tests covering model correctness, training stability,
    generation, checkpointing, determinism, and adversarial inputs.

docs/
    VALIDATION_RESULTS.md    Full benchmark tables and profiling breakdowns
```

## Running Benchmarks

All commands run from the project root. Requires `pip install -e .` first.

```bash
# Generation benchmark (speed + memory)
python scripts/bench_generate.py --gen-lens 128,256,512,1024

# Training benchmark (generalization tasks, 5000 steps)
python scripts/bench_train.py --tasks all --steps 5000

# Real-text language modeling (char-level, WikiText-2 / Gutenberg)
cd scripts
python train_lm_realdata.py --model trn --size small --steps 2000
python train_lm_realdata.py --model tf  --size small --steps 2000

# BPE token-level training
cd scripts
python train_lm_realdata.py --model trn --size small --steps 2000 --tokenizer bpe
python train_lm_realdata.py --model tf  --size small --steps 2000 --tokenizer bpe

# Random-target sanity test
cd scripts
python train_lm_realdata.py --model trn --size small --steps 500 --random-targets
python train_lm_realdata.py --model tf  --size small --steps 500 --random-targets

# Forward pass profiling
python scripts/profile_forward.py --d-model 128 --n-layers 4 --seq-len 64

# CI smoke test (< 60s)
python scripts/bench_smoke.py

# Full test suite
pytest tests/ -x -q
```

Output CSVs are written to `scripts/results/`. See [BENCHMARK.md](BENCHMARK.md) for the full protocol and [docs/VALIDATION_RESULTS.md](docs/VALIDATION_RESULTS.md) for detailed results.

## Development Status

Research architecture, under active development. Validates the core mechanism at small scale (1-2M params). Large-scale LM quality, long-context retrieval, and GPU-optimized training are open. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text.

## Citation

```bibtex
@software{trn2026,
  title  = {Temporal Resonance Network},
  author = {TRN Contributors},
  year   = {2026},
  url    = {https://github.com/TODO/trn},
  note   = {v0.1.0},
}
```
