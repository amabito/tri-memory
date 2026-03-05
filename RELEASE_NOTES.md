# TRN v0.1.0 -- Initial Release

First public release of the Temporal Resonance Network.

## What is TRN

A sequence model that replaces self-attention with damped oscillator dynamics. Tokens interact through a fixed-size complex exponential state that is updated recurrently, yielding O(n) sequence processing and constant-memory generation without a KV cache.

## This Release Includes

### Core Architecture

- `TRNModel`: embedding -> N x TRNBlock -> RMSNorm -> lm_head
- `TemporalResonanceLayer`: oscillator projection + complex exponential recurrence
- `OscillatorProjection`: learned amplitude, frequency, phase, and decay parameters
- Sequential and parallel (associative) scan implementations
- O(1) memory generation via `step_single` recurrence
- Weight tying, SwiGLU FFN, RMSNorm

### P0 Stabilization Patch Set

Gradient explosion fix for stable training:

- **Resonance output scaling** (`res_scale`): learnable scalar initialized to 0.05, attenuates resonance contribution at early training
- **State normalization** (`state_norm`): per-channel max-abs clamping after each recurrence step
- **Amplitude bound** (`amplitude_max=3.0`): tighter softplus ceiling prevents runaway oscillator amplitudes
- **Gate bias initialization** (`gate_bias_init=0.85`): sigmoid-inverse bias for conservative initial decay rate
- **Log-phase positioning** (`phase_mode="log"`): omega * log(i+1) compresses positional angles, prevents explosion at long sequences
- **Optimizer parameter groups**: omega_base, res_scale, and all bias/norm parameters excluded from weight decay

Result: gradient norms reduced from 3-6 million (pre-P0) to 2-27 (post-P0). Training loss decreases steadily over 50 steps with no NaN or Inf.

### Benchmark Framework

- `bench_generate.py`: generation throughput and memory measurement
- `bench_train.py`: training on 5 generalization tasks (copy, counting, reverse, induction, associative recall)
- `bench_memory_tasks.py`: long-context retrieval tasks with context length sweep
- `train_lm_realdata.py`: real-text language modeling with char-level and BPE tokenization
- `train_lm_100m.py`: 100M-parameter scale training script
- `profile_forward.py`: component-level forward pass profiling with Chrome traces
- `bench_smoke.py`: CI smoke test (< 60s)

### Sanity Validation Suite

- **Random-target recheck**: both TRN and Transformer converge to unigram entropy on shuffled targets, confirming no information leakage
- **BPE token-level training**: TRN trains normally on 8192-vocab BPE data (NLTK Gutenberg), ruling out char-level artifacts
- **Label shift audit**: single causal shift verified for both models
- **Overfit capacity audit**: documents TRN's slower overfitting under default settings
- **Dataset split audit**: train/val overlap detection
- **Learning curve audit**: 1000-step training with gradient and gate statistics

### Test Suite

252 unit tests covering:

- Model forward/backward correctness
- Generation (autoregressive, streaming, edge cases)
- Training stability (P0 regression tests)
- Oscillator parameter ranges and adversarial inputs
- Checkpoint save/load round-trip
- Determinism (same seed = same losses)
- Scan correctness (sequential vs parallel)

## Verified Results

| Property | Status |
|----------|--------|
| O(1) generation memory | Confirmed: 0.001 MB at all context lengths |
| Generation speedup at long context | 4.3x at 1024 tokens (CPU) |
| Gradient stability (P0) | Median grad norm < 30 over 50 steps |
| Random-target sanity | PASS: no leakage detected |
| BPE training sanity | PASS: normal learning on token-level data |
| 252 unit tests | All passing |

## Known Limitations

- Long-context retrieval not yet validated at scale
- Large-scale (100M+) language modeling not yet characterized
- GPU parallel scan requires torch.compile (not yet stable for this workload)
- TRN overfits slower than Transformer under default hyperparameters

## Requirements

- Python 3.10+
- PyTorch 2.1+
- NumPy 1.24+

Optional: `tokenizers` (for BPE mode), `nltk` or `datasets` (for real text data)

## License

Apache License 2.0
