# Changelog

## [0.1.1] -- 2026-03-14 -- Mathematical Correctness Hardening

F.R.I.D.A.Y. 3-body parallel review found 34 bugs across 11 source files.
3-round review-fix loop with 2 consecutive CLEAN rounds verified all fixes.

### Critical Fixes

- **oscillator.py**: Clamp omega to (0, pi) to prevent Nyquist aliasing.
  High-k oscillators were silently aliasing to low frequencies, halving
  representational capacity of the oscillator bank.
- **scan.py**: Clone alpha tensor before second `associative_scan` call.
  Potential in-place mutation was corrupting the imaginary decay sequence.
- **tri_memory.py**: Reject `window_size=0` (produces all-inf mask, NaN in SDPA).
- **tri_memory.py**: Remove dead `StateTokenAdapter` instantiation (allocated
  parameters but never called in `forward()`).
- **hybrid_model.py**: Fix `trn_ratio=1.0` (float) being treated as integer
  count 1 instead of 100% TRN ratio.

### High-Severity Fixes

- **resonance.py**: Replace L-inf (max-abs) state normalization with complex
  modulus. L-inf suppressed the smaller of r_r/r_i, destroying phase information.
- **resonance.py**: Move `alpha.float()` before `one_m_a` computation in both
  `forward()` and `step_single()`. Under bf16 AMP, `alpha=0.99` rounded to
  `1.0`, making `one_m_a=0` (latch bug).
- **resonance.py**: Dynamic `autocast` device type covers CPU bf16, not just CUDA.
- **scan.py**: Fix cumsum-rescale double-correction. Small-alpha positions now
  use `drive * alpha_cum` (correct O(alpha_cum) approximation) instead of raw
  `drive` (incorrect O(1) error).
- **scan.py**: `SafeCumprod.backward` returns `(grad_input, None)` tuple,
  not bare tensor. Fixes autograd contract for explicit `dim` argument.
- **config.py**: Align `gate_bias_init` default from 0.85 (unstable) to 0.65
  (P0 stabilization). Matching defaults also updated in `oscillator.py`.
- **tri_memory.py**: Raise `ValueError` when input length exceeds `max_seq_len`
  in both `forward()` and `forward_with_memory()`.
- **tri_memory.py**: Add `retrieval_temperature` backward-compat shim after
  rename to `retrieval_sharpness`.

### Medium-Severity Fixes

- **oscillator.py**: Replace softplus+tanh amplitude clamp with softplus+clamp
  to prevent gradient vanishing at saturation.
- **oscillator.py**: Add `min=1e-4` lower bound to omega clamp (prevents
  negative omega_base from gradient drift).
- **resonance.py**: Add epsilon (1e-8) inside sqrt in `_apply_state_norm` to
  prevent NaN gradient at zero state.
- **scan.py**: Add `neginf=-1e30` to `nan_to_num` in stats collection.
- **scan.py**: Pass `dim=1` explicitly to `SafeCumprod.apply`.
- **tri_memory.py**: Use `torch.finfo(dtype).min` instead of `-1e4` for
  disabled gate masking (exact zero via softmax).
- **tri_memory.py**: Move batch-size warning outside per-token loop.
- **hybrid_model.py**: Replace `assert` with `ValueError` for production safety.
- **hybrid_model.py**: Guard `cfg.n_layers >= 1`.
- **model.py**: Add `ignore_index=-100` to cross_entropy for padding safety.
- **utils.py**: Fix sinusoidal PE shape mismatch for odd `d_model`.
- **saliency.py**: Use `std(correction=0)` for population std. Apply threshold
  floor unconditionally in `GoalAwareSaliencyArchiver.should_archive`.
- **retrieval.py**: Convert deque to list in `search_with_scores` and
  `search_by_metadata` for O(1) index access and `.sort()` compatibility.
- **consolidation.py**: Document thread-safety constraint on `rescore_and_prune`.

### Benchmark (post-fix, RTX 5090, synthetic data)

| Config | Params | Final Loss | Throughput | Stability (3-seed std) |
|--------|--------|-----------|------------|----------------------|
| Toy (d=128, K=64, L=2) | 509K | 5.5456 | 93K tok/s | 0.001 |
| TRN-100M (d=512, K=256, L=8) | 40.5M | 10.3879 | 6.8K tok/s | -- |
