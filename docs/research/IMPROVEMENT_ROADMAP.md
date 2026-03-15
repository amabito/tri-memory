# Tri-Memory Improvement Roadmap

Generated: 2026-03-14
Source: Karen 3-body research + F.R.I.D.A.Y. 3-body analysis
Files reviewed: 20+ source files, 25+ SOTA architectures, 32 datasets, 42 training techniques

---

## Tier 0: Immediate (0 LOC / config only)

| # | Change | Impact | Source |
|---|--------|--------|--------|
| 0a | `window_size` 64 -> 512 | -0.5 to -1.5 ppl | Griffin uses 2048 |
| 0b | `trn_ratio` 0.5 -> 1/6 (HybridModel) | -0.3 to -0.8 ppl | Jamba 1:7, Zamba 1:6 |
| 0c | `scan_chunk_size` 16 -> 64 | 2x-4x chunked scan speed | Python loop overhead |

---

## Tier 1: Quick Wins (1-line to 1-day fixes)

| # | Change | File | Impact | Effort |
|---|--------|------|--------|--------|
| 1a | SimpleTrainer beta2=0.95 | trainer.py:157 | Training stability | 1 line |
| 1b | `pin_memory=True` for CUDA | data.py:60 | 20-40% data loading | 1 line |
| 1c | `num_workers=4` default | data.py:59 | Eliminate CPU bottleneck | 1 line |
| 1d | Remove `emb.clone()` | tri_memory.py:819 | Free memory | 1 line |
| 1e | Gate SafeCumprod stats behind flag | scan.py:68-93 | 10-30% backward speed | 10 lines |
| 1f | Fuse gate+up in SwiGLU | block.py + tri_memory.py | 1.5x-2x FFN | 20 lines |
| 1g | Output gate (GLA-style) | oscillator.py + resonance.py | -0.5 to -0.8 ppl | 30 lines |
| 1h | Re+Im demodulation output | resonance.py | -0.4 to -0.7 ppl | 20 lines |
| 1i | Window mask as registered buffer | tri_memory.py | 3 kernel launches saved/block | 15 lines |

---

## Tier 2: Short-Term (1-3 days each)

### Architecture

| # | Change | Impact | Effort | SOTA Reference |
|---|--------|--------|--------|----------------|
| 2a | Delta rule (write-with-erase) | +3-5% recall, -0.3 ppl | 2d | Gated DeltaNet (ICLR 2025) |
| 2b | Attention sink tokens | Fix streaming correctness | 1d | StreamingLLM |
| 2c | Sigmoid per-tier gate (beta-mix) | -0.5 to -1.0 ppl | 1d | Infini-Attention |
| 2d | Vectorized batch retrieval | 50-200x search speed | 1d | Matrix cosine similarity |
| 2e | StateTokenAdapter integration | -0.4 to -0.6 ppl | 2d | Hymba meta-tokens |
| 2f | Sparse token bag (dict/sparse) | 200x bag memory reduction | 1d | -- |

### Training

| # | Change | Impact | Effort | SOTA Reference |
|---|--------|--------|--------|----------------|
| 2g | WSD LR schedule | +3-5% benchmarks + branching | 1d | MiniCPM, Llama-3.1 |
| 2h | Document boundary state reset | Fix cross-doc contamination | 2d | Mamba/Griffin standard |
| 2i | AMP training (bf16) | 1.5-2x memory, 30-50% speed | 1d | Standard practice |
| 2j | Sequence length curriculum | +3.7% benchmarks, 22% faster | 2d | SkyLadder (NeurIPS 2025) |
| 2k | Gradient checkpointing | 4x-6x activation memory | 1d | Standard practice |
| 2l | Checkpoint scheduler state | Fix LR spike on resume | 0.5d | -- |

### Evaluation

| # | Change | Impact | Effort | SOTA Reference |
|---|--------|--------|--------|----------------|
| 2m | Associative recall diagnostic | Core SSM capability signal | 1d | Based, Mamba |
| 2n | WikiText-103 PPL benchmark | Publication baseline | 1d | Universal standard |
| 2o | 6-task zero-shot (lm-eval-harness) | Community comparability | 1d | EleutherAI standard |
| 2p | Induction head OOD test | Extrapolation signal | 0.5d | Mamba |

---

## Tier 3: Medium-Term (3-7 days each)

| # | Change | Impact | Effort | SOTA Reference |
|---|--------|--------|--------|----------------|
| 3a | Symplectic integration | -1 to -2 ppl (>8K), removes workarounds | 3d | LinOSS (ICLR 2025 Oral) |
| 3b | Log-space exponential gating | Removes SafeCumprod, better stability | 2d | xLSTM |
| 3c | Learned saliency scorer | Major on real vocab | 3d | Titans |
| 3d | Auxiliary retrieval loss (InfoNCE) | +5-10% retrieval precision | 2d | REALM, RAG-Token |
| 3e | Landmark token aggregation | +3-5% recall | 2d | InfLLM, Landmark Attn |
| 3f | FLA Triton chunkwise kernel | 3-8x scan speed | 5d | fla-org/flash-linear-attention |
| 3g | FlashAttention-2 sliding window | 4-16x attn memory | 3d | flash-attn package |
| 3h | Surprise-weighted updates | +3-6% long-range recall | 2d | Titans |
| 3i | TinyStories + real data pipeline | Enable proper benchmarks | 2d | Standard practice |

---

## Tier 4: Long-Term Research

| # | Change | Impact | Effort | SOTA Reference |
|---|--------|--------|--------|----------------|
| 4a | Outer-product state expansion | Free state expressiveness | 1w | HGRN2 |
| 4b | Position-aware gate | -0.3 to -0.5 ppl | 1d | GLA |
| 4c | MQAR multi-query benchmark | Core 3-tier diagnostic | 3d | Based |
| 4d | BABILong/LongMemEval evaluation | Agent memory validation | 3d | BABILong, LongMemEval |
| 4e | Chinchilla-scale sweep (70M-1B) | Scaling law characterization | 2w | -- |
| 4f | MoE integration | Parameter efficiency at scale | 2w | Jamba |
| 4g | Parallel Attn+TRN CUDA streams | +10-25% throughput | 1d | Hymba |

---

## Recommended Datasets (Priority Order)

### Training Data
1. **TinyStories** -- architecture iteration (1 hour to converge)
2. **WikiText-103** -- PPL baseline (2 days)
3. **PG-19** -- long-document PPL for T3 validation (2-3 days)
4. **FineWeb-10BT** -- large-scale pretraining sample
5. **Common Pile v0.1** -- legally clean, 8TB, 30 sources

### Evaluation Data
1. **Associative Recall** -- synthetic, core SSM diagnostic
2. **WikiText-103** -- canonical PPL comparison
3. **BABILong @ 100K** -- T2+T3 integration test
4. **NIAH** -- T3 smoke test
5. **LongMemEval-S** -- agent memory evaluation (ICLR 2025)
6. **RULER** -- multi-hop tracing across long context

---

## Key SOTA Comparisons for Publication

| Architecture | Paper | Why Compare |
|---|---|---|
| **LinOSS** | ICLR 2025 Oral | Closest TRN analog (oscillatory SSM) |
| **Gated DeltaNet** | ICLR 2025 | SOTA gated linear recurrence |
| **Titans** | Google 2025 | 3-tier design validation |
| **Mamba-2** | ICML 2024 | SSM baseline |
| **Griffin/RecurrentGemma** | Google 2024 | Hybrid recurrent+attention |
| **Based** | Stanford 2024 | Recall-throughput tradeoff |

---

## Execution Plan (4 weeks)

### Week 1: Foundations
- Tier 0 (all): config changes
- Tier 1 (all): 1-line fixes + output gate + Re+Im + SwiGLU fusion
- 2g (WSD schedule) + 2h (doc boundary reset) + 2i (AMP)
- 2m-2p (evaluation suite)

### Week 2: Architecture
- 2a (delta rule) + 2b (attention sinks) + 2c (beta-mix gate)
- 2d (vectorized retrieval) + 2e (StateTokenAdapter)
- 2j (sequence length curriculum)
- Train on TinyStories, measure all eval metrics

### Week 3: Performance + Data
- 3f (FLA kernel) or 3g (FlashAttention-2)
- 3i (TinyStories + WikiText-103 pipeline)
- 3a (symplectic integration) -- biggest single architectural change
- Full WikiText-103 PPL benchmark

### Week 4: Polish + Publication
- 3c (learned saliency) + 3d (retrieval loss)
- 4c-4d (MQAR + BABILong evaluation)
- LinOSS comparison experiment
- Paper writing: cite all SOTA, report all benchmarks
