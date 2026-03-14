# TRN Limitations

What TRN cannot do and why.

---

## Summary

TRN is a **pattern memory**, not a **content-addressable memory**.

Linear recurrence compresses input history into continuous statistics (amplitude, phase,
frequency). This preserves periodic patterns and temporal trends. It does not preserve
discrete token identity or support index-gated retrieval.

This is a structural property of the recurrence equation, not a training deficiency.
No amount of training, hyperparameter tuning, or model scaling will fix it.

---

## 1. The Recurrence Equation

The TRN state update per oscillator channel k:

```
v_t  = (1 - alpha_t) * A_t * exp(j * (omega_t * t + phi_t))
r_t  = alpha_t * r_{t-1} + v_t
```

Expanded over a full sequence:

```
r_T = sum_{t=0}^{T} (prod_{s=t+1}^{T} alpha_s) * v_t
```

At any position T, state `r_T` is a weighted sum of all past drive vectors. The weight
for token at position t is `prod_{s=t+1}^{T} alpha_s` -- it depends only on gate values
at positions **after** t, not on the content of position t itself.

This means the recurrence cannot selectively retain or discard individual past tokens
based on their content.

---

## 2. What Fails

### Needle-in-Haystack (NiH)

**Task**: Insert a known token at position P in a sequence of noise tokens. After processing
the full sequence, recall the inserted token.

**Result**: TRN recall = 0.0 at all distances (0, 32, 64, 128, 256, 512).

**Why**: Recalling a specific token requires content-addressed lookup: "what token was at
position P?" The TRN state stores a superposition of all past inputs, weighted by
exponential decay. Individual tokens cannot be extracted from the superposition.

### Selective Copy

**Task**: Given a prefix with 8 marked tokens among noise, reproduce the marked tokens
in order after a separator.

**Result**: TRN accuracy = 8.8% (chance level). Transformer = 96.2%.

**Why**: Selective copy requires setting the retention weight to 1 for marker positions
and 0 for noise positions. In the linear recurrence, the weight for position t is
`prod_{s=t+1}^{T} alpha_s`. Setting alpha_{t+1} = 0 to discard a noise token at t also
zeroes out all information from positions before t, including markers. There is no
mechanism to selectively zero out individual past contributions.

### Goal Tracking Beyond KV Window

**Task** (DualMemoryEngine only): Track a goal token over N filler tokens. At distance > W
(where W is the KV window size), recall the goal.

**Result**: Accuracy ~0.25 (chance level for 4-class task).

**Why**: Within the KV window, exact attention retrieves the goal token. Beyond W, only the
TRN state remains. The TRN state does not retain the discrete goal token identity, so
recall drops to chance.

---

## 3. What Works

### Periodic Pattern Detection (PPD)

**Task**: Classify the frequency of a periodic signal from the final hidden state.

**Result**: TRN accuracy = 0.78--1.00 (vs Transformer = 1.00).

**Why**: Periodic patterns are exactly what complex oscillator recurrence captures.
Each oscillator channel has a natural frequency (omega_base). The recurrence acts as a
bank of frequency-selective filters. Resonance amplifies matching frequencies, damping
suppresses non-matching ones.

### Copy Task (Periodic Structure)

**Task**: Repeat a sequence with period P.

**Result**: Both TRN and Transformer reach 100% accuracy.

**Why**: Periodic repetition is a frequency-domain pattern. The TRN state encodes the
repetition period as oscillator phase relationships.

### Constant-Memory Generation

**Task**: Generate tokens from a fixed-size state without KV cache growth.

**Result**: TRN throughput stays flat at ~240 tps as context grows from 1K to 10K.
Transformer+KV degrades from 73.8 to 15.5 tps over the same range.

**Why**: TRN state is O(1) by construction. No history-dependent data structure grows.

---

## 4. Measured Evidence

| Task | TRN | Transformer | DualMemory (W=64) | What Is Measured |
|------|-----|-------------|-------------------|------------------|
| PPD | 1.000 | 1.000 | 0.78--1.00 | Frequency classification (pattern) |
| Copy | 1.000 | 1.000 | -- | Periodic repetition (pattern) |
| NiH | 0.000 | -- | 0.000 | Exact token recall (content) |
| Selective copy | 0.088 | 0.962 | -- | Content-gated retrieval (content) |
| GT (d > W) | -- | -- | ~0.25 | Discrete symbol recall (content) |

Pattern tasks: TRN performs at or near Transformer level.
Content tasks: TRN performs at or below chance level.

---

## 5. DualMemoryEngine Implications

DualMemoryEngine combines windowed attention (exact retrieval within W tokens) with
TRN state (pattern compression for all tokens). The mixer gate `g = sigmoid(W_g * x)`
blends both signals per position.

Within the KV window (distance <= W), quality matches full-attention Transformer.
Beyond the window, only TRN state is available. This means:

- Periodic patterns beyond W: recoverable via TRN state (PPD = 0.78--1.00)
- Exact token recall beyond W: not recoverable (NiH = 0.0, GT = chance)

The DualMemoryEngine is suitable when distant context can tolerate lossy compression
(e.g., agent conversation history, streaming summarization) and recent context requires
exact attention (e.g., last few turns, current instruction).

It is not suitable when exact recall of arbitrary past tokens is required at any distance
(e.g., document QA requiring retrieval of a specific passage from 10K tokens ago).

---

## 6. Theoretical Basis

The fundamental constraint is **linearity in state**. The recurrence `r_t = alpha_t * r_{t-1} + v_t`
is linear in `r_{t-1}`. This linearity enables O(log n) parallel prefix scan during training
(the associativity property), which is the main computational advantage of TRN over attention.

Content-addressed retrieval requires **nonlinear interaction** between the query and stored
keys. Attention implements this via `softmax(Q * K^T / sqrt(d))` -- a nonlinear function
of both the query Q and all stored keys K. No linear recurrence can replicate this operation
because the output of a linear recurrence is always a linear combination of past inputs,
and linear combinations cannot implement content-gated selection.

This is not a limitation specific to TRN. It applies to all linear RNN variants
(S4, Mamba, RWKV, RetNet, etc.) to varying degrees. Some mitigate it with auxiliary
mechanisms (e.g., Mamba's input-dependent gating), but none fully replicate attention's
content-addressed retrieval through linear recurrence alone.

---

## References

- [docs/TRN_ARCHITECTURE_ANALYSIS.md](TRN_ARCHITECTURE_ANALYSIS.md) -- Go/No-Go test details and recurrence analysis
- [docs/PUBLIC_CLAIMS.md](PUBLIC_CLAIMS.md) -- What can and cannot be claimed publicly
- `results/gate_result.md` -- TRN standalone Go/No-Go verdict
- `results/gate_result_dual.md` -- DualMemoryEngine Go/No-Go verdict
