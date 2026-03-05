# TRN Memory Infrastructure — Design Document

**Status**: Draft RFC
**Date**: 2026-03-05
**Authors**: J.A.R.V.I.S. Iron Legion

---

## Table of Contents

1. [Dual-Memory Architecture](#section-1--dual-memory-architecture)
2. [KV Cache Replacement Strategy](#section-2--kv-cache-replacement-strategy)
3. [Integration with Existing LLM Stacks](#section-3-integration-with-existing-llm-stacks)
4. [Agent Runtime Memory](#section-4-agent-runtime-memory)
5. [Benchmark Plan](#section-5-benchmark-plan)
6. [Productization Plan](#section-6-productization-plan)
7. [Risk Analysis](#section-7-risk-analysis)

---

## Section 1 — Dual-Memory Architecture

### 1.1 Problem Statement

Standard Transformer inference maintains a Key-Value (KV) cache that grows linearly
with context length. At 10k tokens with d_model=512, n_heads=8, n_layers=8 (fp32),
the KV cache consumes roughly 156 MB. At 100k tokens this becomes infeasible on
consumer hardware.

TRN (Temporal Resonance Network) maintains a constant-size complex state
(8 KB for K=128, 8 layers, fp32) by compressing all past tokens into a fixed number
of oscillator channels via a learnable first-order linear recurrence:

    r_t = alpha_t * r_{t-1} + (1 - alpha_t) * A_t * exp(j * (omega_t * t + phi_t))

The tradeoff: TRN cannot address arbitrary past tokens. It performs temporal
aggregation, not content retrieval. It detects periodic patterns and smooth
trends but loses token-level precision beyond its state capacity.

The dual-memory architecture combines both:
- Transformer KV cache for the recent window (full attention precision)
- TRN state for the distant past (constant-memory lossy compression)

### 1.2 Architecture Overview

```
Input tokens (streaming)
        |
        v
+-------+-----------------------------------------------+
|       DUAL-MEMORY LM FORWARD                          |
|                                                       |
|  +------------------+   +-------------------------+   |
|  | KV Window        |   | TRN State Compressor    |   |
|  | (last W tokens)  |   | (all tokens beyond W)   |   |
|  |                  |   |                         |   |
|  | K, V tensors     |   | r_real: (L, K) fp32     |   |
|  | (B, H, W, Dh)    |   | r_imag: (L, K) fp32     |   |
|  | ~W * D * 8 bytes |   | 8 KB fixed              |   |
|  +--------+---------+   +----------+--------------+   |
|           |                        |                  |
|           v                        v                  |
|  +--------+--------+    +----------+-----------+      |
|  | Full Attention  |    | TRN Readout          |      |
|  | (W x W causal)  |    | rho = Re(r * e^{-j}) |      |
|  | O(W^2) compute  |    | ctx_vec = W_res(rho)  |      |
|  +--------+--------+    +----------+-----------+      |
|           |                        |                  |
|           +----------+  +----------+                  |
|                      |  |                             |
|                      v  v                             |
|             +--------+--+---------+                   |
|             | Context Mixer       |                   |
|             |                     |                   |
|             | out = gate * attn   |                   |
|             |     + (1-gate) * trn_ctx               |
|             |                     |                   |
|             | gate: learned scalar|                   |
|             | per layer           |                   |
|             +----------+----------+                   |
|                        |                              |
+------------------------+------------------------------+
                         |
                         v
                    Next token logits
```

### 1.3 Component Descriptions

#### 1.3.1 KV Window Manager

Maintains the rolling window of recent tokens as standard KV cache entries.

- **Window size W**: configurable, typically 512-2048 tokens
- **Data structure**: `list[LayerKVCache]` where each entry holds
  `k_cache: (B, n_heads, W, head_dim)` and `v_cache: (B, n_heads, W, head_dim)`
- **Eviction trigger**: when total cached tokens exceeds W, the oldest M tokens
  are evicted (see Section 2)
- **Attention**: standard scaled dot-product attention over the window,
  causal mask enforced; no cross-window attention is needed because TRN covers
  the out-of-window past

#### 1.3.2 TRN State Compressor

Maintains the fixed-size complex resonance state representing all tokens
beyond the KV window.

- **State**: `r_real: (n_layers, K)`, `r_imag: (n_layers, K)`, both fp32
- **Size**: `n_layers * K * 2 * 4 bytes = 8 KB` for K=128, 8 layers
- **Update**: each evicted token is fed through `step_single()` per layer in
  chronological order (Section 2.2)
- **Readout**: per-layer context vector of shape `(d_model,)` via W_res projection
  on the demodulated state
- **Properties**: O(1) memory regardless of context length; deterministic given
  the same token sequence and position indices

#### 1.3.3 Context Mixer

Combines KV attention output and TRN context vector at each layer via a
learned gate, enabling the model to weight short-term precision vs long-term
temporal context.

- **Gate**: learned scalar `g` per layer, initialized to 0.5
- **Fusion**: `out = sigmoid(g) * attn_out + (1 - sigmoid(g)) * trn_ctx`
- **Gradient flow**: both paths receive gradients; gate can specialize to
  prefer KV for factual queries and TRN for temporal/periodic tasks

### 1.4 Forward Pass Pseudocode

```python
def dual_memory_forward(
    token_ids: Tensor,          # (B, n) -- new token batch
    kv_caches: list[LayerKV],   # rolling KV window per layer
    trn_states: TRNStates,      # r_real, r_imag per layer
    position_offset: int,       # absolute position of token_ids[0]
    window_size: int,
) -> tuple[Tensor, list[LayerKV], TRNStates]:

    # --- Phase 1: evict old KV entries into TRN state ---
    kv_caches, trn_states = maybe_evict(
        kv_caches, trn_states, window_size, position_offset
    )

    # --- Phase 2: embed new tokens ---
    x = embedding(token_ids)                     # (B, n, d_model)

    # --- Phase 3: per-layer dual-memory computation ---
    for layer_idx in range(n_layers):
        # 3a. KV attention over the current window
        attn_out = kv_attention(
            x, kv_caches[layer_idx], layer_idx,
        )                                        # (B, n, d_model)

        # 3b. TRN context readout for this layer (constant-memory path)
        trn_ctx = trn_readout(
            trn_states.r_real[layer_idx],
            trn_states.r_imag[layer_idx],
            x,                                   # needed for omega/phi projection
            position_offset,
        )                                        # (B, n, d_model)

        # 3c. Mix
        g = sigmoid(mix_gate[layer_idx])
        x = g * attn_out + (1 - g) * trn_ctx

        # 3d. FFN
        x = x + ffn[layer_idx](norm(x))

    # --- Phase 4: append new tokens to KV window ---
    kv_caches = append_to_kv_window(kv_caches, x, token_ids)

    logits = lm_head(norm_out(x))
    return logits, kv_caches, trn_states
```

---

## Section 2 — KV Cache Replacement Strategy

### 2.1 Eviction Policy

When the KV cache window contains more than `W` tokens, the oldest `M` tokens
are evicted and compressed into the TRN state. This happens before processing
each new batch of tokens.

**Parameters:**
- `W` -- maximum KV window size (e.g., 512 tokens for memory-constrained devices,
  2048 for standard GPU inference)
- `M` -- eviction chunk size (e.g., W/4 = 128 tokens); larger M reduces eviction
  frequency but increases TRN compression loss per batch

**Policy rationale:** Evicting the oldest tokens is the correct default because
TRN's recurrence naturally handles temporal ordering -- the state reflects a
time-decaying weighted sum of all past tokens, with recent tokens weighted
higher via the alpha gate.

```python
def maybe_evict(
    kv_caches: list[LayerKV],
    trn_states: TRNStates,
    window_size: int,
    position_offset: int,
) -> tuple[list[LayerKV], TRNStates]:

    current_len = kv_caches[0].k_cache.shape[2]  # time dimension
    if current_len <= window_size:
        return kv_caches, trn_states              # no eviction needed

    n_evict = current_len - window_size

    # Extract oldest n_evict tokens from each layer cache
    evicted_v = [cache.v_cache[:, :, :n_evict, :] for cache in kv_caches]

    # Compress evicted tokens into TRN state (Section 2.2)
    trn_states = compress_into_trn(
        evicted_v, trn_states,
        start_position=position_offset,
        n_tokens=n_evict,
    )

    # Trim KV caches: keep only the recent window
    trimmed = [
        LayerKV(
            k_cache=cache.k_cache[:, :, n_evict:, :],
            v_cache=cache.v_cache[:, :, n_evict:, :],
        )
        for cache in kv_caches
    ]
    return trimmed, trn_states
```

### 2.2 State Update Algorithm (Eviction -> TRN Compression)

Each evicted token is fed through `step_single()` for each layer sequentially
in chronological order. The position index passed to `step_single()` is the
**absolute** token position in the full sequence (not the window-relative index),
which is critical for phase coherence in the oscillator projection.

```python
def compress_into_trn(
    evicted_v: list[Tensor],     # per layer: (B, n_heads, n_evict, head_dim)
    trn_states: TRNStates,       # mutable: r_real, r_imag per layer
    start_position: int,         # absolute position of first evicted token
    n_tokens: int,
) -> TRNStates:
    """Feed evicted KV tokens into TRN state via step_single().

    Token embeddings are reconstructed from V cache concatenation:
    n_heads * head_dim = d_model (by construction).

    Phase precision note: position must be the ABSOLUTE sequence position.
    Using window-relative positions would break phase coherence as TRN
    oscillators are calibrated to global time.
    """
    for tok_idx in range(n_tokens):
        abs_position = start_position + tok_idx

        for layer_idx in range(n_layers):
            v_tok = evicted_v[layer_idx][:, :, tok_idx, :]  # (B, n_heads, head_dim)
            x_approx = v_tok.reshape(v_tok.shape[0], -1)    # (B, d_model)

            _, trn_states.r_real[layer_idx], trn_states.r_imag[layer_idx] = (
                trn_layers[layer_idx].step_single(
                    x_approx,
                    trn_states.r_real[layer_idx],
                    trn_states.r_imag[layer_idx],
                    position=abs_position,
                )
            )

    return trn_states
```

**Complexity:**
- Per eviction event: O(n_evict * n_layers * K) time, O(n_layers * K) space
- n_evict is bounded by M (chunk size policy), not total context length

### 2.3 TRN Retrieval: Context Vector into Attention

After compression, the TRN state is read out as a context vector that is
mixed into each layer's attention output (see Section 1.3.3, Context Mixer).

```python
def trn_readout(
    r_real: Tensor,          # (B, K) fp32
    r_imag: Tensor,          # (B, K) fp32
    x: Tensor,               # (B, n, d_model) -- current token embeddings
    position_offset: int,
    trn_layer: TemporalResonanceLayer,
) -> Tensor:                 # (B, n, d_model)
    B, n, _ = x.shape

    # Project x to get current oscillator params (content-dependent readout)
    _, omega_t, phi_t, _ = trn_layer.proj(x)      # each (B, n, K)

    # Compute demodulation angle at each position
    positions = torch.arange(n, device=x.device, dtype=torch.float32).view(1, n, 1)
    positions = torch.log1p(positions + position_offset)  # log-phase
    angle = omega_t * positions + phi_t                    # (B, n, K)

    # Demodulate: project state onto carrier at each position
    r_r = r_real.unsqueeze(1)                              # (B, 1, K)
    r_i = r_imag.unsqueeze(1)                              # (B, 1, K)
    rho = r_r * torch.cos(angle) + r_i * torch.sin(angle) # (B, n, K)

    # Project to d_model via W_res
    ctx_vec = trn_layer.res_scale * trn_layer.W_res(rho)   # (B, n, d_model)
    return ctx_vec
```

### 2.4 Quality Preservation: Sliding Window Guarantees

Key invariants:

1. **Recency invariant**: tokens at positions `[current - W, current]` always
   have exact K, V entries in the KV cache. No approximation for recent tokens.

2. **Monotone compression**: a token is either in the KV window (exact) or
   compressed into TRN (lossy). Never both or neither.

3. **Position coherence**: absolute position indices are preserved through
   eviction. TRN's phase encoding `omega * log(i+1)` is calibrated to the
   global sequence position.

4. **State continuity**: TRN state after processing position T equals the state
   from running `step_single()` on all tokens 0..T in order.

**Memory profile at W=512, K=128, n_layers=8, d_model=256:**

```
KV window:   2 * 8 * 1 * 8 * 512 * 32 * 4 = 67.1 MB  (fp32, B=1)
TRN state:   2 * 8 * 128 * 4 = 8.2 KB                 (any context length)
Total:       ~67 MB fixed, regardless of context = 1k or 1M tokens.
```

### 2.5 Eviction Chunk Size Tradeoff

| M (chunk) | Eviction frequency   | step_single() calls per event | TRN loss per event |
|-----------|----------------------|-------------------------------|--------------------|
| 1         | every token          | 1 per layer                   | minimal (always fresh) |
| W/4 = 128 | every 128 tokens     | 128 per layer                 | moderate            |
| W = 512   | every W tokens       | 512 per layer                 | maximal (long gap)  |

Recommended: M = W/8 to W/4. For streaming inference, M=1 (evict on every
overflow) is simplest.

---

## Section 3: Integration with Existing LLM Stacks

TRN's defining property for integration: a fixed-size complex tensor of shape
`(n_layers, K, 2)` in fp32. At `trn_100m` defaults (8 layers, K=256), that is
16 KB per session, independent of sequence length.

### 3.1 vLLM Integration

vLLM's PagedAttention allocates KV cache in fixed-size pages. TRN replaces the
paged block allocator for historical context beyond the sliding window.

```python
class TRNPagedAttentionBackend:
    """
    Hybrid scheduler: TRN state for old context, PagedAttention for recent window.

    Hooks into vllm.worker.model_runner.ModelRunner.execute_model().
    Tokens within last `trn_window` -> standard PagedAttention.
    Tokens older than `trn_window` -> compressed into TRNSessionState.
    """

    def __init__(self, trn_model, n_layers, K, trn_window=1024, device="cuda"):
        self.trn_layers = trn_model
        self.trn_window = trn_window
        self._states: dict[str, TRNSessionState] = {}

    def compress_evicted_tokens(self, session_id, evicted_embeddings):
        """Fold evicted KV tokens into TRN state via step_single()."""
        state = self.get_or_create_state(session_id)
        for layer_idx, trn_layer in enumerate(self.trn_layers):
            for t in range(evicted_embeddings.size(1)):
                _, state.r_real[layer_idx], state.r_imag[layer_idx] = (
                    trn_layer.step_single(
                        evicted_embeddings[:, t],
                        state.r_real[layer_idx].unsqueeze(0),
                        state.r_imag[layer_idx].unsqueeze(0),
                        state.token_position,
                    )
                )
                state.token_position += 1

    def release(self, session_id):
        self._states.pop(session_id, None)
```

**Serving API:**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B",
    memory_backend="trn",        # activates TRNPagedAttentionBackend
    trn_window=1024,             # tokens kept in standard paged KV
    trn_K=256,                   # oscillators per layer
)
```

**Memory savings:** At 10k tokens, standard KV ~156 MB/session. With
`trn_window=1024`: ~15.7 MB/session -- **10x reduction**.

### 3.2 HuggingFace Transformers Integration

```python
class TRNMemoryWrapper(nn.Module, GenerationMixin):
    """
    Wraps any HuggingFace causal LM with TRN-backed long-range memory.

    Recent `window` tokens -> standard attention.
    Tokens older than `window` -> folded into TRN state.

    Usage:
        base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        wrapper = TRNMemoryWrapper(base, trn_layers=32, K=256, window=2048)
        wrapper.generate(input_ids, max_new_tokens=512)
    """

    def __init__(self, base_model, trn_layers, K=256, window=2048, d_model=None):
        super().__init__()
        self.base = base_model
        self.config = base_model.config
        self.K = K
        self.window = window
        d = d_model or base_model.config.hidden_size

        self.trn = nn.ModuleList([
            TemporalResonanceLayer(d_model=d, K=K)
            for _ in range(trn_layers)
        ])

    def forward(self, input_ids, past_key_values=None, **kwargs):
        if self._r_real is None:
            self._init_state(input_ids.device)

        # Compress overflow into TRN state
        if input_ids.size(1) > self.window and past_key_values is None:
            overflow = input_ids[:, :-self.window]
            embs = self.base.get_input_embeddings()(overflow)
            for t in range(embs.size(1)):
                self._compress_token(embs[0, t].unsqueeze(0))
            input_ids = input_ids[:, -self.window:]

        return self.base(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
```

**Integration in `generate()` loop:**

```
generate() [GenerationMixin]
  +-- prepare_inputs_for_generation()   <- delegates to base
  +-- forward(input_ids, past_key_values)  <- TRNMemoryWrapper intercepts
  |     +-- overflow check: if len > window, compress via step_single()
  |     +-- base.forward(truncated_input_ids)
  +-- logits -> sample -> next_token
  +-- [repeat]
```

### 3.3 llama.cpp Integration

C struct for TRN state:

```c
typedef struct {
    float * r_real;   /* [K] fp32 */
    float * r_imag;   /* [K] fp32 */
} trn_layer_state_t;

typedef struct {
    trn_layer_state_t * layers;  /* [n_layers] */
    int32_t n_layers;
    int32_t K;
    int64_t position;
} trn_session_state_t;
```

step_single in C:

```c
void trn_step_single(
    const float * x,          /* [d_model] input embedding */
    const float * omega,      /* [K] learnable frequencies */
    const float * phi,        /* [K] learnable phases */
    const float * alpha,      /* [K] learnable decay gates (0..1) */
    const float * A,          /* [K] learnable amplitudes */
    float       * r_real,     /* [K] state real (updated in-place) */
    float       * r_imag,     /* [K] state imag (updated in-place) */
    float       * out,        /* [K] demodulated output */
    int32_t       K,
    int64_t       position,
    int           log_phase
) {
    double pos = log_phase ? log1p((double)position) : (double)position;

    for (int k = 0; k < K; k++) {
        double angle    = omega[k] * pos + phi[k];
        double one_m_a  = 1.0 - alpha[k];
        float  v_r      = (float)(one_m_a * A[k] * cos(angle));
        float  v_i      = (float)(one_m_a * A[k] * sin(angle));

        r_real[k] = alpha[k] * r_real[k] + v_r;
        r_imag[k] = alpha[k] * r_imag[k] + v_i;

        /* Per-channel max-abs normalization */
        float max_abs = fabsf(r_real[k]) > fabsf(r_imag[k])
                        ? fabsf(r_real[k]) : fabsf(r_imag[k]);
        if (max_abs > 1.0f) {
            r_real[k] /= max_abs;
            r_imag[k] /= max_abs;
        }

        out[k] = r_real[k] * (float)cos(angle) + r_imag[k] * (float)sin(angle);
    }
}
```

**GGUF metadata:**

```
trn.n_oscillators     = 256    (uint32)
trn.n_layers          = 8      (uint32)
trn.log_phase         = true   (bool)
blk.{i}.trn.omega    shape=[K]         f32
blk.{i}.trn.W_res    shape=[d_model,K] f32
blk.{i}.trn.res_scale scalar           f32
```

---

## Section 4: Agent Runtime Memory

### 4.1 Design

Agent frameworks today store conversation history via full token replay (O(n) VRAM),
vector DB retrieval (latency spikes), or sliding window truncation (permanent loss).

TRN offers a fourth option: **constant-state compression** via resonance recurrence.
Each turn is folded into fixed 8-16 KB state via `step_single()`. The state encodes
a frequency-domain summary weighted by recency via alpha decay gates.

The tradeoff: TRN state is write-only. You cannot extract a verbatim turn.
TRN is appropriate for *context injection* (biasing toward remembered themes)
rather than *exact recall*. For exact recall, pair with a lightweight key-value store.

### 4.2 API

```python
class AgentMemory:
    """
    Persistent agent memory backed by TRN resonance state.

    State size: n_layers * K * 2 * 4 bytes.
    For n_layers=8, K=256: 16 KB per session regardless of conversation length.
    """

    def __init__(self, d_model=512, n_oscillators=256, n_layers=8,
                 tokenizer=None, embedding=None, device="cpu"):
        self.trn_layers = nn.ModuleList([
            TemporalResonanceLayer(d_model=d_model, K=n_oscillators)
            for _ in range(n_layers)
        ])
        self._reset_state()

    def add_turn(self, role: str, content: str) -> None:
        """Compress a conversation turn into TRN state."""
        text = f"[{role.upper()}] {content}"
        emb = self._tokenize_and_embed(text)
        self._compress_embedding(emb)

    def add_tool_result(self, tool: str, result: Any) -> None:
        """Serialize tool result to tokens and compress into state."""
        payload = json.dumps(result, ensure_ascii=False)
        text = f"[TOOL:{tool}] {payload}"
        emb = self._tokenize_and_embed(text)
        self._compress_embedding(emb)

    def get_context(self) -> torch.Tensor:
        """Return (d_model,) context vector for prompt injection."""
        ctx = self.r_real.mean(dim=0)                 # (K,)
        return self.trn_layers[0].W_res(ctx)          # (d_model,)

    def save(self, path: str) -> None:
        torch.save({"r_real": self.r_real, "r_imag": self.r_imag,
                     "position": self.position, "metadata": self.metadata}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.r_real = ckpt["r_real"]
        self.r_imag = ckpt["r_imag"]
        self.position = ckpt["position"]
```

**Usage:**

```python
memory = AgentMemory(d_model=512, n_oscillators=256, n_layers=8,
                     tokenizer=tokenizer, embedding=model.embedding)

memory.add_turn(role="user", content="What is the capital of France?")
memory.add_turn(role="assistant", content="The capital of France is Paris.")
memory.add_tool_result(tool="search", result={"query": "Paris population", "top": "2.1M"})

context = memory.get_context()  # (d_model,) vector for injection
memory.save("session_001.pt")   # 16 KB checkpoint
```

### 4.3 Multi-session Scalability

```
Per session:   16 KB (TRN-100M, K=256, 8 layers)
1,000 sessions: 16 MB  (TRN)  vs  156 GB  (KV cache @ 10k tokens)
```

| Dimension              | TRN AgentMemory       | Vector DB (FAISS)         | Full KV Cache              |
|------------------------|-----------------------|---------------------------|----------------------------|
| Memory per session     | 16 KB (constant)      | ~10-100 MB (index + embs) | O(n_tokens) -- 156 MB @ 10k |
| Memory, 1000 sessions  | 16 MB                 | 10-100 GB                 | ~156 GB                    |
| Write cost per token   | O(1) -- one recurrence | O(log n) -- FAISS insert  | O(1) -- append to KV       |
| Read / retrieval       | O(1) -- get_context() | O(log n) -- ANN search    | O(n) -- full attention     |
| Recall type            | Implicit / soft bias  | Approximate top-k         | Exact (within window)      |
| Context-length scaling | Constant              | Linear (index grows)      | Linear (KV grows)          |
| Information loss       | Yes (lossy)           | Partial (top-k only)      | No (within window)         |

**Redis-backed state store:**

```python
class RedisAgentMemoryStore:
    def save(self, session_id: str, memory: AgentMemory) -> None:
        key = f"trn:session:{session_id}"
        self.r.hset(key, mapping={
            "r_real": memory.r_real.cpu().numpy().tobytes(),  # 16 KB
            "r_imag": memory.r_imag.cpu().numpy().tobytes(),
            "position": str(memory.position),
        })
        self.r.expire(key, self.ttl)
```

**Deployment topology:**

```
HTTP Request -----> Serving Process (GPU)
  (session_id)        AgentMemory.load(sid)   <- Redis (16 KB)
                      memory.add_turn(...)     <- O(1) per token
                      ctx = memory.get_context()
                      model(prompt, ctx)       <- standard forward()
                      memory.save(sid)         -> Redis (16 KB)

Redis cluster: 1000 sessions * 16 KB = 16 MB (trivial)
```

---

## Section 5: Benchmark Plan

### 5.1 Benchmark Suite

| Benchmark | Metric | Baselines | Context Lengths |
|-----------|--------|-----------|-----------------|
| **THROUGHPUT-LONG** | tps, peak_memory_mb | Full KV cache, StreamingLLM | 1k, 4k, 10k, 32k, 100k |
| **NEEDLE-QA** | accuracy@depth | Full KV, Sliding Window, H2O | 4k, 8k, 32k, 100k |
| **AGENT-SIM** | recall_accuracy, tps | Full KV, Vector DB (FAISS) | 100 turns x ~200 tok = 20k |
| **STREAM-PRED** | MSE, latency_ms | LSTM-256, Transformer-4L | Rolling 512-token windows |

### 5.2 Methodology

**Hardware:**

| Tier | Hardware | Purpose |
|------|----------|---------|
| Primary | A100-80GB SXM (2x) | Full model suite (TRN-100M, 400M, 1B) |
| Secondary | RTX 4090 (24GB) | Consumer-grade memory pressure tests |
| Edge | NVIDIA Jetson AGX Orin | Edge inference feasibility |

**Model Configurations** (from `config.py`):

| Config | d_model | n_layers | K | State size |
|--------|---------|----------|---|------------|
| `trn_100m` | 512 | 8 | 256 | 16 KB |
| `trn_400m` | 1024 | 16 | 512 | 64 KB |
| `trn_1b` | 2048 | 24 | 512 | 96 KB |

Compare: KV cache at 10k tokens for a 7B model = ~5.2 GB. TRN-1B state = 96 KB (**~55,000x smaller**).

**Reproducibility:** Fixed seeds, deterministic CUDA, 5 repeats with mean +/- std.

### 5.3 Quality Metrics

| Metric | Pass Threshold |
|--------|---------------|
| Perplexity degradation | < 10% vs Full KV cache |
| BLEU-4 (generation) | > 90% of Full KV baseline |
| ROUGE-L (summarization) | > 92% of Full KV baseline |
| Needle recall accuracy | > 0.60 at depth=75% |
| Agent turn recall | > 0.70 at turn 100 |

**Known hard failure:** Selective copy accuracy 0.088 vs Transformer 0.962. Structural limitation.

---

## Section 6: Productization Plan

### 6.1 Product Stack

```
+-------------------------------------------------------------+
|                       trn-cloud                               |
|   OpenAI-compatible API + memory extensions                   |
|   Multi-tenant, per-session billing, Redis state store        |
+-------------------------------------------------------------+
|                       trn-server                              |
|   REST/gRPC memory server -- self-hosted                      |
|   Stateless compute nodes + external state store              |
+-------------------------------------------------------------+
|                       trn-core (OSS, MIT)                     |
|   pip install trn-core                                        |
|   TRNModel, HybridModel, TRNConfig, generate(), step_single  |
|   State serialization, benchmark scripts                      |
+-------------------------------------------------------------+
```

**trn-server REST API:**

```
POST   /v1/sessions                      # Create session
POST   /v1/sessions/{id}/tokens          # Compress tokens into state
GET    /v1/sessions/{id}/context         # Retrieve context vector
POST   /v1/sessions/{id}/generate        # Generate from state
GET    /v1/sessions/{id}/state           # Export raw state
PUT    /v1/sessions/{id}/state           # Import state
DELETE /v1/sessions/{id}                 # Cleanup
```

**Horizontal scaling:**

```
          Load Balancer (L7, sticky by session_id)
               |            |            |
          Compute-1    Compute-2    Compute-3   (stateless)
               |            |            |
               +-----Redis Cluster------+
                   Key: session_id
                   Value: TRNState (16-96 KB)
                   TTL: 24h
```

**trn-cloud (managed):**

```python
# OpenAI-compatible with TRN extension
response = client.chat.completions.create(
    model="trn-400m",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"trn_session_id": "sess_abc123"},  # state persists across calls
)
```

### 6.2 Target Customers

| Segment | Pain Point | TRN Value |
|---------|-----------|-----------|
| Agent platforms (LangChain, CrewAI) | Memory fills up after 100+ tool calls | Constant 96KB state |
| Chatbot providers | OOM on long sessions | Bounded memory, predictable cost |
| Edge inference (mobile, IoT) | KV cache exceeds device RAM | 16KB fits MCU SRAM |
| Streaming processors | Transformers need fixed windows | TRN native O(1) streaming |

**Not a fit for:** verbatim recall tasks, <4k context, interpretable attention patterns.

### 6.3 Pricing Model

| Tier | Pricing |
|------|---------|
| trn-core (OSS) | Free (MIT) |
| trn-server (self-hosted) | Free; support $500-5000/month |
| trn-cloud Starter | $0.50 / 1M tokens |
| trn-cloud Pro | $2.00 / 1M tokens + $0.01 / session-hour |
| trn-cloud Enterprise | Custom |

```
Revenue density:
  Sessions per A100 (80GB):  ~835 TRN sessions vs ~15 KV sessions
  = 55x more sessions at same token price
```

### 6.4 Competitive Positioning

| Competitor | TRN Advantage | Competitor Advantage |
|-----------|---------------|---------------------|
| Mamba (Selective SSM) | Smaller constant state, simpler architecture | Better selective recall |
| RWKV (Linear Attention) | Smaller per-layer state (2K floats vs d_model^2) | Larger ecosystem, pretrained models |
| Griffin (Gated Linear Recurrence) | Complex oscillators capture phase info | Proven at scale (DeepMind) |
| StreamingLLM (Sink + Window) | Retains all history (with distortion) vs drops | No training modification needed |

---

## Section 7: Risk Analysis

### 7.1 Risk Matrix

| Risk | Likelihood | Impact | Priority | Mitigation |
|------|-----------|--------|----------|------------|
| Information loss at long context | HIGH | HIGH | P0 | Hybrid KV+TRN, quality monitoring |
| Selective recall failure | HIGH | HIGH | P0 | Do not use pure TRN for verbatim retrieval |
| Memory state collapse / explosion | MEDIUM | HIGH | P1 | State norm + alpha clamping + gradient clipping |
| Training instability | MEDIUM | MEDIUM | P1 | Distillation from pretrained Transformer |
| Cold-start quality degradation | HIGH | MEDIUM | P1 | Server-side warmup, prompt pre-processing |
| Integration complexity | LOW | MEDIUM | P2 | Wrapper approach, minimal host changes |
| Competitive displacement | MEDIUM | HIGH | P2 | Focus on constant-state niche |
| API/state format breaks | LOW | HIGH | P2 | Version-tagged state format, migration tooling |

### 7.2 Detailed Risk Analysis

**RISK-1: Information Loss (HIGH/HIGH)**

TRN compresses unbounded history into fixed state. With `alpha < 1`, early
information decays exponentially. Evidence: selective copy accuracy 0.088
vs Transformer 0.962.

Mitigation (priority order):
1. Hybrid architecture: attention layers for exact retrieval, TRN for compression
2. External episodic memory: pair TRN with vector store for verbatim facts
3. Quality monitoring: canary probes that inject known sequences and verify recall
4. Documentation: pure TRN is architecturally unsuitable for exact retrieval

**RISK-2: State Collapse/Explosion (MEDIUM/HIGH)**

Two failure modes:
- Collapse: state -> zero (alpha -> 0). Model ignores recurrent context.
- Explosion: state norm unbounded (alpha -> 1, high amplitude). NaN in generation.

Current mitigations (all in `config.py`):
- `amplitude_max = 3.0` (prevents large A)
- `state_norm = True` (per-channel max-abs normalization)
- `res_scale_init = 0.05` (small initial output scale)
- `gate_bias_init = 0.85` (gates lean toward retention)
- `phase_mode = "log"` (prevents high-frequency resonance at large t)

**RISK-3: Cold-Start (HIGH/MEDIUM)**

`generate()` zero-initializes states. First ~128 tokens get no recurrent benefit.

Mitigations:
1. Server-side warmup: process warmup_prompt at session creation
2. State checkpointing: pre-compute post-warmup state for common system prompts
3. Prompt processing in `generate()` already implemented (iterates prompt tokens)

### 7.3 Go/No-Go Criteria for Production

**Mandatory (all must pass):**

| Criterion | Target |
|-----------|--------|
| Perplexity degradation | < 10% vs Full KV |
| Memory footprint | State < 1% of KV cache at 10k context |
| Generation latency p99 | < 50% of KV-cached generation |
| Throughput at 10k context | TRN tps >= 2x Transformer |
| Quality regression | No regression on >= 90% of eval tasks |
| State serialization round-trip | Identical logits after save/load |

**Disqualifying conditions (immediate No-Go):**
- NaN or Inf in state during normal generation
- `state.abs().max() > 1e4` for any layer
- Memory leak (exceeds model weights + 2x state size)
- Test coverage below 80%

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (4 weeks)

- [ ] Implement `DualMemoryModel` with KV Window Manager + TRN State Compressor + Context Mixer
- [ ] Implement eviction policy with configurable W and M
- [ ] Unit tests for state continuity invariant (evicted state == sequential state)
- [ ] Benchmark: THROUGHPUT-LONG on TRN-100M

### Phase 2: Integration Layer (4 weeks)

- [ ] `TRNMemoryWrapper` for HuggingFace Transformers
- [ ] `TRNPagedAttentionBackend` for vLLM (prototype)
- [ ] `AgentMemory` class with save/load/Redis backend
- [ ] Benchmark: AGENT-SIM on TRN-100M

### Phase 3: Quality Validation (3 weeks)

- [ ] NEEDLE-QA benchmark suite
- [ ] STREAM-PRED benchmark suite
- [ ] Go/No-Go evaluation against mandatory criteria
- [ ] Distillation pipeline: Transformer teacher -> TRN student

### Phase 4: Productization (6 weeks)

- [ ] trn-core PyPI package (v1.0)
- [ ] trn-server Docker image with REST/gRPC API
- [ ] llama.cpp integration (C struct + step_single)
- [ ] Documentation: API reference, integration guides, limitations

### Phase 5: Scale Validation (4 weeks)

- [ ] TRN-400M and TRN-1B training + distillation
- [ ] A100 benchmark suite
- [ ] Multi-session load testing (1000+ concurrent sessions)
- [ ] trn-cloud prototype deployment

---

## Appendix: State Size Reference

```
TRN State = n_layers * n_oscillators * 2 * sizeof(float32)

TRN-100M:  8 * 256 * 2 * 4 =  16 KB per session
TRN-400M: 16 * 512 * 2 * 4 =  64 KB per session
TRN-1B:   24 * 512 * 2 * 4 =  96 KB per session

KV Cache (7B Transformer, fp16) at various context lengths:
  1k tokens:   524 MB
  10k tokens:  5.2 GB
  100k tokens: 51.2 GB

Memory ratio (TRN-1B / KV@10k): 96 KB / 5.2 GB = 0.0018%
```
