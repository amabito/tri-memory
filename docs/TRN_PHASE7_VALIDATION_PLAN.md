# TRN Phase 7 — Definitive Validation Plan

**Status**: Draft RFC
**Date**: 2026-03-05
**Authors**: J.A.R.V.I.S. Iron Legion (Mark-1, Mark-2, Mark-3)
**Objective**: Determine whether TRN can become a production-grade constant-state long-memory layer for LLM and multi-agent systems.

---

## Table of Contents

1. [GPU Benchmark Design](#section-1--gpu-benchmark-design)
2. [Information Retention Evaluation](#section-2--information-retention-evaluation)
3. [Selective Recall Mitigation](#section-3--selective-recall-mitigation)
4. [Multi-Agent Scalability Simulation](#section-4--multi-agent-scalability-simulation)
5. [Agent Runtime Integration](#section-5--agent-runtime-integration)
6. [Go/No-Go Criteria](#section-6--gono-go-criteria)
7. [Final Deliverables and Roadmap](#section-7--final-deliverables-and-roadmap)

---

## Section 1 — GPU Benchmark Design

### 1.1 Objective

Quantify TRN's inference efficiency advantage over Transformer+KV-cache at production
context lengths (4K-128K tokens), using real model scales (trn_400m, trn_1b) and
reference architectures (Llama-3 8B proxy).

### 1.2 Measurement Matrix

#### 1.2.1 Model Variants Under Test

| Variant         | d_model | n_layers | K (n_oscillators) | Notes                          |
|-----------------|---------|----------|-------------------|--------------------------------|
| trn_100m        | 512     | 8        | 256               | Warm-up / smoke test           |
| trn_400m        | 1024    | 16       | 512               | Primary benchmark target       |
| trn_1b          | 2048    | 24       | 512               | Stress test, OOM boundary      |
| llama3_8b_proxy | 4096    | 32       | N/A               | KV-cache reference (TF-only)   |
| hybrid_400m_50  | 1024    | 16       | 512               | HybridModel trn_ratio=0.5      |
| hybrid_400m_25  | 1024    | 16       | 512               | HybridModel trn_ratio=0.25     |

The Llama-3 8B proxy uses: 32 layers, 32 heads, d_model=4096, head_dim=128, GQA 8 KV-heads.

#### 1.2.2 Context Length Sweep

```
context_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
gen_tokens   = 256   # decode phase length (fixed across all runs)
batch_size   = 1     # latency-critical path; separate batch throughput sweep
n_repeats    = 5     # take median, discard min/max
warmup_steps = 2     # excluded from timing
```

#### 1.2.3 Metrics Collected

Per (model, context_len, batch_size):

| Metric              | Unit       | How Measured                                              |
|---------------------|------------|-----------------------------------------------------------|
| prefill_latency_ms  | ms         | wall-clock: model(prompt) first forward pass              |
| decode_tps          | tokens/s   | gen_tokens / total_decode_time                            |
| decode_latency_ms   | ms/tok     | total_decode_time / gen_tokens                            |
| peak_vram_mb        | MB         | torch.cuda.max_memory_allocated() after generation        |
| state_memory_mb     | MB         | TRN: 2 * n_layers * K * 4 bytes; KV: formula below       |
| speedup_vs_kv       | ratio      | decode_tps / KV-cache baseline decode_tps at same ctx_len |

### 1.3 Memory Estimation Formulas

#### 1.3.1 TRN Resonance State

```
TRN_state_bytes = n_layers * K * 2 * sizeof(float32)
                = n_layers * K * 2 * 4
```

| Model    | n_layers | K   | State Size   |
|----------|----------|-----|--------------|
| trn_100m | 8        | 256 | 16 KB        |
| trn_400m | 16       | 512 | 64 KB        |
| trn_1b   | 24       | 512 | 96 KB        |

State size is **constant** regardless of context_len.

#### 1.3.2 KV Cache (Standard Transformer)

```
KV_bytes = n_layers * n_heads * context_len * head_dim * 2 * sizeof(fp16)
```

For Llama-3 8B (GQA, n_kv_heads=8):

| Context (T) | Llama3-8B KV (GQA=8) | trn_1b State | Ratio    |
|-------------|----------------------|--------------|----------|
| 4K          | 512 MB               | 96 KB        | 5,461x   |
| 16K         | 2,048 MB             | 96 KB        | 21,845x  |
| 64K         | 8,192 MB             | 96 KB        | 87,381x  |
| 128K        | 16,384 MB            | 96 KB        | 174,763x |

#### 1.3.3 Activation Memory During Decode

For TRN `step_single` (one decode step):

```
Per layer (B=1, K oscillators):
  OscillatorProjection: 4K * 2 bytes
  angle, v_r, v_i, rho: K * 4 bytes each
  Total per layer: ~10 * K * 4 bytes

trn_400m all 16 layers: 320 KB activation peak per decode step
```

KV-cache attention decode: O(T * head_dim) per layer -- grows with context.

### 1.4 Benchmark Script Architecture

```
scripts/bench_phase7_gpu.py
|
+-- BenchmarkConfig (dataclass)
|   +-- models, context_lens, gen_tokens, batch_sizes, n_repeats, device
|
+-- ModelFactory.build(name) -> nn.Module
+-- measure_prefill(model, prompt) -> PrefillResult
+-- measure_decode(model, prompt, gen_tokens) -> DecodeResult
+-- BenchmarkRunner.run() -> DataFrame
    +-- warm_up, for model x ctx x batch: measure, save CSV
```

Pseudocode: Core TRN Decode Loop

```python
def measure_decode_trn(model: TRNModel, prompt: Tensor, gen_tokens: int) -> DecodeResult:
    model.eval()
    B, T = prompt.shape
    with torch.no_grad():
        _ = model(prompt)  # prefill

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=gen_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps = gen_tokens * B / elapsed
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return DecodeResult(tps=tps, latency_ms=elapsed * 1000 / gen_tokens, peak_vram_mb=peak_vram)
```

### 1.5 GPU Architecture Analysis

```
RTX 5090 (sm_120, 32 GB VRAM, CUDA 12.8):
  Memory Bandwidth: ~1.8 TB/s (fp16)
  FP16 Tensor Core: ~838 TFLOPS

TRN step_single (trn_400m, B=1):
  State size: 64 KB << L2 cache (96 MB)
  Per-step FLOPs: 16 layers * ~3 * 512 * 2 = ~49K FLOPs
  Bottleneck: bandwidth-bound for B=1

KV-cache at T=32K (trn_400m equiv TF):
  KV read per step: 16 * 16 * 32768 * 64 * 4 = 2,147 MB per decode step
  At 1.8 TB/s: ~1.2 ms per token just for memory reads
  TRN state read: 64 KB -> negligible
```

### 1.6 Pass/No-Go Thresholds (Section 1)

| Metric                                   | Pass Threshold     |
|------------------------------------------|--------------------|
| TRN decode TPS vs KV-cache at ctx=16K    | >= 2.5x faster     |
| TRN peak VRAM at ctx=128K                | < 500 MB           |
| TRN state size (trn_400m, B=1)           | 64 KB +/- 5%       |
| trn_1b decode TPS at ctx=32K             | >= 500 tok/s on RTX 5090 |

---

## Section 2 — Information Retention Evaluation

### 2.1 Objective

Quantify how much semantic information TRN's fixed-size resonance state retains
from tokens at varying distances. Characterize what TRN preserves (trends,
frequencies, aggregate statistics) vs what it loses (exact tokens, random access).

### 2.2 Theoretical Basis

The resonance recurrence:

```
v_t  = (1 - alpha_t) * A_t * exp(j * (omega_t * t + phi_t))
r_t  = alpha_t * r_{t-1} + v_t
y_t  = Re(r_t * exp(-j * (omega_t * t + phi_t)))
```

Effective memory horizon per channel k:

```
tau_k ~ -1 / log(E[alpha_k])
```

With gate_bias_init=0.85, sigmoid(0.85) ~ 0.70, initial tau ~ 3 tokens.
After training, alpha converges toward 1.0 for low-frequency channels.

With K=512 oscillators (trn_400m):
- High-freq channels (omega near pi): last 10-50 tokens
- Mid-freq channels (omega near pi/8): last 100-500 tokens
- Low-freq channels (omega near 0): global topic/genre signal

### 2.3 Evaluation Protocols

#### 2.3.1 Needle-in-Haystack (NiH)

```
[PREFIX: n_prefix filler tokens]
[NEEDLE: 5-10 tokens, distinctive fact]
[SUFFIX: n_suffix filler tokens]
[QUERY: prompt asking for needle fact]
```

Sweep: context_lens = [512, 1024, 2048, 4096, 8192, 16384]
       needle_depths = [0.1, 0.25, 0.5, 0.75, 0.9]
       n_trials = 100 per cell

Expected heatmap (trn_400m):

```
Depth    0.1   0.25  0.5   0.75  0.9
ctx=512  0.85  0.80  0.72  0.78  0.88
ctx=2K   0.60  0.55  0.40  0.50  0.75
ctx=8K   0.35  0.30  0.20  0.28  0.60
ctx=32K  0.15  0.12  0.08  0.12  0.45
```

TRN retains recent tokens (depth=0.9) better than middle (depth=0.5).
TF+KV: uniform accuracy across depth.

#### 2.3.2 Token Reconstruction Probe

Directly probe what information is in the resonance state via learned reconstruction:

```python
class ResonanceProbe(nn.Module):
    def __init__(self, n_layers, K, d_model, n_lags):
        # Input: concatenated [r_real, r_imag] from all layers
        # Shape: (B, 2 * n_layers * K)
        self.linear = nn.Linear(2 * n_layers * K, d_model * n_lags)
```

Train probe per lag in [1, 4, 16, 64, 256, 1024, 4096].
Measure cosine similarity between predicted and actual embeddings.

Expected profile:
- cos_sim drops below 0.5 at lag ~256 tokens = "effective memory horizon"
- Target for trn_400m: effective_horizon > 256 tokens

#### 2.3.3 Periodic Pattern Detection

TRN's oscillatory basis should outperform attention on periodic sequences.

```python
periods = [3, 7, 13, 29, 64, 128, 256, 512]
lengths = [1024, 4096, 16384]
```

Expected TRN advantage:

```
period=64, length=16384: TRN=0.65, TF+KV=0.40  (TRN 1.6x)
period=256, length=16384: TRN=0.55, TF+KV=0.31 (TRN 1.8x)
```

Pass threshold: TRN accuracy >= 1.5x TF+KV for period > KV_window_size.

#### 2.3.4 Multi-Hop Reasoning

```
Fact chain: A -> B -> C, with hop_distance tokens between facts.
Query at end: "What is C?"

hop_distance = [512, 1024, 2048, 4096]
n_hops = [2, 3, 4]
```

Expected (2-hop):
- hop_dist=2048: TRN=0.52, TF+KV=0.35 (KV window saturated)
- hop_dist=4096: TRN=0.45, TF+KV=0.12 (KV window miss)

### 2.4 Hybrid Model Analysis

HybridModel (trn_ratio=0.5, 16 layers: T-A-T-A...):

| Configuration         | NiH@depth=0.5,ctx=4K | NiH@depth=0.9,ctx=4K |
|-----------------------|----------------------|----------------------|
| pure_trn_400m         | ~0.35                | ~0.75                |
| hybrid_400m_50        | ~0.55                | ~0.82                |
| hybrid_400m_25        | ~0.65                | ~0.85                |
| pure_tf_400m_kv       | ~0.80                | ~0.83                |

Hypothesis: hybrid_50 recovers 55% of KV accuracy gap vs pure TRN while
maintaining 80% of TRN's memory advantage.

### 2.5 Channel Utilization Analysis

Measure effective channel count: channels where var(r) > epsilon.
Target: > 70% utilization for trn_400m. If < 50%, reduce K and re-benchmark.

### 2.6 Pass/No-Go Thresholds (Section 2)

| Test                                       | Pass Threshold                             |
|--------------------------------------------|--------------------------------------------|
| NiH accuracy, ctx=2K, depth=0.5           | >= 0.50                                    |
| NiH accuracy, ctx=2K, depth=0.9           | >= 0.80                                    |
| Token probe cosine sim, lag=64            | >= 0.50                                    |
| Periodic detection, period=64, len=16K    | TRN >= 1.5x TF+KV                          |
| Multi-hop 2-hop, hop_dist=4K              | TRN >= TF+KV accuracy                      |
| Channel utilization (trn_400m)            | >= 70%                                     |
| Hybrid-50 NiH, ctx=4K, depth=0.5          | >= 0.50                                    |

---

## Section 3 — Selective Recall Mitigation

### 3.1 Problem Statement

Phase 4 measurement: TRN selective copy accuracy = **8.8%**.

Root causes:
1. **Spectral interference**: Many tokens overwrite the same oscillator channels.
2. **Irreversibility**: `r_t = alpha_t * r_{t-1} + v_t` is non-invertible.
3. **No index-addressed retrieval**: Demodulation reconstructs global aggregate, not specific items.

### 3.2 Strategy A — Sparse Key-Value Episodic Store (SKES)

Maintain a small external dictionary mapping learned keys to KV values of important tokens.

```
At eviction:
  TRN state update (all tokens)
  + Importance scorer: I(t) = norm(W_k * x_t)
  + If I(t) > threshold: write k_t -> v_t to SKES store

At query:
  q = W_q * x_current
  scores = softmax(q @ K_store.T)
  mem_ctx = scores @ V_store
  out = gate_a * trn_ctx + gate_b * mem_ctx + gate_c * attn_out
```

Memory: S=256 entries, d_model=512 -> 1 MB (negligible vs KV cache).
Expected accuracy: 8.8% -> **40-60%** at S=256.

### 3.3 Strategy B — Oscillator Channel Reservation (OCR)

Reserve K_r out of K channels as "episodic" -- only updated for high-importance tokens.

```
For channel k in [0, K_r):  -- EPISODIC
  r_k_t = g_t * (alpha * r_{t-1} + v_t) + (1-g_t) * r_{t-1}  // write if important

For channel k in [K_r, K):  -- TEMPORAL
  r_k_t = alpha * r_{t-1} + v_t  // always update
```

Memory overhead: zero (same state size). Extra params: W_imp (16 KB).
Expected accuracy: 8.8% -> **25-35%**.

### 3.4 Strategy C — Position-Indexed State Snapshot (PISS)

Periodically snapshot TRN state; at query time, replay from nearest snapshot.

```
Snapshots: every P tokens, save (r_real, r_imag)
Memory: (T/P) * n_layers * K * 2 * 4 bytes
  P=256, T=10K, trn_100m: 1.28 MB
Query: replay at most P steps from nearest snapshot
```

No training change required -- pure inference-time technique.
Expected accuracy: **60-80%** (within-window retrieval).

### 3.5 Strategy D — Multi-Scale Oscillator Hierarchy (MSOH)

Two TRN layers at different temporal scales:

```
Fast layer: alpha~0.7, K_fast=128, half-life ~2 tokens
Slow layer: alpha~0.99, K_slow=512, half-life ~69 tokens
-> Mixer MLP (2*d_model -> d_model)
```

State: 40 KB (trn_100m) vs 16 KB baseline.
Expected accuracy: 8.8% -> **30-45%**.

### 3.6 Strategy Comparison

| Strategy | Accuracy (projected) | Memory Overhead | Training | Latency |
|----------|---------------------|-----------------|----------|---------|
| Baseline | 8.8% | -- | -- | -- |
| A: SKES (S=256) | 40-60% | +1 MB/agent | FT 5K steps | O(S) retrieval |
| B: OCR (K_r=64) | 25-35% | +16 KB params | FT 5K steps | Negligible |
| C: PISS (P=256) | 60-80% | +3 MB/agent | None | O(P) replay |
| D: MSOH | 30-45% | +24 KB state | PT+FT 20K | ~1.5x/layer |

**Recommendation**: Implement A (SKES) + C (PISS) in parallel.
- SKES: content-addressed retrieval (factual queries)
- PISS: position-addressed retrieval (temporal queries)
- Orthogonal and combinable.

### 3.7 Evaluation Protocol

Three tasks: verbatim token recall (gap=[64,256,1024,4096,16384]),
semantic slot-fill recall, multi-hop recall. 100 trials per gap.

---

## Section 4 — Multi-Agent Scalability Simulation

### 4.1 Memory Architecture

**Transformer KV cache** (Llama-3 8B):

| Context | KV/agent | Agents on A100 80GB |
|---------|----------|---------------------|
| 4K      | 2.0 GB   | ~31                 |
| 16K     | 8.0 GB   | ~7                  |
| 128K    | 16.0 GB  | ~3                  |

**TRN state** (constant):

| Model    | State/agent | Agents on A100 80GB |
|----------|-------------|---------------------|
| trn_100m | 16 KB       | ~4.9M (theoretical) |
| trn_400m | 64 KB       | ~1.2M               |
| trn_1b   | 96 KB       | ~810K               |

### 4.2 Practical Agent Capacity (A100 80GB)

With TRN 1B model (~2 GB bf16) + CUDA overhead (2 GB):

```
Available: 76 GB

Pure TRN state (96 KB/agent): 81,604 agents
TRN + W=128 KV window (64 MB/agent): 1,187 agents
TRN + W=512 KV window (256 MB/agent): 296 agents

Llama-3 8B (16 GB model, 2 GB/agent @4K ctx): 31 agents
```

Agent density ratio (TRN vs Llama-3 8B):
- Pure TRN: **2,632x** more agents
- TRN+W=128: **38x** more agents
- TRN+W=512: **9.5x** more agents

### 4.3 Cost-per-Agent Analysis

Using Lambda Labs A100 80GB at $2.49/hour:

```
Llama-3 8B @4K: $0.0803/agent-hour (31 agents/GPU)
TRN+W=512:      $0.0084/agent-hour (296 agents/GPU)  -> 9.5x cheaper
TRN+W=128:      $0.0021/agent-hour (1187 agents/GPU) -> 38x cheaper
TRN pure:       $0.00003/agent-hour (81K agents/GPU)  -> 2,632x cheaper
```

**10,000 agents, 24/7 operation:**

| Configuration | GPUs Required | Monthly Cost |
|---------------|---------------|-------------|
| Llama-3 8B @4K | 323 A100s | $578,836/mo |
| TRN+W=128 KV | 9 A100s | $16,135/mo |
| TRN pure state | 1 A100 | $1,793/mo |
| **Savings (TRN+W=128)** | | **$562,701/mo (97.2%)** |

### 4.4 Multi-Agent Benchmark Script Design

```python
# scripts/bench_multi_agent.py

@dataclass
class AgentState:
    agent_id: int
    states_r: list[Tensor]  # per layer: (1, K) fp32
    states_i: list[Tensor]  # per layer: (1, K) fp32
    position: int = 0

    def state_bytes(self) -> int:
        return sum(r.numel() * r.element_size() * 2
                   for r in self.states_r)

def run_multi_agent_benchmark(
    n_agents_list: list[int],  # [1, 8, 64, 256, 1024]
    history_tokens: int,
    gen_tokens: int,
    device: str,
) -> None:
    # Create N agent states, interleaved generation, measure:
    # - Total state MB
    # - Aggregate throughput (total tps)
    # - Per-agent latency (ms/token)
    ...
```

### 4.5 State Serialization

Per agent (trn_1b): 96 KB binary, ~384 KB JSON.
10,000-agent fleet checkpoint: 960 MB (fits in 1 GB Redis hash).

### 4.6 Validation Targets

| Test | Pass Condition |
|------|---------------|
| Memory linearity | state_mb = N * 0.096 +/- 5% |
| Throughput scaling | throughput > N * 0.8 * single_tps |
| State isolation | diverging inputs -> different states |
| Serialization roundtrip | identical outputs (deterministic) |
| GPU capacity (A100, trn_1b) | >= 1,000 agents with W=128 KV |
| Cost efficiency vs Llama-3 | >= 10x more agents/dollar |

---

## Section 5 — Agent Runtime Integration

### 5.1 Core API: AgentMemory

```python
class AgentMemory:
    def __init__(self, d_model, n_oscillators, n_layers, tokenizer, embedding, device):
        ...
    def add_turn(self, role: str, content: str) -> None:
        # tokenize -> embed -> step_single per token -> O(1) state update
    def add_tool_result(self, tool: str, result: str) -> None:
        ...
    def get_context(self) -> Tensor:
        # demodulate and return context vector (d_model,)
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def to_dict(self) -> TRNStateDict: ...
    @classmethod
    def from_dict(cls, d: TRNStateDict, template) -> "AgentMemory": ...
```

### 5.2 LangGraph Integration

**TRNMemoryNode**: Compresses latest turn into TRN state after each LLM/tool invocation.
**TRNContextInjectionNode**: Injects TRN context vector into system message.

```python
# Graph topology:
# START -> context_inject -> llm -> route
# route -> [tools -> trn_compress -> context_inject]  (tool call)
# route -> [trn_compress -> END]                       (final answer)

# State: TRNAgentState with messages, trn_memory (dict), tool_results, turn_count
# Memory per checkpoint key: 16 KB (trn_100m)
```

### 5.3 AutoGen Integration

**TRNConversableAgent**: Extends ConversableAgent with persistent TRN memory.
Overrides `generate_reply()` to compress messages and inject context.

**TRNGroupChatManager**: Manages per-agent TRN state in multi-agent GroupChat.
Checkpoint: N * 16 KB per session.

### 5.4 CrewAI Integration

**TRNLongTermMemory**: Drop-in replacement for CrewAI's SQLite-backed LongTermMemory.
O(1) write cost, 16 KB footprint. `search()` returns soft context bias.

**TRNHybridMemory**: Combines TRNLongTermMemory (temporal context) with SQLite
(verbatim fact lookup) for best of both.

### 5.5 REST API (trn-server)

```
POST   /v1/sessions                    -> create session
POST   /v1/sessions/{id}/compress      -> fold turn into state
GET    /v1/sessions/{id}/context       -> get context vector
POST   /v1/sessions/{id}/generate      -> autoregressive generation
GET    /v1/sessions/{id}/state         -> export raw state
PUT    /v1/sessions/{id}/state         -> import state
DELETE /v1/sessions/{id}               -> release session
```

Latency budget (CPU, trn_100m):
- compress_turn(100 tok): ~0.43 ms
- get_context(): < 0.1 ms
- generate(128 tok): ~555 ms CPU, ~14 ms GPU (estimated)

### 5.6 State Serialization Format

```python
class TRNStateDict(TypedDict):
    r_real: list[list[float]]  # [n_layers][K]
    r_imag: list[list[float]]  # [n_layers][K]
    position: int
    metadata: dict
    version: str  # "1.0"
```

Wire size: 16 KB binary, ~48 KB JSON (trn_100m).

---

## Section 6 — Go/No-Go Criteria

### 6.1 Tier 1 — Mandatory (All Must Pass)

Failure of any = immediate **No-Go**.

| ID | Criterion | Threshold | Measurement |
|----|-----------|-----------|-------------|
| T1-1 | Numerical Stability | 0 NaN/Inf in state over 4096 tokens | `torch.isnan/isinf` |
| T1-2 | O(1) Memory Invariance | < 1% state growth, ctx=512 vs ctx=8192 | state_mb comparison |
| T1-3 | Throughput vs KV @ctx=4096 | >= 3.0x TRN/KV tps | bench_kv_vs_trn.py |
| T1-4 | Agent History @10k tokens | >= 180 tps, < 5% degradation 1k->10k | bench_agent_history.py |
| T1-5 | Serialization Fidelity | logit diff < 1e-5 after save/load | round-trip test |
| T1-6 | Memory Leak Absence | < 1 MB growth over 1000 generate() calls | tracemalloc |

Basis: Tier 1 thresholds are 78-80% of validated CPU baselines.

### 6.2 Tier 2 — Quality (Block Production, Not Experimental)

| ID | Criterion | Threshold | Measurement |
|----|-----------|-----------|-------------|
| T2-1 | Information Retention | >= 0.55 soft recall accuracy (200-turn session) | adapted bench_agent_history |
| T2-2 | Streaming Task Quality | TRN_loss/TF_loss <= 1.03 on timeseries+smoothing | bench_stream_tasks.py |
| T2-3 | GPU Throughput | >= 1500 tps (trn_100m, ctx=4096, GPU) | GPU benchmark |
| T2-4 | Multi-Session (100 agents) | >= 10,000 tps aggregate, p99 <= 10 ms | bench_multi_session.py |
| T2-5 | LangGraph Integration | 16 KB constant state, correct position tracking | integration test |

### 6.3 Tier 3 — Performance Targets (Informational)

| Criterion | Target | Stretch |
|-----------|--------|---------|
| TRN vs KV at ctx=16k | >= 10x | >= 20x |
| TRN vs KV at ctx=100k | >= 50x | >= 100x |
| REST compress_turn latency | <= 5 ms p99 | <= 2 ms |
| State compression ratio @10k | >= 1000x | >= 10000x |
| Cold-start warmup tokens | <= 256 | <= 64 |

### 6.4 Disqualifying Conditions (Immediate No-Go)

1. NaN/Inf in state (alpha/amplitude instability)
2. Memory leak > 10 MB over 1000 calls
3. State grows with context > 5%
4. Test coverage < 80%
5. Serialization logit diff > 1e-3
6. Cross-session state contamination

### 6.5 Decision Matrix

```
ALL Tier 1 pass? --- NO  ---> STOP. No-Go.
      |
     YES
      |
DISQUALIFYING? ---- YES ---> STOP. Architectural review.
      |
     NO
      |
Tier 2 count?
  0-2 pass ---> Experimental only
  3-4 pass ---> Beta (limited production)
  5   pass ---> Production (all integrations)
      |
Tier 3 met?
  < 3  ---> Standard positioning
  >= 4 ---> Performance-competitive (vs Mamba, RWKV, Griffin)
```

---

## Section 7 — Final Deliverables and Roadmap

### 7.1 Code Deliverables

**Core Library (trn-core, MIT)**:
- AgentMemory class (add_turn, get_context, save/load)
- TRNStateDict serialization format
- RedisAgentMemoryStore

**Agent Framework Integrations**:
- TRNMemoryNode + TRNContextInjectionNode (LangGraph)
- TRNConversableAgent + TRNGroupChatManager (AutoGen)
- TRNLongTermMemory + TRNHybridMemory (CrewAI)

**Server and Infrastructure**:
- trn-server FastAPI (7 REST endpoints)
- Docker image
- TRNPagedAttentionBackend (vLLM prototype)
- TRNMemoryWrapper (HuggingFace)
- llama.cpp C bindings + GGUF schema

**Benchmark Scripts** (8 scripts):
- bench_phase7_gpu.py, bench_gpu_memory.py
- bench_needle_haystack.py, bench_long_qa.py, bench_goal_tracking.py
- bench_selective_recall.py, bench_multi_agent_scale.py
- bench_agent_frameworks.py

### 7.2 Implementation Roadmap (8 Weeks)

```
Week  Task                                      Blockers
----  ----------------------------------------  ------------------
  1   AgentMemory class + save/load + Redis     resonance.py (done)
  1   TRNStateDict serialization format         AgentMemory
  1   Unit tests (>= 30 tests)                  AgentMemory
  2   LangGraph integration + tests             AgentMemory
  2   AutoGen TRNConversableAgent               AgentMemory
  3   AutoGen GroupChat + CrewAI integration     AutoGen agent
  3   CrewAI HybridMemory + tests               CrewAI LTM
  4   bench_multi_session (100 concurrent)       AgentMemory, Redis
  4   Go/No-Go Tier 1 + Tier 2 evaluation       all above
  4   >> CHECKPOINT: Tier 1 Go/No-Go decision <<
  5   trn-server FastAPI + Docker                AgentMemory, Redis
  5   HuggingFace TRNMemoryWrapper              AgentMemory
  6   vLLM PagedAttention backend (proto)        HF wrapper
  6   llama.cpp C bindings + GGUF               resonance.py
  6   Phase 7 GPU benchmark suite               GPU access
  7   Tier 3 evaluation on A100                  GPU benchmarks
  7   All documentation + integration guides    all integrations
  7   LIMITATIONS.md + ADRs                     Go/No-Go results
  8   trn-core PyPI v1.0 package                all code + docs
  8   Final test coverage audit (>= 80%)        all
  8   >> FINAL Go/No-Go report <<
```

Critical path: AgentMemory (W1) -> LangGraph (W2) -> bench_multi_session (W4)
-> Tier 1 decision (W4 end) -> GPU benchmarks (W6) -> final report (W8).

Week 4 is a **hard gate**: if Tier 1 fails, Weeks 5-8 pivot to root cause
analysis instead of productization.

### 7.3 Success Criteria

Phase 7 is successful when ALL of:
1. All Tier 1 Go/No-Go criteria pass (CPU + GPU)
2. >= 4/5 Tier 2 criteria pass on CPU baseline
3. Test coverage >= 80% across all new modules
4. LangGraph, AutoGen, CrewAI integration tests pass with real AgentMemory
5. trn-server Docker image starts and serves in < 5 seconds
6. BENCHMARK.md updated with Phase 7 GPU results
7. VALIDATION_RESULTS_PHASE7.md completed with all raw data
8. Limitations documented (selective copy 0.088, verbatim recall unsupported)

### 7.4 Expected Outcomes

| Metric | CPU Validated | GPU Expected (A100) |
|--------|--------------|---------------------|
| TRN tps @ctx=4096 | 249.8 | ~2000-2500 |
| TRN tps @10k history | 230.9 | ~1850-2300 |
| TRN state (trn_100m) | 8.0 KB (K=128) | 16 KB (K=256) |
| TRN vs KV speedup @4096 | 5.47x | 4.0-6.0x |
| 1,000 sessions memory | TRN: 16 MB | KV: 156 GB (9,750x) |

---

## Appendix A — State Size Verification

```python
@pytest.mark.parametrize("preset, expected_kb", [
    ("trn_100m", 16),   # 8 * 256 * 2 * 4 = 16,384 bytes
    ("trn_400m", 64),   # 16 * 512 * 2 * 4 = 65,536 bytes
    ("trn_1b",   96),   # 24 * 512 * 2 * 4 = 98,304 bytes
])
def test_resonance_state_size(preset, expected_kb):
    cfg = getattr(TRNConfig, preset)()
    state_bytes = cfg.n_layers * cfg.n_oscillators * 2 * 4
    assert state_bytes == expected_kb * 1024
    assert state_bytes < 200 * 1024  # fits in L2 cache
```

## Appendix B — Benchmark CLI

```bash
# GPU benchmark
python scripts/bench_phase7_gpu.py \
  --models trn_400m trn_1b hybrid_400m_50 \
  --context-lens 512,2048,8192,32768,131072 \
  --gen-tokens 256 --batch-size 1 --device cuda:0

# Retention evaluation
python scripts/eval_phase7_retention.py \
  --models trn_400m hybrid_400m_50 \
  --tests nih probe periodic multihop \
  --context-lens 512,1024,2048,4096,8192

# Multi-agent scalability
python scripts/bench_multi_agent.py \
  --n-agents 1,8,64,256,1024 --device cuda

# Go/No-Go evaluation
python scripts/eval_go_no_go.py --tier 1 2 3 --device cuda
```
