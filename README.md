# TriMemory -- Memory Architecture for LLM Agents

[![tests](https://img.shields.io/badge/tests-277%20passing-brightgreen)]()
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.1%2B-orange)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)

Three-path memory layer for LLM agents. KV window for recent tokens,
retrieval index for archived chunks, TRN recurrent state for compressed
long-range patterns. 8 KB of state per agent. Flat throughput at 10,000+
token history where a KV cache is 156 MB and 15x slower.

- **KV window** -- recent tokens, exact attention
- **Retrieval index** -- archived chunks, cosine search over hidden states
- **TRN state** -- compressed patterns and periodicity, constant size (8--96 KB)

A learned 3-way softmax gate mixes all three paths per token.

### TRN vs Transformer+KV (CPU, d=256, L=8, K=128)

| History | TRN (tps) | TF+KV (tps) | TRN State | KV Cache (fp32) |
|---------|-----------|-------------|-----------|-----------------|
| 1,000 | 240 | 73.8 | 8 KB | 15.6 MB |
| 5,000 | 244 | 35.9 | 8 KB | 78.1 MB |
| 10,000 | 231 | 15.5 | 8 KB | 156.3 MB |

TRN state is O(K), not O(T). Throughput stays flat while Transformer+KV degrades as history grows.

---

## Quick Demo

```bash
git clone https://github.com/amabito/tri-memory.git
cd tri-memory
pip install -e ".[dev]"
pytest  # 277 tests
```

### TriMemoryEngine -- training loop

```python
import torch
from trimemory import TRNConfig
from trimemory.tri_memory import TriMemoryEngine

cfg = TRNConfig(
    vocab_size=8192, d_model=128, n_oscillators=64,
    n_layers=4, d_ff=512, max_seq_len=1024,
)
model = TriMemoryEngine(
    cfg,
    window_size=64,           # KV window: last 64 tokens
    chunk_size=32,            # eviction granularity
    max_retrieval_chunks=256, # retrieval index capacity
    enable_trn=True,
    enable_retrieval=True,
)

ids = torch.randint(0, cfg.vocab_size, (1, 512))
out = model(ids, labels=ids)
print(f"loss: {out['loss']:.4f}")

mem = model.memory_summary()
print(f"TRN state: {mem['trn_state_bytes']} bytes (constant)")
print(f"KV window: {mem['kv_window_bytes']} bytes (bounded)")
```

### AgentMemory -- stateful per-token streaming

```python
from trimemory import TRNConfig
from trimemory.agent_memory import AgentMemory

mem = AgentMemory(TRNConfig.toy(), device="cpu")

# Feed tokens. State is O(K) -- no KV cache.
mem.add_tokens([1, 2, 3, 4, 5])

state = mem.get_state()
print(f"TRN state: {mem.state_size_bytes()} bytes, "
      f"position: {state['position']}")

# Save and restore across agent turns.
mem.save("turn1.pt")
mem.load("turn1.pt")
```

### Standalone TRN

```python
from trimemory import TRNConfig, TRNModel
import torch

cfg = TRNConfig.trn_100m()
model = TRNModel(cfg)

prompt = torch.randint(0, cfg.vocab_size, (1, 16))
tokens = model.generate(prompt, max_new_tokens=128)
# O(1) memory per step. No KV cache.
```

---

## How it differs from RAG

| | Standard RAG | TriMemory |
|--|--|--|
| Retrieval basis | Semantic similarity | Authority chain + semantic similarity |
| Handles amendment override | No (both chunks retrieved equally) | Yes (structured authority resolution) |
| Memory per agent at 10k context | O(context) -- 156 MB KV cache (fp32) | O(1) -- 8 KB TRN state |
| Throughput at 10k context | 15.5 tps (TF+KV, CPU) | 231 tps (TRN, CPU) |
| Content-addressed retrieval | Yes | No -- TRN recall is 0.0 (honest limitation) |
| Status | Production | Alpha (toy-scale models, N=10 benchmark) |

Authority resolution requires `use_compact_memory_packet=True` and document metadata. Off by default.

---

## Architecture

```
Input
 |-- KV window (last W tokens, exact attention)
 |-- Retrieval index (archived chunks, cosine search)
 |-- TRN state (compressed patterns, constant size)
 |
 v
3-way gate: [g_kv, g_trn, g_ret] = softmax(W_gate * x)
  out = g_kv * kv_out + g_trn * trn_out + g_ret * ret_out
 |
 v
FFN -> logits
```

| Path | What it stores | Size | Access |
|------|---------------|------|--------|
| KV window | Recent W tokens | O(W) per layer | Exact attention |
| Retrieval | Archived chunks with hidden states | Fixed capacity (default 256 chunks) | Cosine similarity search |
| TRN state | Compressed history (amplitude, phase, frequency) | O(K) per layer, constant | Linear recurrence |

Tokens enter KV window. Every C tokens, the oldest chunk gets evicted and scored
for saliency. High-saliency chunks go to the retrieval index. TRN state always updates.
The gate routes each path's output based on what the current token needs.

---

## Benchmark Results

### V5 A/B/C/D (Seeds 1--10, 3000 steps)

| Config | Composite | Strength | Caveat |
|--------|-----------|----------|--------|
| A (KV only) | 0.263 | Baseline | No long-range memory |
| B (KV+TRN) | 0.457 | Pattern detection 0.678 | 2/10 seeds stuck on pattern (D recovers) |
| C (KV+Ret) | 0.369 | Old fact recall 0.433 | No pattern capability |
| **D (Full)** | **0.676** | **Pattern 0.805, Old fact 0.719** | Toy scale only (d=128, L=4) |

D >= max(A,B,C) in 10/10 seeds (mean delta +0.165). H1--H4 all PASS.
See [docs/FINAL_VERDICT.md](docs/FINAL_VERDICT.md).

### Multi-Agent Scaling (trn_100m, T=1000)

| Agents | TRN Total | KV Total (fp32) | Ratio |
|--------|-----------|------------------|-------|
| 10 | 0.16 MB | 312 MB | 2,000x |
| 100 | 1.56 MB | 3,125 MB | 2,000x |
| 1,000 | 15.6 MB | 31,250 MB | 2,000x |

Config-specific. bf16 KV halves the ratio. See [docs/PUBLIC_CLAIMS.md](docs/PUBLIC_CLAIMS.md).

---

## PolicyBench (N=10)

Document authority QA -- 10 samples, Japanese corporate IT security policy, 9 evaluation types:
authority resolution, amendment override, scope-dependent values, transition states,
and more. English language corpus evaluation is planned.

```
data/policybench/policy_v1.jsonl
```

Each sample has multi-document context (base policy + amendments + circulars),
a query, gold answer, authority chain, and failure class annotation.

---

## VERONICA Integration

TriMemory handles memory architecture. Runtime containment (budget enforcement,
circuit breaking, policy governance) is handled by
[VERONICA-core](https://github.com/amabito/veronica-core).
Together: governed knowledge execution for LLM agents.

---

## Known Limitations

- TRN cannot perform content-addressed retrieval. Selective copy accuracy is 8.8% vs Transformer 96.2%. Needle-in-Haystack recall is 0.0. Structural property of linear recurrence.
- All experiments use toy-scale models (1--100M parameters). Scaling behavior at 1B+ is unknown.
- B config (KV+TRN) shows seed-dependent pattern failure (2/10 stuck), though D recovers.
- PolicyBench is N=10. Validation on larger corpora is future work.
- CompactMemoryPacket (authority resolution, conflict detection) is off by default. Requires `use_compact_memory_packet=True` and document metadata with authority chains.
- Retrieval index is not persistent across process restarts.
- Gate telemetry is diagnostic only -- gate ratios vary by input distribution.
- Alpha status. No production deployment.

See [docs/TRN_LIMITATIONS.md](docs/TRN_LIMITATIONS.md) and [docs/PUBLIC_CLAIMS.md](docs/PUBLIC_CLAIMS.md).

---

## Repository Structure

```
src/trimemory/
    tri_memory.py    TriMemoryEngine (KV + TRN + Retrieval)
    retrieval.py     RetrievalIndex (bag/hidden/hybrid search)
    model.py         TRNModel (standalone)
    resonance.py     TemporalResonanceLayer (oscillator recurrence)
    baseline.py      TransformerModel (A/B comparison)
    saliency.py      SaliencyArchiver (chunk scoring)
    router.py        Retrieval router / gate
    config.py        TRNConfig (toy, trn_100m, trn_400m, trn_1b)
    agent_memory.py  Streaming agent inference wrapper

scripts/
    eval_go_no_go.py                   Go/No-Go gate evaluation
    run_trimemory_streaming_eval.py    Streaming evaluation with telemetry
    bench_phase7_gpu.py                GPU benchmark (TRN vs TF+KV)

data/policybench/
    policy_v1.jsonl                    PolicyBench N=10

tests/     277 unit tests
docs/      Architecture, limitations, public claims audit
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
