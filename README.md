# TriMemory

[![tests](https://img.shields.io/badge/tests-277%20passing-brightgreen)]()
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.1%2B-orange)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)

Role-specialized memory architecture for language models. Three memory paths -- KV window, retrieval index, TRN state -- each handling a different kind of remembering.

```
pip install -e .
```

## Quickstart

```python
import torch
from trn import TRNConfig
from trn.tri_memory import TriMemoryEngine

cfg = TRNConfig(
    vocab_size=8192, d_model=128, n_oscillators=64,
    n_layers=4, d_ff=512, max_seq_len=1024,
)
model = TriMemoryEngine(
    cfg, window_size=64, chunk_size=32, max_retrieval_chunks=256,
    enable_trn=True, enable_retrieval=True,
)

ids = torch.randint(0, cfg.vocab_size, (1, 512))
out = model(ids, labels=ids)
print(f"loss: {out['loss']:.4f}")
```

### Standalone TRN

```python
from trn import TRNConfig, TRNModel

cfg = TRNConfig.trn_100m()
model = TRNModel(cfg)

prompt = torch.randint(0, cfg.vocab_size, (1, 16))
tokens = model.generate(prompt, max_new_tokens=128)
# O(1) memory per step. No KV cache.
```

### Streaming Evaluation

```bash
python scripts/run_trimemory_streaming_eval.py \
    --model trimemory --device cuda --seeds 0,1,2 \
    --search-mode hidden --num-episodes 128
```

## Architecture

```
Input
 |-- KV window (last W tokens, exact attention)
 |-- Retrieval index (archived chunks, cosine search)
 |-- TRN state (compressed patterns, constant size)
 |
 v
Mixer gate: g = sigmoid(W_g * x)
  out = g * attn + (1-g) * trn + retrieval_context
 |
 v
FFN -> logits
```

KV handles recent context. Retrieval handles old facts that got evicted. TRN handles compressed patterns and periodicity. Each path gets what it is good at.

### Memory Paths

| Path | What it stores | Size | Access |
|------|---------------|------|--------|
| KV window | Recent W tokens | O(W) per layer | Exact attention |
| Retrieval | Archived chunks with hidden states | Fixed capacity (default 256 chunks) | Cosine similarity search |
| TRN state | Compressed history (amplitude, phase, frequency) | O(K) per layer, constant | Linear recurrence |

### Retrieval Search Modes

Three modes for chunk retrieval. `hidden` is default.

| Mode | Method | Measured gold containment |
|------|--------|--------------------------|
| `bag` | Bag-of-token cosine | 0.323 |
| `hidden` | Hidden-state cosine | 0.415 |
| `hybrid` | Weighted combination | 0.404 |

Hidden-state search improved gold containment by 29% over bag-of-token. The bottleneck shifted from "finding the right chunk" to "using it correctly in decoding".

### Token Lifecycle

Tokens enter KV window. Every C tokens, the oldest chunk gets evicted. TRN state always updates. If the chunk scores high on saliency, it goes to the retrieval index. When the model needs old information, the router gates retrieval context into the main path.

## Features

- Three-path memory: KV window + retrieval index + TRN state
- Hidden-state, bag-of-token, and hybrid retrieval search
- Gated mixer for path combination
- Saliency-based chunk archival
- Constant-size TRN state (8--96 KB depending on config)
- Streaming inference with O(1) memory per step
- Multi-agent support (16 KB per agent for trn_100m)
- Score breakdown logging for failure analysis

## Benchmark Results

### Generation Throughput (CPU, d=256, L=8, K=128)

| History | TRN (tps) | TF+KV (tps) | TRN State | KV Cache (fp32) |
|---------|-----------|-------------|-----------|-----------------|
| 1,000 | 240 | 73.8 | 8 KB | 15.6 MB |
| 5,000 | 244 | 35.9 | 8 KB | 78.1 MB |
| 10,000 | 231 | 15.5 | 8 KB | 156.3 MB |

TRN throughput stays flat. Transformer+KV degrades as O(1/T).

### Multi-Agent Scaling (trn_100m, T=1000)

| Agents | TRN Total | KV Total | Ratio |
|--------|-----------|----------|-------|
| 10 | 0.16 MB | 312 MB | 2,000x |
| 100 | 1.56 MB | 3,125 MB | 2,000x |
| 1,000 | 15.6 MB | 31,250 MB | 2,000x |

Config-specific. bf16 KV halves it. See [docs/PUBLIC_CLAIMS.md](docs/PUBLIC_CLAIMS.md).

### Quality (Toy Config)

| Task | TRN | Transformer |
|------|-----|-------------|
| Periodic Pattern Detection | 0.78--1.00 | 1.00 |
| Copy task | 1.00 | 1.00 |
| Selective copy | 0.088 | 0.962 |
| Needle-in-Haystack | 0.00 | -- |

TRN cannot do content-addressed retrieval. 8.8% selective copy vs 96.2% for Transformer. That is why TriMemory pairs it with an explicit retrieval path.

### TriMemory Streaming Eval (Search Mode Comparison)

| Mode | old_fact_acc | retrieval_hit | gold_in_topk | Type A failure |
|------|-------------|---------------|--------------|----------------|
| bag | 0.125 | 0.256 | 0.323 | 67.9% |
| hidden | 0.135 | 0.320 | 0.415 | 58.5% |
| hybrid | 0.130 | 0.309 | 0.404 | 59.3% |

32 episodes x 3 seeds, 300 steps, bf16, CUDA. Hidden search finds better chunks, but decode success is still 0.000 -- the current bottleneck is the decoder/mixer integration, not search.

## Current Status

Retrieval path: validated. Gold chunk selection works. Hidden-state search outperforms bag-of-token.

Copy-mix (additive `main_logits + alpha * copy_logits`): confirmed effective for old fact recovery at the token level.

Full model (KV + Retrieval + TRN): `D > max(A,B,C)` observed in composite score under specific settings.

TRN standalone: pattern learning grows at 3000 steps. Seed dependence remains.

Decoder/mixer integration: current bottleneck. Even when the correct chunk is retrieved and the gate uses it, the model often fails to produce the correct token. This is the next target.

See [ROADMAP.md](ROADMAP.md) for details.

## Known Limitations

- TRN cannot perform content-addressed retrieval. Structural property of linear recurrence.
- Decoder/mixer does not yet reliably convert retrieved chunks into correct tokens (decode success = 0.000 in streaming eval).
- All experiments use 1--100M parameter models. Scaling behavior at 1B+ is unknown.
- TRN seed dependence is still high for pattern tasks.

See [docs/TRN_LIMITATIONS.md](docs/TRN_LIMITATIONS.md).

## Repository Structure

```
src/trn/
    tri_memory.py    TriMemoryEngine (KV + TRN + Retrieval)
    retrieval.py     RetrievalIndex (bag/hidden/hybrid search)
    model.py         TRNModel (standalone)
    resonance.py     TemporalResonanceLayer (oscillator recurrence)
    baseline.py      TransformerModel (A/B comparison)
    saliency.py      SaliencyArchiver (chunk scoring)
    router.py        Retrieval router / gate
    config.py        TRNConfig (toy, trn_100m, trn_400m, trn_1b)
    agent_memory.py  Streaming agent inference wrapper
    integrations/    vLLM backend, llama.cpp, LangChain, AutoGen, CrewAI

scripts/
    run_trimemory_streaming_eval.py    Streaming evaluation with telemetry
    analyze_trimemory_oldfact_failures.py  Failure classification (Type A--E)
    bench_phase7_gpu.py                GPU benchmark (TRN vs TF+KV)
    eval_go_no_go.py                   Go/No-Go gate evaluation

tests/     277 unit tests
docs/      Architecture, limitations, integration guides
artifacts/ Timestamped experiment runs
```

## Install

```bash
git clone https://github.com/TODO/trn.git
cd trn
pip install -e ".[dev]"
pytest
```

Requires Python 3.10+, PyTorch 2.1+.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{trimemory2026,
  title  = {TriMemory},
  author = {TriMemory Contributors},
  year   = {2026},
  url    = {https://github.com/TODO/trn},
}
```
