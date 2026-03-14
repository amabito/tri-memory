# vLLM Integration: DualMemoryEngine

Dual-memory architecture combining windowed KV attention with TRN resonance.
Provides O(1) decode memory growth while retaining exact attention for recent tokens.

## Architecture

```
Input Tokens (B, T)
       |
  Embedding + sinusoidal PE
       |
       v
+------------------------------------------+  x N_LAYERS
|           DualMemoryBlock                |
|                                          |
|   x = RMSNorm(x)                        |
|        /              \                  |
|   Windowed             TRN               |
|   KV Attn             Resonance          |
|   (last W tokens)     (all tokens,       |
|                        O(1) state)       |
|        \              /                  |
|   g = sigmoid(W_gate * x + b)           |
|   out = g * attn_out + (1-g) * trn_out  |
|                                          |
|   x = x + Dropout(out)                  |
|   x = x + Dropout(FFN(RMSNorm(x)))      |
+------------------------------------------+
       |
  RMSNorm -> LM head
       |
   logits (B, T, vocab_size)
```

Gate initialisation: `sigmoid(0) = 0.5` — equal initial weight for both paths.
Weight tying: `lm_head.weight = embedding.weight` when `cfg.tie_weights=True`.

---

## Memory Comparison

### KV Cache Memory

```
KV cache per layer = 2 * n_heads * head_dim * T * 4 bytes  (fp32)
Full KV (no window) grows linearly with T.
Windowed KV is capped at 2 * n_heads * head_dim * W * 4 bytes.
```

Numerical examples (trn_100m: d=512, 8 heads, head_dim=64, fp32):

| Context T | Full KV (8 layers) | Windowed KV W=256 | TRN state |
|-----------|--------------------|-------------------|-----------|
| 256       | 4 MB               | 4 MB              | 16 KB     |
| 1,024     | 16 MB              | 4 MB              | 16 KB     |
| 4,096     | 64 MB              | 4 MB              | 16 KB     |
| 16,384    | 256 MB             | 4 MB              | 16 KB     |
| 100,000   | 1.5 GB             | 4 MB              | 16 KB     |

TRN state formula: `n_layers * K * 2 * 4` bytes (always constant).
trn_100m: `8 * 256 * 2 * 4 = 16,384` bytes = 16 KB.

### Combined DualMemoryEngine Memory

Total decode memory = windowed KV + TRN state.
At T=100,000 with W=256: `4 MB + 16 KB` vs `1.5 GB` for full KV.

---

## Quick Start

### Installation

```bash
# From repo root
pip install -e src/          # or: uv add trn --editable
```

### Training on Copy Task

```python
from trn.config import TRNConfig
from trn.integrations.vllm_backend import DualMemoryEngine
import torch

cfg = TRNConfig.toy()        # d_model=128, K=64, L=2
model = DualMemoryEngine(cfg, window_size=64).cuda()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop (copy task)
for step in range(500):
    ids    = torch.randint(1, cfg.vocab_size, (4, 32), device="cuda")
    result = model(ids, labels=ids)
    result["loss"].backward()
    optim.step()
    optim.zero_grad()
```

### Autoregressive Generation

```python
import torch
from trn.config import TRNConfig
from trn.integrations.vllm_backend import DualMemoryEngine

cfg   = TRNConfig.trn_100m()
model = DualMemoryEngine(cfg, window_size=256).eval()
# model.load_state_dict(torch.load("checkpoint.pt")["model"])

prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
tokens = model.generate(prompt, max_new_tokens=128, temperature=0.8, top_k=50)
# tokens: (B, 128)
```

Generation allocates `n_layers` windowed KV caches (capped at `window_size`)
plus `n_layers` TRN state tensors (each `(B, K)` fp32).

---

## API Reference

### DualMemoryEngine

```python
class DualMemoryEngine(nn.Module):
    def __init__(self, cfg: TRNConfig, window_size: int = 256) -> None: ...

    def forward(
        self,
        input_ids: Tensor,               # (B, T)
        labels:    Optional[Tensor] = None,  # (B, T), next-token targets
    ) -> dict:                           # {"logits": Tensor, "loss"?: Tensor}

    def generate(
        self,
        prompt_ids:     Tensor,          # (B, prompt_len)
        max_new_tokens: int   = 128,
        temperature:    float = 1.0,
        top_k:          int   = 50,
    ) -> Tensor:                         # (B, max_new_tokens)

    def configure_optimizer_param_groups(
        self,
        weight_decay: float = 0.1,
    ) -> list[dict]:                     # decay / no-decay param groups

    def num_parameters(
        self,
        non_embedding: bool = True,
    ) -> int
```

### WindowedKVCache

```python
@dataclass
class WindowedKVCache:
    k_cache:     Tensor   # (B, n_heads, T_stored, head_dim)
    v_cache:     Tensor   # (B, n_heads, T_stored, head_dim)
    window_size: int

    def append(self, k_new: Tensor, v_new: Tensor) -> WindowedKVCache:
        """Append new K/V, evict oldest tokens if T_stored > window_size."""
```

---

## Benchmark

```bash
# Decode throughput vs context length
python scripts/bench_dual_memory.py \
    --config toy \
    --device cuda \
    --window-size 256 \
    --context-lengths 256 512 1024 2048 4096

# Expected output columns:
#   context_len | backend    | tokens/s | ms/token | peak_mem_MB
```

Script: `scripts/bench_dual_memory.py` (created by Task #1).

Artifacts written to `results/bench_dual_memory/`:
- `summary.csv` — throughput table
- `memory.csv`  — peak GPU memory per context length and backend

---

## Optimizer Configuration

`DualMemoryEngine.configure_optimizer_param_groups()` separates parameters
into decay and no-decay groups following the same convention as `TRNModel`:

No-decay (weight_decay=0.0):
- `omega_base` — learnable base frequencies
- `res_scale` — learnable resonance output scale
- `*.bias` — all biases
- Norm layers
- Embedding weights

```python
param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
optim = torch.optim.AdamW(param_groups, lr=3e-4)
```

---

## Training Notes

### Banded Causal Mask

Training uses a banded causal mask (window size W). Token at position `i` attends
to tokens in `[max(0, i-W+1), i]`. This matches the KV cache behaviour at generation
time.

The TRN scan processes the full sequence regardless of window size — it integrates
information from all past tokens through its decaying resonance state.

### Mixed-Precision

TRN resonance state and alpha gates are kept in fp32 even under bf16/fp16 AMP.
(bf16 cannot represent values close to 1.0 accurately; 0.99 rounds to 1.0 in bf16,
causing the decay gate to become a latch.)

```python
model = DualMemoryEngine(cfg, window_size=256).cuda()
scaler = torch.cuda.amp.GradScaler()

with torch.autocast("cuda", dtype=torch.bfloat16):
    result = model(input_ids, labels=labels)
    # Resonance state arithmetic is still fp32 inside TemporalResonanceLayer.
```

### Gradient Checkpointing

Wrap blocks for activation checkpointing:

```python
from torch.utils.checkpoint import checkpoint_sequential

# Replace forward loop with checkpointed blocks
# (requires the blocks to be stateless, which they are at training time)
```

---

## TRNConfig Parameters Relevant to DualMemoryEngine

| Parameter       | Default | Effect in DualMemoryEngine                        |
|-----------------|---------|---------------------------------------------------|
| `d_model`       | 768     | embedding dimension shared by attn and TRN        |
| `n_oscillators` | 256     | K, resonance channels per layer                   |
| `n_layers`      | 12      | number of DualMemoryBlock layers                  |
| `phase_mode`    | "log"   | `log` = `omega * log1p(t)`, `linear` = `omega * t` |
| `state_norm`    | True    | per-channel max-abs normalisation after state update |
| `amplitude_max` | 3.0     | softplus clamp ceiling for A                      |
| `res_scale_init`| 0.05    | initial value of learnable res_scale              |
| `gate_bias_init`| 0.85    | initial alpha gate bias (sigmoid(b) target)       |

`window_size` is a `DualMemoryEngine.__init__` argument (not in `TRNConfig`).

---

## Troubleshooting

### OOM at large context during training

The banded mask `(T, T)` is created on-device per batch and step.
At T=2048, this is `2048^2 * 4 = 16 MB` per layer.

Options:
- Reduce `T` (sequence length) or `window_size`.
- Use gradient checkpointing to trade compute for memory.
- Set `use_parallel_scan=False` in TRNConfig to save GPU VRAM
  (parallel scan doubles intermediate tensors).

### dtype mismatch in generation

If `model.forward()` is called under `torch.autocast` but `step_single` is called
outside autocast, the projection dtype may differ.

Both `TRNModel.generate()` and `DualMemoryEngine.generate()` cast the embedding
to `param_dtype` before each block, so AMP does not affect generation correctness.

### Gate always near 0 or 1 after training

The mixer gate collapses when one path (attn or TRN) dominates.
Signs: gate weights in `gate_proj.weight` have large magnitude.

Mitigations:
- Reduce `window_size` to force the model to use TRN for distant context.
- Add L2 regularisation on `gate_proj.weight` (include in decay param group).
- Lower `gate_bias_init` from default 0.0 to a small negative value to bias toward TRN.

### Loss does not decrease

Check that labels are shifted correctly. The model computes next-token loss:
predict position `t` from tokens `0..t-1`.
If you pass `labels=input_ids`, the causal shift is applied internally —
do not shift manually.
