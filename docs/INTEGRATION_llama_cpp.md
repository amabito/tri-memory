# llama.cpp Integration: C TRN Resonance Layer

Pure C implementation of `TemporalResonanceLayer.step_single()` for deployment
in llama.cpp-style inference engines.

## Overview

```
PyTorch model (.pt)
       |
       | export_weights.py
       v
 resonance_layer_N.bin   (binary, float32, little-endian)
       |
       | trn_resonance_load()
       v
  TRNResonanceLayer*     (C struct, no external deps)
       |
       | trn_resonance_step()  -- O(1) per token per layer
       v
    out[d_model]         (float32 output embedding)
```

No external dependencies in C. Only `stdlib.h`, `math.h`, `string.h`, `stdint.h`.

---

## Binary Weight Format

Header (32 bytes = 8 x 4-byte words, little-endian):

| Offset | Type    | Field         | Notes                      |
|--------|---------|---------------|----------------------------|
| 0      | uint32  | magic         | `0x54524E31` ("TRN1")      |
| 4      | uint32  | version       | `1`                        |
| 8      | int32   | d_model       | embedding dimension        |
| 12     | int32   | K             | oscillator count           |
| 16     | int32   | phase_mode    | `0`=log, `1`=linear        |
| 20     | int32   | state_norm    | `0`=off, `1`=on            |
| 24     | float32 | amplitude_max | softplus clamp ceiling     |
| 28     | float32 | res_scale     | learnable output scale     |

Data (all float32, row-major, immediately after header):

| Section        | Shape              | Size (bytes)      |
|----------------|--------------------|-------------------|
| proj_weight    | [4K, d_model]      | 4K × d_model × 4  |
| proj_bias      | [4K]               | 4K × 4            |
| omega_base     | [K]                | K × 4             |
| w_res_weight   | [d_model, K]       | d_model × K × 4   |

Total file size: `32 + (4K·d_model + 4K + K + d_model·K) × 4` bytes.

---

## Weight Export Workflow

```python
import torch
from trn.integrations.llamacpp.export_weights import export_layer, export_model

# Single layer
state_dict = torch.load("checkpoint.pt", map_location="cpu")["model"]
export_layer(
    state_dict,
    layer_idx=0,
    output_path="resonance_layer_000.bin",
    phase_mode="log",      # must match TRNConfig.phase_mode
    state_norm=True,       # must match TRNConfig.state_norm
    amplitude_max=3.0,     # must match TRNConfig.amplitude_max
)

# All layers at once
paths = export_model(state_dict, output_dir="weights/", phase_mode="log")
```

State-dict key layout (per block index `i`):

| Key                                          | Shape         |
|----------------------------------------------|---------------|
| `blocks.{i}.resonance.proj.proj.weight`      | [4K, d_model] |
| `blocks.{i}.resonance.proj.proj.bias`        | [4K]          |
| `blocks.{i}.resonance.proj.omega_base`       | [K]           |
| `blocks.{i}.resonance.W_res.weight`          | [d_model, K]  |
| `blocks.{i}.resonance.res_scale`             | scalar        |

---

## Compilation

### Linux / WSL2 (recommended)

```bash
cd src/trn/integrations/llamacpp
gcc -O2 -shared -fPIC -o trn_resonance.so trn_resonance.c -lm
```

### Windows (MSVC)

From a VS Developer Command Prompt:

```cmd
cd src\trn\integrations\llamacpp
cl /O2 /LD trn_resonance.c /Fe:trn_resonance.dll
```

### CMake

```cmake
add_library(trn_resonance SHARED trn_resonance.c)
target_link_libraries(trn_resonance m)
target_compile_options(trn_resonance PRIVATE -O2)
```

---

## C API Reference

```c
#include "trn_resonance.h"

/* Load weights from binary file. Returns NULL on failure. */
TRNResonanceLayer * trn_resonance_load(const char * path);

/* Free all resources. */
void trn_resonance_free(TRNResonanceLayer * layer);

/* Allocate zero-initialised state for batch_size sequences.
   Returns float* of size batch_size * K * 2 (r_real then r_imag).
   Caller manages memory; free with trn_resonance_state_free(). */
float * trn_resonance_state_alloc(const TRNResonanceLayer * layer, int batch_size);
void    trn_resonance_state_free(float * state);

/* Single-step forward pass (O(1) per token).
   x:         [batch_size * d_model]  input embeddings
   r_real:    [batch_size * K]        real state (updated in-place)
   r_imag:    [batch_size * K]        imag state (updated in-place)
   position:  absolute token index, 0-based
   out:       [batch_size * d_model]  output embeddings
   Returns 0 on success, non-zero on error. */
int trn_resonance_step(
    const TRNResonanceLayer * layer,
    const float * x, float * r_real, float * r_imag,
    int position, float * out, int batch_size);

/* Metadata accessors */
int   trn_resonance_d_model(const TRNResonanceLayer * layer);
int   trn_resonance_K(const TRNResonanceLayer * layer);
int   trn_resonance_phase_mode(const TRNResonanceLayer * layer);
int   trn_resonance_state_norm(const TRNResonanceLayer * layer);
float trn_resonance_amplitude_max(const TRNResonanceLayer * layer);
float trn_resonance_res_scale(const TRNResonanceLayer * layer);
```

---

## Integration Pattern (Custom Inference Loop)

```c
#include "trn_resonance.h"
#include <stdlib.h>

/* Load all layers */
TRNResonanceLayer * layers[N_LAYERS];
for (int i = 0; i < N_LAYERS; i++) {
    char path[256];
    snprintf(path, sizeof(path), "weights/resonance_layer_%03d.bin", i);
    layers[i] = trn_resonance_load(path);
    if (!layers[i]) { /* handle error */ }
}

int K      = trn_resonance_K(layers[0]);
int d      = trn_resonance_d_model(layers[0]);
int BATCH  = 1;

/* Allocate per-layer state */
float * r_real[N_LAYERS], * r_imag[N_LAYERS];
for (int i = 0; i < N_LAYERS; i++) {
    r_real[i] = calloc(BATCH * K, sizeof(float));
    r_imag[i] = calloc(BATCH * K, sizeof(float));
}

float x[D_MODEL], out[D_MODEL];

/* Autoregressive loop */
for (int pos = 0; pos < n_tokens; pos++) {
    embed(token[pos], x, d);                /* your embedding lookup */

    for (int i = 0; i < N_LAYERS; i++) {
        trn_resonance_step(layers[i], x, r_real[i], r_imag[i], pos, out, BATCH);
        ffn_step(out, x, d);                /* your FFN + residual */
    }

    lm_head(x, logits, vocab_size);         /* your LM head */
    token[pos + 1] = sample(logits);
}

/* Cleanup */
for (int i = 0; i < N_LAYERS; i++) {
    trn_resonance_free(layers[i]);
    free(r_real[i]);
    free(r_imag[i]);
}
```

State memory per layer: `batch × K × 2 × 4` bytes (fp32).
For trn_100m (K=256, L=8, batch=1): `256 × 2 × 4 × 8 = 16,384` bytes total.

---

## Python ctypes Interface

```python
from trn.integrations.llamacpp.ctypes_wrapper import TRNResonance
import numpy as np

# Library search order:
#   1. LIBTRN_RESONANCE_PATH env var
#   2. Same directory as ctypes_wrapper.py
layer = TRNResonance.load("resonance_layer_000.bin")

r_real, r_imag = layer.make_state(batch_size=1)  # zeros, (1, K)

for pos, x_np in enumerate(token_embeddings):    # x_np: (1, d_model) float32
    out, r_real, r_imag = layer.step(x_np, r_real, r_imag, position=pos)
    # out: (1, d_model) float32
```

Set `LIBTRN_RESONANCE_PATH=/path/to/trn_resonance.so` when the library is not
in the same directory as the wrapper module.

---

## Correctness Validation

The bench script validates that C output matches Python `step_single()`:

```bash
# Build first
cd src/trn/integrations/llamacpp
gcc -O2 -shared -fPIC -o trn_resonance.so trn_resonance.c -lm

# Correctness check (toy config, 100 tokens, batch=2)
python scripts/bench_llamacpp_trn.py --correctness-only

# Expected output:
#   [PASS] max_abs_err X.XXe-YY < 1.0e-05
```

Measured error (random weights, 100 tokens):

| Config     | d_model | K   | max_abs_err   | max_state_err |
|------------|---------|-----|---------------|---------------|
| toy        | 128     | 64  | 1.69e-09      | 3.93e-06      |
| trn_100m   | 512     | 256 | 4.31e-09      | 3.14e-06      |

Tolerance threshold: `atol=1e-5` (configurable via `--atol`).

---

## Throughput Benchmark

```bash
python scripts/bench_llamacpp_trn.py \
    --config toy \
    --n-tokens 200 \
    --batch-sizes 1 4 8 16
```

Sample output (toy config, WSL2, Ryzen 9 9950X3D):

```
 batch |     tokens/s |   ms/token |   total_ms
------------------------------------------------
     1 |     38,097.5 |      0.026 |        5.2
     4 |     58,138.2 |      0.069 |       13.8
     8 |     60,713.5 |      0.132 |       26.4
    16 |     61,595.3 |      0.260 |       52.0
```

---

## Arithmetic Specification

The C implementation replicates Python activations exactly:

```
softplus(x) = log1p(exp(x))          -- overflow guard: return x when x > 20
sigmoid(x)  = 1 / (1 + exp(-x))

A     = min(softplus(A_raw), amplitude_max)
omega = sigmoid(Om_raw) * pi + omega_base[k]
phi   = tanh(Ph_raw) * pi
alpha = sigmoid(Ga_raw)

pos   = log1p(position)               -- if phase_mode == 0 (log)
        position                       -- if phase_mode == 1 (linear)
angle = omega * pos + phi

v_r   = (1 - alpha) * A * cos(angle)
v_i   = (1 - alpha) * A * sin(angle)

r_real = alpha * r_real + v_r         -- fp32 state update
r_imag = alpha * r_imag + v_i

if state_norm:
    scale  = max(|r_real|, |r_imag|)
    scale  = max(scale, 1.0)           -- per-channel, clamp(min=1.0)
    r_real /= scale
    r_imag /= scale

rho = r_real * cos(angle) + r_imag * sin(angle)
out = res_scale * (w_res_weight @ rho)
```

Matrix-vector multiply is implemented as a plain nested loop (`sgemv`);
no BLAS dependency.

---

## Troubleshooting

### Shared library not found

```
FileNotFoundError: trn_resonance shared library not found.
```

Set the env var: `export LIBTRN_RESONANCE_PATH=/absolute/path/to/trn_resonance.so`

Or place `trn_resonance.so` next to `ctypes_wrapper.py`.

### Precision mismatch larger than expected

Common causes:
1. `phase_mode` in exported header does not match model config.
2. `state_norm` flag mismatch.
3. Input `x` not cast to `float32` before passing to `trn_resonance_step`.

Re-export with the correct flags:
```python
export_layer(..., phase_mode="log", state_norm=True)
```

### MSVC build errors

Use a VS Developer Command Prompt (not a plain cmd or PowerShell):
```
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
cl /O2 /LD trn_resonance.c /Fe:trn_resonance.dll
```

### File size mismatch on load

The loader checks exact file size against `d_model` and `K` from the header.
A mismatch means the file was truncated or the wrong config was used during export.
