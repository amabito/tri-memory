#!/usr/bin/env python3
"""KV-cache benchmark: TRN vs Transformer-with-KV vs Transformer-without-KV.

Three generation modes compared:
  TRN   - model.generate() using O(1) resonance state per step
  TF+KV - Transformer with explicit KV cache (O(n) memory, O(1) compute per step)
  TF    - Transformer without KV cache (O(n) compute per step, no cache)

Metrics per (mode, context_len):
  tps        : tokens per second
  memory_mb  : peak tracemalloc RSS for a single batch_size=1 run
  latency_ms : average wall-clock time per generation call (ms)
  speedup_vs_kv : tps / TF+KV tps (relative to the KV-cache baseline)

Output:
  - Table printed to stdout
  - results/bench_kv_vs_trn.csv (unless --no-csv)

Usage:
    python scripts/bench_kv_vs_trn.py
    python scripts/bench_kv_vs_trn.py \\
        --context-lens 512,1024,2048 --gen-tokens 64 --no-csv --device cpu
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from torch import Tensor

from trn.baseline import TransformerModel
from trn.bench_data import seed_everything
from trn.config import TRNConfig
from trn.model import TRNModel

# ---------------------------------------------------------------------------
# Default benchmark configuration
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_LENS = [512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_GEN_TOKENS = 128
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEED = 42
DEFAULT_DEVICE = "cpu"
DEFAULT_N_REPEATS = 2

# Fixed model configuration per spec
BENCH_D_MODEL = 256
BENCH_N_LAYERS = 8
BENCH_D_FF = 1024
BENCH_N_OSC = 128
BENCH_VOCAB = 256
BENCH_MAX_SEQ_LEN = 16384 + 256  # covers all context lengths + gen buffer


# ---------------------------------------------------------------------------
# KV cache data structures
# ---------------------------------------------------------------------------

@dataclass
class _LayerKVCache:
    k_cache: Tensor  # (B, n_heads, T, head_dim)
    v_cache: Tensor  # (B, n_heads, T, head_dim)


# ---------------------------------------------------------------------------
# KV cache implementation (standalone, does NOT modify baseline.py)
# ---------------------------------------------------------------------------

def _build_kv_cache(
    model: TransformerModel,
    prompt: Tensor,
) -> list[_LayerKVCache]:
    """Prefill: run model(prompt) and capture K,V from each attention layer.

    Uses forward hooks on each CausalSelfAttention to intercept the normed
    hidden state and compute k, v tensors that are stored in the cache.
    """
    model.eval()
    caches: list[Optional[_LayerKVCache]] = [None] * len(model.blocks)

    hooks = []
    for layer_idx, block in enumerate(model.blocks):
        def make_hook(idx: int):
            def hook(module, args, output):
                # args[0] is the normed input x passed to CausalSelfAttention.forward
                x = args[0]  # (B, T, C)
                B, T, C = x.shape
                n_heads = module.n_heads
                head_dim = module.head_dim
                # Recompute qkv from the same input the module will use
                qkv = module.qkv(x)  # (B, T, 3*C)
                _, k, v = qkv.split(C, dim=-1)
                k = k.view(B, T, n_heads, head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
                v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
                caches[idx] = _LayerKVCache(k_cache=k, v_cache=v)
            return hook
        hooks.append(block.attn.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(prompt)

    for h in hooks:
        h.remove()

    return [c for c in caches]  # type: ignore[return-value]


def _kv_decode_step(
    model: TransformerModel,
    token_id: Tensor,       # (B,)
    caches: list[_LayerKVCache],
    position: int,
) -> tuple[Tensor, list[_LayerKVCache]]:
    """Single-token decode step using the KV cache.

    Args:
        model:    TransformerModel (eval mode, no_grad context expected by caller)
        token_id: (B,) long tensor of the current token
        caches:   per-layer KV caches accumulated so far
        position: absolute position index of this token

    Returns:
        logits:     (B, vocab_size) for the decoded position
        new_caches: updated per-layer KV caches
    """
    B = token_id.shape[0]
    n_heads = model.blocks[0].attn.n_heads
    head_dim = model.blocks[0].attn.head_dim
    C = n_heads * head_dim  # d_model

    # Embed + absolute positional encoding
    x = model.embed(token_id.unsqueeze(1))          # (B, 1, d_model)
    x = x + model.pe[position : position + 1]       # broadcast over B

    new_caches: list[_LayerKVCache] = []

    for block, cache in zip(model.blocks, caches):
        # Pre-norm
        h = block.norm1(x)                           # (B, 1, C)

        # Compute qkv for the single new token
        qkv = block.attn.qkv(h)                     # (B, 1, 3*C)
        q, k_new, v_new = qkv.split(C, dim=-1)

        q = q.view(B, 1, n_heads, head_dim).transpose(1, 2)      # (B, n_heads, 1, head_dim)
        k_new = k_new.view(B, 1, n_heads, head_dim).transpose(1, 2)
        v_new = v_new.view(B, 1, n_heads, head_dim).transpose(1, 2)

        # Append to cache
        k_full = torch.cat([cache.k_cache, k_new], dim=2)         # (B, n_heads, T+1, head_dim)
        v_full = torch.cat([cache.v_cache, v_new], dim=2)
        new_caches.append(_LayerKVCache(k_cache=k_full, v_cache=v_full))

        # Attention: query is single position, no causal mask needed (past only)
        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=False
        )  # (B, n_heads, 1, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, C)
        attn_out = block.attn.proj(attn_out)

        # Residual + FFN
        x = x + attn_out
        h2 = block.norm2(x)
        x = x + block.w3(F.silu(block.w1(h2)) * block.w2(h2))

    x = model.norm(x)
    logits = model.lm_head(x[:, 0])  # (B, vocab_size)
    return logits, new_caches


def _tf_kv_generate(
    model: TransformerModel,
    prompt: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
) -> Tensor:
    """Autoregressive generation with KV cache.

    Returns:
        generated tokens (B, max_new_tokens)
    """
    model.eval()
    prompt_len = prompt.size(1)

    with torch.no_grad():
        # Prefill: build KV cache from prompt
        caches = _build_kv_cache(model, prompt)

        # First generated token: take last position logits from full forward
        out = model(prompt)
        first_logits = out["logits"][:, -1, :]  # (B, vocab_size)
        next_tok = torch.argmax(first_logits, dim=-1)  # (B,)

        generated = [next_tok]

        position = prompt_len  # absolute position of the token just generated

        # Decode remaining tokens
        for _ in range(max_new_tokens - 1):
            logits, caches = _kv_decode_step(model, next_tok, caches, position)
            position += 1

            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float("-inf")

            next_tok = torch.argmax(logits, dim=-1)
            generated.append(next_tok)

    return torch.stack(generated, dim=1)  # (B, max_new_tokens)


# ---------------------------------------------------------------------------
# Transformer naive generation (no KV cache) — copied from bench_long_context_scaling
# ---------------------------------------------------------------------------

def _tf_generate(
    model: TransformerModel,
    cfg: TRNConfig,
    prompt: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """Full forward pass per step — O(n) per step, no caching."""
    model.eval()
    generated = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = generated[:, -cfg.max_seq_len :]
            out = model(ctx)
            logits = out["logits"][:, -1, :]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
    return generated[:, prompt.size(1) :]


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _time_generation(
    fn,
    n_repeats: int,
) -> float:
    """Run fn n_repeats times, return average elapsed seconds."""
    times: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def _measure_memory(fn) -> float:
    """Return peak tracemalloc RSS in MB for a single call to fn."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Core benchmark per (mode, context_len)
# ---------------------------------------------------------------------------

_NaN = float("nan")


def measure_trn(
    trn: TRNModel,
    cfg: TRNConfig,
    context_len: int,
    gen_tokens: int,
    batch_size: int,
    n_repeats: int,
    device: torch.device,
) -> tuple[float, float, float]:
    prompt = torch.randint(4, cfg.vocab_size, (batch_size, context_len), device=device)
    try:
        avg_t = _time_generation(
            lambda: trn.generate(prompt, max_new_tokens=gen_tokens),
            n_repeats,
        )
    except (MemoryError, RuntimeError) as e:
        print(f"  [OOM/ERROR] TRN ctx={context_len}: {e}")
        return _NaN, _NaN, _NaN

    tps = gen_tokens * batch_size / avg_t if avg_t > 0 else float("inf")
    lat = avg_t * 1000.0

    mem_prompt = torch.randint(4, cfg.vocab_size, (1, context_len), device=device)
    try:
        mem = _measure_memory(
            lambda: trn.generate(mem_prompt, max_new_tokens=gen_tokens)
        )
    except (MemoryError, RuntimeError):
        mem = _NaN

    return tps, mem, lat


def measure_tf_kv(
    tf: TransformerModel,
    cfg: TRNConfig,
    context_len: int,
    gen_tokens: int,
    batch_size: int,
    n_repeats: int,
    device: torch.device,
) -> tuple[float, float, float]:
    prompt = torch.randint(4, cfg.vocab_size, (batch_size, context_len), device=device)
    try:
        avg_t = _time_generation(
            lambda: _tf_kv_generate(tf, prompt, max_new_tokens=gen_tokens),
            n_repeats,
        )
    except (MemoryError, RuntimeError) as e:
        print(f"  [OOM/ERROR] TF+KV ctx={context_len}: {e}")
        return _NaN, _NaN, _NaN

    tps = gen_tokens * batch_size / avg_t if avg_t > 0 else float("inf")
    lat = avg_t * 1000.0

    mem_prompt = torch.randint(4, cfg.vocab_size, (1, context_len), device=device)
    try:
        mem = _measure_memory(
            lambda: _tf_kv_generate(tf, mem_prompt, max_new_tokens=gen_tokens)
        )
    except (MemoryError, RuntimeError):
        mem = _NaN

    return tps, mem, lat


def measure_tf_no_kv(
    tf: TransformerModel,
    cfg: TRNConfig,
    context_len: int,
    gen_tokens: int,
    batch_size: int,
    n_repeats: int,
    device: torch.device,
) -> tuple[float, float, float]:
    prompt = torch.randint(4, cfg.vocab_size, (batch_size, context_len), device=device)
    try:
        avg_t = _time_generation(
            lambda: _tf_generate(tf, cfg, prompt, max_new_tokens=gen_tokens),
            n_repeats,
        )
    except (MemoryError, RuntimeError) as e:
        print(f"  [OOM/ERROR] TF ctx={context_len}: {e}")
        return _NaN, _NaN, _NaN

    tps = gen_tokens * batch_size / avg_t if avg_t > 0 else float("inf")
    lat = avg_t * 1000.0

    mem_prompt = torch.randint(4, cfg.vocab_size, (1, context_len), device=device)
    try:
        mem = _measure_memory(
            lambda: _tf_generate(tf, cfg, mem_prompt, max_new_tokens=gen_tokens)
        )
    except (MemoryError, RuntimeError):
        mem = _NaN

    return tps, mem, lat


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    context_lens: list[int],
    gen_tokens: int,
    batch_size: int,
    seed: int,
    device_str: str,
    n_repeats: int = DEFAULT_N_REPEATS,
    output_csv: Optional[Path] = None,
) -> None:
    seed_everything(seed)
    device = torch.device(device_str)

    cfg = TRNConfig(
        vocab_size=BENCH_VOCAB,
        d_model=BENCH_D_MODEL,
        n_layers=BENCH_N_LAYERS,
        d_ff=BENCH_D_FF,
        n_oscillators=BENCH_N_OSC,
        max_seq_len=BENCH_MAX_SEQ_LEN,
    )

    trn = TRNModel(cfg).to(device).eval()
    tf = TransformerModel(cfg).to(device).eval()

    print("=" * 84)
    print("KV-Cache Benchmark: TRN vs TF+KV vs TF (no cache)")
    print("=" * 84)
    print(
        f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
        f"d_ff={cfg.d_ff}, n_osc={cfg.n_oscillators}"
    )
    print(
        f"  max_seq_len={cfg.max_seq_len}, gen_tokens={gen_tokens}, "
        f"batch_size={batch_size}, seed={seed}, device={device_str}"
    )
    print(f"  TRN params (non-emb): {trn.num_parameters(non_embedding=True):,}")
    print(f"  TF  params (non-emb): {tf.num_parameters(non_embedding=True):,}")
    print()

    rows: list[dict] = []

    col_w = 10
    header = (
        f"{'ctx_len':>8}  {'mode':>8}  "
        f"{'tps':>{col_w}}  {'mem_mb':>{col_w}}  {'lat_ms':>{col_w}}  "
        f"{'speedup_vs_kv':>14}"
    )
    print(header)
    print("-" * len(header))

    for ctx_len in context_lens:
        trn_tps, trn_mem, trn_lat = measure_trn(
            trn, cfg, ctx_len, gen_tokens, batch_size, n_repeats, device
        )
        kv_tps, kv_mem, kv_lat = measure_tf_kv(
            tf, cfg, ctx_len, gen_tokens, batch_size, n_repeats, device
        )
        tf_tps, tf_mem, tf_lat = measure_tf_no_kv(
            tf, cfg, ctx_len, gen_tokens, batch_size, n_repeats, device
        )

        def speedup(tps: float) -> str:
            if math.isnan(tps) or math.isnan(kv_tps) or kv_tps == 0.0:
                return "N/A"
            return f"{tps / kv_tps:.2f}x"

        def fmt(v: float) -> str:
            return "N/A" if math.isnan(v) else f"{v:.2f}"

        print(
            f"{ctx_len:>8}  {'TRN':>8}  "
            f"{fmt(trn_tps):>{col_w}}  {fmt(trn_mem):>{col_w}}  {fmt(trn_lat):>{col_w}}  "
            f"{speedup(trn_tps):>14}"
        )
        print(
            f"{'':>8}  {'TF+KV':>8}  "
            f"{fmt(kv_tps):>{col_w}}  {fmt(kv_mem):>{col_w}}  {fmt(kv_lat):>{col_w}}  "
            f"{'1.00x':>14}"
        )
        print(
            f"{'':>8}  {'TF':>8}  "
            f"{fmt(tf_tps):>{col_w}}  {fmt(tf_mem):>{col_w}}  {fmt(tf_lat):>{col_w}}  "
            f"{speedup(tf_tps):>14}"
        )

        for mode, tps, mem, lat in [
            ("TRN", trn_tps, trn_mem, trn_lat),
            ("TF+KV", kv_tps, kv_mem, kv_lat),
            ("TF", tf_tps, tf_mem, tf_lat),
        ]:
            rows.append(
                dict(
                    mode=mode,
                    context_len=ctx_len,
                    tps=tps,
                    memory_mb=mem,
                    latency_ms=lat,
                )
            )

    print()

    # ---- summary ----
    valid_rows = {
        mode: [r for r in rows if r["mode"] == mode and not math.isnan(r["tps"])]
        for mode in ("TRN", "TF+KV", "TF")
    }

    print("Summary")
    print("-" * 50)

    trn_rows = valid_rows["TRN"]
    kv_rows = valid_rows["TF+KV"]
    tf_rows = valid_rows["TF"]

    if trn_rows and kv_rows:
        first_ctx = min(r["context_len"] for r in trn_rows)
        last_ctx = max(r["context_len"] for r in trn_rows)
        trn_first = next((r for r in trn_rows if r["context_len"] == first_ctx), None)
        trn_last = next((r for r in trn_rows if r["context_len"] == last_ctx), None)
        kv_first = next((r for r in kv_rows if r["context_len"] == first_ctx), None)
        kv_last = next((r for r in kv_rows if r["context_len"] == last_ctx), None)
        if all(x is not None for x in [trn_first, trn_last, kv_first, kv_last]):
            sp_first = trn_first["tps"] / kv_first["tps"] if kv_first["tps"] else float("inf")
            sp_last = trn_last["tps"] / kv_last["tps"] if kv_last["tps"] else float("inf")
            print(
                f"  TRN vs TF+KV speedup: {sp_first:.2f}x @ ctx={first_ctx} "
                f"-> {sp_last:.2f}x @ ctx={last_ctx}"
            )

    if kv_rows and tf_rows:
        first_ctx = min(r["context_len"] for r in kv_rows)
        last_ctx = max(r["context_len"] for r in kv_rows)
        kv_first = next((r for r in kv_rows if r["context_len"] == first_ctx), None)
        kv_last = next((r for r in kv_rows if r["context_len"] == last_ctx), None)
        tf_first = next((r for r in tf_rows if r["context_len"] == first_ctx), None)
        tf_last = next((r for r in tf_rows if r["context_len"] == last_ctx), None)
        if all(x is not None for x in [kv_first, kv_last, tf_first, tf_last]):
            sp_first = kv_first["tps"] / tf_first["tps"] if tf_first["tps"] else float("inf")
            sp_last = kv_last["tps"] / tf_last["tps"] if tf_last["tps"] else float("inf")
            print(
                f"  TF+KV vs TF speedup:  {sp_first:.2f}x @ ctx={first_ctx} "
                f"-> {sp_last:.2f}x @ ctx={last_ctx}"
            )

    if trn_rows:
        mem_first = trn_rows[0]["memory_mb"]
        mem_last = trn_rows[-1]["memory_mb"]
        ctx_first = trn_rows[0]["context_len"]
        ctx_last = trn_rows[-1]["context_len"]
        if not (math.isnan(mem_first) or math.isnan(mem_last)):
            growth = mem_last / mem_first if mem_first > 0 else float("inf")
            ctx_ratio = ctx_last / ctx_first
            print(
                f"  TRN memory: {mem_first:.3f}MB @ ctx={ctx_first} "
                f"-> {mem_last:.3f}MB @ ctx={ctx_last} "
                f"({growth:.2f}x growth vs {ctx_ratio:.0f}x context)"
            )

    # ---- CSV output ----
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["mode", "context_len", "tps", "memory_mb", "latency_ms"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  CSV saved -> {output_csv}")

    print()
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV-cache benchmark: TRN vs TF+KV vs TF (no cache)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--context-lens",
        type=str,
        default=",".join(str(c) for c in DEFAULT_CONTEXT_LENS),
        help="Comma-separated list of context lengths",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=DEFAULT_GEN_TOKENS,
        help="Number of tokens to generate per call",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for TPS measurement",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="torch device string (cpu, cuda, cuda:0, ...)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
        help="Number of timed repetitions per data point",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_lens = sorted(
        set(int(x.strip()) for x in args.context_lens.split(",") if x.strip())
    )

    project_root = Path(__file__).resolve().parent.parent
    output_csv = (
        None
        if args.no_csv
        else project_root / "results" / "bench_kv_vs_trn.csv"
    )

    run_benchmark(
        context_lens=context_lens,
        gen_tokens=args.gen_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        device_str=args.device,
        n_repeats=args.n_repeats,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
