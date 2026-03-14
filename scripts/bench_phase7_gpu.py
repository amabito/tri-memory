#!/usr/bin/env python3
"""Phase 7 GPU benchmark: TRN vs TF+KV vs TF+Full vs Hybrid across context lengths.

Compares prefill latency, decode throughput, VRAM usage, and analytical
state memory for multiple model scales and context lengths.

Model variants:
    trn_100m        TRN  512d  8L   2048 max_seq_len
    trn_400m        TRN  1024d 16L  4096 max_seq_len
    trn_1b          TRN  2048d 24L  4096 max_seq_len
    llama3_8b_proxy TRN-cfg proxy for LLaMA-3 8B scale
    tf_kv           TransformerModel with explicit KV cache (O(n) memory, O(1) decode)
    tf_full         TransformerModel full forward per step (no cache, O(n) compute)
    hybrid_400m_50  Hybrid 1024d 16L trn_ratio=0.50
    hybrid_400m_25  Hybrid 1024d 16L trn_ratio=0.25

Metrics per (model, context_len):
    prefill_latency_ms  forward pass on full context
    decode_tps          tokens per second for gen_tokens new tokens
    decode_latency_ms   total decode time in ms
    peak_vram_mb        torch.cuda.max_memory_allocated (GPU only)
    state_memory_mb     analytical: TRN state size (constant across context)
    kv_cache_mb         analytical: KV cache formula for this context_len
    speedup_vs_kv       TRN decode_tps / TF+KV decode_tps at same context_len
    status              "ok", "OOM", "TIMEOUT", "SKIP"

Artifacts saved to artifacts/phase7/{timestamp}/:
    results.json   - all rows with full metadata
    env.json       - hardware/software environment
    summary.md     - top results table
    nvidia_smi.txt - nvidia-smi output (if available)

Usage:
    python scripts/bench_phase7_gpu.py --device cuda
    python scripts/bench_phase7_gpu.py --device cpu --context-lens 512,1024
    python scripts/bench_phase7_gpu.py --models trn_100m,tf_kv --context-lens 512 --gen-tokens 16 --device cpu --no-csv
    python scripts/bench_phase7_gpu.py --dtype bf16 --warmup-steps 5 --torch-compile
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trimemory.baseline import TransformerModel
from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.hybrid_model import HybridModel
from trimemory.model import TRNModel

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODELS = "trn_100m,trn_400m,trn_1b"
DEFAULT_CONTEXT_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_GEN_TOKENS = 128
DEFAULT_N_REPEATS = 3
DEFAULT_WARMUP_STEPS = 3
DEFAULT_SEED = 42
DEFAULT_DTYPE = "fp32"

_NaN = float("nan")

# Config for tf_kv and tf_full: use trn_100m dimensions so it's comparable
_TF_BENCH_MAX_SEQ_LEN = 4096  # enough for our default context lens

# ---------------------------------------------------------------------------
# KV cache implementation (copied from bench_kv_vs_trn.py, NOT imported)
# ---------------------------------------------------------------------------


@dataclass
class _LayerKVCache:
    k_cache: Tensor  # (B, n_heads, T, head_dim)
    v_cache: Tensor  # (B, n_heads, T, head_dim)


def _build_kv_cache(
    model: TransformerModel,
    prompt: Tensor,
) -> list[_LayerKVCache]:
    """Prefill: run model(prompt) and capture K,V from each attention layer.

    Uses forward hooks on each CausalSelfAttention to intercept the normed
    hidden state and compute k, v tensors stored in the cache.
    """
    model.eval()
    caches: list[Optional[_LayerKVCache]] = [None] * len(model.blocks)

    hooks = []
    for layer_idx, block in enumerate(model.blocks):
        def make_hook(idx: int):
            def hook(module, args, output):
                x = args[0]  # (B, T, C) — normed input to CausalSelfAttention
                B, T, C = x.shape
                n_heads = module.n_heads
                head_dim = module.head_dim
                qkv = module.qkv(x)  # (B, T, 3*C)
                _, k, v = qkv.split(C, dim=-1)
                k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
                v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
                caches[idx] = _LayerKVCache(k_cache=k, v_cache=v)
            return hook
        hooks.append(block.attn.register_forward_hook(make_hook(layer_idx)))

    with torch.inference_mode():
        model(prompt)

    for h in hooks:
        h.remove()

    return list(caches)  # type: ignore[return-value]


def _kv_decode_step(
    model: TransformerModel,
    token_id: Tensor,       # (B,)
    caches: list[_LayerKVCache],
    position: int,
) -> tuple[Tensor, list[_LayerKVCache]]:
    """Single-token decode step using the KV cache.

    Args:
        model:    TransformerModel (eval mode, inference_mode context expected)
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
    C = n_heads * head_dim

    x = model.embed(token_id.unsqueeze(1))          # (B, 1, d_model)
    x = x + model.pe[position : position + 1]       # broadcast over B

    new_caches: list[_LayerKVCache] = []

    for block, cache in zip(model.blocks, caches):
        h = block.norm1(x)                           # (B, 1, C)
        qkv = block.attn.qkv(h)                     # (B, 1, 3*C)
        q, k_new, v_new = qkv.split(C, dim=-1)

        q = q.view(B, 1, n_heads, head_dim).transpose(1, 2)
        k_new = k_new.view(B, 1, n_heads, head_dim).transpose(1, 2)
        v_new = v_new.view(B, 1, n_heads, head_dim).transpose(1, 2)

        k_full = torch.cat([cache.k_cache, k_new], dim=2)
        v_full = torch.cat([cache.v_cache, v_new], dim=2)
        new_caches.append(_LayerKVCache(k_cache=k_full, v_cache=v_full))

        attn_out = F.scaled_dot_product_attention(
            q, k_full, v_full, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, C)
        attn_out = block.attn.proj(attn_out)

        x = x + attn_out
        h2 = block.norm2(x)
        x = x + block.w3(F.silu(block.w1(h2)) * block.w2(h2))

    x = model.norm(x)
    logits = model.lm_head(x[:, 0])  # (B, vocab_size)
    return logits, new_caches


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# name -> (factory_fn, model_type, n_kv_heads)
# n_kv_heads: used for analytical KV cache formula
_MODEL_REGISTRY: dict[str, tuple] = {
    "trn_100m": (
        lambda: TRNModel(TRNConfig.trn_100m()),
        "trn",
        8,  # d_model=512 / head_dim=64 = 8 heads
    ),
    "trn_400m": (
        lambda: TRNModel(TRNConfig.trn_400m()),
        "trn",
        16,
    ),
    "trn_1b": (
        lambda: TRNModel(TRNConfig.trn_1b()),
        "trn",
        32,
    ),
    "llama3_8b_proxy": (
        lambda: TRNModel(
            TRNConfig(
                vocab_size=32_000,
                d_model=4_096,
                n_layers=32,
                n_oscillators=512,
                d_ff=14_336,
                max_seq_len=8_192,
            )
        ),
        "trn",
        32,
    ),
    "tf_kv": (
        lambda: TransformerModel(
            TRNConfig(
                vocab_size=32_000,
                d_model=512,
                n_layers=8,
                d_ff=2_048,
                max_seq_len=_TF_BENCH_MAX_SEQ_LEN,
            )
        ),
        "tf_kv",
        8,
    ),
    "tf_full": (
        lambda: TransformerModel(
            TRNConfig(
                vocab_size=32_000,
                d_model=512,
                n_layers=8,
                d_ff=2_048,
                max_seq_len=_TF_BENCH_MAX_SEQ_LEN,
            )
        ),
        "tf_full",
        8,
    ),
    "hybrid_400m_50": (
        lambda: HybridModel(TRNConfig.trn_400m(), trn_ratio=0.50),
        "hybrid",
        16,
    ),
    "hybrid_400m_25": (
        lambda: HybridModel(TRNConfig.trn_400m(), trn_ratio=0.25),
        "hybrid",
        16,
    ),
}


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from: {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[dtype_str]


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _hybrid_generate(
    model: HybridModel,
    prompt: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """Greedy autoregressive generation for HybridModel (no KV cache)."""
    model.eval()
    generated = prompt.clone()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(generated)
            next_tok = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
    return generated[:, prompt.size(1):]


def _tf_kv_generate(
    model: TransformerModel,
    prompt: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """Autoregressive generation with KV cache (tf_kv mode)."""
    model.eval()
    prompt_len = prompt.size(1)
    with torch.inference_mode():
        caches = _build_kv_cache(model, prompt)
        out = model(prompt)
        next_tok = out["logits"][:, -1, :].argmax(dim=-1)
        generated = [next_tok]
        position = prompt_len
        for _ in range(max_new_tokens - 1):
            logits, caches = _kv_decode_step(model, next_tok, caches, position)
            position += 1
            next_tok = logits.argmax(dim=-1)
            generated.append(next_tok)
    return torch.stack(generated, dim=1)


def _tf_full_generate(
    model: TransformerModel,
    cfg: TRNConfig,
    prompt: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """Full forward pass per step (no cache, O(n) compute per step)."""
    model.eval()
    generated = prompt.clone()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            ctx = generated[:, -cfg.max_seq_len:]
            out = model(ctx)
            next_tok = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
    return generated[:, prompt.size(1):]


# ---------------------------------------------------------------------------
# GPU sync helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_vram_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


# ---------------------------------------------------------------------------
# Analytical memory formulas
# ---------------------------------------------------------------------------

def _trn_state_mb(cfg: TRNConfig) -> float:
    """TRN resonance state: n_layers * K * 2 * 4 bytes (two fp32 per oscillator per layer)."""
    return cfg.n_layers * cfg.n_oscillators * 2 * 4 / (1024 * 1024)


def _kv_cache_mb(cfg: TRNConfig, n_kv_heads: int, context_len: int) -> float:
    """KV cache: n_layers * n_kv_heads * T * head_dim * 2 * 2 bytes (bf16 K+V)."""
    head_dim = cfg.d_model // max(n_kv_heads, 1)
    return cfg.n_layers * n_kv_heads * context_len * head_dim * 2 * 2 / (1024 * 1024)


# ---------------------------------------------------------------------------
# Core measurement functions
# ---------------------------------------------------------------------------

def _measure_prefill(
    model: nn.Module,
    prompt: Tensor,
    device: torch.device,
    n_repeats: int,
    warmup_steps: int,
    dtype: torch.dtype,
) -> tuple[float, str]:
    """Return (prefill_latency_ms, status)."""
    use_autocast = dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = torch.autocast(device.type, dtype=dtype, enabled=use_autocast)

    # Warmup
    for _ in range(warmup_steps):
        try:
            with torch.inference_mode(), autocast_ctx:
                model(prompt)
            _sync(device)
        except (RuntimeError, MemoryError):
            return _NaN, "OOM"

    times: list[float] = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()
        try:
            with torch.inference_mode(), autocast_ctx:
                model(prompt)
        except (RuntimeError, MemoryError):
            return _NaN, "OOM"
        _sync(device)
        times.append(time.perf_counter() - t0)

    return median(times) * 1000.0, "ok"


def _model_generate(
    model: nn.Module,
    model_type: str,
    cfg: TRNConfig,
    prompt: Tensor,
    gen_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Run generation for the given model type (in-place, result discarded)."""
    use_autocast = dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device.type, dtype=dtype, enabled=use_autocast):
        if model_type == "trn":
            model.generate(prompt, max_new_tokens=gen_tokens)
        elif model_type == "tf_kv":
            _tf_kv_generate(model, prompt, gen_tokens)
        elif model_type == "tf_full":
            _tf_full_generate(model, cfg, prompt, gen_tokens)
        else:  # hybrid
            _hybrid_generate(model, prompt, gen_tokens)


def _measure_decode(
    model: nn.Module,
    model_type: str,
    cfg: TRNConfig,
    prompt: Tensor,
    gen_tokens: int,
    device: torch.device,
    n_repeats: int,
    warmup_steps: int,
    dtype: torch.dtype,
) -> tuple[float, float, str]:
    """Return (decode_tps, decode_latency_ms, status)."""
    # Warmup
    for _ in range(warmup_steps):
        try:
            _model_generate(model, model_type, cfg, prompt, gen_tokens, dtype, device)
            _sync(device)
        except (RuntimeError, MemoryError) as e:
            print(f"    [WARMUP OOM] {e}")
            return _NaN, _NaN, "OOM"

    times: list[float] = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()
        try:
            _model_generate(model, model_type, cfg, prompt, gen_tokens, dtype, device)
        except (RuntimeError, MemoryError) as e:
            print(f"    [DECODE OOM] {e}")
            return _NaN, _NaN, "OOM"
        _sync(device)
        times.append(time.perf_counter() - t0)

    med_t = median(times)
    tps = gen_tokens / med_t if med_t > 0 else _NaN
    lat_ms = med_t * 1000.0
    return tps, lat_ms, "ok"


def _measure_vram(
    model: nn.Module,
    model_type: str,
    cfg: TRNConfig,
    prompt: Tensor,
    gen_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Return peak VRAM in MB for a single decode run."""
    if device.type != "cuda":
        return 0.0

    _reset_vram(device)
    try:
        _model_generate(model, model_type, cfg, prompt, gen_tokens, dtype, device)
    except (RuntimeError, MemoryError):
        return _NaN
    _sync(device)
    return _peak_vram_mb(device)


# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    model_name: str,
    device: torch.device,
    context_lens: list[int],
    gen_tokens: int,
    n_repeats: int,
    warmup_steps: int,
    seed: int,
    dtype: torch.dtype,
    torch_compile: bool,
    # shared dict: ctx_len -> tf_kv decode_tps (for speedup_vs_kv calculation)
    tf_kv_tps_by_ctx: dict[int, float],
) -> list[dict]:
    factory, model_type, n_kv_heads = _MODEL_REGISTRY[model_name]

    print(f"\n  [{model_name}] loading...")
    try:
        model: nn.Module = factory().to(device).to(dtype).eval()
    except Exception as e:
        print(f"  [LOAD ERROR] {model_name}: {e}")
        return []

    if torch_compile:
        try:
            model = torch.compile(model)
            print(f"  [{model_name}] torch.compile applied")
        except Exception as e:
            print(f"  [{model_name}] torch.compile failed (skipping): {e}")

    cfg = model.cfg  # TRNConfig on all model types

    rows: list[dict] = []

    for ctx_len in context_lens:
        # Skip context lengths that exceed model's max_seq_len
        if ctx_len > cfg.max_seq_len:
            print(f"    ctx={ctx_len:>7}  SKIP (max_seq_len={cfg.max_seq_len})")
            state_mb = _trn_state_mb(cfg)
            kv_mb = _kv_cache_mb(cfg, n_kv_heads, ctx_len)
            rows.append(_make_row(
                model_name, ctx_len,
                prefill_ms=_NaN, decode_tps=_NaN, decode_ms=_NaN,
                vram_mb=_NaN, state_mb=state_mb, kv_mb=kv_mb,
                speedup=_NaN, status="SKIP",
            ))
            continue

        seed_everything(seed)
        prompt = torch.randint(4, cfg.vocab_size, (1, ctx_len), device=device)

        # Prefill
        try:
            prefill_ms, prefill_status = _measure_prefill(
                model, prompt, device, n_repeats, warmup_steps, dtype
            )
        except Exception as e:
            print(f"    ctx={ctx_len:>7}  [PREFILL ERROR] {e}")
            prefill_ms, prefill_status = _NaN, "OOM"

        # Decode
        try:
            decode_tps, decode_ms, decode_status = _measure_decode(
                model, model_type, cfg, prompt, gen_tokens,
                device, n_repeats, warmup_steps, dtype
            )
        except Exception as e:
            print(f"    ctx={ctx_len:>7}  [DECODE ERROR] {e}")
            decode_tps, decode_ms, decode_status = _NaN, _NaN, "OOM"

        # VRAM
        vram_mb = _measure_vram(model, model_type, cfg, prompt, gen_tokens, device, dtype)

        # Analytical memory
        state_mb = _trn_state_mb(cfg)
        kv_mb = _kv_cache_mb(cfg, n_kv_heads, ctx_len)

        # speedup_vs_kv: compare decode_tps to TF+KV decode_tps at same ctx_len
        if model_type == "tf_kv" and not math.isnan(decode_tps):
            tf_kv_tps_by_ctx[ctx_len] = decode_tps

        kv_ref_tps = tf_kv_tps_by_ctx.get(ctx_len, _NaN)
        if not math.isnan(decode_tps) and not math.isnan(kv_ref_tps) and kv_ref_tps > 0:
            speedup = decode_tps / kv_ref_tps
        else:
            speedup = _NaN

        # Determine overall status
        overall_status = "ok"
        if prefill_status == "OOM" or decode_status == "OOM":
            overall_status = "OOM"
        elif prefill_status == "TIMEOUT" or decode_status == "TIMEOUT":
            overall_status = "TIMEOUT"

        def fmt(v: float) -> str:
            return "N/A" if math.isnan(v) else f"{v:.2f}"

        print(
            f"    ctx={ctx_len:>7}  "
            f"prefill={fmt(prefill_ms):>8}ms  "
            f"decode={fmt(decode_tps):>8}tps  "
            f"vram={fmt(vram_mb):>7}MB  "
            f"state={fmt(state_mb):>6}MB  "
            f"kv={fmt(kv_mb):>7}MB  "
            f"speedup={fmt(speedup):>6}x  "
            f"[{overall_status}]"
        )

        rows.append(_make_row(
            model_name, ctx_len,
            prefill_ms=prefill_ms, decode_tps=decode_tps, decode_ms=decode_ms,
            vram_mb=vram_mb, state_mb=state_mb, kv_mb=kv_mb,
            speedup=speedup, status=overall_status,
        ))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows


def _make_row(
    model: str,
    context_len: int,
    prefill_ms: float,
    decode_tps: float,
    decode_ms: float,
    vram_mb: float,
    state_mb: float,
    kv_mb: float,
    speedup: float,
    status: str,
) -> dict:
    return dict(
        model=model,
        context_len=context_len,
        prefill_latency_ms=prefill_ms,
        decode_tps=decode_tps,
        decode_latency_ms=decode_ms,
        peak_vram_mb=vram_mb,
        state_memory_mb=state_mb,
        kv_cache_mb=kv_mb,
        speedup_vs_kv=speedup,
        status=status,
    )


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

def _collect_env(
    seed: int,
    dtype: str,
    gen_tokens: int,
    n_repeats: int,
    warmup_steps: int,
    device: torch.device,
) -> dict:
    env: dict = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "seed": seed,
        "dtype": dtype,
        "gen_tokens": gen_tokens,
        "n_repeats": n_repeats,
        "warmup_steps": warmup_steps,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(device)
        try:
            env["driver_version"] = torch.cuda.get_device_properties(device).driver_version
        except AttributeError:
            env["driver_version"] = "N/A"

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        env["git_hash"] = git_hash
    except Exception:
        env["git_hash"] = "N/A"

    return env


def _save_nvidia_smi(artifact_dir: Path) -> None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode()
        (artifact_dir / "nvidia_smi.txt").write_text(output, encoding="utf-8")
    except Exception:
        pass


def _save_summary_md(
    artifact_dir: Path,
    all_rows: list[dict],
    env: dict,
) -> None:
    lines = [
        "# Phase 7 GPU Benchmark Summary",
        "",
        f"**Timestamp**: {env.get('timestamp', 'N/A')}",
        f"**GPU**: {env.get('gpu_name', 'CPU')}",
        f"**dtype**: {env.get('dtype', 'N/A')}",
        f"**gen_tokens**: {env.get('gen_tokens', 'N/A')}",
        "",
        "## Results",
        "",
        "| model | ctx | prefill_ms | decode_tps | vram_mb | state_mb | kv_mb | speedup_vs_kv | status |",
        "|-------|-----|-----------|-----------|---------|---------|-------|--------------|--------|",
    ]

    def fmt(v) -> str:
        if isinstance(v, float) and math.isnan(v):
            return "N/A"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    for r in all_rows:
        lines.append(
            f"| {r['model']} | {r['context_len']} "
            f"| {fmt(r['prefill_latency_ms'])} "
            f"| {fmt(r['decode_tps'])} "
            f"| {fmt(r['peak_vram_mb'])} "
            f"| {fmt(r['state_memory_mb'])} "
            f"| {fmt(r['kv_cache_mb'])} "
            f"| {fmt(r['speedup_vs_kv'])} "
            f"| {r['status']} |"
        )

    (artifact_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _rows_to_json_safe(rows: list[dict]) -> list[dict]:
    """Replace NaN/inf with None for JSON serialization."""
    result = []
    for row in rows:
        safe_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                safe_row[k] = None
            else:
                safe_row[k] = v
        result.append(safe_row)
    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    model_names: list[str],
    context_lens: list[int],
    gen_tokens: int,
    device_str: str,
    n_repeats: int,
    warmup_steps: int,
    seed: int,
    dtype_str: str,
    output_csv: Optional[Path],
    artifact_root: Optional[Path],
    torch_compile: bool,
) -> None:
    device = torch.device(device_str)
    dtype = _resolve_dtype(dtype_str)
    seed_everything(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Phase 7 GPU Benchmark: TRN / TF+KV / Hybrid memory vs context length")
    print("=" * 80)
    print(f"  device={device_str}  dtype={dtype_str}  gen_tokens={gen_tokens}")
    print(f"  n_repeats={n_repeats}  warmup_steps={warmup_steps}  seed={seed}")
    print(f"  torch_compile={torch_compile}")
    print(f"  models: {', '.join(model_names)}")
    print(f"  context_lens: {context_lens}")
    print()

    # Collect environment info first
    env = _collect_env(seed, dtype_str, gen_tokens, n_repeats, warmup_steps, device)

    all_rows: list[dict] = []

    # tf_kv_tps_by_ctx is shared across all model runs so TRN speedup_vs_kv
    # can reference TF+KV decode_tps measured at the same context length.
    # If tf_kv is not in the run, speedup stays NaN.
    tf_kv_tps_by_ctx: dict[int, float] = {}

    # If tf_kv is in model_names, run it first so its tps are available
    ordered_names = []
    if "tf_kv" in model_names:
        ordered_names.append("tf_kv")
    for n in model_names:
        if n != "tf_kv":
            ordered_names.append(n)

    for model_name in ordered_names:
        if model_name not in _MODEL_REGISTRY:
            print(f"  [WARN] Unknown model '{model_name}', skipping.")
            continue
        rows = benchmark_model(
            model_name=model_name,
            device=device,
            context_lens=context_lens,
            gen_tokens=gen_tokens,
            n_repeats=n_repeats,
            warmup_steps=warmup_steps,
            seed=seed,
            dtype=dtype,
            torch_compile=torch_compile,
            tf_kv_tps_by_ctx=tf_kv_tps_by_ctx,
        )
        all_rows.extend(rows)

    # CSV output
    if output_csv is not None and all_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "model", "context_len",
            "prefill_latency_ms", "decode_tps", "decode_latency_ms",
            "peak_vram_mb", "state_memory_mb", "kv_cache_mb",
            "speedup_vs_kv", "status",
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  CSV saved -> {output_csv}")

    # Artifacts
    if artifact_root is not None and all_rows:
        artifact_dir = artifact_root / timestamp
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # results.json
        (artifact_dir / "results.json").write_text(
            json.dumps(_rows_to_json_safe(all_rows), indent=2),
            encoding="utf-8",
        )

        # env.json
        (artifact_dir / "env.json").write_text(
            json.dumps(env, indent=2),
            encoding="utf-8",
        )

        # nvidia_smi.txt
        _save_nvidia_smi(artifact_dir)

        # summary.md
        _save_summary_md(artifact_dir, all_rows, env)

        print(f"  Artifacts saved -> {artifact_dir}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7 GPU benchmark: TRN vs TF+KV vs Hybrid across context lengths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODELS,
        help="Comma-separated model names. Choices: "
             + ", ".join(_MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--context-lens",
        type=str,
        default=",".join(str(c) for c in DEFAULT_CONTEXT_LENS),
        help="Comma-separated context lengths",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=DEFAULT_GEN_TOKENS,
        help="Tokens to generate per decode measurement",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device string (cpu, cuda, cuda:0, ...)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
        help="Timed repetitions per data point (excluding warmup), median is used",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Warmup iterations before timing (not counted)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=list(_DTYPE_MAP.keys()),
        help="Model dtype (fp32/fp16/bf16). fp16/bf16 use torch.autocast",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Apply torch.compile to each model before benchmarking",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV output",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip artifact directory creation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    context_lens = sorted(
        set(int(x.strip()) for x in args.context_lens.split(",") if x.strip())
    )

    project_root = Path(__file__).resolve().parent.parent
    output_csv = (
        None
        if args.no_csv
        else project_root / "results" / "bench_phase7_gpu.csv"
    )
    artifact_root = (
        None
        if args.no_artifacts
        else project_root / "artifacts" / "phase7"
    )

    run_benchmark(
        model_names=model_names,
        context_lens=context_lens,
        gen_tokens=args.gen_tokens,
        device_str=args.device,
        n_repeats=args.n_repeats,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        dtype_str=args.dtype,
        output_csv=output_csv,
        artifact_root=artifact_root,
        torch_compile=args.torch_compile,
    )


if __name__ == "__main__":
    main()
