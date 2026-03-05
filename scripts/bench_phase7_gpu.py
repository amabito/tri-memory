#!/usr/bin/env python3
"""Phase 7 GPU benchmark: TRN vs TF+KV vs Hybrid across context lengths.

Compares prefill latency, decode throughput, VRAM usage, and analytical
state memory for multiple model scales and context lengths.

Model variants:
    trn_100m        TRN  512d  8L   256K
    trn_400m        TRN  1024d 16L  512K
    trn_1b          TRN  2048d 24L  512K
    llama3_8b_proxy TRN-cfg proxy for LLaMA-3 8B scale
    hybrid_400m_50  Hybrid 1024d 16L trn_ratio=0.50
    hybrid_400m_25  Hybrid 1024d 16L trn_ratio=0.25

Metrics per (model, context_len):
    prefill_latency_ms  forward pass on full context
    decode_tps          tokens per second for gen_tokens new tokens
    decode_latency_ms   total decode time in ms
    peak_vram_mb        torch.cuda.max_memory_allocated (GPU only)
    state_memory_mb     analytical: TRN state or KV cache formula
    speedup_vs_kv       analytical KV memory / TRN state memory ratio

Usage:
    python scripts/bench_phase7_gpu.py --device cuda
    python scripts/bench_phase7_gpu.py --device cpu --context-lens 512,1024
    python scripts/bench_phase7_gpu.py --models trn_100m,hybrid_400m_50
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch import Tensor

from trn.baseline import TransformerModel
from trn.bench_data import seed_everything
from trn.config import TRNConfig
from trn.hybrid_model import HybridModel
from trn.model import TRNModel

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODELS = "trn_100m,trn_400m,trn_1b"
DEFAULT_CONTEXT_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_GEN_TOKENS = 128
DEFAULT_N_REPEATS = 3
DEFAULT_SEED = 42

_NaN = float("nan")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# n_kv_heads is used only for analytical KV memory formula
_MODEL_REGISTRY: dict[str, tuple] = {
    # name -> (factory_fn, model_type, n_kv_heads)
    "trn_100m": (
        lambda: TRNModel(TRNConfig.trn_100m()),
        "trn",
        8,
    ),
    "trn_400m": (
        lambda: TRNModel(TRNConfig.trn_400m()),
        "trn",
        8,
    ),
    "trn_1b": (
        lambda: TRNModel(TRNConfig.trn_1b()),
        "trn",
        8,
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
        8,
    ),
    "hybrid_400m_50": (
        lambda: HybridModel(TRNConfig.trn_400m(), trn_ratio=0.50),
        "hybrid",
        8,
    ),
    "hybrid_400m_25": (
        lambda: HybridModel(TRNConfig.trn_400m(), trn_ratio=0.25),
        "hybrid",
        8,
    ),
}


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
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(generated)
            next_tok = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
    return generated[:, prompt.size(1):]


# ---------------------------------------------------------------------------
# GPU sync helper
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
    """TRN resonance state: n_layers * K * 2 * 4 bytes (two fp32 per layer)."""
    return cfg.n_layers * cfg.n_oscillators * 2 * 4 / (1024 * 1024)


def _kv_cache_mb(cfg: TRNConfig, n_kv_heads: int, context_len: int) -> float:
    """KV cache: n_layers * n_kv_heads * T * head_dim * 2 * 2 bytes (bf16 K+V)."""
    head_dim = cfg.d_model // max(n_kv_heads, 1)
    return cfg.n_layers * n_kv_heads * context_len * head_dim * 2 * 2 / (1024 * 1024)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def _measure_prefill(
    model: nn.Module,
    prompt: Tensor,
    device: torch.device,
    n_repeats: int,
) -> float:
    """Return average prefill latency in ms."""
    # warmup
    with torch.no_grad():
        try:
            model(prompt)
        except Exception:
            return _NaN
    _sync(device)

    times: list[float] = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(prompt)
        _sync(device)
        times.append(time.perf_counter() - t0)

    return (sum(times) / len(times)) * 1000.0


def _measure_decode(
    model: nn.Module,
    model_type: str,
    prompt: Tensor,
    gen_tokens: int,
    device: torch.device,
    n_repeats: int,
) -> tuple[float, float]:
    """Return (decode_tps, decode_latency_ms)."""
    def _gen():
        if model_type == "trn":
            return model.generate(prompt, max_new_tokens=gen_tokens)
        else:
            return _hybrid_generate(model, prompt, gen_tokens)

    # warmup
    try:
        with torch.no_grad():
            _gen()
    except (RuntimeError, MemoryError) as e:
        print(f"    [WARMUP OOM] {e}")
        return _NaN, _NaN
    _sync(device)

    times: list[float] = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _gen()
        _sync(device)
        times.append(time.perf_counter() - t0)

    avg_t = sum(times) / len(times)
    tps = gen_tokens / avg_t if avg_t > 0 else _NaN
    lat_ms = avg_t * 1000.0
    return tps, lat_ms


def _measure_vram(
    model: nn.Module,
    model_type: str,
    prompt: Tensor,
    gen_tokens: int,
    device: torch.device,
) -> float:
    """Return peak VRAM in MB for a single decode run."""
    if device.type != "cuda":
        return 0.0

    _reset_vram(device)
    try:
        with torch.no_grad():
            if model_type == "trn":
                model.generate(prompt, max_new_tokens=gen_tokens)
            else:
                _hybrid_generate(model, prompt, gen_tokens)
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
    seed: int,
) -> list[dict]:
    factory, model_type, n_kv_heads = _MODEL_REGISTRY[model_name]

    print(f"\n  [{model_name}] loading...")
    try:
        model: nn.Module = factory().to(device).eval()
    except Exception as e:
        print(f"  [LOAD ERROR] {model_name}: {e}")
        return []

    cfg = model.cfg  # TRNConfig on all model types

    rows: list[dict] = []

    for ctx_len in context_lens:
        # Skip context lengths that exceed model's max_seq_len for prefill
        if ctx_len > cfg.max_seq_len:
            print(
                f"    ctx={ctx_len:>7}  SKIP (max_seq_len={cfg.max_seq_len})"
            )
            state_mb = _trn_state_mb(cfg)
            kv_mb = _kv_cache_mb(cfg, n_kv_heads, ctx_len)
            rows.append(dict(
                model=model_name,
                context_len=ctx_len,
                prefill_latency_ms=_NaN,
                decode_tps=_NaN,
                decode_latency_ms=_NaN,
                peak_vram_mb=_NaN,
                state_memory_mb=state_mb,
                speedup_vs_kv=kv_mb / state_mb if state_mb > 0 else _NaN,
            ))
            continue

        seed_everything(seed)
        prompt = torch.randint(
            4, cfg.vocab_size, (1, ctx_len), device=device
        )

        # Prefill
        try:
            prefill_ms = _measure_prefill(model, prompt, device, n_repeats)
        except (RuntimeError, MemoryError) as e:
            print(f"    ctx={ctx_len:>7}  [PREFILL OOM] {e}")
            prefill_ms = _NaN

        # Decode
        try:
            decode_tps, decode_ms = _measure_decode(
                model, model_type, prompt, gen_tokens, device, n_repeats
            )
        except (RuntimeError, MemoryError) as e:
            print(f"    ctx={ctx_len:>7}  [DECODE OOM] {e}")
            decode_tps, decode_ms = _NaN, _NaN

        # VRAM
        vram_mb = _measure_vram(model, model_type, prompt, gen_tokens, device)

        # Analytical memory
        state_mb = _trn_state_mb(cfg)
        kv_mb = _kv_cache_mb(cfg, n_kv_heads, ctx_len)
        speedup = kv_mb / state_mb if state_mb > 0 else _NaN

        def fmt(v: float) -> str:
            return "N/A" if math.isnan(v) else f"{v:.2f}"

        print(
            f"    ctx={ctx_len:>7}  "
            f"prefill={fmt(prefill_ms):>8}ms  "
            f"decode={fmt(decode_tps):>8}tps  "
            f"vram={fmt(vram_mb):>7}MB  "
            f"state={fmt(state_mb):>6}MB  "
            f"kv_ratio={fmt(speedup):>6}x"
        )

        rows.append(dict(
            model=model_name,
            context_len=ctx_len,
            prefill_latency_ms=prefill_ms,
            decode_tps=decode_tps,
            decode_latency_ms=decode_ms,
            peak_vram_mb=vram_mb,
            state_memory_mb=state_mb,
            speedup_vs_kv=speedup,
        ))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    model_names: list[str],
    context_lens: list[int],
    gen_tokens: int,
    device_str: str,
    n_repeats: int,
    seed: int,
    output_csv: Optional[Path],
) -> None:
    device = torch.device(device_str)
    seed_everything(seed)

    print("=" * 80)
    print("Phase 7 GPU Benchmark: TRN / Hybrid memory vs context length")
    print("=" * 80)
    print(f"  device={device_str}  gen_tokens={gen_tokens}  n_repeats={n_repeats}")
    print(f"  models: {', '.join(model_names)}")
    print(f"  context_lens: {context_lens}")
    print()

    all_rows: list[dict] = []

    for model_name in model_names:
        if model_name not in _MODEL_REGISTRY:
            print(f"  [WARN] Unknown model '{model_name}', skipping.")
            continue
        rows = benchmark_model(
            model_name=model_name,
            device=device,
            context_lens=context_lens,
            gen_tokens=gen_tokens,
            n_repeats=n_repeats,
            seed=seed,
        )
        all_rows.extend(rows)

    # CSV output
    if output_csv is not None and all_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "model", "context_len",
            "prefill_latency_ms", "decode_tps", "decode_latency_ms",
            "peak_vram_mb", "state_memory_mb", "speedup_vs_kv",
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  CSV saved -> {output_csv}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7 GPU benchmark: TRN vs Hybrid across context lengths",
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
        help="Timed repetitions per data point (excluding warmup)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV output",
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

    run_benchmark(
        model_names=model_names,
        context_lens=context_lens,
        gen_tokens=args.gen_tokens,
        device_str=args.device,
        n_repeats=args.n_repeats,
        seed=args.seed,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
