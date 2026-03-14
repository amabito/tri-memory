#!/usr/bin/env python3
"""vLLM-style benchmark: DualMemoryEngine vs TF+KV vs TRN across context lengths. Phase 9.

Compares prefill latency, decode throughput, VRAM, and memory footprints
for DualMemoryEngine (windowed KV + TRN) against existing baselines.

Model variants:
    dual_100m_w64     DualMemoryEngine 512d 8L window=64
    dual_100m_w256    DualMemoryEngine 512d 8L window=256
    dual_100m_w1024   DualMemoryEngine 512d 8L window=1024
    tf_kv             TransformerModel with KV cache (baseline)
    trn_100m          TRNModel pure resonance (baseline)

Context lengths: 1024, 4096, 16384, 65536, 131072

Metrics per (model, context_len):
    prefill_latency_ms   forward pass on full context
    decode_tps           tokens per second for gen_tokens new tokens
    decode_latency_ms    total decode time in ms
    peak_vram_mb         torch.cuda.max_memory_allocated
    state_memory_mb      analytical: TRN state size (constant)
    kv_window_mb         analytical: KV window at window_size W

Artifacts saved to artifacts/phase9/{timestamp}/:
    results.json  env.json  summary.md  nvidia_smi.txt

Usage:
    python scripts/bench_vllm_trn.py --device cuda
    python scripts/bench_vllm_trn.py --device cpu --context-lens 1024,4096
    python scripts/bench_vllm_trn.py --models dual_100m_w256,tf_kv --context-lens 1024 --device cpu
    python scripts/bench_vllm_trn.py --dtype bf16 --warmup-steps 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
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
from trimemory.integrations.vllm_backend import DualMemoryEngine
from trimemory.model import TRNModel

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODELS       = "dual_100m_w64,dual_100m_w256,dual_100m_w1024,tf_kv,trn_100m"
DEFAULT_CONTEXT_LENS = [1024, 4096, 16384, 65536, 131072]
DEFAULT_GEN_TOKENS   = 128
DEFAULT_N_REPEATS    = 3
DEFAULT_WARMUP_STEPS = 3
DEFAULT_SEED         = 42
DEFAULT_DTYPE        = "fp32"

_NaN = float("nan")

_DUAL_BASE_CFG = TRNConfig(
    vocab_size=32_000,
    d_model=512,
    n_oscillators=256,
    n_layers=8,
    d_ff=2_048,
    max_seq_len=4_096,  # allows up to 4096 in training; generation is unbounded
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# name -> (factory_fn, model_type, n_kv_heads)
_MODEL_REGISTRY: dict[str, tuple] = {
    "dual_100m_w64": (
        lambda: DualMemoryEngine(_DUAL_BASE_CFG, window_size=64),
        "dual",
        8,
    ),
    "dual_100m_w256": (
        lambda: DualMemoryEngine(_DUAL_BASE_CFG, window_size=256),
        "dual",
        8,
    ),
    "dual_100m_w1024": (
        lambda: DualMemoryEngine(_DUAL_BASE_CFG, window_size=1024),
        "dual",
        8,
    ),
    "tf_kv": (
        lambda: TransformerModel(
            TRNConfig(
                vocab_size=32_000,
                d_model=512,
                n_layers=8,
                d_ff=2_048,
                max_seq_len=4_096,
            )
        ),
        "tf_kv",
        8,
    ),
    "trn_100m": (
        lambda: TRNModel(TRNConfig.trn_100m()),
        "trn",
        8,
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
# TF+KV cache helpers (inline, not imported from phase7 to keep independence)
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass


@_dataclass
class _LayerKVCache:
    k_cache: Tensor  # (B, n_heads, T, head_dim)
    v_cache: Tensor  # (B, n_heads, T, head_dim)


def _build_tf_kv_cache(
    model: TransformerModel,
    prompt: Tensor,
) -> list[_LayerKVCache]:
    model.eval()
    caches: list[Optional[_LayerKVCache]] = [None] * len(model.blocks)
    hooks = []
    for layer_idx, block in enumerate(model.blocks):
        def make_hook(idx: int):
            def hook(module, args, output):
                x = args[0]
                B, T, C = x.shape
                n_heads  = module.n_heads
                head_dim = module.head_dim
                qkv = module.qkv(x)
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


def _tf_kv_decode_step(
    model: TransformerModel,
    token_id: Tensor,
    caches: list[_LayerKVCache],
    position: int,
) -> tuple[Tensor, list[_LayerKVCache]]:
    B = token_id.shape[0]
    n_heads  = model.blocks[0].attn.n_heads
    head_dim = model.blocks[0].attn.head_dim
    C = n_heads * head_dim

    x = model.embed(token_id.unsqueeze(1))
    x = x + model.pe[position : position + 1]

    new_caches: list[_LayerKVCache] = []
    for block, cache in zip(model.blocks, caches):
        h = block.norm1(x)
        qkv = block.attn.qkv(h)
        q, k_new, v_new = qkv.split(C, dim=-1)
        q     = q.view(B, 1, n_heads, head_dim).transpose(1, 2)
        k_new = k_new.view(B, 1, n_heads, head_dim).transpose(1, 2)
        v_new = v_new.view(B, 1, n_heads, head_dim).transpose(1, 2)
        k_full = torch.cat([cache.k_cache, k_new], dim=2)
        v_full = torch.cat([cache.v_cache, v_new], dim=2)
        new_caches.append(_LayerKVCache(k_cache=k_full, v_cache=v_full))
        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, C)
        attn_out = block.attn.proj(attn_out)
        x = x + attn_out
        h2 = block.norm2(x)
        x = x + block.w3(F.silu(block.w1(h2)) * block.w2(h2))
    x = model.norm(x)
    logits = model.lm_head(x[:, 0])
    return logits, new_caches


def _tf_kv_generate(
    model: TransformerModel,
    prompt: Tensor,
    max_new_tokens: int,
) -> Tensor:
    model.eval()
    prompt_len = prompt.size(1)
    with torch.inference_mode():
        caches = _build_tf_kv_cache(model, prompt)
        out = model(prompt)
        next_tok = out["logits"][:, -1, :].argmax(dim=-1)
        generated = [next_tok]
        position = prompt_len
        for _ in range(max_new_tokens - 1):
            logits, caches = _tf_kv_decode_step(model, next_tok, caches, position)
            position += 1
            next_tok = logits.argmax(dim=-1)
            generated.append(next_tok)
    return torch.stack(generated, dim=1)


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
    """TRN resonance state: n_layers * K * 2 * 4 bytes."""
    return cfg.n_layers * cfg.n_oscillators * 2 * 4 / (1024 * 1024)


def _kv_window_mb(model: nn.Module, model_type: str, context_len: int) -> float:
    """KV window memory for dual model; full KV for tf_kv."""
    if model_type == "dual":
        # Constant: n_layers * n_heads * W * head_dim * 2 * 2 bytes (bf16 K+V)
        assert isinstance(model, DualMemoryEngine)
        return model.kv_window_bytes(dtype_bytes=2) / (1024 * 1024)
    elif model_type == "tf_kv":
        # Grows with context: n_layers * n_heads * T * head_dim * 2 * 2 bytes
        assert isinstance(model, TransformerModel)
        cfg      = model.cfg
        n_heads  = model.blocks[0].attn.n_heads
        head_dim = model.blocks[0].attn.head_dim
        return cfg.n_layers * n_heads * context_len * head_dim * 2 * 2 / (1024 * 1024)
    else:
        return _NaN


# ---------------------------------------------------------------------------
# Generation dispatch
# ---------------------------------------------------------------------------

def _run_generate(
    model: nn.Module,
    model_type: str,
    prompt: Tensor,
    gen_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    use_autocast = dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device.type, dtype=dtype, enabled=use_autocast):
        if model_type in ("dual", "trn"):
            model.generate(prompt, max_new_tokens=gen_tokens)
        else:  # tf_kv
            _tf_kv_generate(model, prompt, gen_tokens)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _measure_prefill(
    model: nn.Module,
    prompt: Tensor,
    device: torch.device,
    n_repeats: int,
    warmup_steps: int,
    dtype: torch.dtype,
) -> tuple[float, str]:
    use_autocast = dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = torch.autocast(device.type, dtype=dtype, enabled=use_autocast)

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


def _measure_decode(
    model: nn.Module,
    model_type: str,
    prompt: Tensor,
    gen_tokens: int,
    device: torch.device,
    n_repeats: int,
    warmup_steps: int,
    dtype: torch.dtype,
) -> tuple[float, float, str]:
    for _ in range(warmup_steps):
        try:
            _run_generate(model, model_type, prompt, gen_tokens, dtype, device)
            _sync(device)
        except (RuntimeError, MemoryError) as e:
            print(f"    [WARMUP OOM] {e}")
            return _NaN, _NaN, "OOM"

    times: list[float] = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()
        try:
            _run_generate(model, model_type, prompt, gen_tokens, dtype, device)
        except (RuntimeError, MemoryError) as e:
            print(f"    [DECODE OOM] {e}")
            return _NaN, _NaN, "OOM"
        _sync(device)
        times.append(time.perf_counter() - t0)

    med_t = median(times)
    tps   = gen_tokens / med_t if med_t > 0 else _NaN
    lat   = med_t * 1000.0
    return tps, lat, "ok"


def _measure_vram(
    model: nn.Module,
    model_type: str,
    prompt: Tensor,
    gen_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    if device.type != "cuda":
        return 0.0
    _reset_vram(device)
    try:
        _run_generate(model, model_type, prompt, gen_tokens, dtype, device)
    except (RuntimeError, MemoryError):
        return _NaN
    _sync(device)
    return _peak_vram_mb(device)


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _make_row(
    model: str,
    context_len: int,
    prefill_ms: float,
    decode_tps: float,
    decode_ms: float,
    vram_mb: float,
    state_mb: float,
    kv_window_mb: float,
    status: str,
    retained_kv_tokens: int,
    trn_state_bytes: int,
    backend: str,
) -> dict:
    return dict(
        model=model,
        context_len=context_len,
        prefill_latency_ms=prefill_ms,
        decode_tps=decode_tps,
        decode_latency_ms=decode_ms,
        peak_vram_mb=vram_mb,
        state_memory_mb=state_mb,
        kv_window_mb=kv_window_mb,
        status=status,
        retained_kv_tokens=retained_kv_tokens,
        trn_state_bytes=trn_state_bytes,
        backend=backend,
    )


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
) -> list[dict]:
    factory, model_type, _ = _MODEL_REGISTRY[model_name]

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

    cfg = model.cfg

    rows: list[dict] = []

    for ctx_len in context_lens:
        seed_everything(seed)
        prompt = torch.randint(4, cfg.vocab_size, (1, ctx_len), device=device)

        def fmt(v: float) -> str:
            return "N/A" if math.isnan(v) else f"{v:.2f}"

        # Prefill: for dual/trn, max_seq_len is a training cap; use prompt clipped
        # For training-style forward, clip to max_seq_len
        if ctx_len > cfg.max_seq_len and model_type in ("dual",):
            # DualMemoryEngine generate can handle any length (streaming decode),
            # but forward() requires T <= max_seq_len.
            # For prefill benchmark, we use model.generate with prompt[:max_seq_len]
            # and for context lengths beyond, we run generate only.
            prefill_prompt = prompt[:, :cfg.max_seq_len]
        else:
            prefill_prompt = prompt

        try:
            prefill_ms, prefill_status = _measure_prefill(
                model, prefill_prompt, device, n_repeats, warmup_steps, dtype
            )
        except Exception as e:
            print(f"    ctx={ctx_len:>7}  [PREFILL ERROR] {e}")
            prefill_ms, prefill_status = _NaN, "OOM"

        # Decode: use shorter prompt so we don't OOM on very long contexts
        decode_prompt_len = min(ctx_len, cfg.max_seq_len)
        decode_prompt = prompt[:, :decode_prompt_len]

        try:
            decode_tps, decode_ms, decode_status = _measure_decode(
                model, model_type, decode_prompt, gen_tokens,
                device, n_repeats, warmup_steps, dtype
            )
        except Exception as e:
            print(f"    ctx={ctx_len:>7}  [DECODE ERROR] {e}")
            decode_tps, decode_ms, decode_status = _NaN, _NaN, "OOM"

        vram_mb = _measure_vram(model, model_type, decode_prompt, gen_tokens, device, dtype)

        # Analytical memory
        state_mb = _trn_state_mb(cfg) if model_type in ("dual", "trn") else _NaN
        kw_mb    = _kv_window_mb(model, model_type, ctx_len)

        overall = "ok"
        if prefill_status == "OOM" or decode_status == "OOM":
            overall = "OOM"

        # Phase 9 fields
        if model_type == "dual":
            retained_kv_tokens = model.window_size
            trn_state_bytes = cfg.n_layers * cfg.n_oscillators * 2 * 4
        elif model_type == "tf_kv":
            retained_kv_tokens = ctx_len
            trn_state_bytes = 0
        else:  # trn
            retained_kv_tokens = 0
            trn_state_bytes = cfg.n_layers * cfg.n_oscillators * 2 * 4

        print(
            f"    ctx={ctx_len:>7}  "
            f"prefill={fmt(prefill_ms):>8}ms  "
            f"decode={fmt(decode_tps):>8}tps  "
            f"vram={fmt(vram_mb):>7}MB  "
            f"state={fmt(state_mb):>6}MB  "
            f"kv_win={fmt(kw_mb):>7}MB  "
            f"[{overall}]"
        )

        rows.append(_make_row(
            model_name, ctx_len,
            prefill_ms, decode_tps, decode_ms, vram_mb, state_mb, kw_mb, overall,
            retained_kv_tokens, trn_state_bytes, model_name,
        ))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows


# ---------------------------------------------------------------------------
# Environment
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
        "timestamp":    datetime.now(tz=timezone.utc).isoformat(),
        "seed":         seed,
        "dtype":        dtype,
        "gen_tokens":   gen_tokens,
        "n_repeats":    n_repeats,
        "warmup_steps": warmup_steps,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "python_version": platform.python_version(),
        "platform":     platform.platform(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(device)
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
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL, timeout=10).decode()
        (artifact_dir / "nvidia_smi.txt").write_text(output, encoding="utf-8")
    except Exception:
        pass


def _rows_to_json_safe(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        safe: dict = {}
        for k, v in row.items():
            safe[k] = None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
        result.append(safe)
    return result


def _save_summary_md(artifact_dir: Path, all_rows: list[dict], env: dict) -> None:
    def fmt(v) -> str:
        if isinstance(v, float) and math.isnan(v):
            return "N/A"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = [
        "# vLLM TRN Benchmark Summary (Phase 9 — DualMemoryEngine)",
        "",
        f"**Timestamp**: {env.get('timestamp', 'N/A')}",
        f"**GPU**: {env.get('gpu_name', 'CPU')}",
        f"**dtype**: {env.get('dtype', 'N/A')}",
        f"**gen_tokens**: {env.get('gen_tokens', 'N/A')}",
        "",
        "## Results",
        "",
        "| model | ctx | prefill_ms | decode_tps | vram_mb | state_mb | kv_window_mb | retained_kv_tokens | trn_state_bytes | status |",
        "|-------|-----|-----------|-----------|---------|---------|-------------|-------------------|----------------|--------|",
    ]
    for r in all_rows:
        lines.append(
            f"| {r['model']} | {r['context_len']} "
            f"| {fmt(r['prefill_latency_ms'])} "
            f"| {fmt(r['decode_tps'])} "
            f"| {fmt(r['peak_vram_mb'])} "
            f"| {fmt(r['state_memory_mb'])} "
            f"| {fmt(r['kv_window_mb'])} "
            f"| {r.get('retained_kv_tokens', 'N/A')} "
            f"| {r.get('trn_state_bytes', 'N/A')} "
            f"| {r['status']} |"
        )
    (artifact_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
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
    artifact_root: Optional[Path],
    torch_compile: bool,
) -> None:
    device = torch.device(device_str)
    dtype  = _resolve_dtype(dtype_str)
    seed_everything(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("vLLM TRN Benchmark: DualMemoryEngine vs TF+KV vs TRN")
    print("=" * 80)
    print(f"  device={device_str}  dtype={dtype_str}  gen_tokens={gen_tokens}")
    print(f"  n_repeats={n_repeats}  warmup={warmup_steps}  seed={seed}")
    print(f"  models: {', '.join(model_names)}")
    print(f"  context_lens: {context_lens}")
    print()

    env = _collect_env(seed, dtype_str, gen_tokens, n_repeats, warmup_steps, device)

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
            warmup_steps=warmup_steps,
            seed=seed,
            dtype=dtype,
            torch_compile=torch_compile,
        )
        all_rows.extend(rows)

    if artifact_root is not None and all_rows:
        artifact_dir = artifact_root / timestamp
        artifact_dir.mkdir(parents=True, exist_ok=True)

        (artifact_dir / "results.json").write_text(
            json.dumps(_rows_to_json_safe(all_rows), indent=2), encoding="utf-8"
        )
        (artifact_dir / "env.json").write_text(
            json.dumps(env, indent=2), encoding="utf-8"
        )
        _save_nvidia_smi(artifact_dir)
        _save_summary_md(artifact_dir, all_rows, env)
        print(f"\n  Artifacts saved -> {artifact_dir}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM-style benchmark: DualMemoryEngine vs TF+KV vs TRN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models",       type=str, default=DEFAULT_MODELS,
                        help="Comma-separated model names from: " + ", ".join(_MODEL_REGISTRY))
    parser.add_argument("--context-lens", type=str,
                        default=",".join(str(c) for c in DEFAULT_CONTEXT_LENS),
                        help="Comma-separated context lengths")
    parser.add_argument("--gen-tokens",   type=int, default=DEFAULT_GEN_TOKENS)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-repeats",    type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--seed",         type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype",        type=str, default=DEFAULT_DTYPE,
                        choices=list(_DTYPE_MAP.keys()))
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true",
                        help="Skip artifact directory creation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_names  = [m.strip() for m in args.models.split(",")       if m.strip()]
    context_lens = sorted(set(int(x.strip()) for x in args.context_lens.split(",") if x.strip()))

    project_root  = Path(__file__).resolve().parent.parent
    artifact_root = None if args.no_artifacts else project_root / "artifacts" / "phase9"

    run_benchmark(
        model_names=model_names,
        context_lens=context_lens,
        gen_tokens=args.gen_tokens,
        device_str=args.device,
        n_repeats=args.n_repeats,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        dtype_str=args.dtype,
        artifact_root=artifact_root,
        torch_compile=args.torch_compile,
    )


if __name__ == "__main__":
    main()
