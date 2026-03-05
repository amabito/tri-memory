#!/usr/bin/env python3
"""Agent history benchmark: TRN vs Transformer with KV cache vs full context.

Simulates an AI agent maintaining long conversation history across checkpoints.
At each history length T, measures three modes:
  TRN      -- TRNModel.generate() (O(1) state, constant memory)
  TF_kv    -- TransformerModel with KV cache decode (O(T) cache, O(1) per step)
  TF_full  -- TransformerModel full recompute (O(T^2), clamped to max_seq_len)

Output:
  Table: history_tokens, trn_tps, trn_state_kb, tf_kv_tps, tf_kv_cache_mb, tf_full_tps
  results/bench_agent_history.csv (unless --no-csv)

Usage:
    python scripts/bench_agent_history.py
    python scripts/bench_agent_history.py --checkpoints 500,1000 --gen-tokens 16 --no-csv
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

from trn.baseline import CausalSelfAttention, TransformerModel
from trn.bench_data import seed_everything
from trn.config import TRNConfig
from trn.model import TRNModel

# Default benchmark configuration
DEFAULT_CHECKPOINTS = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
DEFAULT_GEN_TOKENS = 64
DEFAULT_TURN_LEN = 32
DEFAULT_SEED = 42
DEFAULT_DEVICE = "cpu"

# Model configuration (matches bench_long_context_scaling.py)
BENCH_D_MODEL = 256
BENCH_N_LAYERS = 8
BENCH_D_FF = 1024
BENCH_N_OSC = 128
BENCH_VOCAB = 256
BENCH_MAX_SEQ_LEN = 16640


# ---------------------------------------------------------------------------
# Dialogue token generation
# ---------------------------------------------------------------------------

def _generate_dialogue_tokens(
    total_tokens: int,
    turn_len: int = 32,
    seed: int = 42,
) -> Tensor:
    """Generate alternating user/assistant dialogue tokens.

    User turns: tokens in [10, 99], assistant turns: tokens in [100, 199].
    Turns are separated by token 3 (turn boundary).
    Returns shape (1, total_tokens).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    tokens: list[int] = []
    is_user = True
    while len(tokens) < total_tokens:
        lo, hi = (10, 100) if is_user else (100, 200)
        turn = torch.randint(lo, hi, (turn_len,), generator=rng).tolist()
        tokens.extend(turn)
        tokens.append(3)  # turn boundary
        is_user = not is_user
    return torch.tensor(tokens[:total_tokens], dtype=torch.long).unsqueeze(0)


# ---------------------------------------------------------------------------
# Naive full-recompute Transformer generation (copy from bench_long_context_scaling)
# ---------------------------------------------------------------------------

def _tf_generate(
    model: TransformerModel,
    cfg: TRNConfig,
    prompt: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """Autoregressive generation for TransformerModel without KV cache.

    Context is clamped to max_seq_len to avoid positional encoding overflow.
    """
    model.eval()
    generated = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = generated[:, -cfg.max_seq_len:]
            out = model(ctx)
            logits = out["logits"][:, -1, :]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
    return generated[:, prompt.size(1):]


# ---------------------------------------------------------------------------
# KV cache implementation for TransformerModel
# ---------------------------------------------------------------------------

def _build_kv_cache(
    model: TransformerModel,
    prompt: Tensor,
) -> tuple[list[Tensor], list[Tensor], Tensor]:
    """Prefill KV cache from prompt tokens using forward hooks.

    Returns:
        k_caches: list of (1, n_heads, T, head_dim) per layer
        v_caches: list of (1, n_heads, T, head_dim) per layer
        last_logits: (1, vocab_size) logits at final prompt position
    """
    n_heads = max(1, model.cfg.d_model // 64)
    head_dim = model.cfg.d_model // n_heads
    B, T = prompt.shape
    device = prompt.device

    k_caches: list[Optional[Tensor]] = [None] * model.cfg.n_layers
    v_caches: list[Optional[Tensor]] = [None] * model.cfg.n_layers

    hooks = []

    def _make_hook(layer_idx: int):
        def hook(module: CausalSelfAttention, args, output):
            # Capture k, v from the attention input (args[0] is the normed input x)
            x = args[0]
            C = x.size(-1)
            q_raw, k_raw, v_raw = module.qkv(x).split(C, dim=-1)
            k_caches[layer_idx] = k_raw.view(B, T, n_heads, head_dim).transpose(1, 2)
            v_caches[layer_idx] = v_raw.view(B, T, n_heads, head_dim).transpose(1, 2)
        return hook

    for i, block in enumerate(model.blocks):
        h = block.attn.register_forward_hook(_make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        out = model(prompt)
    last_logits = out["logits"][:, -1, :]  # (1, vocab_size)

    for h in hooks:
        h.remove()

    return k_caches, v_caches, last_logits


def _kv_decode_step(
    model: TransformerModel,
    token: Tensor,
    k_caches: list[Tensor],
    v_caches: list[Tensor],
    position: int,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Single-token decode step using KV cache.

    Args:
        token: (1,) int tensor of the current token
        k_caches: per-layer (1, n_heads, T, head_dim)
        v_caches: per-layer (1, n_heads, T, head_dim)
        position: absolute position index of the new token

    Returns:
        logits: (1, vocab_size)
        updated k_caches
        updated v_caches
    """
    n_heads = max(1, model.cfg.d_model // 64)
    head_dim = model.cfg.d_model // n_heads
    device = token.device

    with torch.no_grad():
        # Embed single token and add positional encoding
        x = model.embed(token.unsqueeze(0))        # (1, 1, d_model)
        x = x + model.pe[position:position + 1]   # (1, 1, d_model)

        for i, block in enumerate(model.blocks):
            # Pre-norm
            h = block.norm1(x)                     # (1, 1, d_model)

            # Compute q, k, v for new token
            qkv = block.attn.qkv(h)                # (1, 1, 3*d_model)
            C = model.cfg.d_model
            q_new, k_new, v_new = qkv.split(C, dim=-1)
            q_new = q_new.view(1, 1, n_heads, head_dim).transpose(1, 2)  # (1,H,1,D)
            k_new = k_new.view(1, 1, n_heads, head_dim).transpose(1, 2)
            v_new = v_new.view(1, 1, n_heads, head_dim).transpose(1, 2)

            # Append to caches
            k_full = torch.cat([k_caches[i], k_new], dim=2)  # (1,H,T+1,D)
            v_full = torch.cat([v_caches[i], v_new], dim=2)
            k_caches[i] = k_full
            v_caches[i] = v_full

            # Attention: q is (1,H,1,D), k/v are (1,H,T+1,D) — causal already
            import torch.nn.functional as F
            attn_out = F.scaled_dot_product_attention(q_new, k_full, v_full, is_causal=False)
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, C)
            attn_out = block.attn.proj(attn_out)

            x = x + attn_out

            # FFN (SwiGLU)
            h2 = block.norm2(x)
            x = x + block.w3(F.silu(block.w1(h2)) * block.w2(h2))

        x = model.norm(x)                          # (1, 1, d_model)
        logits = model.lm_head(x[:, -1, :])        # (1, vocab_size)

    return logits, k_caches, v_caches


# ---------------------------------------------------------------------------
# TRN state size (analytical)
# ---------------------------------------------------------------------------

def _trn_state_kb(cfg: TRNConfig) -> float:
    """Analytical TRN state size in KB (constant regardless of history length)."""
    K = cfg.n_oscillators
    # n_layers * 2 states (real, imag) * K oscillators * 4 bytes (fp32)
    return cfg.n_layers * 2 * K * 4 / 1024.0


# ---------------------------------------------------------------------------
# KV cache size (analytical)
# ---------------------------------------------------------------------------

def _kv_cache_mb(cfg: TRNConfig, context_len: int) -> float:
    """Analytical KV cache size in MB for given context length."""
    n_heads = max(1, cfg.d_model // 64)
    head_dim = cfg.d_model // n_heads
    # n_layers * 2 (K, V) * n_heads * T * head_dim * 4 bytes (fp32)
    return cfg.n_layers * 2 * n_heads * context_len * head_dim * 4 / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------

def _measure_trn(
    model: TRNModel,
    history: Tensor,
    gen_tokens: int,
    cfg: TRNConfig,
) -> float:
    """Measure TRN generation TPS for given history prefix."""
    model.eval()
    t0 = time.perf_counter()
    model.generate(history, max_new_tokens=gen_tokens)
    elapsed = time.perf_counter() - t0
    return gen_tokens / elapsed if elapsed > 0.0 else float("inf")


def _measure_tf_kv(
    model: TransformerModel,
    history: Tensor,
    gen_tokens: int,
    cfg: TRNConfig,
    device: torch.device,
) -> tuple[float, bool]:
    """Measure Transformer KV-cache generation TPS.

    Returns (tps, oom_occurred). On OOM, returns (NaN, True).
    """
    try:
        model.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            # Prefill
            k_caches, v_caches, logits = _build_kv_cache(model, history)
            position = history.size(1)

            # Decode loop
            for _ in range(gen_tokens):
                next_tok = torch.argmax(logits, dim=-1)  # (1,)
                logits, k_caches, v_caches = _kv_decode_step(
                    model, next_tok, k_caches, v_caches, position
                )
                position += 1

            elapsed = time.perf_counter() - t0
        return gen_tokens / elapsed if elapsed > 0.0 else float("inf"), False
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return float("nan"), True


def _measure_tf_full(
    model: TransformerModel,
    history: Tensor,
    gen_tokens: int,
    cfg: TRNConfig,
) -> float:
    """Measure Transformer full-recompute generation TPS (context clamped to max_seq_len)."""
    model.eval()
    clamped = history[:, :cfg.max_seq_len]
    t0 = time.perf_counter()
    _tf_generate(model, cfg, clamped, gen_tokens)
    elapsed = time.perf_counter() - t0
    return gen_tokens / elapsed if elapsed > 0.0 else float("inf")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    checkpoints: list[int],
    gen_tokens: int,
    turn_len: int,
    seed: int,
    device_str: str,
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

    max_checkpoint = max(checkpoints)
    print("Generating dialogue tokens...")
    full_dialogue = _generate_dialogue_tokens(max_checkpoint, turn_len=turn_len, seed=seed)
    full_dialogue = full_dialogue.to(device)

    trn_state_kb = _trn_state_kb(cfg)

    print("=" * 90)
    print("Agent History Benchmark: TRN vs TF_kv vs TF_full")
    print("=" * 90)
    print(
        f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, d_ff={cfg.d_ff}, "
        f"n_osc={cfg.n_oscillators}"
    )
    print(f"  max_seq_len={cfg.max_seq_len}, gen_tokens={gen_tokens}, "
          f"turn_len={turn_len}, seed={seed}, device={device_str}")
    print()

    header = (
        f"{'history_tokens':>14}  "
        f"{'trn_tps':>10}  {'trn_state_kb':>12}  "
        f"{'tf_kv_tps':>10}  {'tf_kv_cache_mb':>14}  "
        f"{'tf_full_tps':>11}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []

    for T in checkpoints:
        history = full_dialogue[:, :T]

        trn_tps = _measure_trn(trn, history, gen_tokens, cfg)
        tf_kv_tps, oom = _measure_tf_kv(tf, history, gen_tokens, cfg, device)
        tf_full_tps = _measure_tf_full(tf, history, gen_tokens, cfg)
        kv_cache_mb = _kv_cache_mb(cfg, T)

        trn_tps_str = f"{trn_tps:>10.1f}"
        tf_kv_str = "       OOM" if oom else f"{tf_kv_tps:>10.1f}"
        tf_full_str = f"{tf_full_tps:>11.1f}"

        print(
            f"{T:>14}  "
            f"{trn_tps_str}  {trn_state_kb:>12.2f}  "
            f"{tf_kv_str}  {kv_cache_mb:>14.3f}  "
            f"{tf_full_str}"
        )

        rows.append(dict(
            history_tokens=T,
            trn_tps=trn_tps,
            trn_state_kb=trn_state_kb,
            tf_kv_tps=tf_kv_tps,
            tf_kv_cache_mb=kv_cache_mb,
            tf_full_tps=tf_full_tps,
        ))

    print()

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "history_tokens", "trn_tps", "trn_state_kb",
            "tf_kv_tps", "tf_kv_cache_mb", "tf_full_tps",
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV saved -> {output_csv}")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent history benchmark: TRN vs TF_kv vs TF_full",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=",".join(str(c) for c in DEFAULT_CHECKPOINTS),
        help="Comma-separated history token checkpoints",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=DEFAULT_GEN_TOKENS,
        help="Number of tokens to generate at each checkpoint",
    )
    parser.add_argument(
        "--turn-len",
        type=int,
        default=DEFAULT_TURN_LEN,
        help="Tokens per dialogue turn (before turn boundary token)",
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
        "--no-csv",
        action="store_true",
        help="Skip CSV output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    checkpoints = sorted(set(checkpoints))

    project_root = Path(__file__).resolve().parent.parent
    output_csv = None if args.no_csv else project_root / "results" / "bench_agent_history.csv"

    run_benchmark(
        checkpoints=checkpoints,
        gen_tokens=args.gen_tokens,
        turn_len=args.turn_len,
        seed=args.seed,
        device_str=args.device,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
