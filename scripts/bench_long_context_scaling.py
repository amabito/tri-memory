#!/usr/bin/env python3
"""Long-context scaling benchmark: TRN vs Transformer generation latency and memory.

Measures at each context length:
- tokens_per_sec: generation throughput
- memory_mb: peak tracemalloc RSS during generation
- latency_ms: average wall-clock time per generation call

Output:
- Scaling table printed to stdout (with TRN speedup ratio vs TF)
- results/long_context_scaling.csv saved to project root

Usage:
    python scripts/bench_long_context_scaling.py
    python scripts/bench_long_context_scaling.py \\
        --context-lens 512,1024,2048,4096,8192,16384 \\
        --gen-tokens 128 --batch-size 1 --seed 42 --device cpu
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch import Tensor

from trn.bench_data import seed_everything
from trn.baseline import TransformerModel
from trn.config import TRNConfig
from trn.model import TRNModel

# Default benchmark configuration
DEFAULT_CONTEXT_LENS = [512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_GEN_TOKENS = 128
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEED = 42
DEFAULT_DEVICE = "cpu"
DEFAULT_N_REPEATS = 2

# Model configuration (fixed per spec)
BENCH_D_MODEL = 256
BENCH_N_LAYERS = 8
BENCH_D_FF = 1024
BENCH_N_OSC = 128
BENCH_MAX_SEQ_LEN = 16384 + 256  # covers all context lengths + gen buffer


# ---------------------------------------------------------------------------
# Transformer naive autoregressive generation (O(n) per step, no KV cache)
# ---------------------------------------------------------------------------

def _tf_generate(
    model: TransformerModel,
    cfg: TRNConfig,
    prompt: Tensor,
    max_new_tokens: int,
) -> Tensor:
    """Autoregressive generation for TransformerModel (no KV cache).

    Re-runs the full forward pass each step to highlight TRN's O(1) advantage.
    Clamps sequence to max_seq_len to avoid positional encoding overflow.
    """
    model.eval()
    generated = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = generated[:, -cfg.max_seq_len:]
            out = model(ctx)
            logits = out["logits"][:, -1, :]  # last position only
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
    return generated[:, prompt.size(1):]


# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------

def measure_generation(
    model: nn.Module,
    cfg: TRNConfig,
    context_len: int,
    gen_tokens: int = 128,
    batch_size: int = 1,
    n_repeats: int = DEFAULT_N_REPEATS,
    is_trn: bool = True,
    device: torch.device | None = None,
) -> tuple[float, float, float]:
    """Measure generation performance for one (model, context_len) combination.

    Returns:
        tokens_per_sec: generation throughput (gen_tokens * batch_size / avg_latency)
        memory_mb:      peak tracemalloc RSS (MB) for a single batch_size=1 call
        latency_ms:     average wall-clock time per generation call (ms)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # ---- latency + TPS: n_repeats timed runs at requested batch_size ----
    prompt = torch.randint(4, cfg.vocab_size, (batch_size, context_len), device=device)

    times: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        if is_trn:
            model.generate(prompt, max_new_tokens=gen_tokens)
        else:
            _tf_generate(model, cfg, prompt, max_new_tokens=gen_tokens)
        times.append(time.perf_counter() - t0)

    avg_t = sum(times) / len(times)
    latency_ms = avg_t * 1000.0
    total_tokens = gen_tokens * batch_size
    tokens_per_sec = total_tokens / avg_t if avg_t > 0.0 else float("inf")

    # ---- memory: single run at batch_size=1 via tracemalloc ----
    mem_prompt = torch.randint(4, cfg.vocab_size, (1, context_len), device=device)
    tracemalloc.start()
    with torch.no_grad():
        if is_trn:
            model.generate(mem_prompt, max_new_tokens=gen_tokens)
        else:
            _tf_generate(model, cfg, mem_prompt, max_new_tokens=gen_tokens)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / (1024 * 1024)

    return tokens_per_sec, memory_mb, latency_ms


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
    output_csv: Path | None = None,
) -> None:
    seed_everything(seed)
    device = torch.device(device_str)

    cfg = TRNConfig(
        vocab_size=256,
        d_model=BENCH_D_MODEL,
        n_layers=BENCH_N_LAYERS,
        d_ff=BENCH_D_FF,
        n_oscillators=BENCH_N_OSC,
        max_seq_len=BENCH_MAX_SEQ_LEN,
    )

    trn = TRNModel(cfg).to(device).eval()
    tf = TransformerModel(cfg).to(device).eval()

    trn_params = trn.num_parameters(non_embedding=True)
    tf_params = tf.num_parameters(non_embedding=True)

    print("=" * 76)
    print("Long-Context Scaling Benchmark: TRN vs Transformer")
    print("=" * 76)
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"d_ff={cfg.d_ff}, n_osc={cfg.n_oscillators}")
    print(f"  max_seq_len={cfg.max_seq_len}, gen_tokens={gen_tokens}, "
          f"batch_size={batch_size}, seed={seed}, device={device_str}")
    print(f"  TRN params (non-emb): {trn_params:,}")
    print(f"  TF  params (non-emb): {tf_params:,}")
    print()

    # ---- collect results ----
    rows: list[dict] = []

    col_w = 10
    header = (
        f"{'ctx_len':>8}  {'model':>6}  "
        f"{'tps':>{col_w}}  {'mem_mb':>{col_w}}  {'lat_ms':>{col_w}}  "
        f"{'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for ctx_len in context_lens:
        trn_tps, trn_mem, trn_lat = measure_generation(
            trn, cfg, ctx_len, gen_tokens, batch_size,
            n_repeats=n_repeats, is_trn=True, device=device,
        )
        tf_tps, tf_mem, tf_lat = measure_generation(
            tf, cfg, ctx_len, gen_tokens, batch_size,
            n_repeats=n_repeats, is_trn=False, device=device,
        )
        speedup = trn_tps / tf_tps if tf_tps > 0.0 else float("inf")

        print(
            f"{ctx_len:>8}  {'TRN':>6}  "
            f"{trn_tps:>{col_w}.1f}  {trn_mem:>{col_w}.3f}  {trn_lat:>{col_w}.1f}  "
            f"{'':>8}"
        )
        print(
            f"{'':>8}  {'TF':>6}  "
            f"{tf_tps:>{col_w}.1f}  {tf_mem:>{col_w}.3f}  {tf_lat:>{col_w}.1f}  "
            f"{speedup:>8.2f}x"
        )

        rows.append(dict(
            model="TRN", context_len=ctx_len,
            tps=trn_tps, memory_mb=trn_mem, latency_ms=trn_lat,
        ))
        rows.append(dict(
            model="TF", context_len=ctx_len,
            tps=tf_tps, memory_mb=tf_mem, latency_ms=tf_lat,
        ))

    print()

    # ---- summary ----
    trn_rows = [r for r in rows if r["model"] == "TRN"]
    tf_rows  = [r for r in rows if r["model"] == "TF"]

    print("Summary")
    print("-" * 50)

    if len(trn_rows) >= 2 and len(tf_rows) >= 2:
        speedup_first = trn_rows[0]["tps"] / tf_rows[0]["tps"]
        speedup_last  = trn_rows[-1]["tps"] / tf_rows[-1]["tps"]
        print(
            f"  TRN/TF speedup: {speedup_first:.2f}x @ ctx={trn_rows[0]['context_len']} "
            f"-> {speedup_last:.2f}x @ ctx={trn_rows[-1]['context_len']}"
        )

    if len(trn_rows) >= 2:
        mem_first = trn_rows[0]["memory_mb"]
        mem_last  = trn_rows[-1]["memory_mb"]
        ctx_first = trn_rows[0]["context_len"]
        ctx_last  = trn_rows[-1]["context_len"]
        growth = mem_last / mem_first if mem_first > 0.0 else float("inf")
        ctx_ratio = ctx_last / ctx_first
        print(
            f"  TRN memory: {mem_first:.3f}MB @ ctx={ctx_first} "
            f"-> {mem_last:.3f}MB @ ctx={ctx_last} "
            f"({growth:.2f}x growth vs {ctx_ratio:.0f}x context)"
        )

    trn_wins = sum(
        1 for t, f in zip(trn_rows, tf_rows) if t["tps"] > f["tps"]
    )
    print(f"  TRN faster than TF: {trn_wins}/{len(trn_rows)} context lengths")

    # ---- CSV output ----
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["model", "context_len", "tps", "memory_mb", "latency_ms"]
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
        description="Long-context scaling benchmark: TRN vs Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--context-lens",
        type=str,
        default=",".join(str(c) for c in DEFAULT_CONTEXT_LENS),
        help="Comma-separated list of context lengths to benchmark",
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
        help="Random seed for reproducibility",
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
    context_lens = [int(x.strip()) for x in args.context_lens.split(",") if x.strip()]
    context_lens = sorted(set(context_lens))

    # CSV path: results/ relative to project root (two levels up from scripts/)
    project_root = Path(__file__).resolve().parent.parent
    output_csv = None if args.no_csv else project_root / "results" / "long_context_scaling.csv"

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
