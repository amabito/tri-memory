#!/usr/bin/env python3
"""Forward-pass component profiling for TRN vs Transformer.

Usage:
    python scripts/profile_forward.py [--steps 20] [--batch-size 4] [--seq-len 64]

Outputs:
    - Wall-time breakdown per component (projection, resonance/attn, FFN, head)
    - CSV: scripts/results/profile_forward.csv
    - torch.profiler trace: scripts/results/traces/
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.baseline import TransformerModel


def time_forward(
    model: nn.Module,
    input_ids: Tensor,
    n_warmup: int = 5,
    n_steps: int = 20,
) -> float:
    """Return average wall-time in ms for one forward pass."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids, labels=input_ids)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_steps):
            t0 = time.perf_counter()
            model(input_ids, labels=input_ids)
            times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times)


def profile_components(
    model: nn.Module,
    input_ids: Tensor,
    model_name: str,
    out_dir: Path,
) -> str:
    """Run torch.profiler and save trace + top-ops summary.

    Returns the key_averages table string.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                model(input_ids, labels=input_ids)

    trace_path = out_dir / f"trace_{model_name}.json"
    prof.export_chrome_trace(str(trace_path))

    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    (out_dir / f"top_ops_{model_name}.txt").write_text(table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile forward pass: TRN vs Transformer")
    parser.add_argument("--steps", type=int, default=20, help="Number of timed steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = results_dir / "traces"

    cfg = TRNConfig(
        vocab_size=256,
        d_model=args.d_model,
        n_oscillators=args.d_model // 2,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_len * 2,
    )

    trn = TRNModel(cfg)
    tf = TransformerModel(cfg)

    input_ids = torch.randint(4, cfg.vocab_size, (args.batch_size, args.seq_len))

    print(
        f"Profiling forward pass: "
        f"bs={args.batch_size} seq_len={args.seq_len} "
        f"d_model={args.d_model} n_layers={args.n_layers}"
    )
    print(f"n_steps={args.steps} (warmup=5)")
    print()

    trn_ms = time_forward(trn, input_ids, n_steps=args.steps)
    tf_ms = time_forward(tf, input_ids, n_steps=args.steps)

    print(f"TRN forward: {trn_ms:.2f} ms/batch")
    print(f"TF  forward: {tf_ms:.2f} ms/batch")
    ratio = trn_ms / tf_ms if tf_ms > 0 else float("inf")
    print(f"Ratio TRN/TF: {ratio:.2f}x")
    print()

    print("=== TRN Component Breakdown ===")
    trn_table = profile_components(trn, input_ids, "trn", traces_dir)
    print(trn_table)

    print("=== Transformer Component Breakdown ===")
    tf_table = profile_components(tf, input_ids, "tf", traces_dir)
    print(tf_table)

    csv_path = results_dir / "profile_forward.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "batch_size", "seq_len", "d_model", "n_layers", "ms_per_batch"])
        w.writerow(["TRN", args.batch_size, args.seq_len, args.d_model, args.n_layers, f"{trn_ms:.4f}"])
        w.writerow(["TF", args.batch_size, args.seq_len, args.d_model, args.n_layers, f"{tf_ms:.4f}"])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
