#!/usr/bin/env python3
"""Component-level profiling: TRN layer timing breakdown.

Usage:
    python profile.py
    python profile.py --d-model 256 --n-layers 4 --seq-len 128
    python profile.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trn.bench_data import seed_everything
from trn.config import TRNConfig
from trn.model import TRNModel
from trn.baseline import TransformerModel


def wall_time_microbench(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    n_warmup: int = 5,
    n_measure: int = 20,
    label: str = "",
) -> float:
    """Returns mean forward pass time in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids)

    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            t0 = time.perf_counter()
            model(input_ids)
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    tokens_per_sec = (input_ids.numel() / (mean_ms / 1000))
    print(f"  {label:<20} {mean_ms:8.2f} ms/step  {tokens_per_sec:10.0f} tok/s")
    return mean_ms


def profile_components(cfg: TRNConfig, seq_len: int, batch_size: int, device: str) -> None:
    """Profile individual TRN components using torch.profiler."""
    try:
        from torch.profiler import profile, ProfilerActivity, record_function
    except ImportError:
        print("[WARN] torch.profiler not available, skipping component profiling")
        return

    seed_everything(42)
    model = TRNModel(cfg).to(device).eval()
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(x)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(5):
                with record_function("model_forward"):
                    model(x)

    # Print top operations by CPU time
    print(f"\nTop 15 ops (TRN, batch={batch_size}, seq={seq_len}):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


def run_wall_time_bench(
    cfg: TRNConfig,
    seq_len: int,
    batch_sizes: list[int],
    device: str,
) -> None:
    """Wall-time forward pass benchmark for TRN and Transformer."""
    print(f"\nWall-time microbench (seq_len={seq_len}, device={device})")
    print("-" * 60)

    for bs in batch_sizes:
        ids = torch.randint(0, cfg.vocab_size, (bs, seq_len), device=device)

        seed_everything(42)
        trn = TRNModel(cfg).to(device)
        wall_time_microbench(trn, ids, label=f"TRN  bs={bs}")

        seed_everything(42)
        tf = TransformerModel(cfg).to(device)
        wall_time_microbench(tf, ids, label=f"TF   bs={bs}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="TRN profiler")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-profiler", action="store_true")
    args = parser.parse_args()

    cfg = TRNConfig(
        vocab_size=256,
        d_model=args.d_model,
        n_oscillators=args.d_model // 2,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_len + 8,
    )

    print(f"TRN config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_oscillators={cfg.n_oscillators}, seq_len={args.seq_len}")
    print(f"TRN params: {TRNModel(cfg).num_parameters():,}")
    print(f"TF  params: {TransformerModel(cfg).num_parameters():,}")

    run_wall_time_bench(cfg, args.seq_len, batch_sizes=[1, 4, 8], device=args.device)

    if not args.skip_profiler:
        profile_components(cfg, args.seq_len, batch_size=4, device=args.device)


if __name__ == "__main__":
    main()
