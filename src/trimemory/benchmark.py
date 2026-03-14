"""Throughput and memory benchmarks for TRNModel.

All benchmarks run on CPU by default so they work without CUDA.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch

from .config import TRNConfig
from .model import TRNModel


@dataclass
class BenchmarkResult:
    """Holds results from a single benchmark run."""

    tokens_per_second: float
    ms_per_step: float
    peak_memory_mb: float
    n_steps: int
    batch_size: int
    seq_len: int
    mode: str  # "forward" | "generate" | "step_single"


def benchmark_forward(
    model: TRNModel,
    batch_size: int = 4,
    seq_len: int = 128,
    n_steps: int = 20,
    warmup: int = 3,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark forward pass throughput (training mode).

    Returns tokens/sec and ms/step averaged over n_steps.
    """
    model = model.to(device).train()
    input_ids = torch.randint(
        0, model.cfg.vocab_size, (batch_size, seq_len), device=device
    )

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    gc.collect()
    start = time.perf_counter()

    for _ in range(n_steps):
        with torch.no_grad():
            model(input_ids)
        if device == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    peak_mem = 0.0
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2

    ms_per_step = elapsed / n_steps * 1000
    tokens_per_second = (batch_size * seq_len * n_steps) / elapsed

    return BenchmarkResult(
        tokens_per_second=tokens_per_second,
        ms_per_step=ms_per_step,
        peak_memory_mb=peak_mem,
        n_steps=n_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        mode="forward",
    )


def benchmark_step_single(
    model: TRNModel,
    batch_size: int = 1,
    n_steps: int = 50,
    warmup: int = 5,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark single-step autoregressive inference throughput.

    This is the critical path for interactive generation.
    Returns tokens/sec (= steps/sec for batch=1).
    """
    model = model.to(device).eval()
    K = model.cfg.n_oscillators
    n_layers = model.cfg.n_layers

    # Initial state — always fp32 per TRN convention
    states_r = [
        torch.zeros(batch_size, K, device=device, dtype=torch.float32)
        for _ in range(n_layers)
    ]
    states_i = [
        torch.zeros(batch_size, K, device=device, dtype=torch.float32)
        for _ in range(n_layers)
    ]

    input_ids = torch.randint(
        0, model.cfg.vocab_size, (batch_size,), device=device
    )
    param_dtype = next(model.parameters()).dtype

    @torch.no_grad()
    def one_step(pos: int) -> None:
        nonlocal states_r, states_i
        x = model.drop_emb(model.embedding(input_ids).to(param_dtype))
        for layer_idx, block in enumerate(model.blocks):
            x_n = block.norm1(x)
            res_out, states_r[layer_idx], states_i[layer_idx] = (
                block.resonance.step_single(
                    x_n, states_r[layer_idx], states_i[layer_idx], pos
                )
            )
            x = x + block.drop(res_out)
            x = x + block.drop(block.ffn(block.norm2(x)))

    # Warmup
    for i in range(warmup):
        one_step(i)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(n_steps):
        one_step(warmup + i)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_step = elapsed / n_steps * 1000
    tokens_per_second = (batch_size * n_steps) / elapsed

    return BenchmarkResult(
        tokens_per_second=tokens_per_second,
        ms_per_step=ms_per_step,
        peak_memory_mb=0.0,
        n_steps=n_steps,
        batch_size=batch_size,
        seq_len=1,
        mode="step_single",
    )


def run_all_benchmarks(
    cfg: Optional[TRNConfig] = None,
    device: str = "cpu",
) -> dict[str, BenchmarkResult]:
    """Run the full benchmark suite and return results by name."""
    if cfg is None:
        cfg = TRNConfig.toy()

    model = TRNModel(cfg)
    results: dict[str, BenchmarkResult] = {}

    results["forward_bs4_seq128"] = benchmark_forward(
        model, batch_size=4, seq_len=128, n_steps=20, device=device
    )
    results["forward_bs1_seq512"] = benchmark_forward(
        model, batch_size=1, seq_len=512, n_steps=10, device=device
    )
    results["step_single_bs1"] = benchmark_step_single(
        model, batch_size=1, n_steps=50, device=device
    )

    return results


def print_benchmark_report(results: dict[str, BenchmarkResult]) -> None:
    """Print a formatted benchmark report."""
    print(f"{'Benchmark':<30} {'tok/s':>10} {'ms/step':>10} {'peak MB':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(
            f"{name:<30} {r.tokens_per_second:>10.1f}"
            f" {r.ms_per_step:>10.2f} {r.peak_memory_mb:>10.1f}"
        )
