#!/usr/bin/env python3
"""Multi-agent memory simulation benchmark.

Simulates N independent agents each maintaining their own TRN state.
Measures actual memory via tracemalloc and compares against analytical
KV cache memory for equivalent context length.

Usage:
    python scripts/bench_multi_agent.py
    python scripts/bench_multi_agent.py --agent-counts 100,1000,10000 --device cpu --no-csv
    python scripts/bench_multi_agent.py --model-config toy --agent-counts 100,500 --device cpu --no-csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from trn.bench_data import seed_everything
from trn.config import TRNConfig
from trn.model import TRNModel

# GPU pricing (USD per hour)
A100_PRICE_PER_HR = 2.50
H100_PRICE_PER_HR = 3.50

# A100/H100 VRAM in GB
A100_VRAM_GB = 80.0
H100_VRAM_GB = 80.0

# Benchmark configuration
DEFAULT_AGENT_COUNTS = [100, 1000, 10000]
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 42
DEFAULT_TOKENS_PER_AGENT = 64   # tokens fed to each agent
DEFAULT_HISTORY_LEN = 1000      # equivalent KV context length for comparison

# Model config choices
MODEL_CONFIG_CHOICES = ["toy", "trn_100m", "trn_400m", "trn_1b"]


def _get_config(name: str) -> TRNConfig:
    """Return TRNConfig preset by name."""
    factories = {
        "toy": TRNConfig.toy,
        "trn_100m": TRNConfig.trn_100m,
        "trn_400m": TRNConfig.trn_400m,
        "trn_1b": TRNConfig.trn_1b,
    }
    return factories[name]()


def _analytical_trn_state_bytes(cfg: TRNConfig) -> int:
    """TRN state size in bytes: n_layers * K * 2 (real+imag) * 4 (fp32)."""
    return cfg.n_layers * cfg.n_oscillators * 2 * 4


def _analytical_kv_cache_bytes(cfg: TRNConfig, context_len: int) -> int:
    """KV cache size in bytes for given context length.

    n_layers * 2 (K,V) * n_heads * context_len * head_dim * 4 (fp32)
    """
    n_heads = max(1, cfg.d_model // 64)
    head_dim = cfg.d_model // n_heads
    return cfg.n_layers * 2 * n_heads * context_len * head_dim * 4


def _model_params_mb(model: TRNModel) -> float:
    """Total model parameter memory in MB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)


def _simulate_agents_trn(
    cfg: TRNConfig,
    n_agents: int,
    tokens_per_agent: int,
    device: torch.device,
    seed: int,
    model: TRNModel,
) -> tuple[float, float, float]:
    """Simulate n_agents each processing tokens_per_agent tokens with TRN.

    Returns (elapsed_seconds, actual_peak_memory_bytes, gpu_vram_mb).
    gpu_vram_mb is 0.0 when device is cpu.
    """
    seed_everything(seed)

    # Generate random token sequences for all agents
    rng_gen = torch.Generator()
    rng_gen.manual_seed(seed)
    tokens = torch.randint(
        0, cfg.vocab_size, (n_agents, tokens_per_agent),
        generator=rng_gen, device=device,
    )

    # Reset GPU peak memory stats before simulation
    gpu_vram_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Start memory tracking
    tracemalloc.start()
    t0 = time.perf_counter()

    # Each agent processes its token sequence independently
    # We batch all agents together for throughput measurement
    with torch.no_grad():
        # Feed tokens one at a time to simulate streaming (O(1) state per agent)
        K = cfg.n_oscillators
        states_r = torch.zeros(n_agents, K, device=device, dtype=torch.float32)
        states_i = torch.zeros(n_agents, K, device=device, dtype=torch.float32)

        for pos in range(tokens_per_agent):
            token = tokens[:, pos]  # (n_agents,)
            x = model.embedding(token)  # (n_agents, d_model)

            for block in model.blocks:
                x_normed = block.norm1(x)
                res_out, states_r, states_i = block.resonance.step_single(
                    x_normed, states_r, states_i, pos
                )
                x = x + res_out
                x = x + block.ffn(block.norm2(x))

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if device.type == "cuda":
        gpu_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return elapsed, float(peak_bytes), gpu_vram_mb


def _gpu_cost_estimate(
    trn_total_bytes: int,
    kv_total_bytes: int,
    hours: float = 1.0,
) -> dict[str, float]:
    """Estimate GPU cost based on actual memory requirements.

    n_gpus = ceil(total_memory_gb / gpu_vram_gb)
    cost = n_gpus * price_per_hr * hours
    """
    trn_total_gb = trn_total_bytes / (1024 ** 3)
    kv_total_gb = kv_total_bytes / (1024 ** 3)

    trn_n_gpus_a100 = math.ceil(trn_total_gb / A100_VRAM_GB) if trn_total_gb > 0 else 1
    kv_n_gpus_a100 = math.ceil(kv_total_gb / A100_VRAM_GB) if kv_total_gb > 0 else 1

    trn_n_gpus_h100 = math.ceil(trn_total_gb / H100_VRAM_GB) if trn_total_gb > 0 else 1
    kv_n_gpus_h100 = math.ceil(kv_total_gb / H100_VRAM_GB) if kv_total_gb > 0 else 1

    return {
        "trn_gpu_count_a100": trn_n_gpus_a100,
        "kv_gpu_count_a100": kv_n_gpus_a100,
        "trn_a100_cost_usd": trn_n_gpus_a100 * A100_PRICE_PER_HR * hours,
        "kv_a100_cost_usd": kv_n_gpus_a100 * A100_PRICE_PER_HR * hours,
        "trn_h100_cost_usd": trn_n_gpus_h100 * H100_PRICE_PER_HR * hours,
        "kv_h100_cost_usd": kv_n_gpus_h100 * H100_PRICE_PER_HR * hours,
    }


def run_benchmark(
    agent_counts: list[int],
    device_str: str,
    tokens_per_agent: int,
    history_len: int,
    seed: int,
    model_config: str,
    output_csv: Optional[Path] = None,
) -> None:
    seed_everything(seed)
    device = torch.device(device_str)

    cfg = _get_config(model_config)

    # Build model once; reuse across agent counts
    model = TRNModel(cfg).to(device).eval()
    params_mb = _model_params_mb(model)

    trn_state_bytes = _analytical_trn_state_bytes(cfg)
    kv_bytes = _analytical_kv_cache_bytes(cfg, history_len)

    n_heads = max(1, cfg.d_model // 64)
    head_dim = cfg.d_model // n_heads

    print("=" * 80)
    print("Multi-Agent Memory Simulation: TRN vs KV Cache")
    print("=" * 80)
    print(f"  Config: {model_config} ({cfg.d_model}d, {cfg.n_layers}L, {cfg.n_oscillators}K)")
    print(f"  Tokens per agent: {tokens_per_agent}")
    print(f"  Equivalent KV context length: {history_len}")
    print(f"  Device: {device_str}")
    print(f"  Model parameters: {params_mb:.2f} MB")
    print()

    # Premise documentation: explicit formulas with values plugged in
    trn_state_computed = cfg.n_layers * cfg.n_oscillators * 2 * 4
    kv_computed = cfg.n_layers * 2 * n_heads * history_len * head_dim * 4
    print("  Formulas (analytical):")
    print(
        f"  TRN state = n_layers({cfg.n_layers}) * K({cfg.n_oscillators}) * 2 * 4"
        f" = {trn_state_computed} bytes"
    )
    print(
        f"  KV cache  = n_layers({cfg.n_layers}) * 2 * n_heads({n_heads})"
        f" * T({history_len}) * head_dim({head_dim}) * 4"
        f" = {kv_computed} bytes"
    )
    print(
        f"  TRN: O(1) constant {trn_state_bytes / 1024:.2f} KB"
        f" | KV: O(n) at T={history_len} = {kv_bytes / (1024 * 1024):.3f} MB"
        f" | ratio = {kv_bytes / trn_state_bytes:.1f}x"
    )
    print()

    header = (
        f"{'n_agents':>10}  "
        f"{'memory_model':>12}  "
        f"{'trn_total_mb':>14}  "
        f"{'trn_per_kb':>12}  "
        f"{'kv_total_mb':>12}  "
        f"{'kv_per_mb':>10}  "
        f"{'kv/trn':>8}  "
        f"{'actual_mb':>10}  "
        f"{'throughput_aps':>14}  "
        f"{'trn_gpus_a100':>14}  "
        f"{'kv_gpus_a100':>13}  "
        f"{'trn_a100_usd/hr':>16}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []

    for n_agents in agent_counts:
        elapsed, actual_peak_bytes, gpu_vram_mb = _simulate_agents_trn(
            cfg, n_agents, tokens_per_agent, device, seed, model
        )
        actual_peak_memory_mb = actual_peak_bytes / (1024 * 1024)

        # TRN memory: analytical (state per agent * n_agents)
        trn_total_bytes = trn_state_bytes * n_agents
        trn_total_mb = trn_total_bytes / (1024 * 1024)
        trn_per_agent_kb = trn_state_bytes / 1024

        # KV cache: analytical (kv per agent * n_agents)
        kv_total_bytes = kv_bytes * n_agents
        kv_total_mb = kv_total_bytes / (1024 * 1024)
        kv_per_agent_mb = kv_bytes / (1024 * 1024)

        memory_reduction = kv_total_bytes / max(trn_total_bytes, 1)

        # Throughput: agents processed per second
        throughput_aps = n_agents / elapsed if elapsed > 0 else float("inf")

        costs = _gpu_cost_estimate(trn_total_bytes, kv_total_bytes)

        print(
            f"{n_agents:>10}  "
            f"{'O(1)_TRN':>12}  "
            f"{trn_total_mb:>14.3f}  "
            f"{trn_per_agent_kb:>12.2f}  "
            f"{kv_total_mb:>12.1f}  "
            f"{kv_per_agent_mb:>10.3f}  "
            f"{memory_reduction:>8.1f}x  "
            f"{actual_peak_memory_mb:>10.3f}  "
            f"{throughput_aps:>14.1f}  "
            f"{costs['trn_gpu_count_a100']:>14}  "
            f"{costs['kv_gpu_count_a100']:>13}  "
            f"{costs['trn_a100_cost_usd']:>16.4f}"
        )

        rows.append(dict(
            n_agents=n_agents,
            memory_model="O(1)_TRN",
            model_config=model_config,
            model_params_mb=params_mb,
            trn_total_memory_mb=trn_total_mb,
            trn_per_agent_state_kb=trn_per_agent_kb,
            kv_total_memory_mb=kv_total_mb,
            kv_per_agent_mb=kv_per_agent_mb,
            memory_reduction_x=memory_reduction,
            actual_peak_memory_mb=actual_peak_memory_mb,
            gpu_vram_mb=gpu_vram_mb,
            throughput_agents_per_sec=throughput_aps,
            trn_gpu_count_a100=costs["trn_gpu_count_a100"],
            kv_gpu_count_a100=costs["kv_gpu_count_a100"],
            trn_a100_cost_usd_per_hr=costs["trn_a100_cost_usd"],
            kv_a100_cost_usd_per_hr=costs["kv_a100_cost_usd"],
            trn_h100_cost_usd_per_hr=costs["trn_h100_cost_usd"],
            kv_h100_cost_usd_per_hr=costs["kv_h100_cost_usd"],
            elapsed_sec=elapsed,
        ))

    print()

    # Summary
    if rows:
        max_agents = rows[-1]
        print("Summary:")
        print(
            f"  At {max_agents['n_agents']:,} agents: "
            f"TRN={max_agents['trn_total_memory_mb']:.1f}MB [{max_agents['memory_model']}] vs "
            f"KV={max_agents['kv_total_memory_mb']:.1f}MB [O(n)_KV] "
            f"({max_agents['memory_reduction_x']:.0f}x reduction)"
        )
        print(
            f"  Actual peak (tracemalloc): {max_agents['actual_peak_memory_mb']:.3f} MB"
        )
        print(
            f"  Throughput: {max_agents['throughput_agents_per_sec']:.1f} agents/sec"
        )
        print(
            f"  TRN GPU cost (A100): {max_agents['trn_gpu_count_a100']} GPU(s) "
            f"= ${max_agents['trn_a100_cost_usd_per_hr']:.2f}/hr"
        )
        print(
            f"  KV  GPU cost (A100): {max_agents['kv_gpu_count_a100']} GPU(s) "
            f"= ${max_agents['kv_a100_cost_usd_per_hr']:.2f}/hr"
        )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"\n  CSV saved -> {output_csv}")

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent TRN memory simulation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent-counts",
        type=str,
        default=",".join(str(c) for c in DEFAULT_AGENT_COUNTS),
        help="Comma-separated list of agent counts to simulate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="torch device string (cpu, cuda, cuda:0, ...)",
    )
    parser.add_argument(
        "--tokens-per-agent",
        type=int,
        default=DEFAULT_TOKENS_PER_AGENT,
        help="Number of tokens to feed each agent",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=DEFAULT_HISTORY_LEN,
        help="Equivalent KV cache context length for comparison",
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
    parser.add_argument(
        "--model-config",
        type=str,
        default="trn_100m",
        choices=MODEL_CONFIG_CHOICES,
        help="TRNConfig preset to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_counts = [
        int(x.strip()) for x in args.agent_counts.split(",") if x.strip()
    ]
    agent_counts = sorted(set(agent_counts))

    project_root = Path(__file__).resolve().parent.parent
    output_csv = (
        None if args.no_csv
        else project_root / "results" / "bench_multi_agent.csv"
    )

    run_benchmark(
        agent_counts=agent_counts,
        device_str=args.device,
        tokens_per_agent=args.tokens_per_agent,
        history_len=args.history_len,
        seed=args.seed,
        model_config=args.model_config,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
