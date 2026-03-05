#!/usr/bin/env python3
"""Multi-agent memory simulation benchmark.

Simulates N independent agents each maintaining their own TRN state.
Measures actual memory via tracemalloc and compares against analytical
KV cache memory for equivalent context length.

Usage:
    python scripts/bench_multi_agent.py
    python scripts/bench_multi_agent.py --agent-counts 100,1000,10000 --device cpu --no-csv
"""
from __future__ import annotations

import argparse
import csv
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

# Benchmark configuration
DEFAULT_AGENT_COUNTS = [100, 1000, 10000]
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 42
DEFAULT_TOKENS_PER_AGENT = 64   # tokens fed to each agent
DEFAULT_HISTORY_LEN = 1000      # equivalent KV context length for comparison


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


def _simulate_agents_trn(
    cfg: TRNConfig,
    n_agents: int,
    tokens_per_agent: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    """Simulate n_agents each processing tokens_per_agent tokens with TRN.

    Returns (elapsed_seconds, actual_memory_bytes) measured via tracemalloc.
    """
    seed_everything(seed)
    model = TRNModel(cfg).to(device).eval()

    # Generate random token sequences for all agents
    rng_gen = torch.Generator()
    rng_gen.manual_seed(seed)
    tokens = torch.randint(
        0, cfg.vocab_size, (n_agents, tokens_per_agent),
        generator=rng_gen, device=device,
    )

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

    return elapsed, float(peak_bytes)


def _gpu_cost_estimate(memory_gb: float, hours: float = 1.0) -> dict[str, float]:
    """Estimate GPU cost for a given memory footprint and duration."""
    # Rough estimate: memory determines which GPU tier is needed
    # A100 80GB, H100 80GB
    a100_cost = A100_PRICE_PER_HR * hours
    h100_cost = H100_PRICE_PER_HR * hours
    return {
        "a100_cost_usd": a100_cost,
        "h100_cost_usd": h100_cost,
    }


def run_benchmark(
    agent_counts: list[int],
    device_str: str,
    tokens_per_agent: int,
    history_len: int,
    seed: int,
    output_csv: Optional[Path] = None,
) -> None:
    seed_everything(seed)
    device = torch.device(device_str)

    cfg = TRNConfig.trn_100m()

    trn_state_bytes = _analytical_trn_state_bytes(cfg)
    kv_bytes = _analytical_kv_cache_bytes(cfg, history_len)

    print("=" * 80)
    print("Multi-Agent Memory Simulation: TRN vs KV Cache")
    print("=" * 80)
    print(f"  Config: trn_100m ({cfg.d_model}d, {cfg.n_layers}L, {cfg.n_oscillators}K)")
    print(f"  Tokens per agent: {tokens_per_agent}")
    print(f"  Equivalent KV context length: {history_len}")
    print(f"  Device: {device_str}")
    print()
    print(f"  TRN state per agent (analytical): {trn_state_bytes / 1024:.2f} KB")
    print(f"  KV cache per agent @ T={history_len} (analytical): {kv_bytes / (1024 * 1024):.3f} MB")
    print(
        f"  Memory reduction per agent: "
        f"{kv_bytes / trn_state_bytes:.1f}x"
    )
    print()

    header = (
        f"{'n_agents':>10}  "
        f"{'trn_total_mb':>14}  "
        f"{'trn_per_kb':>12}  "
        f"{'kv_total_mb':>12}  "
        f"{'kv_per_mb':>10}  "
        f"{'reduction':>10}  "
        f"{'throughput_aps':>14}  "
        f"{'a100_usd/hr':>12}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []

    for n_agents in agent_counts:
        elapsed, actual_peak_bytes = _simulate_agents_trn(
            cfg, n_agents, tokens_per_agent, device, seed
        )

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
        # (n_agents * tokens_per_agent total tokens processed)
        throughput_aps = n_agents / elapsed if elapsed > 0 else float("inf")

        costs = _gpu_cost_estimate(kv_total_mb / 1024)

        print(
            f"{n_agents:>10}  "
            f"{trn_total_mb:>14.3f}  "
            f"{trn_per_agent_kb:>12.2f}  "
            f"{kv_total_mb:>12.1f}  "
            f"{kv_per_agent_mb:>10.3f}  "
            f"{memory_reduction:>10.1f}x  "
            f"{throughput_aps:>14.1f}  "
            f"{costs['a100_cost_usd']:>12.4f}"
        )

        rows.append(dict(
            n_agents=n_agents,
            trn_total_memory_mb=trn_total_mb,
            trn_per_agent_state_kb=trn_per_agent_kb,
            kv_total_memory_mb=kv_total_mb,
            kv_per_agent_mb=kv_per_agent_mb,
            memory_reduction_x=memory_reduction,
            throughput_agents_per_sec=throughput_aps,
            a100_cost_usd_per_hr=costs["a100_cost_usd"],
            h100_cost_usd_per_hr=costs["h100_cost_usd"],
            elapsed_sec=elapsed,
        ))

    print()

    # Summary
    if rows:
        max_agents = rows[-1]
        print("Summary:")
        print(
            f"  At {max_agents['n_agents']:,} agents: "
            f"TRN={max_agents['trn_total_memory_mb']:.1f}MB vs "
            f"KV={max_agents['kv_total_memory_mb']:.1f}MB "
            f"({max_agents['memory_reduction_x']:.0f}x reduction)"
        )
        print(
            f"  Throughput: {max_agents['throughput_agents_per_sec']:.1f} agents/sec"
        )
        print(
            f"  Cost (A100): ${max_agents['a100_cost_usd_per_hr']:.2f}/hr, "
            f"(H100): ${max_agents['h100_cost_usd_per_hr']:.2f}/hr"
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
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
