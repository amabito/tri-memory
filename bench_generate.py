#!/usr/bin/env python3
"""Long-context generation benchmark: TRN vs Transformer.

Measures tokens/sec and peak memory at increasing sequence lengths.
TRN uses step_single (O(1) per step); Transformer uses naive O(n^2) loop.

Usage:
    python bench_generate.py
    python bench_generate.py --lengths 128 256 512 1024 2048
    python bench_generate.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.generate import generate, GenerationConfig
from trimemory.model import TRNModel
from trimemory.baseline import TransformerModel


DEFAULT_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
PROMPT_LEN = 8
BATCH_SIZE = 1
WARMUP_TOKENS = 32


def _make_cfg(vocab_size: int = 256) -> TRNConfig:
    return TRNConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_oscillators=64,
        n_layers=4,
        d_ff=512,
        max_seq_len=max(DEFAULT_LENGTHS) + PROMPT_LEN + 4,
    )


def _transformer_generate(
    model: TransformerModel,
    prompt: torch.Tensor,
    gen_len: int,
    device: str,
) -> None:
    """Naive O(n^2) autoregressive generation for Transformer."""
    cfg = model.cfg
    ids = prompt.clone()
    with torch.no_grad():
        for _ in range(gen_len):
            out = model(ids)
            next_tok = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_tok], dim=1)
            if ids.shape[1] > cfg.max_seq_len:
                ids = ids[:, -cfg.max_seq_len :]


def _measure(fn, n_warmup: int = 1) -> tuple[float, float]:
    """Returns (elapsed_seconds, peak_bytes)."""
    # Warmup
    for _ in range(n_warmup):
        fn()

    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak


def bench_generation(
    lengths: list[int],
    device: str = "cpu",
    seed: int = 42,
) -> list[dict]:
    seed_everything(seed)
    cfg = _make_cfg()
    vocab = cfg.vocab_size

    trn = TRNModel(cfg).to(device).eval()
    transformer = TransformerModel(cfg).to(device).eval()

    rows = []
    for gen_len in lengths:
        prompt = torch.randint(0, vocab, (BATCH_SIZE, PROMPT_LEN), device=device)

        gen_cfg = GenerationConfig(max_new_tokens=gen_len, temperature=1.0, do_sample=False)

        def trn_fn():
            generate(trn, prompt.clone(), gen_cfg=gen_cfg, device=device)

        def tf_fn():
            _transformer_generate(transformer, prompt.clone(), gen_len, device)

        trn_elapsed, trn_mem = _measure(trn_fn)
        tf_elapsed, tf_mem = _measure(tf_fn)

        trn_tps = (gen_len * BATCH_SIZE) / trn_elapsed
        tf_tps = (gen_len * BATCH_SIZE) / tf_elapsed
        speedup = trn_tps / tf_tps

        rows.append({
            "gen_len": gen_len,
            "trn_tps": trn_tps,
            "tf_tps": tf_tps,
            "speedup": speedup,
            "trn_mem_kb": trn_mem / 1024,
            "tf_mem_kb": tf_mem / 1024,
        })
        print(
            f"  gen_len={gen_len:5d} | "
            f"TRN {trn_tps:8.0f} tps | "
            f"TF {tf_tps:8.0f} tps | "
            f"speedup {speedup:.2f}x | "
            f"TRN mem {trn_mem/1024:.0f} KB | "
            f"TF mem {tf_mem/1024:.0f} KB"
        )
    return rows


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'='*80}")
    print("Long-Context Generation Benchmark: TRN vs Transformer")
    print(f"{'='*80}")
    print(f"{'gen_len':>8} {'TRN tps':>10} {'TF tps':>10} {'speedup':>8} {'TRN mem KB':>12} {'TF mem KB':>10}")
    print(f"{'-'*80}")
    for r in rows:
        print(
            f"{r['gen_len']:>8d} "
            f"{r['trn_tps']:>10.0f} "
            f"{r['tf_tps']:>10.0f} "
            f"{r['speedup']:>8.2f}x "
            f"{r['trn_mem_kb']:>12.0f} "
            f"{r['tf_mem_kb']:>10.0f}"
        )
    print(f"{'='*80}\n")
    # Acceptance check: TRN must beat Transformer for long seqs
    long_rows = [r for r in rows if r["gen_len"] >= 1024]
    if long_rows:
        avg_speedup = sum(r["speedup"] for r in long_rows) / len(long_rows)
        status = "[PASS]" if avg_speedup > 1.0 else "[FAIL]"
        print(f"{status} TRN avg speedup at gen_len>=1024: {avg_speedup:.2f}x (must be >1.0x)")
        # Memory flatness: TRN memory should not grow with sequence length
        mem_ratio = long_rows[-1]["trn_mem_kb"] / max(long_rows[0]["trn_mem_kb"], 1.0)
        mem_status = "[PASS]" if mem_ratio < 3.0 else "[WARN]"
        print(f"{mem_status} TRN memory growth from {long_rows[0]['gen_len']} to {long_rows[-1]['gen_len']}: {mem_ratio:.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-context generation benchmark")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Lengths: {args.lengths}\n")
    rows = bench_generation(args.lengths, args.device, args.seed)
    print_summary(rows)


if __name__ == "__main__":
    main()
