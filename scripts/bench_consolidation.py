#!/usr/bin/env python3
"""Consolidation Benefit Benchmark.

Tests whether replay-based consolidation improves:
  1. Pattern retention (TRN state quality after consolidation)
  2. Retrieval call reduction (fewer lookups needed)
  3. Saliency calibration (better scoring after re-evaluation)
  4. Memory constancy (stable performance over time)

Compares:
  D: KV + TRN + Retrieval (no consolidation)
  F: KV + TRN + Retrieval + Consolidation

Both use the same TriMemoryEngine. F runs consolidation passes
after training to measure improvement on held-out queries.

Output:
  artifacts/trimemory/{timestamp}/consolidation_results.json
  artifacts/trimemory/{timestamp}/consolidation_summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_src = os.path.join(os.path.dirname(__file__), "..", "src")
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _root]
sys.path.insert(0, _src)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.consolidation import ArchiveReweighter, ReplayConsolidator
from trimemory.integrations.vllm_backend import DualMemoryEngine
from trimemory.tri_memory import TriMemoryEngine

WINDOW_SIZE = 64
CHUNK_SIZE = 32
D_MODEL = 128
N_LAYERS = 4
N_OSC = 64
D_FF = 512
VOCAB_SIZE = 256
SEQ_LEN = 256
TRAIN_STEPS = 300
BATCH_SIZE = 16

# Token ranges (same as mixed benchmark)
PATTERN_LOW = 10
PATTERN_HIGH = 80
FACT_LOW = 220
FACT_HIGH = 245
FILLER_LOW = 80
FILLER_HIGH = 200
QUERY_PATTERN = 5
QUERY_FACT = 6


class ConsolidationDataset(Dataset):
    """Dataset designed to benefit from consolidation.

    Contains:
      - Repeated patterns (TRN should absorb after consolidation)
      - Old facts (retrieval targets)
      - Queries for both

    The key test: after consolidation, pattern recall should improve
    even if some retrieval chunks are pruned.
    """

    def __init__(self, n_samples=2000, seq_len=SEQ_LEN, seed=42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        T = self.seq_len
        seq = np.zeros(T, dtype=np.int64)

        # Old fact at position 3
        fact_val = int(self.rng.integers(FACT_LOW, FACT_HIGH))
        seq[3] = fact_val

        # Pattern: positions 20-120
        period = self.rng.integers(4, 10)
        pattern_val = int(self.rng.integers(PATTERN_LOW, PATTERN_HIGH))
        for i in range(20, 120):
            if i % period == 0:
                seq[i] = pattern_val
            else:
                seq[i] = int(self.rng.integers(FILLER_LOW, FILLER_HIGH))

        # More filler 120-240
        for i in range(120, T - 8):
            seq[i] = int(self.rng.integers(FILLER_LOW, FILLER_HIGH))

        # Queries
        q = T - 8
        seq[q] = QUERY_PATTERN
        seq[q + 1] = pattern_val
        seq[q + 2] = QUERY_FACT
        seq[q + 3] = fact_val
        seq[q + 4] = FILLER_LOW
        seq[q + 5] = FILLER_LOW
        seq[q + 6] = FILLER_LOW
        seq[q + 7] = FILLER_LOW

        # Fill remaining zeros
        for i in range(T):
            if seq[i] == 0:
                seq[i] = int(self.rng.integers(FILLER_LOW, FILLER_HIGH))

        return {
            "input_ids": torch.from_numpy(seq),
            "pattern_val": pattern_val,
            "fact_val": fact_val,
        }


def train_model(model, dataset, steps, device, lr=3e-4):
    model = model.to(device).train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader_it = iter(loader)
    records = []
    for step in range(1, steps + 1):
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)
        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        loss = out["loss"]
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % 50 == 0:
            records.append({"step": step, "loss": loss.item()})
    return records


def evaluate_model(model, dataset, device, n_eval=200):
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=min(32, n_eval), shuffle=False)
    pattern_ok = fact_ok = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            B = ids.size(0)
            logits = model(ids)["logits"]
            q = dataset.seq_len - 8

            p_preds = logits[:, q, :].argmax(-1).cpu()
            pattern_ok += (p_preds == batch["pattern_val"]).sum().item()

            f_preds = logits[:, q + 2, :].argmax(-1).cpu()
            fact_ok += (f_preds == batch["fact_val"]).sum().item()

            total += B

    pattern_acc = pattern_ok / max(total, 1)
    fact_acc = fact_ok / max(total, 1)

    return {
        "pattern_retention_acc": pattern_acc,
        "fact_recall_acc": fact_acc,
        "composite_score": 2 * pattern_acc * fact_acc / max(pattern_acc + fact_acc, 1e-6),
        "n_eval": total,
    }


def run_consolidation(model, n_passes=3):
    """Run consolidation on a TriMemoryEngine's retrieval index.

    Returns consolidation stats.
    """
    if not isinstance(model, TriMemoryEngine):
        return {"consolidation": "not_applicable"}

    consolidator = ReplayConsolidator(replay_budget=16, prune_threshold=0.15)
    reweighter = ArchiveReweighter()

    stats_list = []
    for pass_idx in range(n_passes):
        # Select and replay chunks
        chunks = consolidator.select_replay_chunks(model.retrieval_index)

        # Simple scorer function for re-scoring
        def scorer_fn(token_ids, goal_state=None):
            return model.saliency_archiver.score(token_ids)

        stats = consolidator.rescore_and_prune(model.retrieval_index, scorer_fn)

        # Apply frequency boost (simulate retrieval hits)
        reweighter.apply_frequency_boost(model.retrieval_index)

        stats_list.append({
            "pass": pass_idx + 1,
            "chunks_replayed": stats.chunks_replayed,
            "chunks_pruned": stats.chunks_pruned,
            "avg_saliency_before": stats.avg_saliency_before,
            "avg_saliency_after": stats.avg_saliency_after,
            "remaining_chunks": len(model.retrieval_index),
        })

    return {
        "consolidation_passes": stats_list,
        "final_chunks": len(model.retrieval_index),
    }


def make_cfg():
    return TRNConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=SEQ_LEN + 16,
    )


def main():
    parser = argparse.ArgumentParser(description="Consolidation Benefit Benchmark")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--consolidation-passes", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"artifacts/trimemory/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg()
    dataset = ConsolidationDataset(n_samples=2000, seq_len=SEQ_LEN, seed=args.seed)

    print(f"[Consolidation Benchmark] device={args.device} steps={args.steps}")

    results = {}

    # D: Tri-Memory without consolidation
    print("  Training D_no_consolidation...")
    seed_everything(args.seed)
    model_d = TriMemoryEngine(cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE)
    t0 = time.perf_counter()
    loss_d = train_model(model_d, dataset, args.steps, device)
    train_time_d = time.perf_counter() - t0
    eval_d = evaluate_model(model_d, dataset, device)
    results["D_no_consolidation"] = {
        **eval_d, "train_time_s": train_time_d, "loss_curve": loss_d,
    }
    print(f"    pattern={eval_d['pattern_retention_acc']:.3f}"
          f" fact={eval_d['fact_recall_acc']:.3f}"
          f" composite={eval_d['composite_score']:.3f}")

    # F: Tri-Memory with consolidation
    print("  Training F_with_consolidation...")
    seed_everything(args.seed)
    model_f = TriMemoryEngine(cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE)
    t0 = time.perf_counter()
    loss_f = train_model(model_f, dataset, args.steps, device)
    train_time_f = time.perf_counter() - t0

    # Run consolidation passes on model_f
    print(f"  Running {args.consolidation_passes} consolidation passes...")
    consolidation_stats = run_consolidation(model_f, n_passes=args.consolidation_passes)

    eval_f = evaluate_model(model_f, dataset, device)
    results["F_with_consolidation"] = {
        **eval_f, "train_time_s": train_time_f, "loss_curve": loss_f,
        "consolidation": consolidation_stats,
    }
    print(f"    pattern={eval_f['pattern_retention_acc']:.3f}"
          f" fact={eval_f['fact_recall_acc']:.3f}"
          f" composite={eval_f['composite_score']:.3f}")

    # Compute deltas
    pattern_delta = eval_f["pattern_retention_acc"] - eval_d["pattern_retention_acc"]
    fact_delta = eval_f["fact_recall_acc"] - eval_d["fact_recall_acc"]
    composite_delta = eval_f["composite_score"] - eval_d["composite_score"]

    results["deltas"] = {
        "pattern_retention_gain": pattern_delta,
        "fact_recall_gain": fact_delta,
        "composite_gain": composite_delta,
    }

    with open(out_dir / "consolidation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    md_lines = [
        "# Consolidation Benefit Benchmark Results",
        "",
        f"Timestamp: {timestamp}",
        f"Consolidation passes: {args.consolidation_passes}",
        "",
        "## Comparison",
        "",
        "| Model | Pattern | Fact | Composite |",
        "|-------|---------|------|-----------|",
        f"| D (no consol.) | {eval_d['pattern_retention_acc']:.3f}"
        f" | {eval_d['fact_recall_acc']:.3f} | {eval_d['composite_score']:.3f} |",
        f"| F (w/ consol.) | {eval_f['pattern_retention_acc']:.3f}"
        f" | {eval_f['fact_recall_acc']:.3f} | {eval_f['composite_score']:.3f} |",
        "",
        "## Deltas",
        "",
        f"- Pattern retention gain: {pattern_delta:+.3f}",
        f"- Fact recall gain: {fact_delta:+.3f}",
        f"- Composite gain: {composite_delta:+.3f}",
        "",
        "## Consolidation Stats",
        "",
    ]
    if "consolidation_passes" in consolidation_stats:
        for s in consolidation_stats["consolidation_passes"]:
            md_lines.append(
                f"- Pass {s['pass']}: pruned={s['chunks_pruned']}"
                f" saliency {s['avg_saliency_before']:.3f} -> {s['avg_saliency_after']:.3f}"
                f" remaining={s['remaining_chunks']}"
            )

    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "- Consolidation should improve pattern retention (TRN absorbs patterns)",
        "- Consolidation may slightly reduce fact recall (pruned chunks)",
        "- Net composite should be positive or neutral",
        "- Consolidation effect is stronger with more passes and longer sequences",
    ])

    with open(out_dir / "consolidation_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[DONE] Results saved to {out_dir}")
    print(f"\n=== DELTAS ===")
    print(f"  Pattern: {pattern_delta:+.3f}")
    print(f"  Fact:    {fact_delta:+.3f}")
    print(f"  Composite: {composite_delta:+.3f}")


if __name__ == "__main__":
    main()
