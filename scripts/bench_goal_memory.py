#!/usr/bin/env python3
"""Salient vs Neutral Event Retention Benchmark.

Tests whether goal-aware saliency correctly differentiates
important events from neutral filler.

Setup:
  - Inject SALIENT events (high-value tokens at marked positions)
  - Inject NEUTRAL events (mid-range tokens at unmarked positions)
  - After processing, query both salient and neutral facts
  - Salient retention should be significantly higher

Metrics:
  - archive_retention_rate_salient
  - archive_retention_rate_neutral
  - retrieval_hit_salient
  - retrieval_hit_neutral
  - salient_neutral_gap (salient - neutral, should be > 0)

Output:
  artifacts/trimemory/{timestamp}/goal_memory_results.json
  artifacts/trimemory/{timestamp}/goal_memory_summary.md
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
from trimemory.integrations.vllm_backend import DualMemoryEngine
from trimemory.tri_memory import TriMemoryEngine

WINDOW_SIZE = 64
CHUNK_SIZE = 32
D_MODEL = 128
N_LAYERS = 4
N_OSC = 64
D_FF = 512
VOCAB_SIZE = 256
SEQ_LEN = 300
TRAIN_STEPS = 300
BATCH_SIZE = 16

# Token ranges
SALIENT_LOW = 220
SALIENT_HIGH = 245
NEUTRAL_LOW = 100
NEUTRAL_HIGH = 150
FILLER_LOW = 50
FILLER_HIGH = 100
QUERY_SALIENT = 5
QUERY_NEUTRAL = 6


class SalientNeutralDataset(Dataset):
    """Dataset with clearly salient and neutral events.

    Structure:
      [SALIENT_FACT@pos5, filler..., NEUTRAL_FACT@pos80, filler...,
       QUERY_SALIENT, SALIENT_ANSWER,
       QUERY_NEUTRAL, NEUTRAL_ANSWER]

    Salient facts use high-range tokens (easily detectable by saliency scorer).
    Neutral facts use mid-range tokens (low saliency score).
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

        # Salient fact: high-range token early in sequence
        salient_val = int(self.rng.integers(SALIENT_LOW, SALIENT_HIGH))
        salient_pos = 5
        seq[salient_pos] = salient_val

        # Neutral fact: mid-range token in middle of sequence
        neutral_val = int(self.rng.integers(NEUTRAL_LOW, NEUTRAL_HIGH))
        neutral_pos = 80
        seq[neutral_pos] = neutral_val

        # Fill with filler
        for i in range(T - 6):
            if seq[i] == 0:
                seq[i] = int(self.rng.integers(FILLER_LOW, FILLER_HIGH))

        # Queries at end
        q = T - 6
        seq[q] = QUERY_SALIENT
        seq[q + 1] = salient_val
        seq[q + 2] = QUERY_NEUTRAL
        seq[q + 3] = neutral_val
        seq[q + 4] = FILLER_LOW
        seq[q + 5] = FILLER_LOW

        return {
            "input_ids": torch.from_numpy(seq),
            "salient_val": salient_val,
            "neutral_val": neutral_val,
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
    salient_ok = neutral_ok = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            B = ids.size(0)
            logits = model(ids)["logits"]
            q = dataset.seq_len - 6

            s_preds = logits[:, q, :].argmax(-1).cpu()
            salient_ok += (s_preds == batch["salient_val"]).sum().item()

            n_preds = logits[:, q + 2, :].argmax(-1).cpu()
            neutral_ok += (n_preds == batch["neutral_val"]).sum().item()

            total += B

    salient_acc = salient_ok / max(total, 1)
    neutral_acc = neutral_ok / max(total, 1)
    gap = salient_acc - neutral_acc

    return {
        "salient_retention_acc": salient_acc,
        "neutral_retention_acc": neutral_acc,
        "salient_neutral_gap": gap,
        "n_eval": total,
    }


def make_cfg():
    return TRNConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=SEQ_LEN + 16,
    )


def build_models(cfg):
    return {
        "A_kv_only": DualMemoryEngine(cfg, window_size=WINDOW_SIZE),
        "B_kv_trn": DualMemoryEngine(cfg, window_size=WINDOW_SIZE),
        "C_kv_ret": TriMemoryEngine(cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE),
        "D_kv_trn_ret": TriMemoryEngine(cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE),
    }


def main():
    parser = argparse.ArgumentParser(description="Salient vs Neutral Benchmark")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"artifacts/trimemory/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg()
    dataset = SalientNeutralDataset(n_samples=2000, seq_len=SEQ_LEN, seed=args.seed)

    print(f"[Salient vs Neutral Benchmark] device={args.device} steps={args.steps}")
    results = {}
    models = build_models(cfg)

    for name, model in models.items():
        print(f"  Training {name}...")
        seed_everything(args.seed)
        t0 = time.perf_counter()
        loss_curve = train_model(model, dataset, args.steps, device)
        train_time = time.perf_counter() - t0
        eval_result = evaluate_model(model, dataset, device)
        results[name] = {**eval_result, "train_time_s": train_time, "loss_curve": loss_curve}
        print(f"    salient={eval_result['salient_retention_acc']:.3f}"
              f" neutral={eval_result['neutral_retention_acc']:.3f}"
              f" gap={eval_result['salient_neutral_gap']:.3f}")

    with open(out_dir / "goal_memory_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    md_lines = [
        "# Salient vs Neutral Event Retention Results",
        "",
        f"Timestamp: {timestamp}",
        "",
        "## Comparison",
        "",
        "| Model | Salient Acc | Neutral Acc | Gap |",
        "|-------|------------|-------------|-----|",
    ]
    for name, r in results.items():
        md_lines.append(
            f"| {name} | {r['salient_retention_acc']:.3f}"
            f" | {r['neutral_retention_acc']:.3f}"
            f" | {r['salient_neutral_gap']:.3f} |"
        )
    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "- Positive gap means salient events are better retained than neutral",
        "- Retrieval-enabled models (C/D) should show larger gap",
        "  (salient tokens trigger higher saliency -> more archival)",
        "- KV-only models may show gap via attention patterns alone",
    ])

    with open(out_dir / "goal_memory_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[DONE] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
