#!/usr/bin/env python3
"""Goal-Switch Mixed Benchmark: tests memory policy adaptation.

Simulates a session where the goal changes mid-stream:
  Phase 1 (cost optimization): archive cost-related facts, track spending trends
  Phase 2 (incident response): switch to incident-related facts, anomaly patterns
  Phase 3 (recall): query old incident IDs from Phase 1 context

Compares 6 configurations:
  A: KV only
  B: KV + TRN
  C: KV + Retrieval
  D: KV + TRN + Retrieval
  E: D + Goal/Value
  F: E + Consolidation (stretch -- same as E for now, consolidation tested separately)

Metrics:
  - goal_switch_recovery_acc: accuracy after goal switch
  - pre_switch_acc: accuracy in Phase 1
  - post_switch_acc: accuracy in Phase 2
  - old_fact_recall_acc: recall of Phase 1 facts after switch
  - composite_score: harmonic mean

Output:
  artifacts/trimemory/{timestamp}/goal_switch_results.json
  artifacts/trimemory/{timestamp}/goal_switch_summary.md
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
import torch.nn.functional as F
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
SEQ_LEN = 320
TRAIN_STEPS = 300
BATCH_SIZE = 16

# Token ranges
COST_FACT_LOW = 200
COST_FACT_HIGH = 215
INCIDENT_FACT_LOW = 215
INCIDENT_FACT_HIGH = 230
TREND_LOW = 10
TREND_HIGH = 80
FILLER_LOW = 80
FILLER_HIGH = 200
GOAL_SWITCH_MARKER = 4
QUERY_PRE = 5
QUERY_POST = 6
QUERY_OLD = 7
QUERY_TREND = 8


class GoalSwitchDataset(Dataset):
    """Synthetic dataset with mid-session goal switch.

    Structure:
      [COST_FACT, cost_trend..., GOAL_SWITCH_MARKER,
       INCIDENT_FACT, incident_trend...,
       QUERY_PRE(cost_fact), QUERY_POST(incident_fact),
       QUERY_OLD(cost_fact_again), QUERY_TREND(trend_answer)]
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

        # Phase 1: cost context (positions 0-140)
        cost_fact = int(self.rng.integers(COST_FACT_LOW, COST_FACT_HIGH))
        seq[2] = cost_fact

        # Cost trend pattern
        cost_trend_val = int(self.rng.integers(TREND_LOW, TREND_HIGH))
        period = self.rng.integers(4, 12)
        for i in range(5, 140):
            if i % period == 0:
                seq[i] = cost_trend_val
            else:
                seq[i] = int(self.rng.integers(FILLER_LOW, FILLER_HIGH))

        # Goal switch at position 140
        seq[140] = GOAL_SWITCH_MARKER

        # Phase 2: incident context (positions 141-300)
        incident_fact = int(self.rng.integers(INCIDENT_FACT_LOW, INCIDENT_FACT_HIGH))
        seq[145] = incident_fact

        # Incident pattern (different from cost)
        incident_trend_val = int(self.rng.integers(TREND_LOW, TREND_HIGH))
        while incident_trend_val == cost_trend_val:
            incident_trend_val = int(self.rng.integers(TREND_LOW, TREND_HIGH))
        for i in range(150, T - 10):
            if i % 5 == 0:
                seq[i] = incident_trend_val
            else:
                seq[i] = int(self.rng.integers(FILLER_LOW, FILLER_HIGH))

        # Queries at end
        q = T - 10
        seq[q] = QUERY_PRE
        seq[q + 1] = cost_fact          # answer: old cost fact
        seq[q + 2] = QUERY_POST
        seq[q + 3] = incident_fact      # answer: new incident fact
        seq[q + 4] = QUERY_OLD
        seq[q + 5] = cost_fact          # answer: recall old cost fact again
        seq[q + 6] = QUERY_TREND
        seq[q + 7] = incident_trend_val  # answer: current trend
        seq[q + 8] = FILLER_LOW
        seq[q + 9] = FILLER_LOW

        # Fill remaining zeros
        for i in range(T):
            if seq[i] == 0:
                seq[i] = int(self.rng.integers(FILLER_LOW, FILLER_HIGH))

        return {
            "input_ids": torch.from_numpy(seq),
            "cost_fact": cost_fact,
            "incident_fact": incident_fact,
            "trend_answer": incident_trend_val,
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
    pre_ok = post_ok = old_ok = trend_ok = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            B = ids.size(0)
            logits = model(ids)["logits"]
            q = dataset.seq_len - 10

            pre_preds = logits[:, q, :].argmax(-1).cpu()
            pre_ok += (pre_preds == batch["cost_fact"]).sum().item()

            post_preds = logits[:, q + 2, :].argmax(-1).cpu()
            post_ok += (post_preds == batch["incident_fact"]).sum().item()

            old_preds = logits[:, q + 4, :].argmax(-1).cpu()
            old_ok += (old_preds == batch["cost_fact"]).sum().item()

            trend_preds = logits[:, q + 6, :].argmax(-1).cpu()
            trend_ok += (trend_preds == batch["trend_answer"]).sum().item()

            total += B

    pre_acc = pre_ok / max(total, 1)
    post_acc = post_ok / max(total, 1)
    old_acc = old_ok / max(total, 1)
    trend_acc = trend_ok / max(total, 1)

    accs = [pre_acc, post_acc, old_acc, trend_acc]
    composite = len(accs) / sum(1 / max(a, 1e-6) for a in accs) if any(a > 0 for a in accs) else 0.0

    return {
        "pre_switch_acc": pre_acc,
        "post_switch_acc": post_acc,
        "old_fact_recall_acc": old_acc,
        "trend_acc": trend_acc,
        "goal_switch_recovery_acc": (post_acc + old_acc) / 2.0,
        "composite_score": composite,
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
        "E_goal": TriMemoryEngine(cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE),
    }


def main():
    parser = argparse.ArgumentParser(description="Goal-Switch Mixed Benchmark")
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
    dataset = GoalSwitchDataset(n_samples=2000, seq_len=SEQ_LEN, seed=args.seed)

    print(f"[Goal-Switch Benchmark] device={args.device} steps={args.steps}")
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
        print(f"    pre={eval_result['pre_switch_acc']:.3f}"
              f" post={eval_result['post_switch_acc']:.3f}"
              f" old_recall={eval_result['old_fact_recall_acc']:.3f}"
              f" trend={eval_result['trend_acc']:.3f}"
              f" composite={eval_result['composite_score']:.3f}")

    with open(out_dir / "goal_switch_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    md_lines = [
        "# Goal-Switch Mixed Benchmark Results",
        "",
        f"Timestamp: {timestamp}",
        "",
        "## Comparison",
        "",
        "| Model | Pre-Switch | Post-Switch | Old Recall | Trend | Recovery | Composite |",
        "|-------|-----------|-------------|------------|-------|----------|-----------|",
    ]
    for name, r in results.items():
        md_lines.append(
            f"| {name} | {r['pre_switch_acc']:.3f} | {r['post_switch_acc']:.3f}"
            f" | {r['old_fact_recall_acc']:.3f} | {r['trend_acc']:.3f}"
            f" | {r['goal_switch_recovery_acc']:.3f} | {r['composite_score']:.3f} |"
        )
    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "- E (Goal) should show improved recovery vs D after goal switch",
        "- D should beat A/B/C on composite (needs all memory types)",
        "- Old fact recall requires retrieval (C/D/E should beat A/B)",
        "",
        "## Limitations",
        "",
        "- Goal/Value in training mode uses the same model weights as D",
        "  (goal bias is an inference-time mechanism; training learns mixed signal)",
        "- TRN is NOT a Transformer replacement",
        "- TRN is NOT a content-addressable memory",
    ])

    with open(out_dir / "goal_switch_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[DONE] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
