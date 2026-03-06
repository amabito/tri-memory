#!/usr/bin/env python3
"""Long Conversation Mixed Memory Benchmark.

Consumer-oriented demo: simulates a long conversation where:
  - recent dialogue consistency -> KV
  - session topic drift / latent state -> TRN
  - old explicit fact / preference recall -> Retrieval

Comparisons: A/B/C/D (KV only / KV+TRN / KV+Ret / KV+TRN+Ret)

Output:
  artifacts/trimemory/{timestamp}/conversation_results.json
  artifacts/trimemory/{timestamp}/conversation_summary.md
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

from trn.bench_data import seed_everything
from trn.config import TRNConfig
from trn.integrations.vllm_backend import DualMemoryEngine
from trn.tri_memory import TriMemoryEngine

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
TOPIC_LOW = 10
TOPIC_HIGH = 50
DIALOGUE_LOW = 50
DIALOGUE_HIGH = 150
PREFERENCE_LOW = 200
PREFERENCE_HIGH = 230
FACT_LOW = 230
FACT_HIGH = 250
QUERY_CONSISTENCY = 5
QUERY_TOPIC = 6
QUERY_OLD_FACT = 7


class ConversationDataset(Dataset):
    """Synthetic long conversation with three memory requirements.

    Structure:
      [OLD_PREF, topic_A_tokens..., topic_shift, topic_B_tokens...,
       recent_dialogue...,
       QUERY_OLD_FACT, OLD_PREF_ANSWER,
       QUERY_CONSISTENCY, RECENT_ANSWER,
       QUERY_TOPIC, TOPIC_ANSWER]

    - OLD_PREF: user preference stated early (requires retrieval)
    - Topic drift: gradual shift in token distribution (requires TRN)
    - Recent dialogue: last few exchanges (requires KV)
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

        # Old preference at position 2
        pref_val = int(self.rng.integers(PREFERENCE_LOW, PREFERENCE_HIGH))
        seq[2] = pref_val

        # Topic A: positions 5-100
        topic_a = int(self.rng.integers(TOPIC_LOW, TOPIC_HIGH))
        for i in range(5, 100):
            if self.rng.random() < 0.3:
                seq[i] = topic_a
            else:
                seq[i] = int(self.rng.integers(DIALOGUE_LOW, DIALOGUE_HIGH))

        # Topic B: positions 120-250 (shift)
        topic_b = int(self.rng.integers(TOPIC_LOW, TOPIC_HIGH))
        while topic_b == topic_a:
            topic_b = int(self.rng.integers(TOPIC_LOW, TOPIC_HIGH))
        for i in range(120, T - 10):
            if self.rng.random() < 0.3:
                seq[i] = topic_b
            else:
                seq[i] = int(self.rng.integers(DIALOGUE_LOW, DIALOGUE_HIGH))

        # Transition zone 100-120: mix of both
        for i in range(100, 120):
            blend = (i - 100) / 20.0
            if self.rng.random() < blend:
                seq[i] = topic_b if self.rng.random() < 0.3 else int(self.rng.integers(DIALOGUE_LOW, DIALOGUE_HIGH))
            else:
                seq[i] = topic_a if self.rng.random() < 0.3 else int(self.rng.integers(DIALOGUE_LOW, DIALOGUE_HIGH))

        # Recent distinctive value
        recent_val = int(self.rng.integers(FACT_LOW, FACT_HIGH))
        seq[T - 20] = recent_val

        # Fill zeros with filler
        for i in range(T - 8):
            if seq[i] == 0:
                seq[i] = int(self.rng.integers(DIALOGUE_LOW, DIALOGUE_HIGH))

        # Queries
        q = T - 8
        seq[q] = QUERY_OLD_FACT
        seq[q + 1] = pref_val
        seq[q + 2] = QUERY_CONSISTENCY
        seq[q + 3] = recent_val
        seq[q + 4] = QUERY_TOPIC
        seq[q + 5] = topic_b  # current topic
        seq[q + 6] = DIALOGUE_LOW
        seq[q + 7] = DIALOGUE_LOW

        return {
            "input_ids": torch.from_numpy(seq),
            "pref_val": pref_val,
            "recent_val": recent_val,
            "topic_answer": topic_b,
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
    pref_ok = recent_ok = topic_ok = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            B = ids.size(0)
            logits = model(ids)["logits"]
            q = dataset.seq_len - 8

            pref_preds = logits[:, q, :].argmax(-1).cpu()
            pref_ok += (pref_preds == batch["pref_val"]).sum().item()

            recent_preds = logits[:, q + 2, :].argmax(-1).cpu()
            recent_ok += (recent_preds == batch["recent_val"]).sum().item()

            topic_preds = logits[:, q + 4, :].argmax(-1).cpu()
            topic_ok += (topic_preds == batch["topic_answer"]).sum().item()

            total += B

    pref_acc = pref_ok / max(total, 1)
    recent_acc = recent_ok / max(total, 1)
    topic_acc = topic_ok / max(total, 1)
    accs = [pref_acc, recent_acc, topic_acc]
    composite = len(accs) / sum(1 / max(a, 1e-6) for a in accs) if any(a > 0 for a in accs) else 0.0

    return {
        "old_fact_acc": pref_acc,
        "recent_consistency_acc": recent_acc,
        "topic_state_acc": topic_acc,
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
    }


def main():
    parser = argparse.ArgumentParser()
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
    dataset = ConversationDataset(n_samples=2000, seq_len=SEQ_LEN, seed=args.seed)

    print(f"[Conversation Benchmark] device={args.device} steps={args.steps}")
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
        print(f"    old_fact={eval_result['old_fact_acc']:.3f}"
              f" recent={eval_result['recent_consistency_acc']:.3f}"
              f" topic={eval_result['topic_state_acc']:.3f}"
              f" composite={eval_result['composite_score']:.3f}")

    # Save
    with open(out_dir / "conversation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    md_lines = [
        "# Long Conversation Benchmark Results",
        "",
        f"Timestamp: {timestamp}",
        "",
        "## 4-Way Comparison",
        "",
        "| Model | Old Fact | Recent | Topic State | Composite |",
        "|-------|----------|--------|-------------|-----------|",
    ]
    for name, r in results.items():
        md_lines.append(
            f"| {name} | {r['old_fact_acc']:.3f} | {r['recent_consistency_acc']:.3f}"
            f" | {r['topic_state_acc']:.3f} | {r['composite_score']:.3f} |"
        )

    with open(out_dir / "conversation_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[DONE] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
