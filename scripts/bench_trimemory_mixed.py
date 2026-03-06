#!/usr/bin/env python3
"""Mixed Memory Benchmark: 4-way comparison of memory architectures.

Tests tasks that require ALL three memory types simultaneously:
  - recent exact recall (KV window)
  - long-range pattern detection (TRN state)
  - old exact fact retrieval (Retrieval index)

Comparisons:
  A: KV only (TransformerModel with window mask)
  B: KV + TRN (DualMemoryEngine)
  C: KV + Retrieval (TriMemoryEngine with TRN gate zeroed)
  D: KV + TRN + Retrieval (TriMemoryEngine full)

Output:
  artifacts/trimemory/{timestamp}/mixed_results.json
  artifacts/trimemory/{timestamp}/mixed_summary.md

Usage:
    python scripts/bench_trimemory_mixed.py
    python scripts/bench_trimemory_mixed.py --device cuda --steps 500
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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from trn.bench_data import seed_everything
from trn.config import TRNConfig
from trn.tri_memory import TriMemoryEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 64
CHUNK_SIZE = 32
D_MODEL = 128
N_LAYERS = 4
N_OSC = 64
D_FF = 512
VOCAB_SIZE = 256
SEQ_LEN = 256   # long enough to require all memory types
TRAIN_STEPS = 300
BATCH_SIZE = 16
LR = 3e-4

# Task token ranges
RECENT_VALUE_LOW = 200
RECENT_VALUE_HIGH = 220
OLD_FACT_LOW = 220
OLD_FACT_HIGH = 240
PATTERN_TOKEN_LOW = 10
PATTERN_TOKEN_HIGH = 100
FILLER_LOW = 100
FILLER_HIGH = 200
QUERY_RECENT = 5
QUERY_OLD_FACT = 6
QUERY_PATTERN = 7


# ---------------------------------------------------------------------------
# Mixed Memory Dataset
# ---------------------------------------------------------------------------

class MixedMemoryDataset(Dataset):
    """Synthetic dataset requiring all three memory types.

    Sequence structure:
      [OLD_FACT, filler, PATTERN_BLOCK, filler, RECENT_VALUES,
       QUERY_OLD_FACT, OLD_FACT_ANSWER,
       QUERY_RECENT, RECENT_ANSWER,
       QUERY_PATTERN, PATTERN_ANSWER]

    - OLD_FACT: single high-value token placed early (requires retrieval)
    - PATTERN_BLOCK: repeated pattern (requires TRN for long-range trend)
    - RECENT_VALUES: last few values before queries (requires KV window)
    """

    def __init__(
        self,
        n_samples: int = 2000,
        seq_len: int = 256,
        old_fact_distance: int = 150,
        pattern_period: int = 8,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.old_fact_distance = old_fact_distance
        self.pattern_period = pattern_period
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        seq = torch.randint(
            FILLER_LOW, FILLER_HIGH, (self.seq_len,), generator=self.rng
        )

        # Old fact at position 0
        old_fact = torch.randint(
            OLD_FACT_LOW, OLD_FACT_HIGH, (1,), generator=self.rng
        ).item()
        seq[0] = old_fact

        # Pattern block: positions 30-90, repeated pattern
        pattern_len = self.pattern_period
        pattern = torch.randint(
            PATTERN_TOKEN_LOW, PATTERN_TOKEN_HIGH,
            (pattern_len,), generator=self.rng,
        )
        for i in range(30, 90):
            seq[i] = pattern[i % pattern_len]
        # Pattern answer: the dominant token or next in pattern
        pattern_answer = pattern[0].item()

        # Recent values: positions -20 to -10
        recent_val = torch.randint(
            RECENT_VALUE_LOW, RECENT_VALUE_HIGH, (1,), generator=self.rng
        ).item()
        seq[-20] = recent_val

        # Query positions (last 6 tokens)
        query_start = self.seq_len - 6
        seq[query_start] = QUERY_OLD_FACT
        seq[query_start + 1] = old_fact         # answer
        seq[query_start + 2] = QUERY_RECENT
        seq[query_start + 3] = recent_val       # answer
        seq[query_start + 4] = QUERY_PATTERN
        seq[query_start + 5] = pattern_answer   # answer

        return {
            "input_ids": seq,
            "old_fact": old_fact,
            "recent_val": recent_val,
            "pattern_answer": pattern_answer,
        }


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_model(
    model: torch.nn.Module,
    dataset: MixedMemoryDataset,
    steps: int,
    device: torch.device,
    lr: float = LR,
) -> list[dict]:
    """Train a model on mixed-memory task. Returns loss curve."""
    model = model.to(device).train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    records = []
    loader_it = iter(loader)
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

        if step % 50 == 0 or step == steps:
            records.append({"step": step, "loss": loss.item()})

    return records


def evaluate_model(
    model: torch.nn.Module,
    dataset: MixedMemoryDataset,
    device: torch.device,
    n_eval: int = 200,
) -> dict:
    """Evaluate model on mixed-memory queries.

    Returns accuracy for each memory type:
      - recent_exact_acc: query at QUERY_RECENT position
      - old_fact_acc: query at QUERY_OLD_FACT position
      - long_pattern_acc: query at QUERY_PATTERN position
      - composite_score: harmonic mean of all three
    """
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=min(32, n_eval), shuffle=False)

    recent_correct = 0
    old_fact_correct = 0
    pattern_correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            B = ids.size(0)
            out = model(ids)
            logits = out["logits"]  # (B, T, V)

            query_start = dataset.seq_len - 6

            # Old fact: logits at QUERY_OLD_FACT position -> predict old_fact
            old_preds = logits[:, query_start, :].argmax(dim=-1).cpu()
            old_fact_correct += (old_preds == batch["old_fact"]).sum().item()

            # Recent: logits at QUERY_RECENT position -> predict recent_val
            recent_preds = logits[:, query_start + 2, :].argmax(dim=-1).cpu()
            recent_correct += (recent_preds == batch["recent_val"]).sum().item()

            # Pattern: logits at QUERY_PATTERN position -> predict pattern_answer
            pattern_preds = logits[:, query_start + 4, :].argmax(dim=-1).cpu()
            pattern_correct += (pattern_preds == batch["pattern_answer"]).sum().item()

            total += B

    recent_acc = recent_correct / max(total, 1)
    old_fact_acc = old_fact_correct / max(total, 1)
    pattern_acc = pattern_correct / max(total, 1)

    # Harmonic mean (avoids one zero killing everything)
    accs = [recent_acc, old_fact_acc, pattern_acc]
    nonzero = [a for a in accs if a > 0]
    if nonzero:
        composite = len(accs) / sum(1.0 / max(a, 1e-6) for a in accs)
    else:
        composite = 0.0

    return {
        "recent_exact_acc": recent_acc,
        "old_fact_acc": old_fact_acc,
        "long_pattern_acc": pattern_acc,
        "composite_score": composite,
        "n_eval": total,
    }


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_cfg() -> TRNConfig:
    return TRNConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_oscillators=N_OSC,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN + 16,
    )


def build_models(cfg: TRNConfig) -> dict:
    """Build all 4 comparison models using the same base architecture.

    All models use TriMemoryEngine with different paths enabled/disabled
    for a fair ablation comparison:
      A: KV only (TRN disabled, Retrieval disabled)
      B: KV + TRN (Retrieval disabled)
      C: KV + Retrieval (TRN disabled)
      D: KV + TRN + Retrieval (all enabled)
    """
    return {
        "A_kv_only": TriMemoryEngine(
            cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE,
            enable_trn=False, enable_retrieval=False,
        ),
        "B_kv_trn": TriMemoryEngine(
            cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE,
            enable_trn=True, enable_retrieval=False,
        ),
        "C_kv_ret": TriMemoryEngine(
            cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE,
            enable_trn=False, enable_retrieval=True,
        ),
        "D_kv_trn_ret": TriMemoryEngine(
            cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE,
            enable_trn=True, enable_retrieval=True,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed Memory Benchmark")
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
    dataset = MixedMemoryDataset(n_samples=2000, seq_len=SEQ_LEN, seed=args.seed)

    print(f"[Mixed Memory Benchmark] device={args.device} steps={args.steps}")
    print(f"  KV window={WINDOW_SIZE} chunk_size={CHUNK_SIZE}")
    print(f"  seq_len={SEQ_LEN} vocab={VOCAB_SIZE}")
    print()

    results = {}
    models = build_models(cfg)

    for name, model in models.items():
        print(f"  Training {name}...")
        seed_everything(args.seed)
        t0 = time.perf_counter()
        loss_curve = train_model(model, dataset, args.steps, device)
        train_time = time.perf_counter() - t0

        print(f"    Train time: {train_time:.1f}s, final loss: {loss_curve[-1]['loss']:.4f}")

        eval_result = evaluate_model(model, dataset, device)
        n_params = model.num_parameters()

        # Memory stats
        mem_bytes = model.total_memory_bytes()

        results[name] = {
            **eval_result,
            "final_loss": loss_curve[-1]["loss"],
            "train_time_s": train_time,
            "n_params": n_params,
            "memory_bytes": mem_bytes,
            "loss_curve": loss_curve,
        }

        print(f"    recent_exact={eval_result['recent_exact_acc']:.3f}"
              f"  old_fact={eval_result['old_fact_acc']:.3f}"
              f"  pattern={eval_result['long_pattern_acc']:.3f}"
              f"  composite={eval_result['composite_score']:.3f}")
        print()

    # Save results
    # Strip loss_curve for summary (keep in full results)
    summary = {}
    for name, r in results.items():
        summary[name] = {k: v for k, v in r.items() if k != "loss_curve"}

    with open(out_dir / "mixed_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate summary markdown
    md_lines = [
        "# Mixed Memory Benchmark Results",
        "",
        f"Timestamp: {timestamp}",
        f"Device: {args.device}",
        f"Steps: {args.steps}",
        "",
        "## 4-Way Comparison",
        "",
        "| Model | Recent Exact | Old Fact | Pattern | Composite | Memory (KB) |",
        "|-------|-------------|----------|---------|-----------|-------------|",
    ]
    for name, r in summary.items():
        md_lines.append(
            f"| {name} | {r['recent_exact_acc']:.3f} | {r['old_fact_acc']:.3f} "
            f"| {r['long_pattern_acc']:.3f} | {r['composite_score']:.3f} "
            f"| {r['memory_bytes'] / 1024:.1f} |"
        )
    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "- A (KV only): baseline, good at recent, poor at old facts beyond window",
        "- B (KV+TRN): adds pattern memory, should improve pattern score",
        "- C (KV+Ret): adds retrieval, should improve old fact score",
        "- D (KV+TRN+Ret): should be best composite if architecture works",
        "",
        "## Limitations",
        "",
        "- TRN is NOT a content-addressable memory",
        "- TRN is NOT a Transformer replacement",
        "- Retrieval is NOT always-on (gated by router)",
    ])

    with open(out_dir / "mixed_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"[DONE] Results saved to {out_dir}")

    # Print final comparison table
    print("\n=== COMPARISON TABLE ===")
    print(f"{'Model':<20} {'Recent':>8} {'OldFact':>8} {'Pattern':>8} {'Composite':>10}")
    print("-" * 60)
    for name, r in summary.items():
        print(f"{name:<20} {r['recent_exact_acc']:>8.3f} {r['old_fact_acc']:>8.3f}"
              f" {r['long_pattern_acc']:>8.3f} {r['composite_score']:>10.3f}")


if __name__ == "__main__":
    main()
