#!/usr/bin/env python3
"""Long-context memory benchmark: 4 synthetic tasks requiring distant token retrieval.

Tasks:
  1. induction_head    -- [A, B, noise(L-4), A] -> predict B
  2. assoc_recall      -- K key-value pairs + query -> retrieve value
  3. multi_needle      -- 2 needles + query token -> retrieve requested needle
  4. copy_distractors  -- interleaved real/distractor, predict next real token

Model: tiny (d_model=64, n_layers=2, K=32) for CPU speed.
TRN context lengths: 128, 256, 512, 1024, 2048, 4096
TF  context lengths: 128, 256, 512, 1024  (O(n^2) makes 2048+ impractical on CPU)

Usage:
    python scripts/bench_memory_tasks.py [--tasks all] [--steps 300] [--batch-size 8]
                                         [--trn-lens 128,256,512,1024,2048,4096]
                                         [--tf-lens 128,256,512,1024]
                                         [--device cpu]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse, csv, time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.baseline import TransformerModel
from trimemory.bench_data import seed_everything

# Token ranges -- keep separated to avoid collisions
A_LOW, A_HIGH = 10, 50          # 40 possible A tokens
B_OFFSET       = 50              # B = A + B_OFFSET -> B in [60, 110)
NOISE_LOW, NOISE_HIGH = 120, 200
KEY_LOW, KEY_HIGH  = 10, 50
VAL_LOW, VAL_HIGH  = 50, 100
NEEDLE_LOW, NEEDLE_HIGH = 200, 240
FILLER_LOW, FILLER_HIGH = 10, 180
REAL_LOW, REAL_HIGH = 10, 60
DIST_LOW, DIST_HIGH = 100, 200


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class InductionHeadDataset(Dataset):
    """[A, B, noise(L-4), A, PAD] -> predict B at last position.

    Length = context_len. Layout:
      pos 0:   A
      pos 1:   B (= A + B_OFFSET)
      pos 2..L-3: noise
      pos L-2: A  (repeat)
      pos L-1: PAD (query -- model should predict B here)
    Labels: 0 everywhere except pos L-1 = B.
    """
    def __init__(self, context_len: int, n_examples: int, seed: int = 42) -> None:
        assert context_len >= 6, "context_len must be >= 6"
        rng = torch.Generator(); rng.manual_seed(seed)
        A = torch.randint(A_LOW, A_HIGH, (n_examples,), generator=rng)
        B = A + B_OFFSET
        noise = torch.randint(NOISE_LOW, NOISE_HIGH, (n_examples, context_len - 4), generator=rng)

        ids = torch.zeros(n_examples, context_len, dtype=torch.long)
        ids[:, 0] = A
        ids[:, 1] = B
        ids[:, 2:context_len-2] = noise
        ids[:, context_len-2] = A   # repeat A
        ids[:, context_len-1] = 4   # query token (PAD/placeholder)

        labels = torch.zeros_like(ids)
        labels[:, context_len-1] = B

        self.input_ids = ids
        self.labels    = labels
        self.B         = B

    def __len__(self) -> int: return len(self.input_ids)
    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i],
                "target": self.B[i]}


class AssocRecallDataset(Dataset):
    """K key-value pairs + SEP + query_key + PAD -> predict value.

    K=4 pairs, total content = 2*K + 2 = 10 tokens.
    Rest of context_len filled with noise.
    """
    K = 4

    def __init__(self, context_len: int, n_examples: int, seed: int = 42) -> None:
        min_len = 2 * self.K + 3  # 11 tokens
        assert context_len >= min_len, f"context_len must be >= {min_len}"
        rng = torch.Generator(); rng.manual_seed(seed)

        keys    = torch.randint(KEY_LOW, KEY_HIGH,  (n_examples, self.K), generator=rng)
        vals    = torch.randint(VAL_LOW, VAL_HIGH,  (n_examples, self.K), generator=rng)
        q_idx   = torch.randint(0, self.K,          (n_examples,),        generator=rng)
        noise   = torch.randint(NOISE_LOW, NOISE_HIGH, (n_examples, context_len), generator=rng)

        ids    = noise.clone()
        labels = torch.zeros(n_examples, context_len, dtype=torch.long)

        for i in range(n_examples):
            pos = 0
            for j in range(self.K):
                ids[i, pos]   = keys[i, j]
                ids[i, pos+1] = vals[i, j]
                pos += 2
            ids[i, pos]   = 3    # SEP
            ids[i, pos+1] = keys[i, q_idx[i]]  # query key
            ids[i, pos+2] = 4    # placeholder -> predict value
            labels[i, pos+2] = vals[i, q_idx[i]]

        self.input_ids = ids
        self.labels    = labels
        self.targets   = vals[torch.arange(n_examples), q_idx]
        self._last_pos = 2 * self.K + 2  # position of the answer

    def __len__(self) -> int: return len(self.input_ids)
    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i],
                "target": self.targets[i], "last_pos": self._last_pos}


class MultiNeedleDataset(Dataset):
    """2 needles N0, N1 at random positions. Query token selects which needle to recall.

    Layout: [filler..., N0/N1 at random positions, filler..., QUERY, PAD]
    QUERY = 5 (want N0) or 6 (want N1).
    Last position label = selected needle.
    """
    def __init__(self, context_len: int, n_examples: int, seed: int = 42) -> None:
        assert context_len >= 6
        rng = torch.Generator(); rng.manual_seed(seed)

        needles = torch.randint(NEEDLE_LOW, NEEDLE_HIGH, (n_examples, 2), generator=rng)
        filler  = torch.randint(FILLER_LOW, FILLER_HIGH, (n_examples, context_len), generator=rng)
        which   = torch.randint(0, 2, (n_examples,), generator=rng)  # 0=want N0, 1=want N1

        # Random positions for N0, N1 in [1, context_len-3]
        p0 = torch.randint(1, context_len-3, (n_examples,), generator=rng)
        p1 = (p0 + torch.randint(1, context_len//4, (n_examples,), generator=rng)) % (context_len-3) + 1

        ids = filler.clone()
        labels = torch.zeros(n_examples, context_len, dtype=torch.long)

        for i in range(n_examples):
            ids[i, p0[i]] = needles[i, 0]
            ids[i, p1[i]] = needles[i, 1]
            ids[i, context_len-2] = 5 + which[i].item()  # query: 5=want N0, 6=want N1
            ids[i, context_len-1] = 4    # placeholder
            labels[i, context_len-1] = needles[i, which[i].item()]

        self.input_ids = ids
        self.labels    = labels
        self.targets   = needles[torch.arange(n_examples), which]

    def __len__(self) -> int: return len(self.input_ids)
    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i],
                "target": self.targets[i]}


class CopyDistractorsDataset(Dataset):
    """Even positions: real tokens [10,60). Odd positions: distractors [100,200).
    Label at each even position t: real token at position t+2 (next real token).
    Label at odd positions: 0 (ignored in metric).
    """
    def __init__(self, context_len: int, n_examples: int, seed: int = 42) -> None:
        assert context_len % 2 == 0, "context_len must be even"
        rng = torch.Generator(); rng.manual_seed(seed)

        n_real = context_len // 2
        real = torch.randint(REAL_LOW, REAL_HIGH, (n_examples, n_real), generator=rng)
        dist = torch.randint(DIST_LOW, DIST_HIGH, (n_examples, n_real), generator=rng)

        # Interleave: even=real, odd=distractors
        ids = torch.zeros(n_examples, context_len, dtype=torch.long)
        ids[:, 0::2] = real   # even positions
        ids[:, 1::2] = dist   # odd positions

        # Labels: at each even position t, label = real token at position t+2
        # At last even position (context_len-2), label = 0 (no next real)
        labels = torch.zeros_like(ids)
        labels[:, 0::2] = torch.roll(real, -1, dims=1)  # shift real tokens left by 1
        labels[:, -2]   = 0   # last even position: no next real token (don't predict)

        # Which positions matter for metric: even positions 0..context_len-4
        self.eval_positions = list(range(0, context_len-2, 2))

        self.input_ids = ids
        self.labels    = labels
        self.real      = real

    def __len__(self) -> int: return len(self.input_ids)
    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


# ---------------------------------------------------------------------------
# Model forward helpers
# ---------------------------------------------------------------------------

def get_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Get logits from model. Handles both {logits, loss} and {loss}-only outputs."""
    out = model(input_ids)
    if "logits" in out:
        return out["logits"]  # (B, L, V)
    # If no logits key, re-run without labels trick: some models need labels=None explicitly
    try:
        out2 = model(input_ids, labels=None)
        if "logits" in out2:
            return out2["logits"]
    except Exception:
        pass
    raise RuntimeError(
        f"Model {type(model).__name__} does not return 'logits'. "
        f"Available keys: {list(out.keys())}"
    )


def recall_at_last(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    last_pos: int = -1,
) -> float:
    """Compute recall@1 at the last sequence position (or specified position)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            ids     = batch["input_ids"].to(device)
            targets = batch["target"].to(device)
            logits  = get_logits(model, ids)      # (B, L, V)
            preds   = logits[:, last_pos].argmax(-1)  # (B,)
            correct += (preds == targets).sum().item()
            total   += ids.size(0)
    model.train()
    return correct / max(total, 1)


def accuracy_copy_distractors(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    eval_positions: list[int],
) -> float:
    """Accuracy on even (real-token) positions only."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = get_logits(model, ids)  # (B, L, V)
            for pos in eval_positions:
                tgt  = labels[:, pos]          # (B,)
                mask = tgt > 0                 # ignore pad labels
                if mask.sum() == 0:
                    continue
                pred = logits[:, pos].argmax(-1)
                correct += (pred[mask] == tgt[mask]).sum().item()
                total   += mask.sum().item()
    model.train()
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "induction_head":   InductionHeadDataset,
    "assoc_recall":     AssocRecallDataset,
    "multi_needle":     MultiNeedleDataset,
    "copy_distractors": CopyDistractorsDataset,
}


def _collate(batch: list[dict]) -> dict:
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = torch.tensor(vals)
    return result


def make_loaders(task: str, context_len: int, batch_size: int, seed: int):
    cls = TASK_REGISTRY[task]
    train_ds = cls(context_len, n_examples=512, seed=seed)
    val_ds   = cls(context_len, n_examples=128, seed=seed + 99)
    kw = dict(batch_size=batch_size, drop_last=True, collate_fn=_collate)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    return train_loader, val_loader, train_ds


def train_and_eval(
    task: str,
    model_name: str,
    context_len: int,
    n_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> float:
    seed_everything(seed)

    cfg = TRNConfig(
        vocab_size    = 256,
        d_model       = 64,
        n_oscillators = 32,
        n_layers      = 2,
        d_ff          = 256,
        max_seq_len   = context_len + 16,
    )

    train_loader, val_loader, train_ds = make_loaders(task, context_len, batch_size, seed)
    model = (TRNModel(cfg) if model_name == "TRN" else TransformerModel(cfg)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    data_iter = cycle(train_loader)

    model.train()
    optimizer.zero_grad()
    for _ in range(n_steps):
        batch = next(data_iter)
        ids   = batch["input_ids"].to(device)
        out   = model(ids, labels=ids)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Evaluation
    if task == "copy_distractors":
        return accuracy_copy_distractors(model, val_loader, device, train_ds.eval_positions)
    elif task == "assoc_recall":
        last_pos = train_ds._last_pos if hasattr(train_ds, '_last_pos') else -1
        return recall_at_last(model, val_loader, device, last_pos=last_pos)
    else:
        return recall_at_last(model, val_loader, device, last_pos=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_NAMES = ["induction_head", "assoc_recall", "multi_needle", "copy_distractors"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",    type=str, default="all")
    parser.add_argument("--steps",    type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--device",   type=str, default="cpu")
    parser.add_argument("--trn-lens", type=str, default="128,256,512,1024,2048,4096")
    parser.add_argument("--tf-lens",  type=str, default="128,256,512,1024")
    parser.add_argument("--output-dir", type=str, default="scripts/results")
    args = parser.parse_args()

    tasks     = TASK_NAMES if args.tasks == "all" else args.tasks.split(",")
    trn_lens  = [int(x) for x in args.trn_lens.split(",")]
    tf_lens   = [int(x) for x in args.tf_lens.split(",")]
    out_dir   = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    random_baseline = {
        "induction_head":   1 / (A_HIGH - A_LOW),            # 1/40 = 2.5%
        "assoc_recall":     1 / (VAL_HIGH - VAL_LOW),        # 1/50 = 2.0%
        "multi_needle":     1 / (NEEDLE_HIGH - NEEDLE_LOW),  # 1/40 = 2.5%
        "copy_distractors": 1 / (REAL_HIGH - REAL_LOW),      # 1/50 = 2.0%
    }

    print(f"Memory Tasks Benchmark -- {args.steps} steps, lr={args.lr}, bs={args.batch_size}")
    print(f"Model: d_model=64, n_layers=2, K=32 (tiny, CPU-fast)")
    print()

    rows = []
    for task in tasks:
        print(f"=== {task} (random baseline: {random_baseline[task]*100:.1f}%) ===")
        print(f"{'ctx_len':>8} | {'TRN recall':>12} | {'TF recall':>12}")
        print("-" * 38)

        all_lens = sorted(set(trn_lens + tf_lens))
        for ctx_len in all_lens:
            trn_acc = tf_acc = None
            if ctx_len in trn_lens:
                t0 = time.perf_counter()
                trn_acc = train_and_eval(
                    task, "TRN", ctx_len, args.steps,
                    args.batch_size, args.lr, args.seed, args.device,
                )
                trn_t = time.perf_counter() - t0
            if ctx_len in tf_lens:
                t0 = time.perf_counter()
                tf_acc = train_and_eval(
                    task, "TF", ctx_len, args.steps,
                    args.batch_size, args.lr, args.seed, args.device,
                )
                tf_t = time.perf_counter() - t0

            trn_str = f"{trn_acc*100:>10.1f}%" if trn_acc is not None else f"{'(GPU only)':>11}"
            tf_str  = f"{tf_acc*100:>10.1f}%"  if tf_acc  is not None else f"{'(O(n^2) skip)':>13}"
            print(f"{ctx_len:>8} | {trn_str} | {tf_str}")

            if trn_acc is not None:
                rows.append({"task": task, "model": "TRN", "context_len": ctx_len,
                             "recall_at_1": trn_acc, "n_steps": args.steps})
            if tf_acc is not None:
                rows.append({"task": task, "model": "TF", "context_len": ctx_len,
                             "recall_at_1": tf_acc, "n_steps": args.steps})
        print()

    print("Note: TF omitted at context_len>1024 -- O(n^2) attention makes CPU training impractical.")
    print("      Run with --device cuda for TF at longer contexts.")

    csv_path = out_dir / "bench_memory_tasks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "context_len", "recall_at_1", "n_steps"])
        for r in rows:
            w.writerow([r["task"], r["model"], r["context_len"],
                        f"{r['recall_at_1']:.4f}", r["n_steps"]])
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
