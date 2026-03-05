#!/usr/bin/env python3
"""Go/No-Go decision test for TRN architecture.

Runs two critical tasks to determine whether TRN can proceed to 100M scale:

1. Copy Task (5000 steps): Can TRN retain and reproduce past tokens?
   - PASS: loss < 0.01 or accuracy > 0.99 within 500 steps
   - MARGINAL: within 500-2000 steps
   - FAIL: loss > 1.0 or accuracy < 0.5 at 5000 steps

2. Selective Copy Task (5000 steps): Can TRN selectively remember marked tokens?
   - PASS: accuracy > 0.90 within 2000 steps
   - MARGINAL: within 2000-5000 steps
   - FAIL: accuracy < 0.3 at 5000 steps

Both tasks run TRN and Transformer baselines side by side.

Usage:
    cd scripts
    python go_nogo_test.py [--steps 5000] [--device cpu]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from trn.bench_data import seed_everything, NextTokenCopyDataset
from trn.config import TRNConfig
from trn.model import TRNModel
from trn.baseline import TransformerModel
from trn.scheduler import CosineWithWarmup

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────

LR = 3e-4
LR_MIN = 3e-5
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.1  # 10% warmup
BATCH_SIZE = 32
D_MODEL = 128
N_LAYERS = 4
N_OSC = 64
SEED = 42
EVAL_INTERVAL = 50


# ── Selective Copy Dataset (marker-in-noise) ─────────────────────────────────

class MarkerSelectiveCopyDataset(Dataset):
    """Markers interspersed with noise; reproduce markers in order after separator.

    Layout (seq_len tokens total):
        [front: half tokens with n_markers markers at random positions, rest noise]
        [SEP token]
        [back: markers in order, padded with noise]

    For next-token prediction, input_ids = seq[:-1], labels = seq[:-1].
    The model must learn to output markers in order at the separator boundary.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        seq_len: int = 128,
        n_markers: int = 8,
        vocab_size: int = 32,
        seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.n_markers = n_markers
        self.vocab_size = vocab_size
        self.sep_id = vocab_size - 1
        marker_hi = vocab_size // 2  # markers from [4, marker_hi)
        noise_lo = marker_hi         # noise from [marker_hi, sep_id)
        noise_hi = self.sep_id

        rng = random.Random(seed)

        half = seq_len // 2
        full_len = seq_len + 1  # +1 for next-token prediction shift
        back_len = full_len - half - 1  # -1 for separator

        all_seqs = []
        self.marker_start_positions = []  # position where markers begin in output

        for _ in range(n_samples):
            # Generate markers
            markers = [rng.randint(4, marker_hi - 1) for _ in range(n_markers)]

            # Front: noise with markers at random positions
            front = [rng.randint(noise_lo, noise_hi - 1) for _ in range(half)]
            positions = sorted(rng.sample(range(half), min(n_markers, half)))
            for i, pos in enumerate(positions):
                front[pos] = markers[i]

            # Back: markers in order + padding noise
            back = [rng.randint(noise_lo, noise_hi - 1) for _ in range(back_len)]
            for i in range(n_markers):
                if i < back_len:
                    back[i] = markers[i]

            seq = front + [self.sep_id] + back
            seq = seq[:full_len]
            all_seqs.append(seq)
            self.marker_start_positions.append(half + 1)  # after SEP

        self.data = torch.tensor(all_seqs, dtype=torch.long)
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        chunk = self.data[idx]
        return {
            "input_ids": chunk[:-1].clone(),
            "labels": chunk[:-1].clone(),
            "marker_start": self.marker_start_positions[idx],
        }


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_copy_task(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    period: int,
    max_batches: int = 50,
) -> tuple[float, float]:
    """Evaluate copy task. Returns (val_loss, accuracy).

    Accuracy = fraction of correctly predicted next tokens.
    For periodic copy, every position is predictable after seeing one period.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        total_loss += out["loss"].item()

        # Accuracy: predict next token for positions >= period
        logits = out["logits"]
        preds = logits.argmax(dim=-1)  # (B, seq_len)
        # Shifted: preds[:, t] predicts token at t+1
        # Compare preds[:, t] with ids[:, t+1] for t >= period-1
        targets = ids[:, 1:]  # (B, seq_len-1)
        pred_shifted = preds[:, :-1]  # (B, seq_len-1)
        # Only count positions after the first period
        start = max(period - 1, 0)
        if start < targets.size(1):
            correct = (pred_shifted[:, start:] == targets[:, start:]).sum().item()
            total_count = targets[:, start:].numel()
            total_correct += correct
            total_tokens += total_count

        n += 1

    model.train()
    avg_loss = total_loss / max(n, 1)
    accuracy = total_correct / max(total_tokens, 1)
    return avg_loss, accuracy


@torch.no_grad()
def eval_selective_copy(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_markers: int,
    max_batches: int = 50,
) -> tuple[float, float]:
    """Evaluate selective copy task. Returns (val_loss, accuracy).

    Accuracy = fraction of correctly predicted marker tokens after separator.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_markers = 0
    n = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        marker_start = batch["marker_start"][0].item()  # same for all in batch
        out = model(ids, labels=ids)
        total_loss += out["loss"].item()

        logits = out["logits"]
        preds = logits.argmax(dim=-1)  # (B, seq_len)

        # Marker accuracy: positions [marker_start, marker_start + n_markers)
        # preds[:, t] predicts token at t+1
        # So to predict the marker at position marker_start in the sequence,
        # we look at preds[:, marker_start - 1]
        for k in range(n_markers):
            pred_pos = marker_start - 1 + k  # pred position for marker k
            target_pos = marker_start + k     # target position in ids
            if pred_pos < preds.size(1) and target_pos < ids.size(1):
                correct = (preds[:, pred_pos] == ids[:, target_pos]).sum().item()
                total_correct += correct
                total_markers += ids.size(0)

        n += 1

    model.train()
    avg_loss = total_loss / max(n, 1)
    accuracy = total_correct / max(total_markers, 1)
    return avg_loss, accuracy


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_steps: int,
    device: str,
    label: str,
    eval_fn,
    eval_kwargs: dict,
) -> list[dict]:
    """Train model for n_steps. Returns list of eval records."""
    model.to(device).train()
    warmup_steps = max(1, int(n_steps * WARMUP_FRAC))

    param_groups = model.configure_optimizer_param_groups(WEIGHT_DECAY)
    opt = torch.optim.AdamW(param_groups, lr=LR, betas=BETAS)
    sched = CosineWithWarmup(opt, warmup_steps=warmup_steps, max_steps=n_steps, lr=LR, min_lr=LR_MIN)

    loader_iter = iter(train_loader)
    records: list[dict] = []
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        sched.step(step)
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        loss = out["loss"]

        if not torch.isfinite(loss):
            print(f"  [{label}] step {step}: non-finite loss, skipping", flush=True)
            opt.zero_grad()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad()

        if step % EVAL_INTERVAL == 0 or step == n_steps:
            val_loss, accuracy = eval_fn(model, val_loader, device, **eval_kwargs)
            elapsed = time.perf_counter() - t0
            records.append({
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "accuracy": accuracy,
            })
            print(
                f"  [{label}] step {step:5d}/{n_steps} "
                f"train={loss.item():.4f} val={val_loss:.4f} acc={accuracy:.4f} "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    return records


# ── Task runners ─────────────────────────────────────────────────────────────

def run_copy_task(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """Run copy task for TRN and Transformer. Returns (trn_records, tf_records)."""
    print("\n" + "=" * 60, flush=True)
    print("  COPY TASK  (seq_len=64, vocab=32, period=8)", flush=True)
    print("=" * 60, flush=True)

    period = 8
    train_ds = NextTokenCopyDataset(n_samples=2000, seq_len=64, vocab_size=32, period=period, seed=SEED)
    val_ds = NextTokenCopyDataset(n_samples=500, seq_len=64, vocab_size=32, period=period, seed=SEED + 1000)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    cfg = TRNConfig(
        vocab_size=32, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_MODEL * 4, max_seq_len=72,
    )

    # TRN
    print("\n--- TRN ---", flush=True)
    seed_everything(SEED)
    trn = TRNModel(cfg)
    trn_params = sum(p.numel() for p in trn.parameters())
    print(f"  TRN params: {trn_params:,}", flush=True)
    trn_records = train_model(
        trn, train_loader, val_loader, args.steps, args.device,
        "TRN-copy", eval_copy_task, {"period": period},
    )

    # Transformer
    print("\n--- Transformer ---", flush=True)
    seed_everything(SEED)
    tf = TransformerModel(cfg)
    tf_params = sum(p.numel() for p in tf.parameters())
    print(f"  TF params: {tf_params:,}", flush=True)
    tf_records = train_model(
        tf, train_loader, val_loader, args.steps, args.device,
        "TF-copy", eval_copy_task, {"period": period},
    )

    return trn_records, tf_records


def run_selective_copy_task(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """Run selective copy task for TRN and Transformer."""
    print("\n" + "=" * 60, flush=True)
    print("  SELECTIVE COPY TASK  (seq_len=128, markers=8, vocab=32)", flush=True)
    print("=" * 60, flush=True)

    n_markers = 8
    train_ds = MarkerSelectiveCopyDataset(
        n_samples=10000, seq_len=128, n_markers=n_markers,
        vocab_size=32, seed=SEED,
    )
    val_ds = MarkerSelectiveCopyDataset(
        n_samples=1000, seq_len=128, n_markers=n_markers,
        vocab_size=32, seed=SEED + 1000,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    cfg = TRNConfig(
        vocab_size=32, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_MODEL * 4, max_seq_len=136,
    )

    # TRN
    print("\n--- TRN ---", flush=True)
    seed_everything(SEED)
    trn = TRNModel(cfg)
    trn_records = train_model(
        trn, train_loader, val_loader, args.steps, args.device,
        "TRN-selcopy", eval_selective_copy, {"n_markers": n_markers},
    )

    # Transformer
    print("\n--- Transformer ---", flush=True)
    seed_everything(SEED)
    tf = TransformerModel(cfg)
    tf_records = train_model(
        tf, train_loader, val_loader, args.steps, args.device,
        "TF-selcopy", eval_selective_copy, {"n_markers": n_markers},
    )

    return trn_records, tf_records


# ── Verdict ──────────────────────────────────────────────────────────────────

def find_threshold_step(records: list[dict], key: str, threshold: float, direction: str = "below") -> str:
    """Find first step where metric crosses threshold."""
    for r in records:
        if direction == "below" and r[key] < threshold:
            return str(r["step"])
        if direction == "above" and r[key] > threshold:
            return str(r["step"])
    return "not reached"


def judge_copy(records: list[dict]) -> str:
    """Judge copy task verdict."""
    loss_step = find_threshold_step(records, "val_loss", 0.01, "below")
    acc_step = find_threshold_step(records, "accuracy", 0.99, "above")

    # Find earliest success step
    earliest = None
    for s in [loss_step, acc_step]:
        if s != "not reached":
            v = int(s)
            if earliest is None or v < earliest:
                earliest = v

    if earliest is not None and earliest <= 500:
        return "PASS"

    if earliest is not None and earliest <= 2000:
        return "MARGINAL"

    final = records[-1] if records else {"val_loss": float("inf"), "accuracy": 0.0}
    if final["val_loss"] > 1.0 or final["accuracy"] < 0.5:
        return "FAIL"

    return "MARGINAL"


def judge_selective_copy(records: list[dict]) -> str:
    """Judge selective copy task verdict."""
    acc_step = find_threshold_step(records, "accuracy", 0.90, "above")

    if acc_step != "not reached" and int(acc_step) <= 2000:
        return "PASS"

    if acc_step != "not reached" and int(acc_step) <= 5000:
        return "MARGINAL"

    final = records[-1] if records else {"accuracy": 0.0}
    if final["accuracy"] < 0.3:
        return "FAIL"

    return "MARGINAL"


def print_task_results(
    task_name: str,
    trn_records: list[dict],
    tf_records: list[dict],
    judge_fn,
    loss_threshold: float = 0.01,
    acc_threshold: float = 0.99,
) -> str:
    """Print results for one task. Returns TRN verdict."""

    trn_final = trn_records[-1] if trn_records else {"val_loss": float("inf"), "accuracy": 0.0}
    tf_final = tf_records[-1] if tf_records else {"val_loss": float("inf"), "accuracy": 0.0}

    trn_loss_step = find_threshold_step(trn_records, "val_loss", loss_threshold, "below")
    trn_acc_step = find_threshold_step(trn_records, "accuracy", acc_threshold, "above")
    tf_loss_step = find_threshold_step(tf_records, "val_loss", loss_threshold, "below")
    tf_acc_step = find_threshold_step(tf_records, "accuracy", acc_threshold, "above")

    verdict = judge_fn(trn_records)

    print(f"\n=== {task_name} ===", flush=True)
    print(f"Model: TRN", flush=True)
    print(f"Steps to loss < {loss_threshold}: {trn_loss_step}", flush=True)
    print(f"Steps to accuracy > {acc_threshold}: {trn_acc_step}", flush=True)
    print(f"Final val_loss: {trn_final['val_loss']:.6f}", flush=True)
    print(f"Final accuracy: {trn_final['accuracy']:.6f}", flush=True)
    print(f"Verdict: {verdict}", flush=True)
    print(flush=True)
    print(f"Model: Transformer", flush=True)
    print(f"Steps to loss < {loss_threshold}: {tf_loss_step}", flush=True)
    print(f"Steps to accuracy > {acc_threshold}: {tf_acc_step}", flush=True)
    print(f"Final val_loss: {tf_final['val_loss']:.6f}", flush=True)
    print(f"Final accuracy: {tf_final['accuracy']:.6f}", flush=True)

    return verdict


def save_csv(records: list[dict], path: str) -> None:
    """Save records to CSV."""
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="TRN Go/No-Go Decision Test")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    global SEED
    SEED = args.seed

    print("=" * 60, flush=True)
    print("  TRN Go/No-Go Decision Test", flush=True)
    print(f"  Steps: {args.steps}, Device: {args.device}, Seed: {args.seed}", flush=True)
    print(f"  Model: d_model={D_MODEL}, n_layers={N_LAYERS}, n_osc={N_OSC}", flush=True)
    print("=" * 60, flush=True)

    t_start = time.perf_counter()

    # Task 1: Copy
    trn_copy, tf_copy = run_copy_task(args)
    save_csv(trn_copy, "results/go_nogo_copy_trn.csv")
    save_csv(tf_copy, "results/go_nogo_copy_tf.csv")

    # Task 2: Selective Copy
    trn_selcopy, tf_selcopy = run_selective_copy_task(args)
    save_csv(trn_selcopy, "results/go_nogo_selcopy_trn.csv")
    save_csv(tf_selcopy, "results/go_nogo_selcopy_tf.csv")

    total_time = time.perf_counter() - t_start

    # Print results
    copy_verdict = print_task_results(
        "COPY TASK", trn_copy, tf_copy, judge_copy,
        loss_threshold=0.01, acc_threshold=0.99,
    )

    selcopy_verdict = print_task_results(
        "SELECTIVE COPY TASK", trn_selcopy, tf_selcopy, judge_selective_copy,
        loss_threshold=0.01, acc_threshold=0.90,
    )

    # Overall
    print(f"\n{'=' * 60}", flush=True)
    print("=== OVERALL GO/NO-GO ===", flush=True)
    print(f"Copy Task:      {copy_verdict}", flush=True)
    print(f"Selective Copy: {selcopy_verdict}", flush=True)

    if copy_verdict == "PASS" and selcopy_verdict in ("PASS", "MARGINAL"):
        recommendation = "GO"
    elif copy_verdict == "FAIL" or selcopy_verdict == "FAIL":
        recommendation = "NO-GO"
    else:
        recommendation = "CONDITIONAL GO"

    print(f"Recommendation: {recommendation}", flush=True)
    print(f"Total time: {total_time:.0f}s", flush=True)
    print(f"{'=' * 60}", flush=True)

    return 0 if recommendation == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
