#!/usr/bin/env python3
"""Deterministic 5k-step training benchmark: TRN vs Transformer across 4 tasks.

Tasks:
  copy         -- NextTokenCopyDataset (periodic sequence, period=8)
  counting     -- CountingDataset (predict running count)
  reverse      -- ReverseDataset (reverse the second half)
  induction    -- InductionHeadDataset (two-hop bigram retrieval)
  assoc_recall -- AssociativeRecallDataset (K key-value pairs)
  all          -- Run all 5 tasks

Usage:
    python scripts/bench_train.py [--tasks all] [--steps 5000] [--batch-size 32]
                                  [--d-model 128] [--n-layers 4] [--n-osc 64]
                                  [--seed 42] [--output-dir scripts/results]

Exit codes:
    0: all tasks pass (TRN/TF loss ratio <= 1.20 at final step)
    1: any task fails
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse, csv, time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from trn.bench_data import (
    seed_everything,
    CountingDataset,
    ReverseDataset,
    InductionHeadDataset,
    AssociativeRecallDataset,
    NextTokenCopyDataset,
)
from trn.config import TRNConfig
from trn.model import TRNModel
from trn.baseline import TransformerModel
from trn.scheduler import CosineWithWarmup


# Fixed hyperparameters
LR = 3e-4
LR_MIN = 3e-5
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 500
LOG_STEPS = [100, 500, 1000, 2500, 5000]

TASK_NAMES = ["copy", "counting", "reverse", "induction", "assoc_recall"]


@dataclass
class TaskResult:
    task: str
    model_name: str
    loss_at_steps: dict[int, float]   # step -> val_loss
    final_val_loss: float


def _make_dataset(task: str, seed: int) -> tuple[Dataset, Dataset, int]:
    """Returns (train_ds, val_ds, vocab_size)."""
    if task == "copy":
        train_ds = NextTokenCopyDataset(n_samples=2000, seq_len=64, vocab_size=32, period=8, seed=seed)
        val_ds   = NextTokenCopyDataset(n_samples=500,  seq_len=64, vocab_size=32, period=8, seed=seed + 1000)
        return train_ds, val_ds, 32

    elif task == "counting":
        train_ds = CountingDataset(vocab_size=64, seq_len=16, n_examples=2000, seed=seed)
        val_ds   = CountingDataset(vocab_size=64, seq_len=16, n_examples=500,  seed=seed + 1000)
        return train_ds, val_ds, 64

    elif task == "reverse":
        train_ds = ReverseDataset(vocab_size=64, seq_len=16, n_examples=2000, seed=seed)
        val_ds   = ReverseDataset(vocab_size=64, seq_len=16, n_examples=500,  seed=seed + 1000)
        return train_ds, val_ds, 64

    elif task == "induction":
        train_ds = InductionHeadDataset(vocab_size=64, seq_len=32, n_examples=2000, seed=seed)
        val_ds   = InductionHeadDataset(vocab_size=64, seq_len=32, n_examples=500,  seed=seed + 1000)
        return train_ds, val_ds, 64

    elif task == "assoc_recall":
        train_ds = AssociativeRecallDataset(vocab_size=64, seq_len=32, K=4, n_examples=2000, seed=seed)
        val_ds   = AssociativeRecallDataset(vocab_size=64, seq_len=32, K=4, n_examples=500,  seed=seed + 1000)
        return train_ds, val_ds, 64

    else:
        raise ValueError(f"Unknown task: {task!r}")


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader: DataLoader, device: str, max_batches: int = 50) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        total += out["loss"].item()
        n += 1
    model.train()
    return total / max(n, 1)


def _train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_steps: int,
    device: str,
    seed: int,
    label: str,
    log_steps: list[int],
) -> TaskResult:
    """Train model for n_steps, log val_loss at specified steps."""
    seed_everything(seed)
    model.to(device).train()

    param_groups = model.configure_optimizer_param_groups(WEIGHT_DECAY)
    opt = torch.optim.AdamW(param_groups, lr=LR, betas=BETAS, eps=EPS)
    sched = CosineWithWarmup(opt, warmup_steps=WARMUP_STEPS, max_steps=n_steps, lr=LR, min_lr=LR_MIN)

    loader_iter = iter(train_loader)
    log_at = set(log_steps)
    loss_at_steps: dict[int, float] = {}
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
            print(f"  [{label}] [WARN] non-finite loss at step {step}, skipping")
            opt.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad()

        if step in log_at:
            val_loss = _evaluate(model, val_loader, device)
            elapsed = time.perf_counter() - t0
            loss_at_steps[step] = val_loss
            print(f"  [{label}] step {step:5d}/{n_steps} "
                  f"train={loss.item():.4f} val={val_loss:.4f} ({elapsed:.1f}s)")

    final_val = loss_at_steps.get(n_steps, float("inf"))
    return TaskResult(
        task="",
        model_name=label,
        loss_at_steps=loss_at_steps,
        final_val_loss=final_val,
    )


def run_task(
    task: str,
    n_steps: int,
    batch_size: int,
    d_model: int,
    n_layers: int,
    n_osc: int,
    seed: int,
    device: str,
    out_dir: Path,
    log_steps: list[int],
) -> tuple[TaskResult, TaskResult]:
    """Train TRN and TF on one task. Returns (trn_result, tf_result)."""
    print(f"\n=== Task: {task} ===")
    train_ds, val_ds, vocab_size = _make_dataset(task, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Get seq_len from first sample
    sample = train_ds[0]
    seq_len = sample["input_ids"].shape[0]

    cfg = TRNConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_oscillators=n_osc,
        n_layers=n_layers,
        d_ff=d_model * 4,
        max_seq_len=seq_len + 8,
    )

    print(f"  vocab={vocab_size} seq_len={seq_len} d_model={d_model} n_layers={n_layers} n_osc={n_osc}")

    seed_everything(seed)
    trn = TRNModel(cfg)
    trn_result = _train_model(trn, train_loader, val_loader, n_steps, device, seed, "TRN", log_steps)
    trn_result.task = task

    seed_everything(seed)
    tf = TransformerModel(cfg)
    tf_result = _train_model(tf, train_loader, val_loader, n_steps, device, seed, "TF", log_steps)
    tf_result.task = task

    # Save CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task}_curves.csv"
    all_steps = sorted(set(trn_result.loss_at_steps) | set(tf_result.loss_at_steps))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "TRN_val_loss", "TF_val_loss"])
        for s in all_steps:
            w.writerow([s,
                        f"{trn_result.loss_at_steps.get(s, ''):.4f}" if s in trn_result.loss_at_steps else "",
                        f"{tf_result.loss_at_steps.get(s, ''):.4f}" if s in tf_result.loss_at_steps else ""])
    print(f"  Saved: {csv_path}")

    return trn_result, tf_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Training benchmark TRN vs Transformer")
    parser.add_argument("--tasks", type=str, default="all",
                        help=f"Comma-separated tasks or 'all'. Options: {', '.join(TASK_NAMES)}")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-osc", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="scripts/results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 200 steps, small model (fast CI check)")
    args = parser.parse_args()

    if args.quick:
        args.steps = 200
        args.d_model = 64
        args.n_layers = 2
        args.n_osc = 32
        args.batch_size = 16

    if args.tasks == "all":
        tasks = TASK_NAMES
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    out_dir = Path(args.output_dir)
    log_steps = [s for s in LOG_STEPS if s <= args.steps]
    if args.steps not in log_steps:
        log_steps.append(args.steps)
    log_steps = sorted(set(log_steps))

    all_results: list[tuple[TaskResult, TaskResult]] = []
    for task in tasks:
        trn_res, tf_res = run_task(
            task=task,
            n_steps=args.steps,
            batch_size=args.batch_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_osc=args.n_osc,
            seed=args.seed,
            device=args.device,
            out_dir=out_dir,
            log_steps=log_steps,
        )
        all_results.append((trn_res, tf_res))

    # Summary table
    print(f"\n{'='*75}")
    print(f"Summary: {args.steps} steps, d_model={args.d_model}, n_layers={args.n_layers}")
    print(f"{'='*75}")
    print(f"{'task':<14} | {'TRN loss@final':>14} | {'TF loss@final':>13} | {'ratio':>6} | status")
    print("-" * 75)

    failures: list[str] = []
    for trn_res, tf_res in all_results:
        trn_loss = trn_res.final_val_loss
        tf_loss  = tf_res.final_val_loss
        ratio = trn_loss / tf_loss if tf_loss > 0 else float("inf")
        ok = ratio <= 1.20
        status = "PASS" if ok else "FAIL"
        print(f"{trn_res.task:<14} | {trn_loss:>14.4f} | {tf_loss:>13.4f} | {ratio:>6.3f} | {status}")
        if not ok:
            failures.append(f"{trn_res.task}: TRN/TF ratio={ratio:.3f} > 1.20")

    print(f"{'='*75}")
    print()
    if failures:
        print(f"FAILED ({len(failures)} tasks):")
        for fail in failures:
            print(f"  - {fail}")
        return 1
    else:
        print("ALL TASKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
