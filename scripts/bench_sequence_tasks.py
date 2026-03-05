#!/usr/bin/env python3
"""Sequence task benchmark: TRN vs Transformer vs Hybrid across synthetic tasks.

Tasks:
  copy         -- NextTokenCopyDataset (periodic sequence memorisation)
  counting     -- CountingDataset (predict running count, tests counter state)
  reverse      -- ReverseDataset (reverse second half, tests state recall)
  induction    -- InductionHeadDataset (two-hop bigram retrieval)
  assoc_recall -- AssociativeRecallDataset (key-value store in recurrent state)
  selective    -- SelectiveCopyDataset (associative recall with SEP token)
  all          -- Run all tasks

Models compared:
  TRN    -- pure TRN (TemporalResonanceLayer blocks)
  TF     -- pure Transformer (CausalSelfAttention blocks)
  Hybrid -- interleaved TRN + Attention (trn_ratio=0.5 by default)

Usage:
    python scripts/bench_sequence_tasks.py [--tasks all] [--steps 2000]
        [--batch-size 32] [--d-model 128] [--n-layers 4] [--n-osc 64]
        [--trn-ratio 0.5] [--seed 42] [--device cpu]
        [--output-dir scripts/results/sequence]

Exit codes:
    0: all models trained without error
    1: any runtime error
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from trn.bench_data import (
    seed_everything,
    NextTokenCopyDataset,
    CountingDataset,
    ReverseDataset,
    InductionHeadDataset,
    AssociativeRecallDataset,
    SelectiveCopyDataset,
)
from trn.config import TRNConfig
from trn.model import TRNModel
from trn.baseline import TransformerModel
from trn.hybrid_model import HybridModel
from trn.scheduler import CosineWithWarmup


# ── Fixed hyperparameters ─────────────────────────────────────────────────────

LR = 3e-4
LR_MIN = 3e-5
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 100

TASK_NAMES = ["copy", "counting", "reverse", "induction", "assoc_recall", "selective"]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task: str
    model_name: str
    loss_at_steps: dict[int, float] = field(default_factory=dict)
    final_val_loss: float = float("inf")
    wall_time_s: float = 0.0
    n_params: int = 0


# ── Dataset factory ───────────────────────────────────────────────────────────

def _make_datasets(task: str, seed: int) -> tuple[Dataset, Dataset, int, int]:
    """Return (train_ds, val_ds, vocab_size, seq_len)."""
    if task == "copy":
        kw = dict(n_samples=2000, seq_len=64, vocab_size=32, period=8, seed=seed)
        val_kw = {**kw, "n_samples": 500, "seed": seed + 1000}
        return NextTokenCopyDataset(**kw), NextTokenCopyDataset(**val_kw), 32, 64

    if task == "counting":
        kw = dict(vocab_size=64, seq_len=16, n_examples=2000, seed=seed)
        val_kw = {**kw, "n_examples": 500, "seed": seed + 1000}
        return CountingDataset(**kw), CountingDataset(**val_kw), 64, 16

    if task == "reverse":
        kw = dict(vocab_size=64, seq_len=16, n_examples=2000, seed=seed)
        val_kw = {**kw, "n_examples": 500, "seed": seed + 1000}
        return ReverseDataset(**kw), ReverseDataset(**val_kw), 64, 16

    if task == "induction":
        kw = dict(vocab_size=64, seq_len=32, n_examples=2000, seed=seed)
        val_kw = {**kw, "n_examples": 500, "seed": seed + 1000}
        return InductionHeadDataset(**kw), InductionHeadDataset(**val_kw), 64, 32

    if task == "assoc_recall":
        kw = dict(vocab_size=64, seq_len=32, K=4, n_examples=2000, seed=seed)
        val_kw = {**kw, "n_examples": 500, "seed": seed + 1000}
        return AssociativeRecallDataset(**kw), AssociativeRecallDataset(**val_kw), 64, 32

    if task == "selective":
        kw = dict(n_samples=2000, n_vals=8, vocab_size=32, seed=seed)
        val_kw = {**kw, "n_samples": 500, "seed": seed + 1000}
        train_ds = SelectiveCopyDataset(**kw)
        val_ds = SelectiveCopyDataset(**val_kw)
        seq_len = train_ds.seq_len
        return train_ds, val_ds, 32, seq_len

    raise ValueError(f"Unknown task: {task!r}")


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_models(
    cfg: TRNConfig,
    trn_ratio: float,
    seed: int,
) -> list[tuple[str, torch.nn.Module]]:
    """Return [(name, model)] for TRN, TF, and Hybrid."""
    seed_everything(seed)
    trn = TRNModel(cfg)

    seed_everything(seed)
    tf = TransformerModel(cfg)

    seed_everything(seed)
    hybrid = HybridModel(cfg, trn_ratio=trn_ratio)

    return [("TRN", trn), ("TF", tf), (f"Hybrid({trn_ratio:.1f})", hybrid)]


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int = 50,
) -> float:
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


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_model(
    model: torch.nn.Module,
    model_name: str,
    task: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_steps: int,
    device: str,
    seed: int,
    log_steps: list[int],
) -> TaskResult:
    seed_everything(seed)
    model.to(device).train()

    n_params = model.num_parameters(non_embedding=True) if hasattr(model, "num_parameters") else 0

    param_groups = model.configure_optimizer_param_groups(WEIGHT_DECAY)
    opt = torch.optim.AdamW(param_groups, lr=LR, betas=BETAS, eps=EPS)
    sched = CosineWithWarmup(
        opt,
        warmup_steps=min(WARMUP_STEPS, n_steps // 4),
        max_steps=n_steps,
        lr=LR,
        min_lr=LR_MIN,
    )

    loader_iter = iter(train_loader)
    log_set = set(log_steps)
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
            print(f"  [{model_name}] WARN: non-finite loss at step {step}, skipping")
            opt.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad()

        if step in log_set:
            val_loss = _evaluate(model, val_loader, device)
            elapsed = time.perf_counter() - t0
            loss_at_steps[step] = val_loss
            print(
                f"  [{model_name:>18}] step {step:5d}/{n_steps}"
                f"  train={loss.item():.4f}  val={val_loss:.4f}"
                f"  ({elapsed:.1f}s)"
            )

    wall_time = time.perf_counter() - t0
    final_val = loss_at_steps.get(n_steps, float("inf"))

    return TaskResult(
        task=task,
        model_name=model_name,
        loss_at_steps=loss_at_steps,
        final_val_loss=final_val,
        wall_time_s=wall_time,
        n_params=n_params,
    )


# ── Per-task runner ───────────────────────────────────────────────────────────

def run_task(
    task: str,
    n_steps: int,
    batch_size: int,
    d_model: int,
    n_layers: int,
    n_osc: int,
    trn_ratio: float,
    seed: int,
    device: str,
    out_dir: Path,
    log_steps: list[int],
) -> list[TaskResult]:
    print(f"\n{'='*70}")
    print(f"Task: {task}")
    print(f"{'='*70}")

    train_ds, val_ds, vocab_size, seq_len = _make_datasets(task, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    cfg = TRNConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_oscillators=n_osc,
        n_layers=n_layers,
        d_ff=d_model * 4,
        max_seq_len=seq_len + 8,
    )

    print(
        f"  vocab={vocab_size}  seq_len={seq_len}  d_model={d_model}"
        f"  n_layers={n_layers}  n_osc={n_osc}  trn_ratio={trn_ratio}"
    )

    models = _make_models(cfg, trn_ratio, seed)
    results: list[TaskResult] = []

    for model_name, model in models:
        print(f"\n  -- Training {model_name} --")
        result = _train_model(
            model=model,
            model_name=model_name,
            task=task,
            train_loader=train_loader,
            val_loader=val_loader,
            n_steps=n_steps,
            device=device,
            seed=seed,
            log_steps=log_steps,
        )
        results.append(result)

    # Save per-task CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{task}_seq_curves.csv"
    all_steps = sorted({s for r in results for s in r.loss_at_steps})
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["step"] + [r.model_name for r in results]
        w.writerow(header)
        for s in all_steps:
            row = [s] + [
                f"{r.loss_at_steps[s]:.4f}" if s in r.loss_at_steps else ""
                for r in results
            ]
            w.writerow(row)
    print(f"\n  Saved: {csv_path}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequence task benchmark: TRN vs TF vs Hybrid"
    )
    parser.add_argument(
        "--tasks", type=str, default="all",
        help=f"Comma-separated tasks or 'all'. Options: {', '.join(TASK_NAMES)}",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-osc", type=int, default=64)
    parser.add_argument("--trn-ratio", type=float, default=0.5,
                        help="Fraction of Hybrid layers that are TRN (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir", type=str, default="scripts/results/sequence"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 200 steps, small model (fast sanity check)",
    )
    args = parser.parse_args()

    if args.quick:
        args.steps = 200
        args.d_model = 64
        args.n_layers = 2
        args.n_osc = 32
        args.batch_size = 16

    tasks = TASK_NAMES if args.tasks == "all" else [
        t.strip() for t in args.tasks.split(",")
    ]

    # Build log steps
    default_checkpoints = [50, 100, 200, 500, 1000, 2000]
    log_steps = sorted({s for s in default_checkpoints if s <= args.steps} | {args.steps})

    out_dir = Path(args.output_dir)
    all_results: list[list[TaskResult]] = []

    for task in tasks:
        task_results = run_task(
            task=task,
            n_steps=args.steps,
            batch_size=args.batch_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_osc=args.n_osc,
            trn_ratio=args.trn_ratio,
            seed=args.seed,
            device=args.device,
            out_dir=out_dir,
            log_steps=log_steps,
        )
        all_results.append(task_results)

    # ── Summary table ─────────────────────────────────────────────────────────
    col_w = 20
    header_width = 14 + col_w * 3 + 10
    print(f"\n{'='*header_width}")
    print(f"Summary  |  steps={args.steps}  d_model={args.d_model}  n_layers={args.n_layers}")
    print(f"{'='*header_width}")

    # Collect model names from first task
    if all_results:
        model_names = [r.model_name for r in all_results[0]]
    else:
        model_names = []

    header = f"{'task':<14}" + "".join(f"{n:>{col_w}}" for n in model_names) + f"{'best':>8}"
    print(header)
    print("-" * header_width)

    # Write summary CSV
    summary_csv = out_dir / "sequence_summary.csv"
    summary_rows: list[list] = []

    for task_results in all_results:
        task_name = task_results[0].task if task_results else "?"
        losses = {r.model_name: r.final_val_loss for r in task_results}
        best = min(losses, key=losses.get) if losses else "-"
        row_str = f"{task_name:<14}"
        csv_row: list = [task_name]
        for name in model_names:
            v = losses.get(name, float("inf"))
            row_str += f"{v:>{col_w}.4f}"
            csv_row.append(f"{v:.4f}" if v != float("inf") else "")
        row_str += f"{best:>8}"
        csv_row.append(best)
        print(row_str)
        summary_rows.append(csv_row)

    print(f"{'='*header_width}\n")

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task"] + model_names + ["best"])
        w.writerows(summary_rows)

    print(f"Summary saved: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
