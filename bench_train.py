#!/usr/bin/env python3
"""Deterministic training benchmark: TRN vs Transformer.

Fixed hyperparameters (identical for both models):
  optimizer:    AdamW, betas=(0.9, 0.95), eps=1e-8
  lr:           3e-4 (cosine to 3e-5 with warmup)
  weight_decay: 0.1 (weight matrices only)
  grad_clip:    1.0
  batch_size:   32
  dropout:      0.0

Usage:
  python bench_train.py --task copy --steps 5000
  python bench_train.py --task selective --steps 5000 --seq-len 16
  python bench_train.py --task corpus --steps 5000 --seq-len 128
  python bench_train.py --task copy --steps 1000 --quick   # quick smoke run
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

# Remove '' (cwd) from sys.path before torch imports to avoid shadowing stdlib
# profile.py in the project root shadows the stdlib cProfile module.
_project_root = str(Path(__file__).parent)
if "" in sys.path:
    sys.path.remove("")
if _project_root in sys.path:
    sys.path.remove(_project_root)
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader

from trn.bench_data import (
    seed_everything,
    NextTokenCopyDataset,
    SelectiveCopyDataset,
    TinyCorpusDataset,
    TINY_CORPUS,
)
from trn.config import TRNConfig
from trn.model import TRNModel
from trn.baseline import TransformerModel
from trn.scheduler import CosineWithWarmup
from trn.tokenizer import CharTokenizer


# ── Fixed hyperparameters ─────────────────────────────────────────────────────
LR = 3e-4
LR_MIN = 3e-5
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
DROPOUT = 0.0
BATCH_SIZE = 32
WARMUP_FRAC = 0.1  # 10% of total steps


def _bench_cfg(vocab_size: int, seq_len: int) -> TRNConfig:
    """Comparable model: ~same param count for TRN and Transformer."""
    return TRNConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_oscillators=64,
        n_layers=4,
        d_ff=512,
        max_seq_len=max(seq_len + 8, 64),
        dropout=DROPOUT,
    )


def _make_dataset(task: str, seq_len: int, split: str, seed: int):
    if task == "copy":
        return NextTokenCopyDataset(
            n_samples=2000, seq_len=seq_len, vocab_size=32, period=min(8, seq_len), seed=seed,
        )
    elif task == "selective":
        n_vals = min(8, seq_len - 3)
        return SelectiveCopyDataset(n_samples=2000, n_vals=n_vals, vocab_size=32, seed=seed)
    elif task == "corpus":
        tok = CharTokenizer().fit(TINY_CORPUS)
        return TinyCorpusDataset(seq_len=seq_len, split=split, tokenizer=tok)
    else:
        raise ValueError(f"Unknown task: {task!r}. Choose: copy, selective, corpus")


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int = 50) -> float:
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


def train_one_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_steps: int,
    device: str,
    seed: int,
    eval_every: int,
    label: str,
) -> tuple[list[tuple[int, float, float]], float]:
    """Train model for n_steps. Returns (curve, final_val_loss)."""
    seed_everything(seed)
    model.to(device).train()

    warmup_steps = max(1, int(n_steps * WARMUP_FRAC))
    param_groups = model.configure_optimizer_param_groups(WEIGHT_DECAY)
    opt = torch.optim.AdamW(param_groups, lr=LR, betas=BETAS, eps=EPS)
    sched = CosineWithWarmup(
        opt, warmup_steps=warmup_steps, max_steps=n_steps, lr=LR, min_lr=LR_MIN,
    )

    loader_iter = iter(train_loader)
    curve: list[tuple[int, float, float]] = []  # (step, train_loss, val_loss)
    t0 = time.perf_counter()

    for step in range(n_steps):
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

        if step % eval_every == 0 or step == n_steps - 1:
            val_loss = evaluate(model, val_loader, device)
            elapsed = time.perf_counter() - t0
            curve.append((step, loss.item(), val_loss))
            print(
                f"  [{label}] step {step:5d}/{n_steps} "
                f"train={loss.item():.4f} val={val_loss:.4f} "
                f"({elapsed:.1f}s)"
            )

    final_val = curve[-1][2] if curve else float("inf")
    return curve, final_val


def save_curves(
    trn_curve: list,
    tf_curve: list,
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "trn_train", "trn_val", "tf_train", "tf_val"])
        trn_map = {s: (tl, vl) for s, tl, vl in trn_curve}
        tf_map = {s: (tl, vl) for s, tl, vl in tf_curve}
        all_steps = sorted(set(trn_map) | set(tf_map))
        for s in all_steps:
            trn_tl, trn_vl = trn_map.get(s, ("", ""))
            tf_tl, tf_vl = tf_map.get(s, ("", ""))
            writer.writerow([s, trn_tl, trn_vl, tf_tl, tf_vl])
    print(f"  Loss curves saved to {out_path}")


def print_summary_table(
    task: str,
    n_steps: int,
    trn_final: float,
    tf_final: float,
    trn_params: int,
    tf_params: int,
) -> None:
    ratio = trn_final / tf_final if tf_final > 0 else float("inf")
    within_20pct = ratio <= 1.20
    status = "[PASS]" if within_20pct else "[FAIL]"

    print(f"\n{'='*70}")
    print(f"Benchmark Summary: task={task}, steps={n_steps}")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'TRN':>15} {'Transformer':>15}")
    print(f"{'-'*70}")
    print(f"{'Parameters':<30} {trn_params:>15,} {tf_params:>15,}")
    print(f"{'Final val loss':<30} {trn_final:>15.4f} {tf_final:>15.4f}")
    print(f"{'TRN/TF loss ratio':<30} {ratio:>15.4f}")
    print(f"{'='*70}")
    print(f"{status} TRN loss ratio = {ratio:.3f} (threshold <=1.20 for PASS)")
    print(f"{'='*70}\n")


def run_benchmark(args: argparse.Namespace) -> None:
    device = args.device
    n_steps = args.steps
    seq_len = args.seq_len
    eval_every = max(1, n_steps // 20)  # 20 eval points total
    seed = args.seed

    # Build datasets
    train_ds = _make_dataset(args.task, seq_len, "train", seed)
    val_ds = _make_dataset(args.task, seq_len, "val", seed + 1000)

    # Determine vocab_size
    vocab_size = getattr(train_ds, "vocab_size", 32)
    if args.task == "copy":
        vocab_size = 32
    elif args.task == "selective":
        vocab_size = 32
    # corpus: uses tokenizer.vocab_size

    cfg = _bench_cfg(vocab_size, seq_len)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"\nBenchmark: task={args.task}, steps={n_steps}, seq_len={seq_len}, device={device}")
    print(f"Hyperparams: lr={LR}, warmup={int(n_steps*WARMUP_FRAC)}, "
          f"wd={WEIGHT_DECAY}, clip={GRAD_CLIP}, bs={BATCH_SIZE}")
    print(f"vocab_size={vocab_size}, d_model={cfg.d_model}, n_layers={cfg.n_layers}\n")

    # Train TRN
    print("--- Training TRN ---")
    seed_everything(seed)
    trn = TRNModel(cfg)
    trn_curve, trn_final = train_one_model(
        trn, train_loader, val_loader, n_steps, device, seed, eval_every, "TRN",
    )

    # Train Transformer (fresh loaders, same seed)
    print("\n--- Training Transformer ---")
    seed_everything(seed)
    tf = TransformerModel(cfg)
    tf_curve, tf_final = train_one_model(
        tf, train_loader, val_loader, n_steps, device, seed, eval_every, "TF",
    )

    # Save curves
    out_dir = Path("bench_results")
    out_dir.mkdir(exist_ok=True)
    save_curves(
        trn_curve, tf_curve,
        out_dir / f"curves_{args.task}_{n_steps}steps.csv",
    )

    # Print summary
    print_summary_table(
        args.task, n_steps, trn_final, tf_final,
        trn.num_parameters(), tf.num_parameters(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TRN vs Transformer training benchmark")
    parser.add_argument(
        "--task", choices=["copy", "selective", "corpus"], default="copy",
        help="Benchmark task",
    )
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke run (100 steps, small batch)",
    )
    args = parser.parse_args()

    if args.quick:
        args.steps = 100

    run_benchmark(args)


if __name__ == "__main__":
    main()
