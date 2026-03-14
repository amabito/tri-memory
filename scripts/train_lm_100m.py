#!/usr/bin/env python3
"""Train a ~100M parameter TRN or Transformer language model.

Configuration (100M parameter target):
    d_model = 768, n_layers = 12, n_oscillators = 256, d_ff = 3072
    vocab_size = 50257 (GPT-2 tokenizer), max_seq_len = 2048

Data formats supported:
    synthetic:      random token sequences (for smoke tests and profiling)
    path/to/file:   packed uint16 binary file (GPT-2 tokenized, same as
                    nanoGPT / OpenWebText format)

Full training:
    Requires GPU with CUDA. Estimated ~72h on A100 for 2B tokens.
    SlimPajama preprocessing: https://huggingface.co/datasets/cerebras/SlimPajama-627B
    Tokenize with tiktoken (cl100k_base or gpt2) and pack into .bin files.

Usage (smoke test, CPU):
    python scripts/train_lm_100m.py --model trn --data synthetic \
        --steps 50 --batch-size 2 --seq-len 256 --log-interval 10

Usage (full run, GPU):
    python scripts/train_lm_100m.py --model trn \
        --data /path/to/train.bin --val-data /path/to/val.bin \
        --steps 100000 --batch-size 8 --grad-accum 8 \
        --seq-len 2048 --device cuda --compile \
        --checkpoint-dir checkpoints/100m_trn/
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import csv
import time
import tracemalloc
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.baseline import TransformerModel
from trimemory.scheduler import CosineWithWarmup
from trimemory.bench_data import seed_everything


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_100m_config(model_name: str, vocab_size: int = 50257, seq_len: int = 2048) -> TRNConfig:
    """Return TRNConfig for ~100M parameter model."""
    return TRNConfig(
        vocab_size=vocab_size,
        d_model=768,
        n_oscillators=256,   # used only by TRN
        n_layers=12,
        d_ff=3072,
        max_seq_len=seq_len,
        tie_weights=True,
    )


def build_model(model_name: str, cfg: TRNConfig, device: str) -> nn.Module:
    if model_name == "trn":
        model = TRNModel(cfg)
    elif model_name == "tf":
        model = TransformerModel(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose 'trn' or 'tf'.")
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Random token sequences for smoke tests."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_examples: int = 1024,
        seed: int = 42,
    ) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (n_examples, seq_len + 1), generator=rng)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        chunk = self.data[i]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


class PackedBinaryDataset(IterableDataset):
    """Streams (seq_len+1)-token chunks from a packed uint16 binary file.

    Compatible with nanoGPT / OpenWebText .bin format.
    """

    def __init__(self, path: str, seq_len: int, seed: int = 42) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.seed = seed
        data = np.memmap(str(self.path), dtype=np.uint16, mode="r")
        self.n_tokens = len(data)
        self.n_chunks = (self.n_tokens - 1) // (seq_len + 1)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        data = np.memmap(str(self.path), dtype=np.uint16, mode="r")
        indices = rng.permutation(self.n_chunks)
        for idx in indices:
            start = idx * (self.seq_len + 1)
            chunk = torch.from_numpy(
                data[start : start + self.seq_len + 1].astype(np.int64)
            )
            yield {"input_ids": chunk[:-1], "labels": chunk[1:]}


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    max_batches: int = 20,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            out = model(input_ids, labels=input_ids)
            total += out["loss"].item()
            n += 1
    model.train()
    return total / max(n, 1)


def estimate_peak_memory_mb(device: str) -> float:
    if device == "cpu":
        _cur, peak = tracemalloc.get_traced_memory()
        return peak / (1024 * 1024)
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a ~100M parameter TRN or Transformer LM"
    )
    parser.add_argument("--model", type=str, default="trn", choices=["trn", "tf"])
    parser.add_argument(
        "--data",
        type=str,
        default="synthetic",
        help="'synthetic' or path to packed uint16 .bin file",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Validation .bin file (optional; skipped for synthetic)",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Warmup steps (0 = 10% of --steps)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (requires PyTorch 2.0+, CUDA recommended)",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/100m")
    parser.add_argument("--output-dir", type=str, default="scripts/results")
    args = parser.parse_args()

    seed_everything(args.seed)

    # ------------------------------------------------------------------
    # Config & model
    # ------------------------------------------------------------------
    # Use smaller vocab for synthetic smoke test to reduce embedding overhead
    vocab_size = 50257 if args.data != "synthetic" else min(50257, 4096)
    cfg = make_100m_config(args.model, vocab_size=vocab_size, seq_len=args.seq_len)

    model = build_model(args.model, cfg, args.device)
    n_params = count_parameters(model)
    print(f"Model: {args.model.upper()}  |  Parameters: {n_params/1e6:.1f}M")
    print(
        f"Config: d_model={cfg.d_model}  n_layers={cfg.n_layers}  "
        f"n_osc={cfg.n_oscillators}  d_ff={cfg.d_ff}  vocab={cfg.vocab_size}"
    )
    print(
        f"seq_len={args.seq_len}  batch_size={args.batch_size}  "
        f"grad_accum={args.grad_accum}"
    )
    print(f"device={args.device}  steps={args.steps}")
    print()

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.data == "synthetic":
        train_ds = SyntheticDataset(cfg.vocab_size, args.seq_len, n_examples=2048, seed=args.seed)
        val_ds = SyntheticDataset(cfg.vocab_size, args.seq_len, n_examples=256, seed=args.seed + 1)
    else:
        train_ds = PackedBinaryDataset(args.data, args.seq_len, seed=args.seed)
        val_ds = (
            PackedBinaryDataset(args.val_data, args.seq_len, seed=args.seed + 1)
            if args.val_data
            else None
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(args.data == "synthetic"),
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        if val_ds is not None
        else None
    )
    data_iter = cycle(train_loader)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    if hasattr(model, "configure_optimizer_param_groups"):
        param_groups = model.configure_optimizer_param_groups(args.weight_decay)
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    warmup = args.warmup_steps if args.warmup_steps > 0 else max(1, args.steps // 10)
    scheduler = CosineWithWarmup(
        optimizer,
        warmup_steps=warmup,
        max_steps=args.steps,
        lr=args.lr,
        min_lr=args.lr * 0.1,
    )

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    tracemalloc.start()

    model.train()
    optimizer.zero_grad()

    rows: list[dict] = []
    t_start = time.perf_counter()
    tokens_seen = 0

    for step in range(args.steps):
        scheduler.step(step)

        accum_loss = 0.0
        for _ in range(args.grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(args.device)
            out = model(input_ids, labels=input_ids)
            loss = out["loss"] / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            tokens_seen += input_ids.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        s = step + 1
        if s % args.log_interval == 0 or step == args.steps - 1:
            elapsed = time.perf_counter() - t_start
            tps = tokens_seen / elapsed
            peak_mb = estimate_peak_memory_mb(args.device)
            val_loss = (
                compute_val_loss(model, val_loader, args.device)
                if val_loader is not None
                else float("nan")
            )
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"step {s:6d}/{args.steps} | loss={accum_loss:.4f} | "
                f"val={val_loss:.4f} | tps={tps:,.0f} | "
                f"mem={peak_mb:.1f}MB | lr={lr_now:.2e}"
            )
            rows.append({
                "step": s,
                "train_loss": accum_loss,
                "val_loss": val_loss,
                "tps": tps,
                "peak_mb": peak_mb,
            })

        if args.save_interval > 0 and s % args.save_interval == 0:
            ckpt = ckpt_dir / f"step_{s:07d}.pt"
            torch.save(
                {
                    "step": s,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt,
            )
            print(f"Checkpoint saved: {ckpt}")

    tracemalloc.stop()

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    final_tps = rows[-1]["tps"] if rows else 0.0
    final_peak_mb = rows[-1]["peak_mb"] if rows else 0.0
    final_loss = rows[-1]["train_loss"] if rows else float("nan")

    print()
    print("=== Summary ===")
    print(f"Model:          {args.model.upper()} ({n_params/1e6:.1f}M params)")
    print(f"Steps:          {args.steps}")
    print(f"Tokens/sec:     {final_tps:,.0f}")
    print(f"Peak memory:    {final_peak_mb:.1f} MB")
    print(f"Final loss:     {final_loss:.4f}")
    if val_loader is not None and rows:
        print(f"Final val loss: {rows[-1]['val_loss']:.4f}")
    print()
    print("Note: Full 2B-token training requires GPU with CUDA.")
    print("  Estimated training time: ~72h on single A100 (80GB).")
    print("  Prepare data: tokenize with tiktoken (gpt2) -> pack into uint16 .bin")
    print(
        f"  Full run: python scripts/train_lm_100m.py --model {args.model} "
        f"--data train.bin"
    )
    print(
        f"            --val-data val.bin --steps 100000 --batch-size 8 --grad-accum 8"
    )
    print(
        f"            --seq-len 2048 --device cuda --compile"
    )

    # Save CSV
    csv_path = out_dir / "train_lm_100m_smoke.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss", "tps", "peak_mb"])
        for r in rows:
            w.writerow([
                r["step"],
                f"{r['train_loss']:.6f}",
                f"{r['val_loss']:.6f}",
                f"{r['tps']:.0f}",
                f"{r['peak_mb']:.1f}",
            ])
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
