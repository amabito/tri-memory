"""Verify that TRN student can overfit a tiny fixed dataset.

Trains on a 2048-token fixed batch for 2000 steps.
PASS: train_loss < 0.5

Usage:
    cd scripts
    python verify_overfit_microset.py --steps 2000 --device cpu
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trn.config import TRNConfig
from trn.model import TRNModel
from trn.scheduler import CosineWithWarmup


PASS_THRESHOLD = 0.5


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run(args: argparse.Namespace) -> bool:
    seed_everything(args.seed)

    # Load teacher tokenizer for vocab size
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    vocab_size = tokenizer.vocab_size

    # Build student
    cfg = TRNConfig(
        vocab_size=vocab_size, d_model=128, n_oscillators=64,
        n_layers=4, d_ff=512, max_seq_len=args.seq_len,
    )
    model = TRNModel(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[student] params={n_params:,}, vocab={vocab_size}")

    # Create fixed 2048-token microset from tokenizer
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "A language model must learn to predict the next token accurately. "
        "Knowledge distillation transfers learned representations. "
        "Temporal resonance networks use oscillatory dynamics. "
    )
    token_ids = tokenizer.encode(text)
    # Repeat to fill seq_len
    while len(token_ids) < args.seq_len:
        token_ids = token_ids + token_ids
    token_ids = token_ids[:args.seq_len]
    fixed_batch = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    # Repeat to batch
    fixed_batch = fixed_batch.expand(args.batch_size, -1).contiguous()

    print(f"[data] Fixed batch: {fixed_batch.shape}, {fixed_batch.numel()} tokens")

    # Optimizer
    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    scheduler = CosineWithWarmup(
        optimizer, warmup_steps=100, max_steps=args.steps,
        lr=args.lr, min_lr=args.lr * 0.01,
    )

    # Train
    model.train()
    log_every = max(1, args.steps // 10)
    final_loss = float("nan")

    for step in range(1, args.steps + 1):
        scheduler.step(step)
        optimizer.zero_grad()

        out = model(fixed_batch, labels=fixed_batch)
        loss = out["loss"]
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        final_loss = loss.item()

        if step % log_every == 0 or step == args.steps:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step={step:5d}/{args.steps}  loss={final_loss:.4f}  lr={lr:.2e}")

    passed = final_loss < PASS_THRESHOLD
    verdict = "PASS" if passed else "FAIL"
    print(f"\n{'='*50}")
    print(f"  Overfit microset: {verdict}")
    print(f"  Final loss: {final_loss:.4f}  (threshold: {PASS_THRESHOLD})")
    print(f"{'='*50}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify TRN overfit on microset")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher", type=str, default="gpt2")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        args.device = "cpu"

    passed = run(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
