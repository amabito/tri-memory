"""Verify that distillation with random targets converges to unigram entropy.

Runs distillation with shuffled labels. Student should NOT learn anything useful.
PASS: |val_loss - H(p)| < 0.5 nats

Usage:
    cd scripts
    python verify_random_targets.py --steps 500 --device cpu
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.scheduler import CosineWithWarmup


PASS_THRESHOLD = 0.5  # nats from H(p)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def run(args: argparse.Namespace) -> bool:
    seed_everything(args.seed)

    # Load teacher tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    vocab_size = tokenizer.vocab_size

    # Tokenize data
    from distill_lm import _load_text, tokenize_dataset
    train_ids, val_ids = tokenize_dataset(tokenizer, args.teacher)

    # Compute unigram entropy
    counts = Counter(train_ids.tolist())
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    unigram_entropy = -sum(p * math.log(p) for p in probs)
    print(f"[random-targets] H_unigram = {unigram_entropy:.4f}", flush=True)

    # Build student
    cfg = TRNConfig(
        vocab_size=vocab_size, d_model=128, n_oscillators=64,
        n_layers=4, d_ff=512, max_seq_len=args.seq_len,
    )
    model = TRNModel(cfg).to(args.device)

    # Optimizer
    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))
    scheduler = CosineWithWarmup(
        optimizer, warmup_steps=50, max_steps=args.steps,
        lr=3e-4, min_lr=3e-5,
    )

    # Load teacher
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher).to(args.device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Training with shuffled labels
    rng = np.random.default_rng(args.seed)
    model.train()
    log_every = max(1, args.steps // 5)

    for step in range(1, args.steps + 1):
        scheduler.step(step)
        max_start = max(1, len(train_ids) - args.seq_len - 1)
        starts = rng.integers(0, max_start, size=args.batch_size)
        seqs = [train_ids[s: s + args.seq_len] for s in starts]
        batch = torch.tensor(np.stack(seqs), dtype=torch.long, device=args.device)

        # Shuffle labels
        labels = batch.clone()
        flat = labels.view(-1)
        perm = flat[torch.randperm(flat.numel(), device=args.device)]
        labels = perm.view_as(labels)

        optimizer.zero_grad()

        # Student forward
        s_out = model(batch)
        s_logits = s_out["logits"]

        # Teacher forward (also with shuffled for KL)
        with torch.no_grad():
            t_out = teacher(batch)
            t_logits = t_out.logits

        # Distillation loss with shuffled labels
        shift_s = s_logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        ce = F.cross_entropy(
            shift_s.view(-1, shift_s.size(-1)),
            shift_labels.view(-1),
        )
        loss = ce  # CE-only with random targets

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == args.steps:
            print(f"  step={step:5d}/{args.steps}  loss={loss.item():.4f}", flush=True)

    # Evaluate
    model.eval()
    eval_rng = np.random.default_rng(0)
    total_loss = 0.0
    n_eval = 50
    with torch.no_grad():
        for _ in range(n_eval):
            max_start = max(1, len(val_ids) - args.seq_len - 1)
            starts = eval_rng.integers(0, max_start, size=args.batch_size)
            seqs = [val_ids[s: s + args.seq_len] for s in starts]
            batch = torch.tensor(np.stack(seqs), dtype=torch.long, device=args.device)

            # Shuffle labels for eval too
            labels = batch.clone()
            flat = labels.view(-1)
            perm = flat[torch.randperm(flat.numel(), device=args.device)]
            labels = perm.view_as(labels)

            out = model(batch, labels=labels)
            total_loss += out["loss"].item()

    val_loss = total_loss / n_eval
    diff = abs(val_loss - unigram_entropy)
    passed = diff < PASS_THRESHOLD

    verdict = "PASS" if passed else "FAIL"
    print(f"\n{'='*50}", flush=True)
    print(f"  Random-target control: {verdict}", flush=True)
    print(f"  val_loss:       {val_loss:.4f}", flush=True)
    print(f"  H_unigram:      {unigram_entropy:.4f}", flush=True)
    print(f"  |difference|:   {diff:.4f}  (threshold: {PASS_THRESHOLD})", flush=True)
    print(f"{'='*50}", flush=True)
    sys.stdout.flush()
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-target distillation control")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher", type=str, default="gpt2")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    passed = run(args)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
