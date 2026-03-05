"""Compare distillation (KL+CE) vs CE-only at equal compute.

Runs two training sessions:
  1. KL+CE distillation (default weights)
  2. CE-only (kl_weight=0, ce_weight=1)

PASS: distilled val_ppl < CE-only val_ppl by at least 10%

Usage:
    cd scripts
    python verify_teacher_ablation.py --steps 2000 --device cpu
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trn.config import TRNConfig
from trn.model import TRNModel
from trn.scheduler import CosineWithWarmup


IMPROVEMENT_THRESHOLD = 0.10  # 10%


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample_batch(
    token_ids: np.ndarray, seq_len: int, batch_size: int,
    rng: np.random.Generator, device: str,
) -> torch.Tensor:
    max_start = max(1, len(token_ids) - seq_len - 1)
    starts = rng.integers(0, max_start, size=batch_size)
    seqs = [token_ids[s: s + seq_len] for s in starts]
    return torch.tensor(np.stack(seqs), dtype=torch.long, device=device)


def eval_ppl(
    model: nn.Module, val_ids: np.ndarray,
    seq_len: int, batch_size: int, device: str,
) -> float:
    model.eval()
    rng = np.random.default_rng(0)
    total_loss = 0.0
    n_eval = 50
    with torch.no_grad():
        for _ in range(n_eval):
            batch = sample_batch(val_ids, seq_len, batch_size, rng, device)
            out = model(batch, labels=batch)
            total_loss += out["loss"].item()
    model.train()
    return math.exp(total_loss / n_eval)


def train_one(
    label: str,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    vocab_size: int,
    teacher: nn.Module,
    kl_weight: float,
    ce_weight: float,
    temperature: float,
    args: argparse.Namespace,
) -> float:
    """Train one configuration. Returns final val_ppl."""
    seed_everything(args.seed)

    cfg = TRNConfig(
        vocab_size=vocab_size, d_model=128, n_oscillators=64,
        n_layers=4, d_ff=512, max_seq_len=args.seq_len,
    )
    model = TRNModel(cfg).to(args.device)

    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))
    scheduler = CosineWithWarmup(
        optimizer, warmup_steps=200, max_steps=args.steps,
        lr=3e-4, min_lr=3e-5,
    )

    rng = np.random.default_rng(args.seed)
    model.train()
    log_every = max(1, args.steps // 5)

    for step in range(1, args.steps + 1):
        scheduler.step(step)
        batch = sample_batch(train_ids, args.seq_len, args.batch_size, rng, args.device)

        optimizer.zero_grad()

        s_out = model(batch)
        s_logits = s_out["logits"]

        # Causal shift
        shift_s = s_logits[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        loss = torch.tensor(0.0, device=args.device)

        if kl_weight > 0.0:
            with torch.no_grad():
                t_out = teacher(batch)
                t_logits = t_out.logits
            shift_t = t_logits[:, :-1].contiguous()

            flat_s = shift_s.view(-1, shift_s.size(-1))
            flat_t = shift_t.view(-1, shift_t.size(-1))
            s_log_probs = F.log_softmax(flat_s / temperature, dim=-1)
            t_probs = F.softmax(flat_t / temperature, dim=-1)
            kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)
            loss = loss + kl_weight * kl

        if ce_weight > 0.0:
            ce = F.cross_entropy(
                shift_s.view(-1, shift_s.size(-1)),
                targets.view(-1),
            )
            loss = loss + ce_weight * ce

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == args.steps:
            print(f"  [{label}] step={step:5d}/{args.steps}  loss={loss.item():.4f}")

    val_ppl = eval_ppl(model, val_ids, args.seq_len, args.batch_size, args.device)
    print(f"  [{label}] final val_ppl={val_ppl:.2f}")
    return val_ppl


def run(args: argparse.Namespace) -> bool:
    # Load teacher
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher).to(args.device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    vocab_size = teacher.config.vocab_size

    # Tokenize
    from distill_lm import tokenize_dataset
    train_ids, val_ids = tokenize_dataset(tokenizer, args.teacher)

    # Run 1: KL+CE distillation
    print("\n--- Run 1: KL+CE distillation ---")
    distill_ppl = train_one(
        "distill", train_ids, val_ids, vocab_size, teacher,
        kl_weight=1.0, ce_weight=0.1, temperature=2.0, args=args,
    )

    # Run 2: CE-only
    print("\n--- Run 2: CE-only ---")
    ce_only_ppl = train_one(
        "ce-only", train_ids, val_ids, vocab_size, teacher,
        kl_weight=0.0, ce_weight=1.0, temperature=2.0, args=args,
    )

    improvement = (ce_only_ppl - distill_ppl) / ce_only_ppl
    passed = improvement >= IMPROVEMENT_THRESHOLD

    verdict = "PASS" if passed else "FAIL"
    print(f"\n{'='*50}")
    print(f"  Teacher ablation: {verdict}")
    print(f"  Distill val_ppl: {distill_ppl:.2f}")
    print(f"  CE-only val_ppl: {ce_only_ppl:.2f}")
    print(f"  Improvement:     {improvement:.1%}  (threshold: {IMPROVEMENT_THRESHOLD:.0%})")
    print(f"{'='*50}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher ablation: distill vs CE-only")
    parser.add_argument("--steps", type=int, default=2000)
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
