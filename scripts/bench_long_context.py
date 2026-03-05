#!/usr/bin/env python3
"""Needle-in-a-Haystack long-context retrieval benchmark.

A 'needle' token is planted at a random position in a haystack of filler
tokens. After training for N steps the model must recall the needle at the
last sequence position.  Measures recall@1 as a function of context length.

CPU-practical lengths: 128, 256, 512, 1024
GPU lengths (4k/8k/16k) require --device cuda with associative_scan enabled.

Usage:
    python scripts/bench_long_context.py [--context-lens 128,256,512,1024]
                                         [--steps 500] [--batch-size 16]
                                         [--seed 42] [--device cpu]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse, csv
from pathlib import Path
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from trn.config import TRNConfig
from trn.model import TRNModel
from trn.baseline import TransformerModel
from trn.bench_data import seed_everything


FILLER_LOW  = 10
FILLER_HIGH = 200
NEEDLE_LOW  = 200
NEEDLE_HIGH = 240
QUERY_TOKEN = 5      # signals "what was the needle?"


class NeedleHaystackDataset(Dataset):
    """Needle-in-a-haystack retrieval dataset.

    Sequence layout:
        [filler ... filler, NEEDLE, filler ... filler, QUERY]
        length = context_len

    The model is trained with GPT-style next-token prediction.
    At eval time, recall@1 is measured at logits[:, -1]:
        given the QUERY token at position L-1, predict NEEDLE as the next token.
    """

    def __init__(
        self,
        context_len: int,
        n_examples: int,
        seed: int = 42,
    ) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)

        needles   = torch.randint(NEEDLE_LOW, NEEDLE_HIGH, (n_examples,), generator=rng)
        fillers   = torch.randint(FILLER_LOW, FILLER_HIGH, (n_examples, context_len), generator=rng)
        # needle position: anywhere in [1, context_len-2] (not first or last)
        needle_pos = torch.randint(1, context_len - 1, (n_examples,), generator=rng)

        input_ids = fillers.clone()
        for i in range(n_examples):
            input_ids[i, needle_pos[i]] = needles[i]
        input_ids[:, -1] = QUERY_TOKEN   # last token is the query

        self.input_ids  = input_ids    # (N, context_len)
        self.needles    = needles      # (N,) ground truth answer after QUERY
        self.n_examples = n_examples

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "needle":    self.needles[i],
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "needle":    torch.stack([x["needle"]    for x in batch]),
    }


def recall_at_1(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> float:
    """Compute recall@1: fraction of examples where top-1 logit at last pos == needle.

    logits[:, -1] is the model's prediction for the token *after* the QUERY token,
    which in our GPT-style autoregressive setup is exactly where we expect NEEDLE.
    """
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)   # (B, L)
            needles   = batch["needle"].to(device)       # (B,)

            # Forward without labels to get logits (no loss computation needed)
            out    = model(input_ids)
            logits = out["logits"]                       # (B, L, V)

            # logits[:, t] predicts the token at position t+1.
            # The QUERY token is at position L-1 (last input position).
            # logits[:, L-2] predicts logits for position L-1 (the QUERY).
            # We want the prediction AFTER QUERY = what comes next.
            # In a GPT model logits[:, -1] is the prediction for position L
            # (the next token beyond the sequence), which is the NEEDLE answer.
            pred = logits[:, -1].argmax(dim=-1)          # (B,)

            correct += (pred == needles).sum().item()
            total   += input_ids.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def train_and_eval(
    context_len: int,
    model_name: str,
    cfg: TRNConfig,
    n_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> float:
    seed_everything(seed)

    train_ds = NeedleHaystackDataset(context_len, n_examples=512, seed=seed)
    val_ds   = NeedleHaystackDataset(context_len, n_examples=128, seed=seed + 99)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate,
    )

    if model_name == "TRN":
        model = TRNModel(cfg).to(device)
    else:
        model = TransformerModel(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
    )
    data_iter = cycle(train_loader)

    model.train()
    optimizer.zero_grad()
    for _ in range(n_steps):
        batch     = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        out       = model(input_ids, labels=input_ids)
        loss      = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return recall_at_1(model, val_loader, device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack benchmark")
    parser.add_argument("--context-lens", type=str,  default="128,256,512,1024")
    parser.add_argument("--steps",        type=int,  default=500)
    parser.add_argument("--batch-size",   type=int,  default=16)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--device",       type=str,  default="cpu")
    parser.add_argument("--d-model",      type=int,  default=128)
    parser.add_argument("--n-layers",     type=int,  default=4)
    parser.add_argument("--output-dir",   type=str,  default="scripts/results")
    args = parser.parse_args()

    context_lens = [int(x) for x in args.context_lens.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TRNConfig(
        vocab_size    = 256,
        d_model       = args.d_model,
        n_oscillators = args.d_model // 2,
        n_layers      = args.n_layers,
        d_ff          = args.d_model * 4,
        max_seq_len   = max(context_lens) + 64,
    )

    needle_range = NEEDLE_HIGH - NEEDLE_LOW  # 40 distinct needle tokens
    print(f"Needle-in-a-Haystack - {args.steps} steps, lr={args.lr}, bs={args.batch_size}")
    print(f"Context lengths: {context_lens}")
    print(f"Random baseline recall@1 = {100.0 / needle_range:.1f}% (1/{needle_range} needle tokens)")
    print()
    print(f"{'context_len':>12} | {'TRN recall@1':>14} | {'TF recall@1':>14}")
    print("-" * 46)

    rows = []
    for ctx_len in context_lens:
        trn_recall = train_and_eval(
            ctx_len, "TRN", cfg, args.steps, args.batch_size, args.lr, args.seed, args.device,
        )
        tf_recall = train_and_eval(
            ctx_len, "TF", cfg, args.steps, args.batch_size, args.lr, args.seed, args.device,
        )
        print(f"{ctx_len:>12} | {trn_recall * 100:>13.1f}% | {tf_recall * 100:>13.1f}%")
        rows.append({"context_len": ctx_len, "TRN_recall": trn_recall, "TF_recall": tf_recall})

    print()
    print("Note: 4k/8k/16k context lengths require --device cuda with parallel associative_scan.")
    print("GPU speedup enables O(log n) scan vs O(n) sequential on CPU.")

    csv_path = out_dir / "bench_long_context.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["context_len", "model", "recall_at_1", "n_steps"])
        for r in rows:
            w.writerow([r["context_len"], "TRN", f"{r['TRN_recall']:.4f}", args.steps])
            w.writerow([r["context_len"], "TF",  f"{r['TF_recall']:.4f}",  args.steps])
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
