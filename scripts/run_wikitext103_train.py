"""Train 44M hybrid TRN on WikiText-103 (118M tokens).

Best config from Phase 2C: d=512, K=256, L=8, hybrid 2 attn, dropout=0.3
Data: packed uint16 binary at data/wikitext103/

Key difference from WikiText-2:
  WT-2: 2.4M tokens, 44M params -> 18:1 param/token (severe overfit)
  WT-103: 118M tokens, 44M params -> 0.37:1 param/token (underfitting expected)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.block import CausalAttnBlock
from trimemory.config import TRNConfig
from trimemory.model import TRNModel


def load_packed(path: str) -> torch.Tensor:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    return torch.from_numpy(data.astype(np.int64))


@torch.inference_mode()
def evaluate(model, data, seq_len, bs, device):
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(data) - seq_len - 1, seq_len * bs):
        batch = []
        for b in range(bs):
            off = s + b * seq_len
            if off + seq_len + 1 > len(data):
                break
            batch.append(data[off : off + seq_len].unsqueeze(0))
        if not batch:
            break
        ids = torch.cat(batch).to(device)
        total += model(ids, labels=ids)["loss"].item()
        n += 1
    return math.exp(total / max(n, 1))


def train_epoch(model, data, seq_len, bs, optimizer, device, max_steps=None):
    model.train()
    n_tokens = len(data)
    total_loss, n_steps = 0.0, 0
    n_examples = (n_tokens - 1) // seq_len
    indices = torch.randperm(n_examples)

    for i in range(0, len(indices) - bs, bs):
        batch = []
        for idx in indices[i : i + bs]:
            off = idx.item() * seq_len
            if off + seq_len + 1 > n_tokens:
                continue
            batch.append(data[off : off + seq_len].unsqueeze(0))
        if len(batch) < bs:
            continue

        ids = torch.cat(batch).to(device)
        optimizer.zero_grad()
        loss = model(ids, labels=ids)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_steps += 1

        if max_steps and n_steps >= max_steps:
            break

    return total_loss / max(n_steps, 1), n_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="Limit steps per epoch for quick test")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Data
    data_dir = Path("data/wikitext103")
    train_data = load_packed(str(data_dir / "train.bin"))
    val_data = load_packed(str(data_dir / "validation.bin"))
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

    # Model (Phase 2C best)
    cfg = TRNConfig(
        vocab_size=50257, d_model=512, n_oscillators=256,
        n_layers=8, d_ff=1024, max_seq_len=256,
        dropout=0.3, gate_bias_init=0.65, state_norm=True, phase_mode="log",
    )
    torch.manual_seed(args.seed)
    model = TRNModel(cfg).to(device)
    model.blocks[2] = CausalAttnBlock(cfg, n_heads=8).to(device)
    model.blocks[6] = CausalAttnBlock(cfg, n_heads=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Optimizer
    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))

    seq_len, bs = 256, 16
    n_epochs = args.epochs
    # Steps per epoch: 118M / 256 / 16 = ~28,800
    steps_per_epoch = (len(train_data) - 1) // seq_len // bs
    print(f"Steps/epoch: ~{steps_per_epoch:,}, Epochs: {n_epochs}")
    print(f"Total tokens: ~{steps_per_epoch * bs * seq_len * n_epochs / 1e6:.0f}M")

    results = {"epochs": [], "config": {
        "d_model": cfg.d_model, "n_oscillators": cfg.n_oscillators,
        "n_layers": cfg.n_layers, "dropout": cfg.dropout,
        "n_params": n_params, "data": "wikitext103",
    }}

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10} | "
          f"{'Steps':>8} | {'Time':>8} | {'tok/s':>8}")
    print("-" * 65)

    best_val_ppl = float("inf")
    t0 = time.perf_counter()

    for ep in range(n_epochs):
        # Cosine LR
        warmup = max(1, n_epochs // 10)
        if ep < warmup:
            lr = 3e-4 * (ep + 1) / warmup
        else:
            p = (ep - warmup) / max(1, n_epochs - warmup)
            lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(p * math.pi))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        ep_start = time.perf_counter()
        train_loss, n_steps = train_epoch(
            model, train_data, seq_len, bs, optimizer, device,
            max_steps=args.max_steps_per_epoch,
        )
        val_ppl = evaluate(model, val_data, seq_len, bs, device)
        ep_time = time.perf_counter() - ep_start
        tok_per_sec = n_steps * bs * seq_len / ep_time

        marker = " *" if val_ppl < best_val_ppl else ""
        best_val_ppl = min(best_val_ppl, val_ppl)

        results["epochs"].append({
            "epoch": ep, "train_loss": round(train_loss, 4),
            "val_ppl": round(val_ppl, 2), "steps": n_steps,
            "time_sec": round(ep_time, 1), "tok_per_sec": round(tok_per_sec, 0),
        })

        print(f"{ep:5d} | {train_loss:10.4f} | {val_ppl:10.2f} | "
              f"{n_steps:8d} | {ep_time/60:7.1f}m | {tok_per_sec:7.0f}{marker}")

    total_time = time.perf_counter() - t0
    results["final"] = {
        "best_val_ppl": round(best_val_ppl, 2),
        "total_time_min": round(total_time / 60, 1),
    }

    print(f"\nBest Val PPL: {best_val_ppl:.2f}")
    print(f"Total time: {total_time / 60:.1f} min")

    out = Path("data") / "wikitext103_train.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
