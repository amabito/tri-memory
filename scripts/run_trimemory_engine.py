"""Train full TriMemoryEngine (3-tier: KV Window + TRN + Retrieval) on WikiText-103.

This is the FIRST test of the actual Tri-Memory architecture.
Previous experiments only used TRNModel (TRN + FFN) + manual CausalAttnBlock.

Usage:
    python scripts/run_trimemory_engine.py --epochs 1 --seed 42
    python scripts/run_trimemory_engine.py --epochs 3 --seed 42
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

from trimemory.config import TRNConfig
from trimemory.tri_memory import TriMemoryEngine


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
        out = model(ids, labels=ids)
        total += out["loss"].item()
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
        out = model(ids, labels=ids)
        loss = out["loss"]
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    # Data
    data_dir = Path("data/wikitext103")
    train_data = load_packed(str(data_dir / "train.bin"))
    val_data = load_packed(str(data_dir / "validation.bin"))
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

    # TriMemoryEngine config -- ~45M to match TRNModel hybrid (43.6M)
    cfg = TRNConfig(
        vocab_size=50257, d_model=384, n_oscillators=128,
        n_layers=4, d_ff=768, max_seq_len=256,
        dropout=0.3, gate_bias_init=0.65, state_norm=True, phase_mode="log",
    )

    torch.manual_seed(args.seed)
    model = TriMemoryEngine(
        cfg,
        window_size=64,         # Tier 1: sliding window attention
        chunk_size=32,          # Retrieval chunk granularity
        retrieval_top_k=4,      # Top-4 retrieved chunks
        max_retrieval_chunks=256,
        enable_trn=True,        # Tier 2: TRN
        enable_retrieval=True,  # Tier 3: Retrieval
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TriMemoryEngine: {n_params:,} params")
    print("  Tier 1: KV Window (W=64)")
    print("  Tier 2: TRN (K=256 oscillators)")
    print("  Tier 3: Retrieval (top-4 from 256 chunks)")
    print("  3-way gate: softmax(g_kv, g_trn, g_ret)")

    # Compile for speed
    model = torch.compile(model)

    # Optimizer
    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))

    seq_len, bs = 256, 16
    n_epochs = args.epochs
    max_steps = 200 if args.smoke else None

    print(f"\nTraining for {n_epochs} epochs" + (" (smoke)" if args.smoke else ""))
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10} | {'Steps':>8} | {'Time':>8}")
    print("-" * 55)

    best_val_ppl = float("inf")
    t0 = time.perf_counter()
    results = {"epochs": [], "config": {
        "model": "TriMemoryEngine", "d_model": cfg.d_model,
        "n_oscillators": cfg.n_oscillators, "n_layers": cfg.n_layers,
        "window_size": 64, "chunk_size": 32, "retrieval_top_k": 4,
        "n_params": n_params,
    }}

    for ep in range(n_epochs):
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
            model, train_data, seq_len, bs, optimizer, device, max_steps=max_steps,
        )
        val_ppl = evaluate(model, val_data, seq_len, bs, device)
        ep_time = time.perf_counter() - ep_start

        marker = " *" if val_ppl < best_val_ppl else ""
        best_val_ppl = min(best_val_ppl, val_ppl)

        results["epochs"].append({
            "epoch": ep, "train_loss": round(train_loss, 4),
            "val_ppl": round(val_ppl, 2), "steps": n_steps,
            "time_sec": round(ep_time, 1),
        })

        print(f"{ep:5d} | {train_loss:10.4f} | {val_ppl:10.2f} | {n_steps:8d} | {ep_time/60:7.1f}m{marker}")

    total_time = time.perf_counter() - t0
    results["final"] = {
        "best_val_ppl": round(best_val_ppl, 2),
        "total_time_min": round(total_time / 60, 1),
    }

    print(f"\nBest Val PPL: {best_val_ppl:.2f}")
    print(f"Total time: {total_time / 60:.1f} min")

    out = Path("data") / "trimemory_engine_wt103.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
