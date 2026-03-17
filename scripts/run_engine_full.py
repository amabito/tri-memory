"""TriMemoryEngine: compiled WT-103 training + recall evaluation.

Priority 1: Train 3-tier model on WT-103
Priority 2: Test exact recall after pretraining
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.config import TRNConfig
from trimemory.tri_memory import TriMemoryEngine


def load_packed(path: str) -> torch.Tensor:
    return torch.from_numpy(np.memmap(path, dtype=np.uint16, mode="r").astype(np.int64))


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


def train_epoch(model, data, seq_len, bs, optimizer, device):
    model.train()
    n_tokens = len(data)
    total_loss, n_steps = 0.0, 0
    indices = torch.randperm((n_tokens - 1) // seq_len)
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
        model(ids, labels=ids)["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += model(ids, labels=ids)["loss"].item() if n_steps == 0 else total_loss
        n_steps += 1
    # Recompute avg loss properly
    model.eval()
    sample_loss = 0.0
    with torch.inference_mode():
        count = 0
        for i in range(0, min(500 * bs, len(indices) - bs), bs):
            batch = []
            for idx in indices[i : i + bs]:
                off = idx.item() * seq_len
                if off + seq_len + 1 > n_tokens:
                    continue
                batch.append(data[off : off + seq_len].unsqueeze(0))
            if len(batch) < bs:
                continue
            ids = torch.cat(batch).to(device)
            sample_loss += model(ids, labels=ids)["loss"].item()
            count += 1
            if count >= 50:
                break
    return sample_loss / max(count, 1), n_steps


@torch.inference_mode()
def test_recall(model, device, vocab=128, n_pairs=8, n_trials=500):
    """Associative recall: KV pairs -> query K -> predict V."""
    model.eval()
    correct = 0
    for _ in range(n_trials):
        keys = torch.arange(n_pairs) % vocab
        vals = torch.randint(0, vocab, (n_pairs,))
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([k.item(), v.item()])
        qi = torch.randint(0, n_pairs, (1,)).item()
        seq.append(vocab)  # sep
        seq.append(keys[qi].item())
        ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        pred = model(ids)["logits"][0, -1, :vocab].argmax().item()
        if pred == vals[qi].item():
            correct += 1
    return correct / n_trials


def main():
    device = "cuda"
    print(f"Device: {device}")

    train_data = load_packed("data/wikitext103/train.bin")
    val_data = load_packed("data/wikitext103/validation.bin")
    print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")

    cfg = TRNConfig(
        vocab_size=50257, d_model=384, n_oscillators=128,
        n_layers=4, d_ff=768, max_seq_len=256,
        dropout=0.3, gate_bias_init=0.65, state_norm=True, phase_mode="log",
    )

    torch.manual_seed(42)
    model = TriMemoryEngine(
        cfg, window_size=64, chunk_size=32,
        retrieval_top_k=4, max_retrieval_chunks=256,
        enable_trn=True, enable_retrieval=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TriMemoryEngine: {n_params:,} params (3-tier)")

    model = torch.compile(model)

    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))

    seq_len, bs = 256, 16
    n_epochs = 10
    results = {"epochs": [], "recall": {}}

    print(f"\n{'Ep':>3} | {'Val PPL':>8} | {'Time':>6}")
    print("-" * 25)

    best_ppl = float("inf")
    t0 = time.perf_counter()

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
        _, n_steps = train_epoch(model, train_data, seq_len, bs, optimizer, device)
        val_ppl = evaluate(model, val_data, seq_len, bs, device)
        ep_time = time.perf_counter() - ep_start

        marker = " *" if val_ppl < best_ppl else ""
        best_ppl = min(best_ppl, val_ppl)
        results["epochs"].append({"epoch": ep, "val_ppl": round(val_ppl, 2), "time_sec": round(ep_time, 1)})
        print(f"{ep:3d} | {val_ppl:8.2f} | {ep_time/60:5.1f}m{marker}")

    total_time = time.perf_counter() - t0
    print(f"\nBest PPL: {best_ppl:.2f}, Total: {total_time/60:.1f} min")

    # Recall test
    print("\n=== Recall Test (after WT-103 pretraining) ===")
    for n_pairs in [4, 8, 16]:
        acc = test_recall(model, device, vocab=128, n_pairs=n_pairs)
        results["recall"][f"pairs_{n_pairs}"] = round(acc, 4)
        print(f"  AR (vocab=128, pairs={n_pairs:2d}): {acc:.4f} (random={1/128:.4f})")

    results["final"] = {"best_val_ppl": round(best_ppl, 2), "total_time_min": round(total_time/60, 1)}

    with open("data/engine_full_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/engine_full_results.json")


if __name__ == "__main__":
    main()
