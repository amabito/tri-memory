"""Recall/Retrieval benchmark: test exact recall ability of TriMemoryEngine vs TRNModel.

Tests:
1. Needle in a Haystack: insert fact, query at end
2. Associative Recall: KV pairs -> query K -> predict V
3. Copy: reproduce input sequence after separator

Compares TriMemoryEngine (3-tier with Retrieval) vs TRNModel hybrid (no Retrieval).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.block import CausalAttnBlock
from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.tri_memory import TriMemoryEngine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_trn_hybrid(cfg, device=DEVICE):
    model = TRNModel(cfg).to(device)
    model.blocks[2] = CausalAttnBlock(cfg, n_heads=8).to(device)
    model.blocks[6] = CausalAttnBlock(cfg, n_heads=8).to(device)
    return model


def build_engine(cfg, device=DEVICE):
    return TriMemoryEngine(
        cfg, window_size=64, chunk_size=32,
        retrieval_top_k=4, max_retrieval_chunks=256,
        enable_trn=True, enable_retrieval=True,
    ).to(device)


def quick_train(model, n_steps=500, vocab=50257, seq_len=256, bs=8):
    """Quick training on random data to get the model into a reasonable state."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(n_steps):
        ids = torch.randint(0, vocab, (bs, seq_len), device=DEVICE)
        optimizer.zero_grad()
        loss = model(ids, labels=ids)["loss"]
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"    train step {step+1}/{n_steps}: loss={loss.item():.4f}")


@torch.inference_mode()
def test_associative_recall(model, vocab_size=64, n_pairs=8, n_trials=200):
    """Insert KV pairs, query K, check if model predicts V."""
    model.eval()
    correct = 0
    for _ in range(n_trials):
        keys = torch.arange(n_pairs) % vocab_size
        vals = torch.randint(0, vocab_size, (n_pairs,))
        seq = []
        for k, v in zip(keys, vals):
            seq.extend([k.item(), v.item()])
        query_idx = torch.randint(0, n_pairs, (1,)).item()
        seq.append(vocab_size)  # separator
        seq.append(keys[query_idx].item())

        ids = torch.tensor(seq, dtype=torch.long, device=DEVICE).unsqueeze(0)
        out = model(ids)
        pred = out["logits"][0, -1, :vocab_size].argmax().item()
        if pred == vals[query_idx].item():
            correct += 1
    return correct / n_trials


@torch.inference_mode()
def test_copy(model, vocab_size=64, seq_len=16, n_trials=200):
    """Input seq + separator + check if model reproduces seq."""
    model.eval()
    sep = vocab_size  # separator token
    total_correct, total_tokens = 0, 0
    for _ in range(n_trials):
        seq = torch.randint(0, vocab_size, (seq_len,))
        # input: [seq, sep, seq[:-1]], predict: seq (shifted)
        full = torch.cat([seq, torch.tensor([sep]), seq])
        ids = full.unsqueeze(0).to(DEVICE)
        out = model(ids)
        # Check predictions for positions after separator
        start = seq_len + 1  # position after sep
        for i in range(seq_len - 1):
            pred = out["logits"][0, start + i, :vocab_size].argmax().item()
            target = seq[i + 1].item()
            if pred == target:
                total_correct += 1
            total_tokens += 1
    return total_correct / max(total_tokens, 1)


def main():
    print(f"Device: {DEVICE}")

    # Small config for fast training
    cfg_trn = TRNConfig(
        vocab_size=128, d_model=256, n_oscillators=128,
        n_layers=8, d_ff=512, max_seq_len=256,
        dropout=0.0, gate_bias_init=0.65, state_norm=True, phase_mode="log",
    )
    cfg_engine = TRNConfig(
        vocab_size=128, d_model=256, n_oscillators=128,
        n_layers=4, d_ff=512, max_seq_len=256,
        dropout=0.0, gate_bias_init=0.65, state_norm=True, phase_mode="log",
    )

    results = {}

    for name, build_fn, cfg in [
        ("TRN-hybrid", lambda c: build_trn_hybrid(c), cfg_trn),
        ("TriMemoryEngine", lambda c: build_engine(c), cfg_engine),
    ]:
        print(f"\n=== {name} ===")
        torch.manual_seed(42)
        model = build_fn(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        print("  Training on associative recall data...")
        # Train specifically on KV pair patterns
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for step in range(1000):
            # Generate KV pair training data
            bs = 16
            batch_ids = []
            for _ in range(bs):
                n_pairs = 8
                keys = torch.arange(n_pairs) % 64
                vals = torch.randint(0, 64, (n_pairs,))
                seq = []
                for k, v in zip(keys, vals):
                    seq.extend([k.item(), v.item()])
                qi = torch.randint(0, n_pairs, (1,)).item()
                seq.append(64)  # sep
                seq.append(keys[qi].item())
                seq.append(vals[qi].item())  # target
                # pad to fixed length
                while len(seq) < 20:
                    seq.append(0)
                batch_ids.append(torch.tensor(seq[:20], dtype=torch.long))
            ids = torch.stack(batch_ids).to(DEVICE)
            optimizer.zero_grad()
            loss = model(ids, labels=ids)["loss"]
            loss.backward()
            optimizer.step()
            if (step + 1) % 200 == 0:
                print(f"    step {step+1}/1000: loss={loss.item():.4f}")

        # Test
        print("  Testing associative recall...")
        ar_acc = test_associative_recall(model, vocab_size=64, n_pairs=8, n_trials=500)
        print(f"  Associative Recall: {ar_acc:.4f}")

        print("  Testing copy...")
        copy_acc = test_copy(model, vocab_size=64, seq_len=8, n_trials=500)
        print(f"  Copy: {copy_acc:.4f}")

        results[name] = {
            "params": n_params,
            "associative_recall": round(ar_acc, 4),
            "copy": round(copy_acc, 4),
        }
        del model
        torch.cuda.empty_cache()

    print("\n=== COMPARISON ===")
    print(f"{'Model':<20} {'Assoc Recall':>12} {'Copy':>12}")
    print("-" * 48)
    for name, r in results.items():
        print(f"{name:<20} {r['associative_recall']:>12.4f} {r['copy']:>12.4f}")

    random_ar = 1.0 / 64
    random_copy = 1.0 / 64
    print(f"{'Random baseline':<20} {random_ar:>12.4f} {random_copy:>12.4f}")

    with open("data/recall_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/recall_benchmark.json")


if __name__ == "__main__":
    main()
