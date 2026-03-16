"""Baseline measurement: train TRN on WikiText-2, measure PPL.

This establishes the pre-improvement baseline on real data.
WikiText-2 is small enough to train to convergence in ~30 minutes on RTX 5090.

Tokenizer: character-level (vocab_size=256) for simplicity.
Future: BPE tokenizer for fair comparison against Transformer baselines.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.config import TRNConfig
from trimemory.model import TRNModel


def prepare_wikitext2_char() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download WikiText-2 and convert to character-level token IDs."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def encode_split(split_name: str) -> np.ndarray:
        text = "\n".join(ds[split_name]["text"])
        # Character-level: each byte is a token (0-255)
        ids = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8)
        return ids.astype(np.uint16)

    train = encode_split("train")
    val = encode_split("validation")
    test = encode_split("test")
    print(f"WikiText-2 char-level: train={len(train):,}, val={len(val):,}, test={len(test):,} tokens")
    return train, val, test


def prepare_wikitext2_bpe() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Download WikiText-2 and tokenize with GPT-2 BPE."""
    from datasets import load_dataset

    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        vocab_size = tokenizer.vocab_size  # 50257
    except ImportError:
        print("[WARN] transformers not installed, falling back to char-level")
        train, val, test = prepare_wikitext2_char()
        return train, val, test, 256

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def encode_split(split_name: str) -> np.ndarray:
        text = "\n".join(ds[split_name]["text"])
        ids = tokenizer.encode(text)
        return np.array(ids, dtype=np.uint16)

    train = encode_split("train")
    val = encode_split("validation")
    test = encode_split("test")
    print(f"WikiText-2 BPE: train={len(train):,}, val={len(val):,}, test={len(test):,} tokens (vocab={vocab_size})")
    return train, val, test, vocab_size


def make_packed_tensor(ids: np.ndarray) -> torch.Tensor:
    """Convert numpy uint16 array to packed torch tensor."""
    return torch.from_numpy(ids.astype(np.int64))


@torch.inference_mode()
def evaluate_ppl(
    model: TRNModel,
    data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: str,
) -> float:
    """Compute perplexity on a packed token sequence."""
    model.eval()
    n_tokens = len(data)
    total_loss = 0.0
    n_batches = 0

    # Stride through data in non-overlapping windows
    for start in range(0, n_tokens - seq_len - 1, seq_len * batch_size):
        batch_ids = []
        for b in range(batch_size):
            offset = start + b * seq_len
            if offset + seq_len + 1 > n_tokens:
                break
            batch_ids.append(data[offset : offset + seq_len].unsqueeze(0))

        if not batch_ids:
            break

        input_ids = torch.cat(batch_ids, dim=0).to(device)
        out = model(input_ids, labels=input_ids)
        total_loss += out["loss"].item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(avg_loss)
    return ppl


def train_epoch(
    model: TRNModel,
    data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> float:
    """Train one epoch, return average loss."""
    model.train()
    n_tokens = len(data)
    total_loss = 0.0
    n_steps = 0

    # Shuffle start positions
    n_examples = (n_tokens - 1) // seq_len
    indices = torch.randperm(n_examples)

    for i in range(0, len(indices) - batch_size, batch_size):
        batch_idx = indices[i : i + batch_size]
        batch_ids = []
        for idx in batch_idx:
            offset = idx.item() * seq_len
            if offset + seq_len + 1 > n_tokens:
                continue
            batch_ids.append(data[offset : offset + seq_len].unsqueeze(0))

        if len(batch_ids) < batch_size:
            continue

        input_ids = torch.cat(batch_ids, dim=0).to(device)

        optimizer.zero_grad()
        out = model(input_ids, labels=input_ids)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    return total_loss / max(n_steps, 1)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # -- Prepare data --
    use_bpe = True
    try:
        from transformers import GPT2TokenizerFast  # noqa: F401
    except ImportError:
        use_bpe = False

    if use_bpe:
        train_ids, val_ids, test_ids, vocab_size = prepare_wikitext2_bpe()
    else:
        train_ids, val_ids, test_ids = prepare_wikitext2_char()
        vocab_size = 256

    train_data = make_packed_tensor(train_ids)
    val_data = make_packed_tensor(val_ids)
    test_data = make_packed_tensor(test_ids)

    # -- Model config --
    seq_len = 256
    batch_size = 32 if vocab_size == 256 else 16

    # Exp 4: d=384 + 1 attention layer (combine 3a+3b)
    cfg = TRNConfig(
        vocab_size=vocab_size,
        d_model=384,
        n_oscillators=192,
        n_layers=4,
        d_ff=1536,
        max_seq_len=seq_len,
        gate_bias_init=0.65,
        state_norm=True,
        phase_mode="log",
        dropout=0.1,
    )

    torch.manual_seed(42)
    model = TRNModel(cfg).to(device)

    # 3b: Replace block 2 (index 2) with causal attention
    use_hybrid = True
    if use_hybrid:
        from trimemory.block import CausalAttnBlock
        attn_block = CausalAttnBlock(cfg, n_heads=4).to(device)
        model.blocks[2] = attn_block
        layout = "hybrid [TRN, TRN, Attn, TRN]"
    else:
        layout = "pure TRN"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params, vocab={vocab_size}, d={cfg.d_model}, "
          f"K={cfg.n_oscillators}, L={cfg.n_layers}, seq={seq_len}, layout={layout}")

    # -- Optimizer --
    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))

    # -- Training --
    n_epochs = 20
    best_val_ppl = float("inf")
    results = {
        "config": {
            "vocab_size": vocab_size,
            "d_model": cfg.d_model,
            "n_oscillators": cfg.n_oscillators,
            "n_layers": cfg.n_layers,
            "n_params": n_params,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "tokenizer": "bpe" if use_bpe else "char",
        },
        "epochs": [],
    }

    print(f"\nTraining for {n_epochs} epochs...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10} | {'Time':>6}")
    print("-" * 45)

    start_total = time.perf_counter()
    for epoch in range(n_epochs):
        # LR schedule (cosine with warmup)
        lr_warmup = 3
        if epoch < lr_warmup:
            lr = 3e-4 * (epoch + 1) / lr_warmup
        else:
            progress = (epoch - lr_warmup) / max(1, n_epochs - lr_warmup)
            lr = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(progress * math.pi))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_start = time.perf_counter()
        train_loss = train_epoch(model, train_data, seq_len, batch_size, optimizer, device)
        val_ppl = evaluate_ppl(model, val_data, seq_len, batch_size, device)
        epoch_time = time.perf_counter() - epoch_start

        results["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "lr": round(lr, 6),
            "time_sec": round(epoch_time, 1),
        })

        marker = " *" if val_ppl < best_val_ppl else ""
        best_val_ppl = min(best_val_ppl, val_ppl)
        print(f"{epoch:5d} | {train_loss:10.4f} | {val_ppl:10.2f} | {epoch_time:5.1f}s{marker}")

    total_time = time.perf_counter() - start_total

    # -- Final test PPL --
    test_ppl = evaluate_ppl(model, test_data, seq_len, batch_size, device)

    results["final"] = {
        "best_val_ppl": round(best_val_ppl, 2),
        "test_ppl": round(test_ppl, 2),
        "total_time_sec": round(total_time, 1),
    }

    print(f"\n{'=' * 45}")
    print(f"Best Val PPL:  {best_val_ppl:.2f}")
    print(f"Test PPL:      {test_ppl:.2f}")
    print(f"Total time:    {total_time:.1f}s")

    # Reference baselines (BPE, from literature):
    if use_bpe:
        print(f"\nReference baselines (BPE, WikiText-2):")
        print(f"  Transformer-XL (41M):  24.0 ppl")
        print(f"  GPT-2 small (117M):    29.4 ppl")
        print(f"  LSTM (35M):            65.9 ppl")

    # Save results
    out_path = Path(__file__).parent.parent / "data" / "baseline_wikitext2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
