"""Transformer baseline on WikiText-103 -- same params/data/tokenizer as TRN.

Pure causal Transformer (GPT-2 style) for fair comparison.
44M params, BPE 50K, WikiText-103, same optimizer/schedule/dropout.

Usage:
    python scripts/run_transformer_baseline.py --epochs 10 --seed 42
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trimemory.utils import build_rms_norm


# ---------------------------------------------------------------------------
# Transformer Model (GPT-2 style, ~44M params)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        std = d_model ** -0.5
        nn.init.normal_(self.qkv.weight, std=std)
        nn.init.normal_(self.proj.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff_hidden: int) -> None:
        super().__init__()
        self.d_ff_hidden = d_ff_hidden
        self.gate_up = nn.Linear(d_model, d_ff_hidden * 2, bias=False)
        self.down = nn.Linear(d_ff_hidden, d_model, bias=False)
        std_in = d_model ** -0.5
        std_out = d_ff_hidden ** -0.5
        nn.init.normal_(self.gate_up.weight, std=std_in)
        nn.init.normal_(self.down.weight, std=std_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gate_up(x)
        gate, up = y.split(self.d_ff_hidden, dim=-1)
        return self.down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff_hidden: int,
                 max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = build_rms_norm(d_model)
        self.norm2 = build_rms_norm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ffn = SwiGLU(d_model, d_ff_hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerLM(nn.Module):
    """GPT-2 style causal LM. Matched to TRN's ~44M param budget.

    Config: d=512, n_heads=8, n_layers=8, d_ff=1024 (d_ff_hidden=768)
    Embedding: 50257 * 512 = 25.7M
    8 blocks: ~2.2M each = 17.6M
    Total: ~43.3M (close to TRN's 43.6M)
    """

    def __init__(self, vocab_size: int = 50257, d_model: int = 512,
                 n_heads: int = 8, n_layers: int = 8, d_ff: int = 1024,
                 max_seq_len: int = 256, dropout: float = 0.3,
                 tie_weights: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.drop_emb = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable position embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        d_ff_hidden = (int(2 / 3 * d_ff) + 255) // 256 * 256  # same as TRN
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff_hidden, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = build_rms_norm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        nn.init.normal_(self.embedding.weight, std=d_model ** -0.5)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels=None) -> dict:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop_emb(self.embedding(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        logits = self.lm_head(x)
        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return result

    def configure_optimizer_param_groups(self, weight_decay: float = 0.1):
        decay = set()
        no_decay = set()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".weight") and "norm" not in name and "emb" not in name:
                decay.add(name)
            else:
                no_decay.add(name)
        return [
            {"params": [self.get_parameter(n) for n in sorted(decay)], "weight_decay": weight_decay},
            {"params": [self.get_parameter(n) for n in sorted(no_decay)], "weight_decay": 0.0},
        ]


# ---------------------------------------------------------------------------
# Data / Eval / Train (reuse from scale_test)
# ---------------------------------------------------------------------------

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Run 2 epochs x 500 steps")
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

    # Model -- matched to TRN's param budget
    torch.manual_seed(args.seed)
    model = TransformerLM(
        vocab_size=50257, d_model=512, n_heads=8, n_layers=8,
        d_ff=1024, max_seq_len=256, dropout=0.3, tie_weights=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer: {n_params:,} params (d=512, heads=8, L=8, d_ff=1024, dropout=0.3)")

    # Optimizer -- same as TRN
    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))

    seq_len, bs = 256, 16
    n_epochs = args.epochs
    max_steps = 500 if args.smoke else None
    if args.smoke:
        n_epochs = 2

    steps_per_epoch = (len(train_data) - 1) // seq_len // bs
    print(f"Steps/epoch: ~{steps_per_epoch:,}, Epochs: {n_epochs}")

    results = {"epochs": [], "config": {
        "model": "transformer", "d_model": 512, "n_heads": 8,
        "n_layers": 8, "d_ff": 1024, "dropout": 0.3,
        "n_params": n_params, "data": "wikitext103",
    }}

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10} | "
          f"{'Steps':>8} | {'Time':>8} | {'tok/s':>8}")
    print("-" * 65)

    best_val_ppl = float("inf")
    t0 = time.perf_counter()

    for ep in range(n_epochs):
        # Cosine LR (same as TRN)
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

    out = Path("data") / "transformer_baseline_wt103.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
