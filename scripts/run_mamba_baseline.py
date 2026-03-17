"""Minimal Mamba-like baseline for fair comparison with Tri-Memory.

Implements the core Mamba architecture:
- Selective SSM (input-dependent A, B, C)
- Short convolution (conv1d, kernel=4)
- SwiGLU FFN

NOT the official Mamba -- no custom CUDA kernels. Uses PyTorch native ops.
Same throughput disadvantage as TRN without compile.

Usage:
    python scripts/run_mamba_baseline.py --epochs 3 --seed 42
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

torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from trimemory.utils import build_rms_norm


class SelectiveSSM(nn.Module):
    """Simplified selective SSM (Mamba-style).

    r_t = A_t * r_{t-1} + B_t * x_t
    y_t = C_t * r_t

    A, B, C are input-dependent (selective).
    Uses Kogge-Stone scan for parallel training.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_state = d_state

        # Input projection: d_model -> 2*d_model (gate + input)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)

        # Conv1d (short convolution before SSM)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, d_model, bias=True)

        # A: fixed log-space initialization (like Mamba)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_model, -1).clone())

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Gate + input
        xz = self.in_proj(x)  # (B, T, 2D)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, T, D)

        # Conv1d
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM parameters from input
        ssm_params = self.x_proj(x_conv)  # (B, T, 2*d_state + 1)
        B_ssm = ssm_params[:, :, :self.d_state]  # (B, T, d_state)
        C_ssm = ssm_params[:, :, self.d_state:2*self.d_state]
        dt = F.softplus(self.dt_proj(ssm_params[:, :, -1:]))  # (B, T, D)

        # Discretized A
        A = -torch.exp(self.A_log)  # (D, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, T, D, d_state)

        # Discretized B * x
        dBx = dt.unsqueeze(-1) * B_ssm.unsqueeze(2) * x_conv.unsqueeze(-1)  # (B, T, D, d_state)

        # Sequential scan (simple, no parallel scan for d_state>1)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dBx[:, t]  # (B, D, d_state)
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, D)

        # Gate and output
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_ff_hidden: int = 0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = build_rms_norm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.norm2 = build_rms_norm(d_model)
        if d_ff_hidden == 0:
            d_ff_hidden = (int(2 / 3 * d_model * 4) + 255) // 256 * 256
        self.gate_up = nn.Linear(d_model, d_ff_hidden * 2, bias=False)
        self.down = nn.Linear(d_ff_hidden, d_model, bias=False)
        self.d_ff_hidden = d_ff_hidden
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.ssm(self.norm1(x)))
        h = self.norm2(x)
        y = self.gate_up(h)
        gate, up = y.split(self.d_ff_hidden, dim=-1)
        x = x + self.drop(self.down(F.silu(gate) * up))
        return x


class MambaLM(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 512,
                 n_layers: int = 8, d_state: int = 16,
                 max_seq_len: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.drop_emb = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm_out = build_rms_norm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, std=d_model ** -0.5)

    def forward(self, input_ids, labels=None):
        x = self.drop_emb(self.embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm_out(x))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        return result

    def configure_optimizer_param_groups(self, weight_decay=0.1):
        decay, no_decay = set(), set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith(".weight") and "norm" not in n and "emb" not in n:
                decay.add(n)
            else:
                no_decay.add(n)
        return [
            {"params": [self.get_parameter(n) for n in sorted(decay)], "weight_decay": weight_decay},
            {"params": [self.get_parameter(n) for n in sorted(no_decay)], "weight_decay": 0.0},
        ]


def load_packed(path):
    return torch.from_numpy(np.memmap(path, dtype=np.uint16, mode="r").astype(np.int64))


@torch.inference_mode()
def evaluate(model, data, sl, bs, device):
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(data) - sl - 1, sl * bs):
        batch = [data[s + b*sl : s + b*sl + sl].unsqueeze(0) for b in range(bs) if s + b*sl + sl + 1 <= len(data)]
        if not batch: break
        ids = torch.cat(batch).to(device)
        total += model(ids, labels=ids)["loss"].item()
        n += 1
    return math.exp(total / max(n, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda"
    train_data = load_packed("data/wikitext103/train.bin")
    val_data = load_packed("data/wikitext103/validation.bin")

    torch.manual_seed(args.seed)
    model = MambaLM(vocab_size=50257, d_model=512, n_layers=8, d_state=16,
                    max_seq_len=256, dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MambaLM: {n_params:,} params")

    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.configure_optimizer_param_groups(0.1), lr=3e-4, betas=(0.9, 0.95))

    sl, bs = 256, 16

    for ep in range(args.epochs):
        lr = 3e-4 if ep == 0 else 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos((ep-1) / max(1, args.epochs-1) * math.pi))
        for pg in optimizer.param_groups: pg["lr"] = lr

        model.train()
        indices = torch.randperm((len(train_data)-1)//sl)
        ns = 0
        t0 = time.perf_counter()
        for i in range(0, len(indices)-bs, bs):
            batch = []
            for idx in indices[i:i+bs]:
                off = idx.item()*sl
                if off+sl+1 > len(train_data): continue
                batch.append(train_data[off:off+sl].unsqueeze(0))
            if len(batch) < bs: continue
            ids = torch.cat(batch).to(device)
            optimizer.zero_grad()
            model(ids, labels=ids)["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ns += 1
        ep_time = time.perf_counter() - t0
        val_ppl = evaluate(model, val_data, sl, bs, device)
        print(f"ep{ep}: PPL={val_ppl:.2f}, {ep_time/60:.1f}m, {ns*bs*sl/ep_time:.0f} tok/s")

    print(f"Final PPL: {val_ppl:.2f}")
    with open("data/mamba_baseline.json", "w") as f:
        json.dump({"params": n_params, "ppl": round(val_ppl, 2), "epochs": args.epochs}, f)


if __name__ == "__main__":
    main()
