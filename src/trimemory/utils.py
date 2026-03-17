from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def build_rms_norm(d_model: int, eps: float = 1e-6) -> nn.Module:
    """Return nn.RMSNorm if available (PyTorch >= 2.4), else custom fallback."""
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model, eps=eps)
    return _RMSNorm(d_model, eps=eps)


class _RMSNorm(nn.Module):
    """RMSNorm compatible with PyTorch < 2.4.

    Computes in fp32 for numerical stability, outputs in input dtype.
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_f  = x.float()
        norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


def build_sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
    """Build sinusoidal positional encoding buffer."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term[:pe[:, 0::2].size(1)])
    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
    return pe


def num_parameters(model: nn.Module, non_embedding: bool = True) -> int:
    """Count trainable parameters, optionally excluding embedding tables."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if non_embedding and hasattr(model, "embedding"):
        total -= model.embedding.weight.numel()
    elif non_embedding and hasattr(model, "embed"):
        total -= model.embed.weight.numel()
    return total


def configure_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    base_lr: float = 3e-4,
) -> list[dict]:
    """Split parameters into decay / no-decay / slow-lr groups.

    Groups:
    - decay:    weight matrices -- weight_decay applied, full LR
    - no_decay: biases, norms, embeddings -- no weight decay, full LR
    - slow_lr:  omega_base, res_scale -- no weight decay, 0.1x LR
                (these are oscillator frequency/scale params that benefit
                from slower updates to avoid destabilizing learned frequencies)
    """
    decay: set[str] = set()
    no_decay: set[str] = set()
    slow_lr: set[str] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "omega_base" in name or "res_scale" in name:
            slow_lr.add(name)
        elif (
            name.endswith(".bias")
            or "norm" in name.lower()
            or "embedding" in name
            or "embed" in name
        ):
            no_decay.add(name)
        else:
            decay.add(name)

    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    groups: list[dict] = [
        {"params": [params[n] for n in sorted(decay)], "weight_decay": weight_decay},
        {"params": [params[n] for n in sorted(no_decay)], "weight_decay": 0.0},
    ]
    if slow_lr:
        groups.append(
            {
                "params": [params[n] for n in sorted(slow_lr)],
                "weight_decay": 0.0,
                "lr": base_lr * 0.1,
            }
        )
    return groups
