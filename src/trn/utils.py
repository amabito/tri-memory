from __future__ import annotations

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
