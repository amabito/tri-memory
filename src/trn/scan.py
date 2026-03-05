from __future__ import annotations

import torch
from torch import Tensor


def _combine(
    x: tuple[Tensor, Tensor],
    y: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    """Associative combine for the first-order linear recurrence s_t = a_t*s_{t-1} + b_t.

    Monoid element (a, b) represents: s = a * s_prev + b.
    Composition law: (a2, b2) o (a1, b1) = (a2*a1,  a2*b1 + b2).

    Uses torch.addcmul for fused multiply-add — numerically identical to
    a2*b1+b2 but avoids a temporary tensor.
    """
    a1, b1 = x
    a2, b2 = y
    return a2 * a1, torch.addcmul(b2, a2, b1)


def parallel_resonance_scan(
    alpha:   Tensor,   # (B, n, K)  fp32 decay in [0, 1)
    drive_r: Tensor,   # (B, n, K)  fp32 real drive
    drive_i: Tensor,   # (B, n, K)  fp32 imaginary drive
) -> tuple[Tensor, Tensor]:
    """O(log n) parallel prefix scan on GPU.

    Requires torch.associative_scan (PyTorch >= 2.1 experimental).
    The API signature changed between minor versions; this wrapper
    catches all failures and delegates to the sequential fallback.
    """
    if not hasattr(torch, "associative_scan"):
        return sequential_resonance_scan(alpha, drive_r, drive_i)

    try:
        # PyTorch >= 2.1: associative_scan(combine_fn, elements, dim)
        # _combine returns (accumulated_alpha, state_s).  We need the state,
        # which is the SECOND element — not the first.
        _, r_r = torch.associative_scan(_combine, (alpha, drive_r), dim=1)
        _, r_i = torch.associative_scan(_combine, (alpha, drive_i), dim=1)
        return r_r, r_i
    except Exception:
        # API changed or CUDA kernel unavailable — fall back gracefully.
        return sequential_resonance_scan(alpha, drive_r, drive_i)


def sequential_resonance_scan(
    alpha:   Tensor,   # (B, n, K)  fp32
    drive_r: Tensor,   # (B, n, K)  fp32
    drive_i: Tensor,   # (B, n, K)  fp32
) -> tuple[Tensor, Tensor]:
    """Sequential (CPU-safe, training-fallback) resonance scan.

    O(n) time.  State vector fits in L1 cache for K <= 256
    (2 * K * 4 bytes = 2 KB at fp32).
    """
    B, n, K = alpha.shape

    r_r = alpha.new_zeros(B, K)
    r_i = alpha.new_zeros(B, K)

    out_r = torch.empty_like(drive_r)
    out_i = torch.empty_like(drive_i)

    for t in range(n):
        a_t  = alpha[:, t]      # (B, K)
        r_r  = a_t * r_r + drive_r[:, t]
        r_i  = a_t * r_i + drive_i[:, t]
        out_r[:, t] = r_r
        out_i[:, t] = r_i

    return out_r, out_i
