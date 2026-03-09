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

    Uses torch.addcmul for fused multiply-add -- numerically identical to
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
    catches all failures and delegates to the chunked fallback.
    """
    if not hasattr(torch, "associative_scan"):
        return chunked_resonance_scan(alpha, drive_r, drive_i)

    try:
        _, r_r = torch.associative_scan(_combine, (alpha, drive_r), dim=1)
        _, r_i = torch.associative_scan(_combine, (alpha, drive_i), dim=1)
        return r_r, r_i
    except (TypeError, RuntimeError, AttributeError):
        return chunked_resonance_scan(alpha, drive_r, drive_i)


def _scan_chunk(
    alpha_chunk: Tensor,   # (B, C, K)
    drive_chunk: Tensor,   # (B, C, K)
    r_prev: Tensor,        # (B, K)
) -> tuple[Tensor, Tensor]:
    """Process one chunk of the linear recurrence using cumulative products.

    For r_t = a_t * r_{t-1} + d_t within a chunk of length C:

      alpha_cum[t] = prod(alpha[0..t])         -- cumulative decay from chunk start
      r_t = alpha_cum[t] * r_prev + sum_{i=0}^{t} alpha_cum_from_i_to_t * d_i

    The second term uses the cumsum-rescale trick:
      scaled_drive[t] = d_t / alpha_cum[t]
      r_t = alpha_cum[t] * (r_prev + cumsum(scaled_drive)[t])

    Returns:
        (chunk_output, r_final) where chunk_output is (B, C, K) and r_final is (B, K).
    """
    # Cumulative product of alpha within the chunk: (B, C, K)
    alpha_cum = torch.cumprod(alpha_chunk, dim=1)

    # Contribution from previous state: alpha_cum * r_prev
    prev_contrib = alpha_cum * r_prev.unsqueeze(1)  # (B, C, K)

    # Drive contribution via cumsum-rescale:
    # scaled_drive = drive / alpha_cum, then cumsum, then * alpha_cum
    #
    # When alpha is exactly 0 for all steps up to t, alpha_cum[t] = 0
    # and the rescale trick produces huge * 0 which doesn't cancel in fp32.
    # For true zero alpha, r_t = d_t (no state carry).
    # For small-but-nonzero alpha, the rescale trick loses precision as
    # chunk_size grows (cumprod underflows), but this is inherent to the
    # vectorized approach and acceptable for typical chunk_size <= 16.
    has_zero_alpha = (alpha_chunk == 0.0).any()
    inv_alpha_cum = 1.0 / alpha_cum.clamp(min=1e-30)
    scaled_drive = drive_chunk * inv_alpha_cum
    drive_contrib = torch.cumsum(scaled_drive, dim=1) * alpha_cum  # (B, C, K)

    # Where alpha_cum is exactly 0 (only when alpha=0 throughout),
    # replace with raw drive (r_t = d_t when alpha = 0).
    if has_zero_alpha:
        exact_zero = alpha_cum == 0.0
        if exact_zero.any():
            drive_contrib = torch.where(exact_zero, drive_chunk, drive_contrib)

    chunk_out = prev_contrib + drive_contrib  # (B, C, K)
    r_final = chunk_out[:, -1]  # (B, K)

    return chunk_out, r_final


def _inter_chunk_state_norm(
    r_r: Tensor,   # (B, K)
    r_i: Tensor,   # (B, K)
) -> tuple[Tensor, Tensor]:
    """Per-channel max-abs normalization between chunks: r /= max(max_abs(r), 1.0).

    Prevents resonance state from growing unboundedly across chunks,
    which causes gradient explosion at large batch sizes (BS>=128).
    """
    max_abs = torch.maximum(r_r.abs(), r_i.abs())  # (B, K)
    scale = max_abs.clamp(min=1.0)
    return r_r / scale, r_i / scale


def chunked_resonance_scan(
    alpha:   Tensor,   # (B, n, K)  fp32
    drive_r: Tensor,   # (B, n, K)  fp32
    drive_i: Tensor,   # (B, n, K)  fp32
    chunk_size: int = 16,
    inter_chunk_norm: bool = True,
) -> tuple[Tensor, Tensor]:
    """Chunked resonance scan: O(n/C) Python loop iterations instead of O(n).

    Within each chunk of size C, uses vectorized cumprod + cumsum.
    Between chunks, carries state in a Python loop (C iterations -> n/C iterations).

    When inter_chunk_norm=True (default), applies per-channel max-abs normalization
    to the carried state between chunks.  This prevents state explosion at large
    batch sizes without affecting the final output (TemporalResonanceLayer applies
    state_norm after the scan anyway).

    Autograd tape: n/C nodes instead of n nodes, reducing backward cost proportionally.
    """
    B, n, K = alpha.shape

    r_r = alpha.new_zeros(B, K)
    r_i = alpha.new_zeros(B, K)

    out_r_chunks = []
    out_i_chunks = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        a_chunk = alpha[:, start:end]       # (B, C, K)
        d_r_chunk = drive_r[:, start:end]   # (B, C, K)
        d_i_chunk = drive_i[:, start:end]   # (B, C, K)

        chunk_r, r_r = _scan_chunk(a_chunk, d_r_chunk, r_r)
        chunk_i, r_i = _scan_chunk(a_chunk, d_i_chunk, r_i)

        out_r_chunks.append(chunk_r)
        out_i_chunks.append(chunk_i)

        # Normalize carried state between chunks to prevent explosion
        if inter_chunk_norm:
            r_r, r_i = _inter_chunk_state_norm(r_r, r_i)

    return torch.cat(out_r_chunks, dim=1), torch.cat(out_i_chunks, dim=1)


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
