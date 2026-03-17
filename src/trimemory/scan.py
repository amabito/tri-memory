from __future__ import annotations

import threading

import torch
from torch import Tensor


# Thread-safe counters for SafeCumprod gradient statistics.
# Accumulated across forward/backward calls; reset by read_and_reset_stats().
# Set _STATS_ENABLED = False for production training to eliminate GPU-CPU syncs.
_STATS_ENABLED = False
_safe_cumprod_lock = threading.Lock()
_safe_cumprod_stats = {
    "total_elements": 0,
    "nonfinite_elements": 0,
    "max_abs_before_guard": 0.0,
    "max_abs_after_guard": 0.0,
    "calls": 0,
}


def read_and_reset_stats() -> dict:
    """Return accumulated SafeCumprod stats and reset counters."""
    with _safe_cumprod_lock:
        snap = dict(_safe_cumprod_stats)
        for k in _safe_cumprod_stats:
            _safe_cumprod_stats[k] = 0 if isinstance(_safe_cumprod_stats[k], int) else 0.0
    return snap


class SafeCumprod(torch.autograd.Function):
    """torch.cumprod with NaN/Inf-safe backward.

    Forward is identical to torch.cumprod(input, dim=1).
    Backward uses the standard cumprod gradient formula:
        grad_input[i] = sum_{j>=i}(grad_output[j] * output[j]) / input[i]
    but replaces any NaN/Inf in grad_input with zero.

    This preserves V5's rich gradient structure for most elements while
    preventing the rare NaN explosion that caused D seed2 collapse.
    """

    @staticmethod
    def forward(ctx, alpha: Tensor, dim: int = 1) -> Tensor:
        result = torch.cumprod(alpha, dim=dim)
        ctx.save_for_backward(alpha, result)
        ctx.dim = dim
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        alpha, result = ctx.saved_tensors
        dim = ctx.dim

        # Standard cumprod backward:
        # grad_input[i] = sum_{j>=i}(grad_output[j] * result[j]) / alpha[i]
        # Implemented as reverse cumsum of (grad_output * result), then / alpha.
        grad_weighted = grad_output * result  # (B, C, K)
        # Reverse cumsum: flip, cumsum, flip back
        grad_sum = torch.flip(
            torch.cumsum(torch.flip(grad_weighted, [dim]), dim=dim), [dim],
        )
        # Guard against near-zero alpha: clamp to 1e-6 to prevent gradient explosion.
        # (Previously 1e-30 which only caught exact zero but let 1e-7 produce 1e20 gradients)
        grad_input = grad_sum / alpha.clamp(min=1e-6)

        # Zero out NaN/Inf unconditionally (no .any() -- avoids graph break under torch.compile)
        nonfinite_mask = ~torch.isfinite(grad_input)
        grad_input = torch.where(nonfinite_mask, torch.zeros_like(grad_input), grad_input)

        # Stats collection (disabled by default -- each .item() forces GPU-CPU sync)
        if _STATS_ENABLED:
            # Note: max_abs_before is measured post-cleanup; for pre-cleanup stats,
            # save grad_sum / alpha before the where() above.
            max_abs_before = grad_input.nan_to_num(nan=0.0, posinf=1e30, neginf=-1e30).abs().max().item()
            n_nonfinite = nonfinite_mask.sum().item()
            max_abs_after = grad_input.abs().max().item()
            with _safe_cumprod_lock:
                _safe_cumprod_stats["total_elements"] += grad_input.numel()
                _safe_cumprod_stats["nonfinite_elements"] += n_nonfinite
                _safe_cumprod_stats["max_abs_before_guard"] = max(
                    _safe_cumprod_stats["max_abs_before_guard"], max_abs_before,
                )
                _safe_cumprod_stats["max_abs_after_guard"] = max(
                    _safe_cumprod_stats["max_abs_after_guard"], max_abs_after,
                )
                _safe_cumprod_stats["calls"] += 1

        return grad_input, None


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


def _kogge_stone_scan(alpha: Tensor, drive: Tensor) -> Tensor:
    """Single-channel O(log n) prefix scan for r_t = a_t * r_{t-1} + d_t.

    Kogge-Stone parallel prefix using the associative monoid:
    (a2, b2) o (a1, b1) = (a2*a1, a2*b1 + b2)

    No division by alpha_cum (numerically stable).
    Autograd-compatible (no in-place ops).
    """
    B, n, K = alpha.shape
    a = alpha
    b = drive

    offset = 1
    while offset < n:
        a_left = torch.cat([a.new_ones(B, offset, K), a[:, :n - offset]], dim=1)
        b_left = torch.cat([b.new_zeros(B, offset, K), b[:, :n - offset]], dim=1)
        b = b + a * b_left
        a = a * a_left
        offset *= 2

    return b


def parallel_resonance_scan(
    alpha:   Tensor,   # (B, n, K)  fp32 decay in [0, 1)
    drive_r: Tensor,   # (B, n, K)  fp32 real drive
    drive_i: Tensor,   # (B, n, K)  fp32 imaginary drive
) -> tuple[Tensor, Tensor]:
    """O(log n) parallel prefix scan on GPU using Kogge-Stone algorithm.

    No Triton/FLA dependency. Pure PyTorch tensor ops.
    7-28x faster than chunked scan depending on sequence length.
    """
    alpha = alpha.float()
    drive_r = drive_r.float()
    drive_i = drive_i.float()
    r_r = _kogge_stone_scan(alpha, drive_r)
    r_i = _kogge_stone_scan(alpha, drive_i)
    return r_r, r_i


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
    # Option B: SafeCumprod -- forward identical to torch.cumprod,
    # backward uses standard cumprod gradient but zeros out NaN/Inf.
    # This preserves V5's gradient structure (rich multiplicative gradients)
    # while preventing the rare NaN that caused D seed2 collapse.
    alpha_cum = SafeCumprod.apply(alpha_chunk, 1)  # (B, C, K)

    # Contribution from previous state: alpha_cum * r_prev
    prev_contrib = alpha_cum * r_prev.unsqueeze(1)  # (B, C, K)

    # Drive contribution via cumsum-rescale:
    # scaled_drive = drive / alpha_cum, then cumsum, then * alpha_cum
    #
    # Use a clamped denominator (min=1e-6) so the cumsum itself always sees
    # correct values. Then override only the final result at positions where
    # alpha_cum < 1e-6 with the O(alpha_cum) near-zero approximation.
    # This avoids the previous bug where torch.where on the pre-cumsum tensor
    # injected ones into non-small positions and corrupted the cumsum.
    safe_alpha_cum = alpha_cum.clamp(min=1e-6)
    scaled_drive = drive_chunk / safe_alpha_cum
    drive_contrib = torch.cumsum(scaled_drive, dim=1) * alpha_cum
    # For positions where alpha_cum < 1e-6, override with near-zero contribution
    # Override small-alpha positions unconditionally (no .any() -- avoids graph break)
    small_alpha = alpha_cum < 1e-6
    drive_contrib = torch.where(small_alpha, drive_chunk * alpha_cum, drive_contrib)

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
    inter_chunk_norm: bool = False,
) -> tuple[Tensor, Tensor]:
    """Chunked resonance scan: O(n/C) Python loop iterations instead of O(n).

    Within each chunk of size C, uses vectorized cumprod + cumsum.
    Between chunks, carries state in a Python loop (C iterations -> n/C iterations).

    WARNING: inter_chunk_norm=True normalizes carried state between chunks, which
    breaks mathematical equivalence with sequential scan. Default is False.
    Enable only if state explosion occurs during CPU fallback training.

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
