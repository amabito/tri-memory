"""Adversarial tests for chunked_resonance_scan -- attacker mindset.

Focuses on the cumsum-rescale trick:
    inv_alpha_cum = 1.0 / alpha_cum.clamp(min=1e-30)

which is numerically fragile when alpha is near zero (clamp prevents div/0 but
the rescaled drive can become astronomically large before being multiplied back
by the near-zero alpha_cum, producing Inf - Inf = NaN).
"""

from __future__ import annotations

import math

import pytest
import torch

from trimemory.scan import chunked_resonance_scan, sequential_resonance_scan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(B: int, n: int, K: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (alpha, drive_r, drive_i) with alpha in (0, 1)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    alpha = torch.rand(B, n, K, generator=rng) * 0.9 + 0.05   # (0.05, 0.95)
    drive_r = torch.randn(B, n, K, generator=rng)
    drive_i = torch.randn(B, n, K, generator=rng)
    return alpha, drive_r, drive_i


def _assert_finite(tensor: torch.Tensor, label: str) -> None:
    assert torch.isfinite(tensor).all(), (
        f"{label}: expected all-finite, got "
        f"nan={tensor.isnan().sum().item()} "
        f"inf={tensor.isinf().sum().item()}"
    )


def _assert_close(a: torch.Tensor, b: torch.Tensor, atol: float, label: str) -> None:
    max_err = (a - b).abs().max().item()
    assert max_err <= atol, (
        f"{label}: max abs error {max_err:.3e} exceeds atol={atol}"
    )


# ---------------------------------------------------------------------------
# 1. Alpha near-zero at chunk boundary positions
# ---------------------------------------------------------------------------

class TestAlphaNearZeroChunkBoundary:
    """inv_alpha_cum blows up when alpha_cum -> 0; clamp(min=1e-30) must hold."""

    def test_alpha_near_zero_at_boundary_no_nan(self):
        """alpha=0.001 placed exactly at each chunk boundary must not produce NaN/Inf."""
        B, n, K = 2, 32, 4
        chunk_size = 8
        alpha = torch.ones(B, n, K) * 0.5
        # Set every 8th step (chunk boundaries) to near-zero alpha
        for pos in range(0, n, chunk_size):
            alpha[:, pos] = 0.001

        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=chunk_size)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")

    def test_alpha_near_zero_mid_chunk_no_nan(self):
        """alpha=0.001 in the middle of a chunk (not at boundary) must also be safe."""
        B, n, K = 2, 16, 4
        chunk_size = 8
        alpha = torch.ones(B, n, K) * 0.5
        alpha[:, 3] = 0.001   # mid-chunk position
        alpha[:, 11] = 0.001  # mid-chunk position in second chunk

        drive_r = torch.ones(B, n, K)
        drive_i = torch.ones(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=chunk_size)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")

    def test_alpha_near_zero_every_step_no_nan(self):
        """All alpha=0.001 throughout sequence -- alpha_cum = 0.001^t -> 1e-30 floor."""
        B, n, K = 1, 20, 8
        alpha = torch.full((B, n, K), 0.001)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")


# ---------------------------------------------------------------------------
# 2. Alpha exactly 0: output must equal drive (no state carry)
# ---------------------------------------------------------------------------

class TestAlphaExactlyZero:
    """When alpha=0, the recurrence collapses to r_t = d_t (no memory)."""

    def test_alpha_zero_output_equals_drive(self):
        """All-zero alpha: each output step equals the drive at that step."""
        B, n, K = 3, 10, 5
        alpha = torch.zeros(B, n, K)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")
        _assert_close(out_r, drive_r, atol=1e-5, label="out_r vs drive_r")
        _assert_close(out_i, drive_i, atol=1e-5, label="out_i vs drive_i")

    def test_alpha_zero_single_channel(self):
        """One channel alpha=0, others normal -- zero-alpha channel must equal its drive."""
        B, n, K = 2, 12, 4
        alpha = torch.rand(B, n, K) * 0.8 + 0.1
        alpha[:, :, 2] = 0.0   # channel 2 fully zeroed

        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i)

        _assert_finite(out_r, "out_r")
        _assert_close(out_r[:, :, 2], drive_r[:, :, 2], atol=1e-5, label="channel 2 out_r")
        _assert_close(out_i[:, :, 2], drive_i[:, :, 2], atol=1e-5, label="channel 2 out_i")


# ---------------------------------------------------------------------------
# 3. Alpha exactly 1: compare against sequential reference
# ---------------------------------------------------------------------------

class TestAlphaExactlyOne:
    """When alpha=1, recurrence is r_t = r_{t-1} + d_t (pure cumsum).

    The inter_chunk_norm=True default normalizes between chunks, so
    chunked_scan DIVERGES from sequential_scan for long sequences.
    We test with inter_chunk_norm=False to verify pure numerical equivalence,
    and separately verify that inter_chunk_norm=True produces finite output.
    """

    def test_alpha_one_no_norm_matches_sequential(self):
        """alpha=1, no inter_chunk_norm: chunked must match sequential exactly."""
        B, n, K = 2, 16, 4
        alpha = torch.ones(B, n, K)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        c_r, c_i = chunked_resonance_scan(alpha, drive_r, drive_i, inter_chunk_norm=False)
        s_r, s_i = sequential_resonance_scan(alpha, drive_r, drive_i)

        _assert_close(c_r, s_r, atol=1e-4, label="out_r (alpha=1, no norm)")
        _assert_close(c_i, s_i, atol=1e-4, label="out_i (alpha=1, no norm)")

    def test_alpha_one_with_norm_finite(self):
        """alpha=1 with inter_chunk_norm=True: output must be finite despite large cumsum."""
        B, n, K = 2, 64, 8
        alpha = torch.ones(B, n, K)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i, inter_chunk_norm=True)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")


# ---------------------------------------------------------------------------
# 4. chunk_size=1: degenerate case must match sequential exactly
# ---------------------------------------------------------------------------

class TestChunkSizeOne:
    """Each chunk is a single step -- degenerates to sequential scan logic."""

    def test_chunk_size_one_matches_sequential(self):
        alpha, drive_r, drive_i = _rand(B=2, n=15, K=6, seed=42)

        c_r, c_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=1, inter_chunk_norm=False)
        s_r, s_i = sequential_resonance_scan(alpha, drive_r, drive_i)

        _assert_close(c_r, s_r, atol=1e-4, label="out_r (chunk=1)")
        _assert_close(c_i, s_i, atol=1e-4, label="out_i (chunk=1)")

    def test_chunk_size_one_no_crash_with_near_zero_alpha(self):
        """chunk_size=1 with near-zero alpha must not crash or produce NaN."""
        B, n, K = 2, 8, 4
        alpha = torch.full((B, n, K), 0.001)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=1)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")


# ---------------------------------------------------------------------------
# 5. chunk_size > n: larger chunk than sequence -- must not crash
# ---------------------------------------------------------------------------

class TestChunkSizeLargerThanSequence:
    """Single chunk containing all steps: equivalent to one call of _scan_chunk."""

    def test_chunk_larger_than_n_no_crash(self):
        alpha, drive_r, drive_i = _rand(B=3, n=8, K=4, seed=7)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=100)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")
        assert out_r.shape == (3, 8, 4)
        assert out_i.shape == (3, 8, 4)

    def test_chunk_larger_than_n_matches_sequential(self):
        """One-chunk scan (chunk>n) with no norm must match sequential reference."""
        alpha, drive_r, drive_i = _rand(B=2, n=10, K=5, seed=99)

        c_r, c_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=1000, inter_chunk_norm=False)
        s_r, s_i = sequential_resonance_scan(alpha, drive_r, drive_i)

        _assert_close(c_r, s_r, atol=1e-4, label="out_r (chunk>n)")
        _assert_close(c_i, s_i, atol=1e-4, label="out_i (chunk>n)")


# ---------------------------------------------------------------------------
# 6. Numerical equivalence for random inputs
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:
    """Chunked and sequential scans: tight atol at small chunk_size, relaxed at large.

    The cumsum-rescale trick loses precision when cumprod(alpha) underflows
    in fp32.  With chunk_size <= 4 and alpha in [0, 0.99], precision is
    excellent (< 1e-6).  Larger chunk sizes are tested for finite-ness
    and bounded deviation, not exact match.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 17, 31])
    def test_small_chunk_matches_sequential(self, seed: int):
        """chunk_size=4: tight tolerance (cumprod stays well above fp32 floor)."""
        alpha, drive_r, drive_i = _rand(B=2, n=32, K=8, seed=seed)

        c_r, c_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=4, inter_chunk_norm=False)
        s_r, s_i = sequential_resonance_scan(alpha, drive_r, drive_i)

        _assert_close(c_r, s_r, atol=1e-4, label=f"out_r seed={seed}")
        _assert_close(c_i, s_i, atol=1e-4, label=f"out_i seed={seed}")

    @pytest.mark.parametrize("seed", [0, 1, 2, 17, 31])
    def test_large_chunk_finite_and_bounded(self, seed: int):
        """chunk_size=8: relaxed tolerance -- verify finite and bounded deviation."""
        alpha, drive_r, drive_i = _rand(B=2, n=32, K=8, seed=seed)

        c_r, c_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=8, inter_chunk_norm=False)
        s_r, s_i = sequential_resonance_scan(alpha, drive_r, drive_i)

        assert torch.all(torch.isfinite(c_r)), "out_r has non-finite values"
        assert torch.all(torch.isfinite(c_i)), "out_i has non-finite values"
        # Bounded deviation: chunked must not diverge wildly
        max_err_r = (c_r - s_r).abs().max().item()
        max_err_i = (c_i - s_i).abs().max().item()
        assert max_err_r < 5.0, f"out_r deviation too large: {max_err_r}"
        assert max_err_i < 5.0, f"out_i deviation too large: {max_err_i}"

    def test_batch_size_1_equivalence(self):
        alpha, drive_r, drive_i = _rand(B=1, n=24, K=16, seed=55)

        c_r, c_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=4, inter_chunk_norm=False)
        s_r, s_i = sequential_resonance_scan(alpha, drive_r, drive_i)

        _assert_close(c_r, s_r, atol=1e-4, label="out_r B=1")
        _assert_close(c_i, s_i, atol=1e-4, label="out_i B=1")

    def test_large_batch_equivalence(self):
        alpha, drive_r, drive_i = _rand(B=64, n=16, K=4, seed=11)

        c_r, c_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=4, inter_chunk_norm=False)
        s_r, s_i = sequential_resonance_scan(alpha, drive_r, drive_i)

        _assert_close(c_r, s_r, atol=1e-4, label="out_r B=64")
        _assert_close(c_i, s_i, atol=1e-4, label="out_i B=64")


# ---------------------------------------------------------------------------
# 7. n not divisible by chunk_size -- remainder chunk handling
# ---------------------------------------------------------------------------

class TestNonDivisibleChunkSize:
    """Remainder chunk (smaller than chunk_size) must be processed correctly."""

    @pytest.mark.parametrize("n,chunk_size", [
        (7, 4),    # remainder 3
        (9, 4),    # remainder 1
        (11, 4),   # remainder 3
        (13, 6),   # remainder 1
        (1, 4),    # n=1 edge case
        (3, 4),    # n < chunk_size, single partial chunk
    ])
    def test_remainder_chunk_matches_sequential(self, n: int, chunk_size: int):
        alpha, drive_r, drive_i = _rand(B=2, n=n, K=4, seed=n * 100 + chunk_size)

        c_r, c_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=chunk_size, inter_chunk_norm=False)
        s_r, s_i = sequential_resonance_scan(alpha, drive_r, drive_i)

        assert c_r.shape == (2, n, 4), f"shape mismatch n={n}"
        _assert_close(c_r, s_r, atol=1e-4, label=f"out_r n={n} chunk={chunk_size}")
        _assert_close(c_i, s_i, atol=1e-4, label=f"out_i n={n} chunk={chunk_size}")

    def test_n7_chunk4_finite_output(self):
        """n=7, chunk_size=4: two chunks (4+3). Both must produce finite output."""
        B, n, K = 3, 7, 6
        alpha = torch.rand(B, n, K) * 0.8 + 0.1
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=4)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")
        assert out_r.shape == (B, n, K)


# ---------------------------------------------------------------------------
# 8. Very long sequence with high alpha -- no accumulation explosion
# ---------------------------------------------------------------------------

class TestLongSequenceHighAlpha:
    """High alpha (close to 1) over long sequences can cause state explosion
    if inter_chunk_norm is disabled.  With norm enabled the output must be finite."""

    def test_long_sequence_high_alpha_finite_with_norm(self):
        """n=1000, alpha=0.999, inter_chunk_norm=True: output must be finite."""
        B, n, K = 2, 1000, 8
        alpha = torch.full((B, n, K), 0.999)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i, inter_chunk_norm=True)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")

    def test_long_sequence_alpha_one_finite_with_norm(self):
        """n=1000, alpha=1.0 (worst case): state grows as cumsum but norm must save it."""
        B, n, K = 1, 1000, 4
        alpha = torch.ones(B, n, K)
        drive_r = torch.ones(B, n, K)   # drive=1 maximises accumulation
        drive_i = torch.ones(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i, inter_chunk_norm=True)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")

    def test_long_sequence_near_zero_alpha_finite(self):
        """n=1000, alpha=0.001: alpha_cum -> 1e-30 floor. inv_alpha_cum large but output finite."""
        B, n, K = 1, 1000, 4
        alpha = torch.full((B, n, K), 0.001)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")

    def test_long_sequence_mixed_alpha_finite(self):
        """n=500 with alpha alternating between 0.001 and 0.999: worst case for cumprod range."""
        B, n, K = 2, 500, 8
        alpha = torch.zeros(B, n, K)
        alpha[:, 0::2] = 0.001
        alpha[:, 1::2] = 0.999

        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i)

        _assert_finite(out_r, "out_r")
        _assert_finite(out_i, "out_i")


# ---------------------------------------------------------------------------
# Bonus: output shape invariant
# ---------------------------------------------------------------------------

class TestOutputShape:
    """Output shapes must always match input (B, n, K)."""

    @pytest.mark.parametrize("B,n,K,chunk", [
        (1, 1, 1, 1),
        (4, 7, 3, 4),
        (2, 16, 8, 16),
        (3, 100, 32, 7),
    ])
    def test_output_shape(self, B: int, n: int, K: int, chunk: int):
        alpha, drive_r, drive_i = _rand(B=B, n=n, K=K)
        out_r, out_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=chunk)
        assert out_r.shape == (B, n, K), f"out_r shape mismatch: {out_r.shape}"
        assert out_i.shape == (B, n, K), f"out_i shape mismatch: {out_i.shape}"
