from __future__ import annotations

import torch
from trimemory.scan import (
    _combine,
    _kogge_stone_scan,
    chunked_resonance_scan,
    parallel_resonance_scan,
    sequential_resonance_scan,
)


# ---------------------------------------------------------------------------
# _combine tests
# ---------------------------------------------------------------------------

def test_combine_associativity():
    # (a o b) o c == a o (b o c) for random tensors
    torch.manual_seed(0)
    shape = (4, 8)
    a = (torch.rand(shape), torch.rand(shape))
    b = (torch.rand(shape), torch.rand(shape))
    c = (torch.rand(shape), torch.rand(shape))

    left  = _combine(_combine(a, b), c)
    right = _combine(a, _combine(b, c))

    assert torch.allclose(left[0], right[0], atol=1e-5)
    assert torch.allclose(left[1], right[1], atol=1e-5)


def test_combine_identity():
    # (1, 0) is left identity: _combine((1,0), (a,b)) == (a,b)
    torch.manual_seed(1)
    shape = (3, 5)
    a_val = torch.rand(shape)
    b_val = torch.rand(shape)

    ones  = torch.ones(shape)
    zeros = torch.zeros(shape)

    out_a, out_b = _combine((ones, zeros), (a_val, b_val))
    assert torch.allclose(out_a, a_val, atol=1e-6)
    assert torch.allclose(out_b, b_val, atol=1e-6)


# ---------------------------------------------------------------------------
# sequential_resonance_scan shape tests
# ---------------------------------------------------------------------------

def test_sequential_scan_shape():
    B, n, K = 2, 16, 8
    alpha   = torch.rand(B, n, K).clamp(0.0, 0.99)
    drive_r = torch.randn(B, n, K)
    drive_i = torch.randn(B, n, K)

    out_r, out_i = sequential_resonance_scan(alpha, drive_r, drive_i)

    assert out_r.shape == (B, n, K)
    assert out_i.shape == (B, n, K)


def test_sequential_scan_zero_drive_zero_state():
    # alpha arbitrary, drive=0 -> output all zeros (starts from zero state)
    B, n, K = 2, 10, 4
    alpha   = torch.rand(B, n, K).clamp(0.0, 0.99)
    drive_r = torch.zeros(B, n, K)
    drive_i = torch.zeros(B, n, K)

    out_r, out_i = sequential_resonance_scan(alpha, drive_r, drive_i)

    assert torch.all(out_r == 0.0)
    assert torch.all(out_i == 0.0)


def test_sequential_scan_unit_alpha_unit_drive():
    # alpha=1.0, drive_r=1.0, drive_i=0 -> out_r[t] = t+1 (cumsum)
    B, n, K = 1, 5, 2
    alpha   = torch.ones(B, n, K)
    drive_r = torch.ones(B, n, K)
    drive_i = torch.zeros(B, n, K)

    out_r, out_i = sequential_resonance_scan(alpha, drive_r, drive_i)

    expected = torch.arange(1, n + 1, dtype=torch.float32).view(1, n, 1).expand(B, n, K)
    assert torch.allclose(out_r, expected, atol=1e-5)
    assert torch.all(out_i == 0.0)


def test_sequential_scan_finite():
    torch.manual_seed(42)
    B, n, K = 3, 32, 16
    alpha   = torch.rand(B, n, K).clamp(0.0, 0.99)
    drive_r = torch.randn(B, n, K)
    drive_i = torch.randn(B, n, K)

    out_r, out_i = sequential_resonance_scan(alpha, drive_r, drive_i)

    assert torch.all(torch.isfinite(out_r)), "out_r has non-finite values"
    assert torch.all(torch.isfinite(out_i)), "out_i has non-finite values"


# ---------------------------------------------------------------------------
# parallel_resonance_scan (Kogge-Stone) tests
# ---------------------------------------------------------------------------

def test_kogge_stone_matches_sequential():
    """Kogge-Stone scan must match sequential scan within fp32 tolerance."""
    torch.manual_seed(10)
    B, n, K = 2, 32, 8
    alpha   = torch.rand(B, n, K).clamp(0.1, 0.9)
    drive_r = torch.randn(B, n, K)
    drive_i = torch.randn(B, n, K)

    r_r_ks, r_i_ks  = parallel_resonance_scan(alpha, drive_r, drive_i)
    r_r_seq, r_i_seq = sequential_resonance_scan(alpha, drive_r, drive_i)

    assert torch.allclose(r_r_ks, r_r_seq, atol=1e-4), \
        f"r_r max diff: {(r_r_ks - r_r_seq).abs().max():.2e}"
    assert torch.allclose(r_i_ks, r_i_seq, atol=1e-4), \
        f"r_i max diff: {(r_i_ks - r_i_seq).abs().max():.2e}"


def test_kogge_stone_closer_to_sequential_than_chunked():
    """Kogge-Stone (pure multiplications) is numerically closer to the sequential
    reference than the chunked scan (which uses SafeCumprod + division rescaling).

    The chunked scan is an approximate algorithm used as a CPU fallback.
    Kogge-Stone is the mathematically exact parallel implementation.
    """
    torch.manual_seed(11)
    B, n, K = 4, 64, 16
    alpha   = torch.rand(B, n, K).clamp(0.1, 0.9)
    drive_r = torch.randn(B, n, K)
    drive_i = torch.randn(B, n, K)

    r_r_ks, r_i_ks     = parallel_resonance_scan(alpha, drive_r, drive_i)
    r_r_seq, r_i_seq   = sequential_resonance_scan(alpha, drive_r, drive_i)
    r_r_ch, r_i_ch     = chunked_resonance_scan(alpha, drive_r, drive_i)

    ks_err  = (r_r_ks - r_r_seq).abs().max().item()
    ch_err  = (r_r_ch - r_r_seq).abs().max().item()

    # Kogge-Stone must be within 1e-4 of sequential (numerically exact parallel scan)
    assert ks_err < 1e-4, f"KS vs sequential r_r max diff: {ks_err:.2e}"
    # Kogge-Stone must be more accurate than chunked
    assert ks_err < ch_err, \
        f"KS err {ks_err:.2e} not less than chunked err {ch_err:.2e}"


def test_kogge_stone_shape():
    """parallel_resonance_scan must return tensors of input shape."""
    B, n, K = 3, 128, 32
    alpha   = torch.rand(B, n, K).clamp(0.1, 0.9)
    drive_r = torch.randn(B, n, K)
    drive_i = torch.randn(B, n, K)

    r_r, r_i = parallel_resonance_scan(alpha, drive_r, drive_i)

    assert r_r.shape == (B, n, K)
    assert r_i.shape == (B, n, K)


def test_kogge_stone_non_power_of_two():
    """Kogge-Stone must handle sequence lengths that are not powers of 2."""
    torch.manual_seed(12)
    for n in [1, 2, 3, 5, 7, 10, 17, 100, 200]:
        B, K = 2, 4
        alpha   = torch.rand(B, n, K).clamp(0.1, 0.9)
        drive_r = torch.randn(B, n, K)
        drive_i = torch.randn(B, n, K)

        r_r_ks, r_i_ks   = parallel_resonance_scan(alpha, drive_r, drive_i)
        r_r_seq, r_i_seq  = sequential_resonance_scan(alpha, drive_r, drive_i)

        assert torch.allclose(r_r_ks, r_r_seq, atol=1e-4), \
            f"n={n}: r_r max diff: {(r_r_ks - r_r_seq).abs().max():.2e}"
        assert torch.allclose(r_i_ks, r_i_seq, atol=1e-4), \
            f"n={n}: r_i max diff: {(r_i_ks - r_i_seq).abs().max():.2e}"


def test_kogge_stone_autograd():
    """Kogge-Stone must be autograd-compatible (no in-place ops)."""
    torch.manual_seed(13)
    B, n, K = 2, 16, 4
    alpha   = torch.rand(B, n, K).clamp(0.1, 0.9).requires_grad_(True)
    drive_r = torch.randn(B, n, K).requires_grad_(True)
    drive_i = torch.randn(B, n, K).requires_grad_(True)

    r_r, r_i = parallel_resonance_scan(alpha, drive_r, drive_i)
    loss = r_r.sum() + r_i.sum()
    loss.backward()  # must not raise

    assert alpha.grad is not None
    assert drive_r.grad is not None
    assert drive_i.grad is not None


def test_kogge_stone_gradcheck():
    """Gradient values must match numerical gradient (not just no-crash)."""
    torch.manual_seed(99)
    B, n, K = 1, 4, 2
    alpha = torch.rand(B, n, K, dtype=torch.float64).clamp(0.2, 0.8).requires_grad_(True)
    drive = torch.randn(B, n, K, dtype=torch.float64).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda a, d: _kogge_stone_scan(a, d), (alpha, drive), eps=1e-6, atol=1e-4,
    )


def test_kogge_stone_alpha_zero():
    """alpha=0: output[t] = drive[t] (no memory carry)."""
    B, n, K = 2, 8, 4
    alpha = torch.zeros(B, n, K)
    drive_r = torch.randn(B, n, K)
    drive_i = torch.randn(B, n, K)
    r_r, r_i = parallel_resonance_scan(alpha, drive_r, drive_i)
    assert torch.allclose(r_r, drive_r, atol=1e-5), f"max diff: {(r_r - drive_r).abs().max():.2e}"
    assert torch.allclose(r_i, drive_i, atol=1e-5)


def test_kogge_stone_alpha_one():
    """alpha=1: output[t] = cumsum(drive)[t]."""
    B, n, K = 1, 8, 2
    alpha = torch.ones(B, n, K)
    drive_r = torch.ones(B, n, K)
    r_r, _ = parallel_resonance_scan(alpha, drive_r, torch.zeros(B, n, K))
    expected = torch.arange(1, n + 1, dtype=torch.float32).view(1, n, 1).expand(B, n, K)
    assert torch.allclose(r_r, expected, atol=1e-4), f"max diff: {(r_r - expected).abs().max():.2e}"


# ---------------------------------------------------------------------------
# addcmul correctness
# ---------------------------------------------------------------------------

def test_addcmul_correctness():
    torch.manual_seed(7)
    shape = (5, 5)
    a2 = torch.rand(shape)
    b1 = torch.rand(shape)
    b2 = torch.rand(shape)

    fused   = torch.addcmul(b2, a2, b1)
    naive   = a2 * b1 + b2

    assert torch.allclose(fused, naive, atol=1e-6)


# ---------------------------------------------------------------------------
# ADVERSARIAL tests
# ---------------------------------------------------------------------------

def test_nan_alpha_propagates():
    # One NaN in alpha -> NaN must appear in output (not silently replaced)
    B, n, K = 1, 5, 2
    alpha   = torch.ones(B, n, K) * 0.5
    alpha[0, 2, 0] = float("nan")
    drive_r = torch.ones(B, n, K)
    drive_i = torch.zeros(B, n, K)

    out_r, _ = sequential_resonance_scan(alpha, drive_r, drive_i)

    # NaN at step 2 should propagate to step 2 and beyond for channel 0
    assert torch.isnan(out_r[0, 2, 0]), "NaN in alpha did not propagate to output"


def test_large_alpha_no_explosion():
    # alpha=0.9999 for n=100, bounded drive -> output must be finite
    B, n, K = 1, 100, 4
    alpha   = torch.full((B, n, K), 0.9999)
    drive_r = torch.ones(B, n, K) * 0.01
    drive_i = torch.zeros(B, n, K)

    out_r, out_i = sequential_resonance_scan(alpha, drive_r, drive_i)

    assert torch.all(torch.isfinite(out_r)), "out_r exploded with alpha=0.9999"
    assert torch.all(torch.isfinite(out_i)), "out_i exploded with alpha=0.9999"
