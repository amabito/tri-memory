from __future__ import annotations

import math
import pytest
import torch
from trimemory.scan import _combine, sequential_resonance_scan


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
# parallel_fallback test
# ---------------------------------------------------------------------------

def test_parallel_fallback(monkeypatch):
    import trimemory.scan as scan_module

    # Remove associative_scan from the torch mock so parallel path falls back
    class _FakeTorch:
        """Minimal torch stand-in without associative_scan."""
        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(scan_module, "torch", _FakeTorch())

    called = []
    original_chunked = scan_module.chunked_resonance_scan

    def _spy(*args, **kwargs):
        called.append(True)
        return original_chunked(*args, **kwargs)

    monkeypatch.setattr(scan_module, "chunked_resonance_scan", _spy)

    B, n, K = 1, 4, 2
    alpha   = torch.rand(B, n, K).clamp(0.0, 0.99)
    drive_r = torch.randn(B, n, K)
    drive_i = torch.randn(B, n, K)

    from trimemory.scan import parallel_resonance_scan
    parallel_resonance_scan(alpha, drive_r, drive_i)

    assert len(called) > 0, "chunked_resonance_scan was not called as fallback"


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
