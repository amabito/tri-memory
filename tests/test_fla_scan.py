"""Tests for fla_scan.py -- verifies all backends against chunked_resonance_scan.

Reference algorithm: chunked_resonance_scan (SafeCumprod + cumsum-rescale).
This is the canonical scan used by TemporalResonanceLayer.

NOTE: sequential_resonance_scan uses a different algorithm (direct step-by-step
recurrence without the cumsum-rescale trick) and diverges numerically from
chunked for long sequences.  It is used only as the reference for the Triton
kernel, which also implements the direct recurrence.
"""

from __future__ import annotations

import pytest
import torch

from trimemory.fla_scan import (
    _BackendTag,
    _compiled_resonance_scan,
    _triton_resonance_scan,
    fla_resonance_scan,
    get_active_backend,
    reset_backend_cache,
)
from trimemory.scan import chunked_resonance_scan, sequential_resonance_scan

# ---- helpers ----------------------------------------------------------------

def _make_inputs(B=2, n=32, K=8, seed=0):
    torch.manual_seed(seed)
    alpha   = torch.rand(B, n, K).clamp(0.0, 0.99)
    drive_r = torch.randn(B, n, K)
    drive_i = torch.randn(B, n, K)
    return alpha, drive_r, drive_i


ATOL = 1e-5  # tolerance vs. chunked reference


# ---- shape tests ------------------------------------------------------------

def test_fla_scan_output_shapes():
    alpha, drive_r, drive_i = _make_inputs()
    out_r, out_i = fla_resonance_scan(alpha, drive_r, drive_i)
    B, n, K = alpha.shape
    assert out_r.shape == (B, n, K)
    assert out_i.shape == (B, n, K)


def test_fla_scan_fp32_output():
    alpha, drive_r, drive_i = _make_inputs()
    out_r, out_i = fla_resonance_scan(alpha, drive_r, drive_i)
    assert out_r.dtype == torch.float32
    assert out_i.dtype == torch.float32


# ---- numerical correctness vs. sequential reference -------------------------

def test_compiled_backend_matches_reference():
    """compile backend must agree with chunked_resonance_scan reference.

    The chunked scan (SafeCumprod + cumsum-rescale) is the canonical reference.
    Sequential scan uses a different algorithm and diverges numerically for
    long sequences -- it is NOT the reference here.

    If torch.compile fails (e.g. no MSVC on PATH), _compiled_resonance_scan
    degrades to the plain chunked scan internally, so results are identical.
    """
    alpha, drive_r, drive_i = _make_inputs(B=2, n=64, K=16, seed=1)
    ref_r, ref_i = chunked_resonance_scan(alpha, drive_r, drive_i, chunk_size=16)
    out_r, out_i = _compiled_resonance_scan(alpha, drive_r, drive_i)
    assert torch.allclose(out_r, ref_r, atol=ATOL), f"max diff r: {(out_r - ref_r).abs().max()}"
    assert torch.allclose(out_i, ref_i, atol=ATOL), f"max diff i: {(out_i - ref_i).abs().max()}"


def test_fla_scan_matches_chunked_reference():
    """fla_resonance_scan (whatever backend) must match chunked scan within atol."""
    alpha, drive_r, drive_i = _make_inputs(B=3, n=48, K=12, seed=2)
    ref_r, ref_i = chunked_resonance_scan(alpha, drive_r, drive_i)
    out_r, out_i = fla_resonance_scan(alpha, drive_r, drive_i)
    assert torch.allclose(out_r, ref_r, atol=ATOL), f"max diff r: {(out_r - ref_r).abs().max()}"
    assert torch.allclose(out_i, ref_i, atol=ATOL), f"max diff i: {(out_i - ref_i).abs().max()}"


# ---- dtype coercion ---------------------------------------------------------

def test_bf16_input_coerced_to_fp32():
    """bf16 alpha/drive must be cast to fp32 before scan."""
    alpha, drive_r, drive_i = _make_inputs(B=1, n=16, K=4, seed=3)
    out_r, out_i = fla_resonance_scan(
        alpha.bfloat16(), drive_r.bfloat16(), drive_i.bfloat16()
    )
    assert out_r.dtype == torch.float32
    assert out_i.dtype == torch.float32


# ---- boundary conditions ----------------------------------------------------

def test_zero_drive_zero_output():
    """With zero drive and zero initial state, output must be all zeros."""
    B, n, K = 2, 16, 4
    alpha   = torch.rand(B, n, K).clamp(0.0, 0.99)
    drive_r = torch.zeros(B, n, K)
    drive_i = torch.zeros(B, n, K)
    out_r, out_i = fla_resonance_scan(alpha, drive_r, drive_i)
    assert torch.all(out_r == 0.0)
    assert torch.all(out_i == 0.0)


def test_unit_alpha_unit_drive_cumsum():
    """alpha=1, drive_r=1 -> out_r[t] = t+1 (cumulative sum)."""
    B, n, K = 1, 8, 2
    alpha   = torch.ones(B, n, K)
    drive_r = torch.ones(B, n, K)
    drive_i = torch.zeros(B, n, K)
    out_r, _ = fla_resonance_scan(alpha, drive_r, drive_i)
    expected = torch.arange(1, n + 1, dtype=torch.float32).view(1, n, 1).expand(B, n, K)
    assert torch.allclose(out_r, expected, atol=ATOL), f"max diff: {(out_r - expected).abs().max()}"


def test_output_is_finite():
    """Outputs must be finite for normal inputs."""
    torch.manual_seed(42)
    alpha, drive_r, drive_i = _make_inputs(B=4, n=64, K=32)
    out_r, out_i = fla_resonance_scan(alpha, drive_r, drive_i)
    assert torch.all(torch.isfinite(out_r)), "out_r contains non-finite values"
    assert torch.all(torch.isfinite(out_i)), "out_i contains non-finite values"


# ---- Triton backend (CPU skip) ----------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton backend requires CUDA",
)
def test_triton_backend_matches_reference_cuda():
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("triton not installed")

    alpha, drive_r, drive_i = _make_inputs(B=2, n=64, K=16, seed=10)
    alpha   = alpha.cuda()
    drive_r = drive_r.cuda()
    drive_i = drive_i.cuda()

    ref_r, ref_i = sequential_resonance_scan(alpha, drive_r, drive_i)
    out_r, out_i = _triton_resonance_scan(alpha, drive_r, drive_i)

    assert torch.allclose(out_r, ref_r, atol=ATOL), f"max diff r: {(out_r - ref_r).abs().max()}"
    assert torch.allclose(out_i, ref_i, atol=ATOL), f"max diff i: {(out_i - ref_i).abs().max()}"


# ---- backend selection & graceful degradation -------------------------------

def test_get_active_backend_returns_string():
    tag = get_active_backend(torch.device("cpu"))
    assert isinstance(tag, str)
    assert tag in (_BackendTag.FLA, _BackendTag.TRITON, _BackendTag.COMPILE, _BackendTag.CHUNKED)


def test_reset_backend_cache_forces_reselect():
    """After reset, get_active_backend must re-probe (not crash)."""
    _ = get_active_backend(torch.device("cpu"))
    reset_backend_cache()
    tag = get_active_backend(torch.device("cpu"))
    assert isinstance(tag, str)


def test_backend_degrades_on_forced_failure(monkeypatch):
    """If active backend raises, fla_resonance_scan must fall back to chunked."""
    import trimemory.fla_scan as fla_module

    # Force compile backend with a broken implementation
    reset_backend_cache()
    monkeypatch.setitem(
        fla_module._backend_cache,
        "cpu",
        _BackendTag.COMPILE,
    )

    def _always_raise(alpha, drive_r, drive_i):
        raise RuntimeError("forced failure")

    monkeypatch.setitem(
        fla_module._BACKEND_DISPATCH,
        _BackendTag.COMPILE,
        _always_raise,
    )

    alpha, drive_r, drive_i = _make_inputs(B=1, n=16, K=4, seed=99)
    # Should not raise; should silently fall back
    out_r, out_i = fla_resonance_scan(alpha, drive_r, drive_i)
    assert out_r.shape == alpha.shape
    # After failure, cache must be updated to chunked
    assert fla_module._backend_cache["cpu"] == _BackendTag.CHUNKED

    reset_backend_cache()


# ---- larger shape stress test -----------------------------------------------

def test_large_sequence_no_explosion():
    """n=512, K=64 -- state must remain finite with alpha close to 1."""
    B, n, K = 1, 512, 64
    alpha   = torch.full((B, n, K), 0.99)
    drive_r = torch.randn(B, n, K) * 0.01
    drive_i = torch.randn(B, n, K) * 0.01
    out_r, out_i = fla_resonance_scan(alpha, drive_r, drive_i)
    assert torch.all(torch.isfinite(out_r))
    assert torch.all(torch.isfinite(out_i))
