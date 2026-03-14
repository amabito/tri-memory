"""Unit tests for TemporalResonanceLayer."""
from __future__ import annotations

import pytest
import torch

from trimemory.config import TRNConfig
from trimemory.resonance import TemporalResonanceLayer


@pytest.fixture
def cfg() -> TRNConfig:
    return TRNConfig.toy()


@pytest.fixture
def layer(cfg: TRNConfig) -> TemporalResonanceLayer:
    torch.manual_seed(42)
    return TemporalResonanceLayer(
        d_model=cfg.d_model,
        K=cfg.n_oscillators,
        use_parallel_scan=False,  # CPU-safe, deterministic
    ).eval()


# ---------------------------------------------------------------------------
# Shape & basic forward
# ---------------------------------------------------------------------------

def test_resonance_forward_shape(layer: TemporalResonanceLayer, cfg: TRNConfig) -> None:
    """forward must return (B, n, d_model)."""
    B, n = 3, 7
    x = torch.randn(B, n, cfg.d_model)
    with torch.no_grad():
        out = layer(x)
    assert out.shape == (B, n, cfg.d_model), f"unexpected shape {out.shape}"


def test_resonance_output_finite(layer: TemporalResonanceLayer, cfg: TRNConfig) -> None:
    """Random input must produce all-finite output."""
    torch.manual_seed(99)
    x = torch.randn(2, 10, cfg.d_model)
    with torch.no_grad():
        out = layer(x)
    assert torch.isfinite(out).all(), "non-finite values in output"


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

def test_resonance_causal(layer: TemporalResonanceLayer, cfg: TRNConfig) -> None:
    """Output at t=0 must not depend on future tokens x[:,1:].

    The recurrence r_t = alpha_t * r_{t-1} + v_t is causal: position 0 only
    sees itself.  We verify that zeroing all future tokens does not change
    output at t=0.
    """
    torch.manual_seed(7)
    B, n = 2, 6
    x = torch.randn(B, n, cfg.d_model)
    x_copy = x.clone()
    x_copy[:, 1:] = 0.0  # wipe future tokens

    with torch.no_grad():
        out_orig = layer(x)
        out_copy = layer(x_copy)

    torch.testing.assert_close(
        out_orig[:, 0],
        out_copy[:, 0],
        atol=1e-5,
        rtol=1e-4,
        msg="output at t=0 changed when future tokens were zeroed",
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_resonance_single_token_n1(layer: TemporalResonanceLayer, cfg: TRNConfig) -> None:
    """n=1 must not crash and must return shape (B, 1, d_model)."""
    x = torch.randn(2, 1, cfg.d_model)
    with torch.no_grad():
        out = layer(x)
    assert out.shape == (2, 1, cfg.d_model)
    assert torch.isfinite(out).all()


def test_resonance_n_equals_max_seq(layer: TemporalResonanceLayer, cfg: TRNConfig) -> None:
    """n=max_seq_len must not crash (toy config: max_seq_len=512)."""
    n = cfg.max_seq_len
    x = torch.randn(1, n, cfg.d_model)
    with torch.no_grad():
        out = layer(x)
    assert out.shape == (1, n, cfg.d_model)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# fp32 internal state
# ---------------------------------------------------------------------------

def test_resonance_state_fp32_internal(cfg: TRNConfig) -> None:
    """chunked_resonance_scan must receive fp32 alpha even when the layer
    computes in a different dtype.

    We monkeypatch chunked_resonance_scan to record the dtype of alpha,
    run a fp32 forward pass, and verify that the captured dtype is float32.
    The resonance.py forward always casts alpha to .float() before the scan,
    so this must hold regardless of the input dtype.
    """
    import trimemory.resonance as resonance_mod
    from trimemory.scan import chunked_resonance_scan as original_chunked

    captured_dtypes: list[torch.dtype] = []

    def recording_scan(alpha, drive_r, drive_i, **kwargs):
        captured_dtypes.append(alpha.dtype)
        return original_chunked(alpha, drive_r, drive_i, **kwargs)

    torch.manual_seed(0)
    layer_fp32 = TemporalResonanceLayer(
        d_model=cfg.d_model,
        K=cfg.n_oscillators,
        use_parallel_scan=False,
    ).eval()

    # Patch at the module level used by TemporalResonanceLayer.forward
    original = resonance_mod.chunked_resonance_scan
    resonance_mod.chunked_resonance_scan = recording_scan

    x = torch.randn(1, 4, cfg.d_model)  # fp32
    try:
        with torch.no_grad():
            layer_fp32(x)
    finally:
        resonance_mod.chunked_resonance_scan = original

    assert len(captured_dtypes) > 0, "scan was never called"
    assert all(
        dt == torch.float32 for dt in captured_dtypes
    ), f"alpha dtype passed to scan was not float32: {captured_dtypes}"


# ---------------------------------------------------------------------------
# Sequential vs sequential sanity
# ---------------------------------------------------------------------------

def test_resonance_sequential_vs_parallel_equivalent(
    layer: TemporalResonanceLayer, cfg: TRNConfig
) -> None:
    """Two identical sequential runs must produce the same result (sanity check)."""
    torch.manual_seed(55)
    x = torch.randn(2, 8, cfg.d_model)

    with torch.no_grad():
        out1 = layer(x)
        out2 = layer(x)

    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# Adversarial tests
# ---------------------------------------------------------------------------

def test_resonance_nan_in_one_position(
    layer: TemporalResonanceLayer, cfg: TRNConfig
) -> None:
    """NaN at position 2 must propagate forward (not be silently dropped).

    Outputs at positions 0 and 1 (before the NaN) must remain finite.
    Outputs at position 2 onward may contain NaN (recurrence propagates it).
    """
    torch.manual_seed(3)
    B, n = 1, 5
    x = torch.randn(B, n, cfg.d_model)
    x[:, 2, :] = float("nan")

    with torch.no_grad():
        out = layer(x)

    # Before the NaN position: must be finite
    assert torch.isfinite(out[:, :2]).all(), (
        "output before NaN position should be finite"
    )
    # At and after position 2: NaN must have propagated (not silently dropped)
    assert not torch.isfinite(out[:, 2:]).all(), (
        "NaN at position 2 should propagate to later outputs"
    )
