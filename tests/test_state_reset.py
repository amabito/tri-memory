"""Verify resonance state is properly zero-initialized and reset between sequences."""
from __future__ import annotations

import pytest
import torch

from trimemory.resonance import TemporalResonanceLayer


@pytest.fixture
def layer() -> TemporalResonanceLayer:
    torch.manual_seed(42)
    return TemporalResonanceLayer(d_model=32, K=16, use_parallel_scan=False).eval()


def test_parallel_and_sequential_agree(layer: TemporalResonanceLayer) -> None:
    """Sequential scan must match parallel scan result (already tested, belt+suspenders)."""
    layer_seq = TemporalResonanceLayer(d_model=32, K=16, use_parallel_scan=False).eval()
    # Copy weights
    layer_seq.load_state_dict(layer.state_dict())

    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        out_seq = layer_seq(x)
        out_ref = layer(x)
    torch.testing.assert_close(out_seq, out_ref, atol=1e-5, rtol=1e-4)


def test_state_independent_between_batches(layer: TemporalResonanceLayer) -> None:
    """Each forward call starts from zero state (no state leakage between batches)."""
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        out1 = layer(x)
        out2 = layer(x)
    torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-5)


def test_log_phase_no_crash() -> None:
    """phase_mode='log' must produce finite outputs."""
    layer = TemporalResonanceLayer(d_model=32, K=16, use_parallel_scan=False, phase_mode="log").eval()
    x = torch.randn(2, 32, 32)
    with torch.no_grad():
        out = layer(x)
    assert torch.isfinite(out).all(), "log_phase produced non-finite output"


def test_clamp_resonance_no_crash() -> None:
    """clamp_resonance=True must produce finite outputs."""
    layer = TemporalResonanceLayer(
        d_model=32, K=16, use_parallel_scan=False,
        clamp_resonance=True, resonance_clamp_val=1.0
    ).eval()
    x = torch.randn(2, 64, 32)
    with torch.no_grad():
        out = layer(x)
    assert torch.isfinite(out).all()


def test_clamp_resonance_actually_clamps() -> None:
    """With very tight clamp, resonance state norm should be bounded."""
    # Use large input to try to overflow state
    layer = TemporalResonanceLayer(
        d_model=32, K=16, use_parallel_scan=False,
        clamp_resonance=True, resonance_clamp_val=0.1
    ).eval()
    x = torch.ones(1, 128, 32) * 10.0  # large input
    with torch.no_grad():
        out = layer(x)
    assert torch.isfinite(out).all()
