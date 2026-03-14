"""Verify that step_single is numerically equivalent to forward at n=1 and n>1."""

from __future__ import annotations

import pytest
import torch

from trimemory.config import TRNConfig
from trimemory.resonance import TemporalResonanceLayer


@pytest.fixture
def layer() -> TemporalResonanceLayer:
    torch.manual_seed(42)
    cfg = TRNConfig.toy()
    return TemporalResonanceLayer(
        d_model           = cfg.d_model,
        K                 = cfg.n_oscillators,
        use_parallel_scan = False,   # force sequential for determinism
    ).eval()


def _zero_state(B: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(B, K), torch.zeros(B, K)


# ---------------------------------------------------------------------------
# Core equivalence tests
# ---------------------------------------------------------------------------

def test_step_single_equals_forward_at_n1(layer: TemporalResonanceLayer) -> None:
    """step_single(x, state=0, pos=0) must match forward(x.unsqueeze(1))."""
    torch.manual_seed(0)
    B = 3
    d = layer.proj.proj.in_features
    K = layer.K

    x = torch.randn(B, d)
    r0, i0 = _zero_state(B, K)

    with torch.no_grad():
        out_step, _, _ = layer.step_single(x, r0, i0, position=0)
        out_fwd        = layer(x.unsqueeze(1)).squeeze(1)  # (B, d_model)

    assert out_step.shape == out_fwd.shape
    torch.testing.assert_close(out_step, out_fwd, atol=1e-5, rtol=1e-4)


def test_step_single_stateful_n3(layer: TemporalResonanceLayer) -> None:
    """Chaining step_single for 3 steps must match forward(x_seq) over 3 tokens."""
    torch.manual_seed(7)
    B = 2
    d = layer.proj.proj.in_features
    K = layer.K
    n = 3

    x_seq = torch.randn(B, n, d)

    with torch.no_grad():
        out_fwd = layer(x_seq)  # (B, n, d_model) — reference

    r, i = _zero_state(B, K)
    outs = []
    with torch.no_grad():
        for t in range(n):
            out_t, r, i = layer.step_single(x_seq[:, t], r, i, position=t)
            outs.append(out_t)

    out_step = torch.stack(outs, dim=1)  # (B, n, d_model)

    torch.testing.assert_close(out_step, out_fwd, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# State evolution
# ---------------------------------------------------------------------------

def test_step_single_state_accumulates(layer: TemporalResonanceLayer) -> None:
    """Resonance states must differ from zero after the first step."""
    B = 2
    d = layer.proj.proj.in_features
    K = layer.K

    x     = torch.randn(B, d)
    r0, i0 = _zero_state(B, K)

    with torch.no_grad():
        _, r1, i1 = layer.step_single(x, r0, i0, position=0)

    assert not torch.allclose(r1, r0), "real state did not update"
    assert not torch.allclose(i1, i0), "imag state did not update"


def test_step_single_state_fp32(layer: TemporalResonanceLayer) -> None:
    """Returned states must always be fp32 even when input is bf16."""
    B = 2
    d = layer.proj.proj.in_features
    K = layer.K

    x     = torch.randn(B, d).to(torch.bfloat16)
    r0, i0 = _zero_state(B, K)

    with torch.no_grad():
        _, r1, i1 = layer.step_single(x, r0, i0, position=0)

    assert r1.dtype == torch.float32, f"r_real dtype: {r1.dtype}"
    assert i1.dtype == torch.float32, f"r_imag dtype: {i1.dtype}"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("position", [0, 1, 99, 1023])
def test_step_single_no_crash_at_positions(
    layer: TemporalResonanceLayer,
    position: int,
) -> None:
    """step_single must not crash and must produce finite outputs at any position."""
    B = 2
    d = layer.proj.proj.in_features
    K = layer.K

    x     = torch.randn(B, d)
    r0, i0 = _zero_state(B, K)

    with torch.no_grad():
        out, _, _ = layer.step_single(x, r0, i0, position=position)

    assert out.shape == (B, layer.W_res.out_features)
    assert torch.isfinite(out).all(), f"non-finite output at position={position}"


def test_step_single_output_dtype_matches_input(layer: TemporalResonanceLayer) -> None:
    """Output dtype must match the input embedding dtype (not the state dtype)."""
    B = 2
    d = layer.proj.proj.in_features
    K = layer.K

    for dtype in (torch.float32, torch.bfloat16):
        x     = torch.randn(B, d, dtype=dtype)
        r0, i0 = _zero_state(B, K)
        with torch.no_grad():
            out, _, _ = layer.step_single(x, r0, i0, position=0)
        assert out.dtype == dtype, f"expected {dtype}, got {out.dtype}"
