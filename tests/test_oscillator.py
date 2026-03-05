"""Unit tests for OscillatorProjection."""
from __future__ import annotations

from math import pi

import pytest
import torch
import torch.nn as nn

from trn.config import TRNConfig
from trn.oscillator import OscillatorProjection


@pytest.fixture
def layer() -> OscillatorProjection:
    torch.manual_seed(0)
    cfg = TRNConfig.toy()
    return OscillatorProjection(d_model=cfg.d_model, K=cfg.n_oscillators)


def _forward(layer: OscillatorProjection, B: int = 2, n: int = 5) -> tuple:
    torch.manual_seed(1)
    d = layer.proj.in_features
    x = torch.randn(B, n, d)
    with torch.no_grad():
        return layer(x), (B, n, layer.K)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_oscillator_output_shapes(layer: OscillatorProjection) -> None:
    """All four outputs must have shape (B, n, K)."""
    B, n = 2, 5
    (A, omega, phi, alpha), (eB, en, eK) = _forward(layer, B, n)
    assert A.shape == (eB, en, eK), f"A shape {A.shape}"
    assert omega.shape == (eB, en, eK), f"omega shape {omega.shape}"
    assert phi.shape == (eB, en, eK), f"phi shape {phi.shape}"
    assert alpha.shape == (eB, en, eK), f"alpha shape {alpha.shape}"


# ---------------------------------------------------------------------------
# Value range tests
# ---------------------------------------------------------------------------

def test_A_positive(layer: OscillatorProjection) -> None:
    """Amplitude A must be strictly positive everywhere."""
    (A, _, _, _), _ = _forward(layer)
    assert (A > 0).all(), "A has non-positive values"


def test_A_clamped(layer: OscillatorProjection) -> None:
    """Amplitude A must not exceed 10.0."""
    (A, _, _, _), _ = _forward(layer)
    assert (A <= 10.0).all(), f"A max = {A.max().item()}"


def test_omega_lower_bound(layer: OscillatorProjection) -> None:
    """omega must be strictly positive (omega_base starts at 0.05*pi > 0)."""
    (_, omega, _, _), _ = _forward(layer)
    assert (omega > 0).all(), f"omega min = {omega.min().item()}"


def test_phi_range(layer: OscillatorProjection) -> None:
    """phi must satisfy |phi| <= pi."""
    (_, _, phi, _), _ = _forward(layer)
    assert (phi.abs() <= pi + 1e-6).all(), f"phi max abs = {phi.abs().max().item()}"


def test_alpha_range(layer: OscillatorProjection) -> None:
    """alpha must be strictly in (0, 1) — sigmoid never reaches endpoints."""
    (_, _, _, alpha), _ = _forward(layer)
    assert (alpha > 0).all(), f"alpha min = {alpha.min().item()}"
    assert (alpha < 1).all(), f"alpha max = {alpha.max().item()}"


# ---------------------------------------------------------------------------
# Parameter tests
# ---------------------------------------------------------------------------

def test_omega_base_is_parameter(layer: OscillatorProjection) -> None:
    """omega_base must be an nn.Parameter."""
    assert isinstance(layer.omega_base, nn.Parameter)


def test_omega_base_requires_grad(layer: OscillatorProjection) -> None:
    """omega_base must require gradients."""
    assert layer.omega_base.requires_grad


def test_gate_bias_default(layer: OscillatorProjection) -> None:
    """With zero linear weights, alpha approx sigmoid(1.73) ≈ 0.85 (slow decay)."""
    cfg = TRNConfig.toy()
    osc = OscillatorProjection(d_model=cfg.d_model, K=cfg.n_oscillators)
    # Zero out all weights — only bias contributes
    with torch.no_grad():
        osc.proj.weight.zero_()

    x = torch.zeros(1, 1, cfg.d_model)
    with torch.no_grad():
        _, _, _, alpha = osc(x)

    expected = torch.sigmoid(torch.tensor(1.73))
    torch.testing.assert_close(alpha.mean(), expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Adversarial tests
# ---------------------------------------------------------------------------

def test_zero_input_no_nan(layer: OscillatorProjection) -> None:
    """Zero input must produce all-finite outputs."""
    d = layer.proj.in_features
    x = torch.zeros(2, 4, d)
    with torch.no_grad():
        A, omega, phi, alpha = layer(x)
    for name, t in [("A", A), ("omega", omega), ("phi", phi), ("alpha", alpha)]:
        assert torch.isfinite(t).all(), f"{name} has non-finite values on zero input"


def test_large_input_clamp(layer: OscillatorProjection) -> None:
    """Very large positive input must cause A to be clamped at amplitude_max."""
    d = layer.proj.in_features
    x = torch.full((1, 1, d), 1e6)
    with torch.no_grad():
        A, _, _, _ = layer(x)
    a_max = layer.amplitude_max
    # softplus(large) is large, but clamp(max=amplitude_max) must cap it
    assert (A <= a_max + 1e-5).all(), f"A not clamped: max={A.max().item()}"
    # At least some values should be near the ceiling
    assert (A >= a_max - 0.1).any(), (
        f"Expected A to hit clamp ceiling ({a_max}) on huge input, got max={A.max().item()}"
    )
