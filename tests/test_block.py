"""Tests for TRNBlock and SwiGLU."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from trn.config import TRNConfig
from trn.block import TRNBlock, SwiGLU


@pytest.fixture
def toy_cfg() -> TRNConfig:
    return TRNConfig.toy()


@pytest.fixture
def toy_block(toy_cfg: TRNConfig) -> TRNBlock:
    torch.manual_seed(0)
    return TRNBlock(toy_cfg)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_block_forward_shape(toy_block: TRNBlock, toy_cfg: TRNConfig) -> None:
    B, n, d = 2, 16, toy_cfg.d_model
    x = torch.randn(B, n, d)
    out = toy_block(x)
    assert out.shape == (B, n, d)


def test_swiglu_hidden_dim(toy_block: TRNBlock, toy_cfg: TRNConfig) -> None:
    """SwiGLU gate/up weights use d_ff_hidden, not d_ff."""
    gate_shape = toy_block.ffn.gate.weight.shape
    up_shape   = toy_block.ffn.up.weight.shape
    assert gate_shape == (toy_cfg.d_ff_hidden, toy_cfg.d_model)
    assert up_shape   == (toy_cfg.d_ff_hidden, toy_cfg.d_model)


def test_d_ff_hidden_ne_d_ff_100m() -> None:
    """For trn_100m, d_ff_hidden (1536) < d_ff (2048)."""
    cfg = TRNConfig.trn_100m()
    assert cfg.d_ff_hidden != cfg.d_ff
    assert cfg.d_ff_hidden == 1536
    assert cfg.d_ff == 2048


# ---------------------------------------------------------------------------
# Residual connection
# ---------------------------------------------------------------------------

def test_block_residual_nonzero(toy_block: TRNBlock, toy_cfg: TRNConfig) -> None:
    """Output should differ from input (residual adds something)."""
    B, n, d = 2, 8, toy_cfg.d_model
    x = torch.randn(B, n, d)
    out = toy_block(x)
    assert not torch.allclose(out, x), "Output equals input — residual is a no-op"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_block_gradient_flow(toy_block: TRNBlock, toy_cfg: TRNConfig) -> None:
    """Backward pass should produce gradients on all parameters."""
    B, n, d = 2, 8, toy_cfg.d_model
    x = torch.randn(B, n, d)
    out = toy_block(x)
    loss = out.sum()
    loss.backward()
    for name, param in toy_block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Dropout identity when dropout=0
# ---------------------------------------------------------------------------

def test_block_dropout_zero(toy_cfg: TRNConfig) -> None:
    """When dropout=0, drop should be nn.Identity (not nn.Dropout)."""
    assert toy_cfg.dropout == 0.0
    block = TRNBlock(toy_cfg)
    assert isinstance(block.drop, nn.Identity)


def test_block_dropout_nonzero() -> None:
    """When dropout>0, drop should be nn.Dropout."""
    cfg = TRNConfig.toy()
    cfg = TRNConfig(
        vocab_size=cfg.vocab_size, d_model=cfg.d_model,
        n_oscillators=cfg.n_oscillators, n_layers=cfg.n_layers,
        d_ff=cfg.d_ff, max_seq_len=cfg.max_seq_len, dropout=0.1,
    )
    block = TRNBlock(cfg)
    assert isinstance(block.drop, nn.Dropout)


# ---------------------------------------------------------------------------
# ADVERSARIAL
# ---------------------------------------------------------------------------

def test_block_zero_input_finite(toy_block: TRNBlock, toy_cfg: TRNConfig) -> None:
    """All-zero input must produce finite output (no NaN / Inf)."""
    B, n, d = 2, 8, toy_cfg.d_model
    x = torch.zeros(B, n, d)
    out = toy_block(x)
    assert torch.isfinite(out).all(), "Non-finite output for zero input"
