"""Tests for TRNModel."""
from __future__ import annotations

import torch
import pytest
from trimemory.config import TRNConfig
from trimemory.model import TRNModel


@pytest.fixture
def toy_cfg() -> TRNConfig:
    return TRNConfig.toy()


@pytest.fixture
def toy_model(toy_cfg: TRNConfig) -> TRNModel:
    torch.manual_seed(42)
    return TRNModel(toy_cfg)


def _make_ids(cfg: TRNConfig, B: int = 2, n: int = 16) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (B, n))


# ---------------------------------------------------------------------------
# Forward output shapes
# ---------------------------------------------------------------------------

def test_model_forward_logits_shape(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    B, n = 2, 16
    ids = _make_ids(toy_cfg, B, n)
    out = toy_model(ids)
    assert out["logits"].shape == (B, n, toy_cfg.vocab_size)


def test_model_forward_with_labels_has_loss(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    ids    = _make_ids(toy_cfg, B=2, n=16)
    labels = _make_ids(toy_cfg, B=2, n=16)
    out    = toy_model(ids, labels=labels)
    assert "loss" in out
    loss = out["loss"]
    assert loss.shape == ()          # scalar
    assert torch.isfinite(loss).item()


def test_model_forward_without_labels_no_loss(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    ids = _make_ids(toy_cfg, B=2, n=16)
    out = toy_model(ids)
    assert "loss" not in out


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------

def test_weight_tying_same_tensor(toy_cfg: TRNConfig) -> None:
    cfg = TRNConfig.toy()  # tie_weights=True by default
    model = TRNModel(cfg)
    assert model.lm_head.weight is model.embedding.weight


def test_weight_tying_disabled() -> None:
    cfg = TRNConfig(
        vocab_size=256, d_model=128, n_oscillators=64,
        n_layers=2, d_ff=512, max_seq_len=512, tie_weights=False,
    )
    model = TRNModel(cfg)
    assert model.lm_head.weight is not model.embedding.weight


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def test_num_parameters(toy_model: TRNModel) -> None:
    n = toy_model.num_parameters(non_embedding=False)
    assert isinstance(n, int)
    assert n > 0


def test_num_parameters_non_embedding_smaller(toy_model: TRNModel) -> None:
    total     = toy_model.num_parameters(non_embedding=False)
    non_embed = toy_model.num_parameters(non_embedding=True)
    assert non_embed < total


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def test_generate_shape(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    B, prompt_len = 2, 4
    max_new = 8
    prompt = _make_ids(toy_cfg, B=B, n=prompt_len)
    generated = toy_model.generate(prompt, max_new_tokens=max_new)
    assert generated.shape == (B, max_new)


def test_generate_valid_tokens(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    B, prompt_len = 2, 4
    prompt = _make_ids(toy_cfg, B=B, n=prompt_len)
    generated = toy_model.generate(prompt, max_new_tokens=8)
    assert (generated >= 0).all()
    assert (generated < toy_cfg.vocab_size).all()


# ---------------------------------------------------------------------------
# Optimizer param groups
# ---------------------------------------------------------------------------

def test_configure_optimizer_groups(toy_model: TRNModel) -> None:
    groups = toy_model.configure_optimizer_param_groups()
    assert isinstance(groups, list)
    # Multi-scale oscillators add a slow_lr group: expect 2 or 3 groups
    # (3 when omega_base / res_scale params exist, 2 for models without them)
    assert len(groups) >= 2
    for g in groups:
        assert "params" in g
        assert "weight_decay" in g


def test_omega_base_in_no_decay_group(toy_model: TRNModel) -> None:
    """omega_base parameters must not have weight decay applied."""
    groups = toy_model.configure_optimizer_param_groups()
    # omega_base may be in the dedicated slow_lr group or the no-decay group --
    # either way weight_decay must be 0.0.
    zero_wd_params: set[int] = set()
    for g in groups:
        if g["weight_decay"] == 0.0:
            for p in g["params"]:
                zero_wd_params.add(id(p))
    for name, param in toy_model.named_parameters():
        if "omega_base" in name:
            assert id(param) in zero_wd_params, (
                f"omega_base param '{name}' has non-zero weight decay"
            )


# ---------------------------------------------------------------------------
# ADVERSARIAL
# ---------------------------------------------------------------------------

def test_model_single_token(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """n=1 should not crash."""
    ids = _make_ids(toy_cfg, B=2, n=1)
    out = toy_model(ids)
    assert out["logits"].shape == (2, 1, toy_cfg.vocab_size)


def test_model_b1(toy_model: TRNModel, toy_cfg: TRNConfig) -> None:
    """B=1 should not crash."""
    ids = _make_ids(toy_cfg, B=1, n=8)
    out = toy_model(ids)
    assert out["logits"].shape == (1, 8, toy_cfg.vocab_size)
