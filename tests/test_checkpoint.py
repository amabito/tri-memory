"""Tests for src/trn/checkpoint.py."""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.checkpoint import save_checkpoint, load_checkpoint


# --- Fixtures ---

@pytest.fixture()
def toy_model() -> TRNModel:
    """Small TRN model for fast tests."""
    cfg = TRNConfig.toy()
    return TRNModel(cfg)


@pytest.fixture()
def toy_optimizer(toy_model: TRNModel) -> optim.Optimizer:
    return optim.AdamW(toy_model.parameters(), lr=1e-3)


# --- save_checkpoint tests ---

def test_save_creates_file(tmp_path: Path, toy_model: TRNModel, toy_optimizer: optim.Optimizer) -> None:
    """save_checkpoint must create a .pt file at the expected path."""
    path = save_checkpoint(toy_model, toy_optimizer, step=10, loss=2.5, checkpoint_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".pt"
    assert path.name == "latest.pt"


def test_save_custom_tag(tmp_path: Path, toy_model: TRNModel, toy_optimizer: optim.Optimizer) -> None:
    """Custom tag should be reflected in the filename."""
    path = save_checkpoint(toy_model, toy_optimizer, step=0, loss=1.0,
                           checkpoint_dir=tmp_path, tag="step_100")
    assert path.name == "step_100.pt"


def test_save_creates_directory(tmp_path: Path, toy_model: TRNModel, toy_optimizer: optim.Optimizer) -> None:
    """save_checkpoint must create checkpoint_dir if it does not exist."""
    nested = tmp_path / "deep" / "nested" / "ckpts"
    save_checkpoint(toy_model, toy_optimizer, step=0, loss=0.0, checkpoint_dir=nested)
    assert nested.is_dir()


def test_save_returns_path(tmp_path: Path, toy_model: TRNModel, toy_optimizer: optim.Optimizer) -> None:
    """Return value must be a Path pointing to the saved file."""
    returned = save_checkpoint(toy_model, toy_optimizer, step=5, loss=1.23,
                               checkpoint_dir=tmp_path)
    assert isinstance(returned, Path)
    assert returned.exists()


# --- load_checkpoint tests ---

def test_load_restores_state(tmp_path: Path, toy_model: TRNModel, toy_optimizer: optim.Optimizer) -> None:
    """load_checkpoint must restore model parameters in-place."""
    path = save_checkpoint(toy_model, toy_optimizer, step=1, loss=3.0, checkpoint_dir=tmp_path)

    # Create fresh model with different weights
    cfg = TRNConfig.toy()
    new_model = TRNModel(cfg)
    torch.nn.init.constant_(new_model.embedding.weight, 0.0)

    load_checkpoint(new_model, optimizer=None, path=path)

    for (n1, p1), (n2, p2) in zip(
        toy_model.named_parameters(), new_model.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"Parameter mismatch for {n1}"


def test_checkpoint_roundtrip(tmp_path: Path, toy_model: TRNModel, toy_optimizer: optim.Optimizer) -> None:
    """Full roundtrip: save -> load -> parameters identical."""
    original_params = {n: p.clone() for n, p in toy_model.named_parameters()}

    path = save_checkpoint(toy_model, toy_optimizer, step=42, loss=1.5, checkpoint_dir=tmp_path)

    cfg = TRNConfig.toy()
    restored_model = TRNModel(cfg)
    restored_opt = optim.AdamW(restored_model.parameters(), lr=1e-3)

    load_checkpoint(restored_model, restored_opt, path=path)

    for name, param in restored_model.named_parameters():
        assert torch.allclose(original_params[name], param), f"Mismatch: {name}"


def test_missing_checkpoint_raises(tmp_path: Path, toy_model: TRNModel) -> None:
    """Loading from a nonexistent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint(toy_model, optimizer=None, path=tmp_path / "ghost.pt")


def test_checkpoint_step_loss_preserved(tmp_path: Path, toy_model: TRNModel, toy_optimizer: optim.Optimizer) -> None:
    """Saved step and loss values must be recoverable from the checkpoint dict."""
    save_checkpoint(toy_model, toy_optimizer, step=99, loss=0.314,
                    checkpoint_dir=tmp_path, tag="epoch1")

    cfg = TRNConfig.toy()
    restored = TRNModel(cfg)
    ckpt = load_checkpoint(restored, optimizer=None, path=tmp_path / "epoch1.pt")

    assert ckpt["step"] == 99
    assert ckpt["loss"] == pytest.approx(0.314)


def test_load_with_optimizer_restores_optimizer_state(tmp_path: Path) -> None:
    """Optimizer state should be restored correctly when optimizer is provided."""
    cfg = TRNConfig.toy()
    model = TRNModel(cfg)
    opt = optim.AdamW(model.parameters(), lr=5e-4)

    # Do a fake gradient step to populate optimizer state
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model(dummy_input, labels=dummy_input)
    out["loss"].backward()
    opt.step()
    opt.zero_grad()

    path = save_checkpoint(model, opt, step=1, loss=out["loss"].detach().item(), checkpoint_dir=tmp_path)

    # Restore into new model + optimizer
    new_model = TRNModel(cfg)
    new_opt = optim.AdamW(new_model.parameters(), lr=5e-4)
    ckpt = load_checkpoint(new_model, new_opt, path=path)

    # Optimizer state should exist for all parameter groups
    assert len(new_opt.state) > 0
    assert ckpt["step"] == 1


def test_load_string_path(tmp_path: Path, toy_model: TRNModel, toy_optimizer: optim.Optimizer) -> None:
    """load_checkpoint must accept str path as well as Path."""
    path = save_checkpoint(toy_model, toy_optimizer, step=0, loss=0.0, checkpoint_dir=tmp_path)
    cfg = TRNConfig.toy()
    new_model = TRNModel(cfg)
    # Pass as string
    load_checkpoint(new_model, optimizer=None, path=str(path))

    for (n1, p1), (n2, p2) in zip(toy_model.named_parameters(), new_model.named_parameters()):
        assert torch.allclose(p1, p2)
