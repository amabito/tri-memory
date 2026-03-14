"""Tests for src/trn/trainer.py."""
from __future__ import annotations
import math
import numpy as np
import pytest
import torch
from pathlib import Path

from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.data import PackedDataset
from trimemory.trainer import Trainer, TrainConfig, SimpleTrainer


# --- Helpers ---

def _make_dataset(tmp_path: Path, n_tokens: int = 500, seq_len: int = 32) -> PackedDataset:
    """Create a small PackedDataset backed by a temp binary file."""
    path = tmp_path / "train.bin"
    rng = np.random.default_rng(seed=42)
    # Vocab size matches TRNConfig.toy() = 256
    data = rng.integers(0, 256, size=n_tokens, dtype=np.uint16)
    data.tofile(str(path))
    return PackedDataset(path, seq_len)


def _make_trainer(
    tmp_path: Path,
    max_steps: int = 10,
    batch_size: int = 4,
    grad_accum: int = 1,
    n_tokens: int = 500,
    seq_len: int = 32,
) -> Trainer:
    cfg_model = TRNConfig.toy()
    model = TRNModel(cfg_model)
    dataset = _make_dataset(tmp_path, n_tokens=n_tokens, seq_len=seq_len)
    train_cfg = TrainConfig(
        max_steps=max_steps,
        warmup_steps=2,
        lr=1e-3,
        lr_min=1e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        batch_size=batch_size,
        grad_accum=grad_accum,
        log_interval=max_steps + 1,  # suppress prints in tests
        save_interval=0,             # disable checkpointing
        device="cpu",
    )
    return Trainer(model, dataset, cfg=train_cfg)


# --- Basic tests ---

def test_trainer_constructs(tmp_path: Path) -> None:
    """Trainer(TRNModel, PackedDataset) should construct without error."""
    trainer = _make_trainer(tmp_path)
    assert trainer.step == 0
    assert len(trainer.loss_history) == 0


def test_trainer_step_counter(tmp_path: Path) -> None:
    """After training N steps, trainer.step == N."""
    n = 5
    trainer = _make_trainer(tmp_path, max_steps=n)
    trainer.train()
    assert trainer.step == n


def test_trainer_loss_history_length(tmp_path: Path) -> None:
    """loss_history must have one entry per step."""
    n = 8
    trainer = _make_trainer(tmp_path, max_steps=n)
    history = trainer.train()
    assert len(history) == n
    assert history is trainer.loss_history


def test_trainer_no_nan_loss(tmp_path: Path) -> None:
    """All recorded losses must be finite (no NaN / inf)."""
    trainer = _make_trainer(tmp_path, max_steps=20)
    history = trainer.train()
    for i, loss in enumerate(history):
        assert math.isfinite(loss), f"Non-finite loss at step {i}: {loss}"


def test_trainer_loss_decreases_100_steps(tmp_path: Path) -> None:
    """Loss should trend downward over 100 steps on a tiny dataset."""
    torch.manual_seed(0)
    trainer = _make_trainer(
        tmp_path,
        max_steps=100,
        batch_size=4,
        n_tokens=500,
        seq_len=32,
    )
    history = trainer.train()

    first_loss = sum(history[:10]) / 10
    last_loss = sum(history[-10:]) / 10
    assert last_loss < first_loss, (
        f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"
    )


def test_trainer_grad_accum(tmp_path: Path) -> None:
    """TrainConfig(grad_accum=2) should complete 20 steps without error."""
    trainer = _make_trainer(tmp_path, max_steps=20, batch_size=4, grad_accum=2)
    history = trainer.train()
    assert len(history) == 20
    for loss in history:
        assert math.isfinite(loss)


# --- Adversarial tests ---

def test_trainer_single_step(tmp_path: Path) -> None:
    """max_steps=1 must complete without crash and produce one loss entry."""
    trainer = _make_trainer(tmp_path, max_steps=1, batch_size=2)
    history = trainer.train()
    assert len(history) == 1
    assert math.isfinite(history[0])


def test_trainer_dataset_smaller_than_batch(tmp_path: Path) -> None:
    """Dataset with barely enough tokens for batch_size=2 should work fine."""
    # seq_len=32 → need at least batch_size * seq_len + 1 tokens
    # batch_size=2, seq_len=32 → 65 tokens minimum
    trainer = _make_trainer(
        tmp_path,
        max_steps=5,
        batch_size=2,
        n_tokens=200,
        seq_len=32,
    )
    history = trainer.train()
    assert len(history) == 5


def test_trainer_checkpoint_saved(tmp_path: Path) -> None:
    """Checkpoints should be saved at the specified interval."""
    ckpt_dir = tmp_path / "ckpts"
    cfg_model = TRNConfig.toy()
    model = TRNModel(cfg_model)
    dataset = _make_dataset(tmp_path, n_tokens=500, seq_len=32)
    train_cfg = TrainConfig(
        max_steps=10,
        warmup_steps=1,
        lr=1e-3,
        lr_min=1e-4,
        batch_size=4,
        grad_accum=1,
        log_interval=100,
        save_interval=5,
        checkpoint_dir=str(ckpt_dir),
        device="cpu",
    )
    trainer = Trainer(model, dataset, cfg=train_cfg)
    trainer.train()

    # step 5 checkpoint should exist
    saved = list(ckpt_dir.glob("*.pt"))
    assert len(saved) > 0, "Expected at least one checkpoint file"


def test_trainer_val_dataset_optional(tmp_path: Path) -> None:
    """val_dataset=None must be accepted without crash."""
    cfg_model = TRNConfig.toy()
    model = TRNModel(cfg_model)
    dataset = _make_dataset(tmp_path)
    train_cfg = TrainConfig(
        max_steps=3,
        warmup_steps=1,
        lr=1e-3,
        lr_min=1e-4,
        batch_size=4,
        grad_accum=1,
        log_interval=100,
        save_interval=0,
        device="cpu",
    )
    trainer = Trainer(model, dataset, val_dataset=None, cfg=train_cfg)
    trainer.train()
    assert trainer.step == 3


# --- SimpleTrainer (TRNConfig-based API) tests ---

def _toy_cfg_small() -> TRNConfig:
    return TRNConfig(
        vocab_size=100,
        d_model=64,
        n_oscillators=32,
        n_layers=2,
        d_ff=256,
        max_seq_len=32,
    )


def test_simple_trainer_synthetic_runs() -> None:
    """SimpleTrainer.train_synthetic completes and returns correct number of losses."""
    cfg = _toy_cfg_small()
    trainer = SimpleTrainer(cfg, device="cpu", log_every=100, save_every=10000)
    losses = trainer.train_synthetic(n_steps=20, batch_size=4, seq_len=16)
    assert len(losses) == 20


def test_simple_trainer_loss_is_finite() -> None:
    """All losses from train_synthetic must be finite."""
    cfg = _toy_cfg_small()
    trainer = SimpleTrainer(cfg, device="cpu", log_every=100, save_every=10000)
    losses = trainer.train_synthetic(n_steps=20, batch_size=4, seq_len=16)
    for i, loss in enumerate(losses):
        assert math.isfinite(loss), f"Non-finite loss at step {i}: {loss}"


def test_simple_trainer_loss_decreases() -> None:
    """Loss at step 20 should be lower than loss at step 1."""
    torch.manual_seed(0)
    cfg = _toy_cfg_small()
    trainer = SimpleTrainer(cfg, device="cpu", lr=5e-3, log_every=100, save_every=10000)
    losses = trainer.train_synthetic(n_steps=20, batch_size=8, seq_len=16)
    first = sum(losses[:3]) / 3
    last = sum(losses[-3:]) / 3
    assert last < first, f"Loss did not decrease: first={first:.4f}, last={last:.4f}"


def test_simple_trainer_saves_checkpoint(tmp_path: Path) -> None:
    """Checkpoint saved when save_every=5 and n_steps=10."""
    cfg = _toy_cfg_small()
    ckpt_dir = tmp_path / "ckpts"
    trainer = SimpleTrainer(
        cfg,
        device="cpu",
        checkpoint_dir=str(ckpt_dir),
        log_every=100,
        save_every=5,
    )
    trainer.train_synthetic(n_steps=10, batch_size=4, seq_len=16)
    saved = list(ckpt_dir.glob("*.pt"))
    assert len(saved) > 0, "Expected at least one checkpoint file"


def test_simple_trainer_with_real_data(tmp_path: Path) -> None:
    """SimpleTrainer.train() on a temp binary file completes without error."""
    data_path = tmp_path / "train.bin"
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, 100, size=2000, dtype=np.uint16)
    tokens.tofile(str(data_path))

    cfg = TRNConfig(
        vocab_size=100,
        d_model=64,
        n_oscillators=32,
        n_layers=2,
        d_ff=256,
        max_seq_len=32,
    )
    trainer = SimpleTrainer(cfg, device="cpu", log_every=100, save_every=10000)
    losses = trainer.train(data_path=str(data_path), n_steps=10, batch_size=4)
    assert len(losses) == 10
    for loss in losses:
        assert math.isfinite(loss)
