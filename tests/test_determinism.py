"""Verify that identical seeds produce identical training outcomes."""
from __future__ import annotations

import pytest
import torch

from trimemory.bench_data import seed_everything, NextTokenCopyDataset
from trimemory.config import TRNConfig
from trimemory.model import TRNModel
from trimemory.baseline import TransformerModel
from torch.utils.data import DataLoader


def _toy_cfg() -> TRNConfig:
    return TRNConfig(
        vocab_size=32, d_model=32, n_oscillators=16, n_layers=1, d_ff=64, max_seq_len=32,
    )


def _run_steps(model_cls, cfg, n_steps: int, seed: int) -> list[float]:
    seed_everything(seed)
    model = model_cls(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ds = NextTokenCopyDataset(seq_len=16, vocab_size=cfg.vocab_size, seed=seed)
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    loader_iter = iter(loader)
    losses = []
    model.train()
    for _ in range(n_steps):
        batch = next(loader_iter)
        ids = batch["input_ids"]
        out = model(ids, labels=ids)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_trn_deterministic():
    """Two runs with same seed must produce identical loss curves."""
    cfg = _toy_cfg()
    losses_a = _run_steps(TRNModel, cfg, n_steps=5, seed=42)
    losses_b = _run_steps(TRNModel, cfg, n_steps=5, seed=42)
    assert losses_a == losses_b, f"Non-deterministic: {losses_a} vs {losses_b}"


def test_transformer_deterministic():
    """Same check for Transformer baseline."""
    cfg = _toy_cfg()
    losses_a = _run_steps(TransformerModel, cfg, n_steps=5, seed=42)
    losses_b = _run_steps(TransformerModel, cfg, n_steps=5, seed=42)
    assert losses_a == losses_b


def test_different_seeds_different_losses():
    """Different seeds must produce different loss curves (sanity)."""
    cfg = _toy_cfg()
    losses_42 = _run_steps(TRNModel, cfg, n_steps=5, seed=42)
    losses_99 = _run_steps(TRNModel, cfg, n_steps=5, seed=99)
    assert losses_42 != losses_99


def test_seed_everything_reproducible():
    """seed_everything followed by torch.rand must give same tensor twice."""
    seed_everything(42)
    t1 = torch.rand(10)
    seed_everything(42)
    t2 = torch.rand(10)
    torch.testing.assert_close(t1, t2)
