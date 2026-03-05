"""Tests for eval.py — perplexity evaluator."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from trn.config import TRNConfig
from trn.data import PackedDataset
from trn.eval import compute_perplexity, evaluate
from trn.model import TRNModel


def make_synthetic_bin(tmp_path, n_tokens: int = 500, name: str = "data.bin"):
    """Write n_tokens random uint16 values to a temp binary file."""
    path = tmp_path / name
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, 256, size=n_tokens, dtype=np.uint16)
    tokens.tofile(path)
    return path


def make_toy_model() -> TRNModel:
    return TRNModel(TRNConfig.toy())


def test_compute_perplexity_returns_float(tmp_path):
    """compute_perplexity returns a float > 0."""
    path = make_synthetic_bin(tmp_path)
    ds = PackedDataset(path, seq_len=32)
    model = make_toy_model()
    ppl = compute_perplexity(model, ds, batch_size=4)
    assert isinstance(ppl, float)
    assert ppl > 0.0


def test_perplexity_finite(tmp_path):
    """Random model on synthetic data → finite perplexity (not inf/nan)."""
    path = make_synthetic_bin(tmp_path, n_tokens=500)
    ds = PackedDataset(path, seq_len=32)
    model = make_toy_model()
    ppl = compute_perplexity(model, ds, batch_size=4)
    assert math.isfinite(ppl)


def test_perplexity_empty_dataset(tmp_path):
    """Dataset with 0 items → returns inf (not a crash)."""
    path = tmp_path / "empty.bin"
    # Only 8 tokens, seq_len=32 → no examples
    np.arange(8, dtype=np.uint16).tofile(path)
    ds = PackedDataset(path, seq_len=32)
    assert len(ds) == 0
    model = make_toy_model()
    ppl = compute_perplexity(model, ds, batch_size=4)
    assert ppl == float("inf")


def test_evaluate_returns_dict(tmp_path):
    """evaluate() returns dict with keys 'loss', 'perplexity', 'n_batches'."""
    path = make_synthetic_bin(tmp_path)
    ds = PackedDataset(path, seq_len=32)
    model = make_toy_model()
    result = evaluate(model, ds, batch_size=4)
    assert "loss" in result
    assert "perplexity" in result
    assert "n_batches" in result


def test_perplexity_ge_1(tmp_path):
    """perplexity >= 1.0 (exp of non-negative cross-entropy loss)."""
    path = make_synthetic_bin(tmp_path, n_tokens=500)
    ds = PackedDataset(path, seq_len=32)
    model = make_toy_model()
    ppl = compute_perplexity(model, ds, batch_size=4)
    assert ppl >= 1.0


def test_model_stays_eval_mode(tmp_path):
    """Model is in eval mode after compute_perplexity."""
    path = make_synthetic_bin(tmp_path)
    ds = PackedDataset(path, seq_len=32)
    model = make_toy_model()
    model.train()  # start in train mode
    compute_perplexity(model, ds, batch_size=4)
    assert not model.training, "Model should be in eval mode after compute_perplexity"


def test_perplexity_overfit(tmp_path):
    """After training on a tiny dataset, perplexity should decrease.

    P0 stabilization (res_scale=0.05, state_norm=True) intentionally starts
    conservative. This test uses standard P0 defaults with enough steps (300)
    and lr=3e-4 with grad clipping to demonstrate learning.
    """
    path = make_synthetic_bin(tmp_path, n_tokens=500)
    ds = PackedDataset(path, seq_len=32)
    model = make_toy_model()

    initial_ppl = compute_perplexity(model, ds, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model.train()
    step = 0
    from itertools import cycle
    for batch in cycle(loader):
        if step >= 300:
            break
        x = batch["input_ids"]
        # Use input_ids as labels — model.forward does the causal shift internally.
        # Using batch["labels"] would double-shift (PackedDataset already shifts by 1).
        out = model(x, labels=x)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    final_ppl = compute_perplexity(model, ds, batch_size=4)
    assert final_ppl < initial_ppl, (
        f"Model failed to overfit: initial={initial_ppl:.2f}, final={final_ppl:.2f}"
    )


import math  # noqa: E402 — needed for test_perplexity_finite

# --- Tests for evaluate_perplexity (DataLoader-based API) ---

from trn.eval import evaluate_perplexity
from torch.utils.data import TensorDataset


def _make_loader(vocab_size=100, seq_len=16, batch_size=4, n_batches=5):
    """Create an in-memory DataLoader with random ids/labels."""
    N = n_batches * batch_size
    ids = torch.randint(0, vocab_size, (N, seq_len))
    labels = torch.randint(0, vocab_size, (N, seq_len))
    ds = TensorDataset(ids, labels)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, y = self.base[i]
            return {"input_ids": x, "labels": y}

    return DataLoader(DictDataset(ds), batch_size=batch_size)


def _toy_cfg():
    return TRNConfig(
        vocab_size=100,
        d_model=64,
        n_oscillators=32,
        n_layers=2,
        d_ff=256,
        max_seq_len=64,
    )


def test_perplexity_random_model():
    """Fresh TRNModel with random weights — ppl must be > 1."""
    model = TRNModel(_toy_cfg())
    loader = _make_loader()
    ppl = evaluate_perplexity(model, loader, device="cpu")
    assert ppl > 1.0


def test_perplexity_finite_and_positive():
    """Result must be finite and positive."""
    model = TRNModel(_toy_cfg())
    loader = _make_loader()
    ppl = evaluate_perplexity(model, loader, device="cpu")
    assert math.isfinite(ppl)
    assert ppl > 0.0


def test_perplexity_max_batches():
    """max_batches=2 → only 2 batches evaluated."""
    call_count = 0
    orig_model = TRNModel(_toy_cfg())

    class CountingModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, labels=None):
            nonlocal call_count
            call_count += 1
            return self.inner(input_ids, labels=labels)

    loader = _make_loader(n_batches=10)
    model = CountingModel(orig_model)
    evaluate_perplexity(model, loader, device="cpu", max_batches=2)
    assert call_count == 2


def test_perplexity_empty_loader():
    """DataLoader with 0 batches → returns inf."""
    model = TRNModel(_toy_cfg())
    loader = _make_loader(n_batches=0)
    ppl = evaluate_perplexity(model, loader, device="cpu")
    assert ppl == float("inf")


def test_perplexity_lower_after_training():
    """After training steps on a fixed tiny batch, perplexity should decrease."""
    cfg = _toy_cfg()
    model = TRNModel(cfg)

    # Fixed single-batch loader — deterministic data so model can overfit
    torch.manual_seed(42)
    ids = torch.randint(0, cfg.vocab_size, (8, 16))
    labels = torch.randint(0, cfg.vocab_size, (8, 16))

    class FixedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            return {"input_ids": ids[i], "labels": labels[i]}

    loader = DataLoader(FixedDataset(), batch_size=8)

    initial_ppl = evaluate_perplexity(model, loader, device="cpu")

    # Train with high lr and enough steps to overfit 8 examples
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(80):
        for batch in loader:
            optimizer.zero_grad()
            # evaluate_perplexity passes input_ids as labels (model does shift internally).
            out = model(batch["input_ids"], labels=batch["input_ids"])
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    final_ppl = evaluate_perplexity(model, loader, device="cpu")
    assert final_ppl < initial_ppl, (
        f"Perplexity did not decrease: initial={initial_ppl:.2f}, final={final_ppl:.2f}"
    )
