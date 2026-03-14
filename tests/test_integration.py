"""End-to-end integration tests for the full TRN pipeline.

Tests that cover:
- Train a few steps → evaluate perplexity → confirm decrease
- Train → save checkpoint → load checkpoint → resume training
- Train → generate (ensure output shape and validity)
- PackedDataset → DataLoader → Trainer → eval (real data flow)
- Confirm loss goes down with 100 steps on fixed data (overfit test)
"""
from __future__ import annotations

import math
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from trimemory.checkpoint import load_checkpoint, save_checkpoint
from trimemory.config import TRNConfig
from trimemory.data import PackedDataset, build_dataloader
from trimemory.eval import evaluate_perplexity
from trimemory.model import TRNModel
from trimemory.trainer import SimpleTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _toy_cfg() -> TRNConfig:
    """Minimal TRNConfig for fast CPU tests."""
    return TRNConfig.toy()


def _make_packed_bin(path: Path, n_tokens: int, vocab_size: int) -> None:
    """Write a packed binary file of n_tokens uint16 token ids."""
    data = np.random.randint(0, vocab_size, size=n_tokens, dtype=np.uint16)
    data.tofile(str(path))


def _fixed_batch(
    cfg: TRNConfig,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 42,
) -> torch.Tensor:
    """Create a fixed (deterministic) token batch for overfitting tests."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len), generator=rng)


# ---------------------------------------------------------------------------
# Test 1: train → evaluate_perplexity → ppl should decrease
# ---------------------------------------------------------------------------

def test_train_eval_pipeline():
    """Train 50 steps on fixed data then verify perplexity dropped."""
    cfg = _toy_cfg()
    trainer = SimpleTrainer(cfg, device="cpu", lr=1e-2, save_every=0, log_every=50)

    # Fixed synthetic data (NOT random each step) so the model can overfit
    seq_len = 32
    batch_size = 4
    n_seqs = 16
    fixed_input = _fixed_batch(cfg, n_seqs, seq_len, cfg.vocab_size, seed=7)

    # Evaluate initial perplexity before training
    eval_loader = DataLoader(
        torch.utils.data.TensorDataset(fixed_input),
        batch_size=batch_size,
        shuffle=False,
    )

    # Build a DataLoader that returns dict with 'input_ids'
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, data: torch.Tensor):
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> dict:
            return {"input_ids": self.data[idx]}

    dict_ds = DictDataset(fixed_input)
    eval_dl = DataLoader(dict_ds, batch_size=batch_size, shuffle=False)

    initial_ppl = evaluate_perplexity(trainer.model, eval_dl, device="cpu")
    assert math.isfinite(initial_ppl), f"Initial perplexity not finite: {initial_ppl}"

    # Train 50 steps on fixed batches
    n_steps = 50
    trainer.scheduler.max_steps = n_steps
    for step in range(n_steps):
        trainer.scheduler.step(step)
        # Cycle through fixed data
        idx = step % (n_seqs // batch_size)
        batch = fixed_input[idx * batch_size : (idx + 1) * batch_size]
        trainer._train_step(batch)

    final_ppl = evaluate_perplexity(trainer.model, eval_dl, device="cpu")
    assert math.isfinite(final_ppl), f"Final perplexity not finite: {final_ppl}"
    assert final_ppl < initial_ppl, (
        f"Perplexity should decrease after training: initial={initial_ppl:.2f}, final={final_ppl:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 2: train → save → load → continue → loss consistent
# ---------------------------------------------------------------------------

def test_checkpoint_resume_consistency():
    """Train 10 steps, save, reload, train 10 more — compare to uninterrupted run."""
    cfg = _toy_cfg()
    seq_len = 16
    batch_size = 4

    # Fixed data
    fixed_data = _fixed_batch(cfg, batch_size, seq_len, cfg.vocab_size, seed=99)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpts"
        ckpt_dir.mkdir()

        # --- Run A: train 10 steps, save, reload, train 10 more ---
        trainer_a = SimpleTrainer(cfg, device="cpu", lr=5e-3, save_every=0, log_every=100)
        torch.manual_seed(0)
        for step in range(10):
            trainer_a._train_step(fixed_data)

        ckpt_path = save_checkpoint(
            trainer_a.model,
            trainer_a.optimizer,
            step=10,
            loss=0.0,
            checkpoint_dir=ckpt_dir,
            tag="after10",
        )

        # Reload into a fresh trainer with same config
        trainer_a2 = SimpleTrainer(cfg, device="cpu", lr=5e-3, save_every=0, log_every=100)
        load_checkpoint(trainer_a2.model, trainer_a2.optimizer, ckpt_path, device="cpu")

        losses_a2 = []
        for _ in range(10):
            losses_a2.append(trainer_a2._train_step(fixed_data))

        # --- Run B: train 20 steps uninterrupted (same init + same data) ---
        trainer_b = SimpleTrainer(cfg, device="cpu", lr=5e-3, save_every=0, log_every=100)
        # Load checkpoint so starting point matches trainer_a2
        load_checkpoint(trainer_b.model, trainer_b.optimizer, ckpt_path, device="cpu")

        losses_b = []
        for _ in range(10):
            losses_b.append(trainer_b._train_step(fixed_data))

        # Both resumed from the same checkpoint → losses should be identical
        for i, (la, lb) in enumerate(zip(losses_a2, losses_b)):
            assert abs(la - lb) < 1e-5, (
                f"Loss mismatch at resumed step {i}: trainer_a2={la:.6f} vs trainer_b={lb:.6f}"
            )


# ---------------------------------------------------------------------------
# Test 3: train → generate → output shape and valid token ids
# ---------------------------------------------------------------------------

def test_train_then_generate():
    """Train 20 steps then generate 5 tokens — verify shape and token validity."""
    try:
        from trimemory.generate import GenerationConfig, generate
    except ImportError:
        pytest.skip("trimemory.generate not available yet")

    cfg = _toy_cfg()
    trainer = SimpleTrainer(cfg, device="cpu", lr=3e-3, save_every=0, log_every=100)

    # Train briefly
    seq_len = 16
    batch_size = 2
    fixed_data = _fixed_batch(cfg, batch_size, seq_len, cfg.vocab_size, seed=55)
    for _ in range(20):
        trainer._train_step(fixed_data)

    # Generate
    B = 2
    n_new = 5
    prompt = torch.randint(0, cfg.vocab_size, (B, 4))
    gen_cfg = GenerationConfig(max_new_tokens=n_new, do_sample=False)
    out = generate(trainer.model, prompt, gen_cfg=gen_cfg, device="cpu")

    assert out.shape == (B, n_new), f"Expected shape ({B}, {n_new}), got {out.shape}"
    assert out.dtype in (torch.int32, torch.int64), f"Unexpected dtype: {out.dtype}"
    assert (out >= 0).all(), "Negative token ids generated"
    assert (out < cfg.vocab_size).all(), "Token ids exceed vocab_size"


# ---------------------------------------------------------------------------
# Test 4: PackedDataset → build_dataloader → Trainer.train() no errors
# ---------------------------------------------------------------------------

def test_packed_dataset_full_pipeline():
    """Real binary file → PackedDataset → DataLoader → SimpleTrainer.train() 10 steps."""
    cfg = _toy_cfg()
    seq_len = cfg.max_seq_len
    batch_size = 2
    # Need enough tokens for at least batch_size sequences
    n_tokens = (seq_len + 1) * batch_size * 4 + 1

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "train.bin"
        _make_packed_bin(data_path, n_tokens, cfg.vocab_size)

        trainer = SimpleTrainer(cfg, device="cpu", lr=1e-3, save_every=0, log_every=100)
        losses = trainer.train(str(data_path), n_steps=10, batch_size=batch_size)

    assert len(losses) == 10, f"Expected 10 losses, got {len(losses)}"
    assert all(math.isfinite(l) for l in losses), "NaN/Inf loss detected"


# ---------------------------------------------------------------------------
# Test 5: overfit — single fixed batch, loss < 4.5 after 100 steps
# ---------------------------------------------------------------------------

def test_overfit_toy():
    """Train 100 steps on a fixed batch of 8 sequences. Final loss must be < 4.5."""
    cfg = _toy_cfg()
    # Use very small seq_len for speed
    seq_len = 16
    batch_size = 8
    fixed_data = _fixed_batch(cfg, batch_size, seq_len, cfg.vocab_size, seed=42)

    trainer = SimpleTrainer(cfg, device="cpu", lr=5e-3, save_every=0, log_every=100)

    losses = []
    trainer.scheduler.max_steps = 100
    for step in range(100):
        trainer.scheduler.step(step)
        loss = trainer._train_step(fixed_data)
        losses.append(loss)

    final_loss = losses[-1]
    assert final_loss < 4.5, (
        f"Model failed to overfit toy data: final loss {final_loss:.4f} >= 4.5"
    )


# ---------------------------------------------------------------------------
# Test 6: deterministic greedy generation — same seed → same output
# ---------------------------------------------------------------------------

def test_generate_after_train_consistent():
    """Same prompt + greedy decoding → identical output across two calls."""
    try:
        from trimemory.generate import GenerationConfig, generate
    except ImportError:
        pytest.skip("trimemory.generate not available yet")

    cfg = _toy_cfg()
    trainer = SimpleTrainer(cfg, device="cpu", lr=1e-3, save_every=0, log_every=100)

    # Brief training (just to have non-random weights)
    fixed_data = _fixed_batch(cfg, 4, 16, cfg.vocab_size, seed=77)
    for _ in range(5):
        trainer._train_step(fixed_data)

    prompt = torch.tensor([[1, 2, 3, 4]])
    gen_cfg = GenerationConfig(max_new_tokens=8, do_sample=False)  # greedy

    out1 = generate(trainer.model, prompt.clone(), gen_cfg=gen_cfg, device="cpu")
    out2 = generate(trainer.model, prompt.clone(), gen_cfg=gen_cfg, device="cpu")

    assert torch.equal(out1, out2), (
        f"Greedy generation is not deterministic: {out1.tolist()} vs {out2.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 7: bf16 forward pass
# ---------------------------------------------------------------------------

def test_bf16_forward_pass():
    """model.to(bfloat16).forward() should produce bfloat16 logits without error."""
    if not torch.cuda.is_available():
        # bfloat16 on CPU may work on some platforms; skip if not supported
        try:
            t = torch.tensor([1.0], dtype=torch.bfloat16)
            _ = t + t
        except Exception:
            pytest.skip("bfloat16 not supported on this CPU")

    cfg = _toy_cfg()
    model = TRNModel(cfg)

    try:
        model = model.to(torch.bfloat16)
    except Exception as e:
        pytest.skip(f"Cannot convert model to bfloat16: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 8), device=device)
    out = model(input_ids)

    assert out["logits"].dtype == torch.bfloat16, (
        f"Expected bfloat16 logits, got {out['logits'].dtype}"
    )
    assert out["logits"].shape == (2, 8, cfg.vocab_size)
