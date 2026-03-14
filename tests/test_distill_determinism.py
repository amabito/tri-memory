"""Verify distillation training determinism: same seed = same losses."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trimemory.config import TRNConfig
from trimemory.model import TRNModel


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _tiny_cfg(vocab_size: int = 256) -> TRNConfig:
    return TRNConfig(
        vocab_size=vocab_size, d_model=32, n_oscillators=16,
        n_layers=1, d_ff=64, max_seq_len=64,
    )


def _make_fake_teacher(vocab_size: int) -> nn.Module:
    """A tiny frozen MLP that acts as a deterministic teacher."""

    class FakeTeacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, 32)
            self.proj = nn.Linear(32, vocab_size)

        def forward(self, input_ids: torch.Tensor) -> object:
            x = self.emb(input_ids)
            logits = self.proj(x)

            class Result:
                pass

            r = Result()
            r.logits = logits
            return r

    teacher = FakeTeacher()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _run_distill(seed: int, n_steps: int = 20) -> list[float]:
    """Run a short distillation loop and return per-step losses."""
    vocab_size = 256
    _seed_all(seed)

    cfg = _tiny_cfg(vocab_size)
    student = TRNModel(cfg)
    teacher = _make_fake_teacher(vocab_size)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    temperature = 2.0
    kl_weight = 1.0
    ce_weight = 0.1

    rng = np.random.default_rng(seed)
    losses = []
    student.train()

    for step in range(n_steps):
        input_ids = torch.tensor(
            rng.integers(0, vocab_size, size=(4, 32)),
            dtype=torch.long,
        )

        optimizer.zero_grad()

        s_out = student(input_ids)
        s_logits = s_out["logits"]

        with torch.no_grad():
            t_out = teacher(input_ids)
            t_logits = t_out.logits

        # Causal shift
        shift_s = s_logits[:, :-1].contiguous()
        shift_t = t_logits[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()

        # KL
        s_log_probs = F.log_softmax(shift_s / temperature, dim=-1)
        t_probs = F.softmax(shift_t / temperature, dim=-1)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)

        # CE
        ce = F.cross_entropy(
            shift_s.view(-1, shift_s.size(-1)),
            targets.view(-1),
        )

        loss = kl_weight * kl + ce_weight * ce
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    return losses


def test_distill_deterministic():
    """Two runs with same seed must produce identical losses."""
    losses_a = _run_distill(seed=42)
    losses_b = _run_distill(seed=42)
    assert losses_a == losses_b, f"Non-deterministic:\n  a={losses_a[:5]}\n  b={losses_b[:5]}"


def test_distill_different_seeds():
    """Different seeds must produce different losses."""
    losses_42 = _run_distill(seed=42)
    losses_99 = _run_distill(seed=99)
    assert losses_42 != losses_99, "Different seeds produced identical losses"


def test_distill_loss_decreases():
    """Loss should generally decrease over 20 steps on a learnable task."""
    losses = _run_distill(seed=42, n_steps=20)
    avg_first_5 = sum(losses[:5]) / 5
    avg_last_5 = sum(losses[-5:]) / 5
    assert avg_last_5 < avg_first_5, (
        f"Loss did not decrease: first_5={avg_first_5:.4f}, last_5={avg_last_5:.4f}"
    )
