"""Tests for src/trn/scheduler.py."""
from __future__ import annotations
import math
import pytest
import torch
import torch.optim as optim

from trn.scheduler import get_lr, CosineWithWarmup


# --- get_lr tests ---

def test_get_lr_warmup_start() -> None:
    """Step 0 during warmup should return 0.0."""
    result = get_lr(0, warmup_steps=100, max_steps=1000, lr=1e-3)
    assert result == 0.0


def test_get_lr_warmup_end() -> None:
    """At step == warmup_steps, LR should equal peak lr."""
    result = get_lr(100, warmup_steps=100, max_steps=1000, lr=1e-3)
    assert result == pytest.approx(1e-3)


def test_get_lr_after_max() -> None:
    """After max_steps, LR should return min_lr."""
    result = get_lr(2000, warmup_steps=100, max_steps=1000, lr=1e-3, min_lr=1e-5)
    assert result == pytest.approx(1e-5)


def test_get_lr_after_max_default_min_lr() -> None:
    """After max_steps with default min_lr=0.0, LR should return 0.0."""
    result = get_lr(5000, warmup_steps=100, max_steps=1000, lr=1e-3)
    assert result == 0.0


def test_get_lr_midpoint() -> None:
    """At midpoint of cosine decay, LR should be midway between lr and min_lr."""
    # warmup=0, max=1000 → cosine from step 0..1000
    # midpoint step = 500 → progress=0.5 → coeff=0.5*(1+cos(pi*0.5))=0.5*(1+0)=0.5
    result = get_lr(500, warmup_steps=0, max_steps=1000, lr=1e-3, min_lr=0.0)
    expected = 0.5 * 1e-3
    assert result == pytest.approx(expected, rel=1e-5)


def test_get_lr_midpoint_with_min_lr() -> None:
    """Midpoint cosine with non-zero min_lr."""
    lr = 1e-3
    min_lr = 1e-4
    result = get_lr(500, warmup_steps=0, max_steps=1000, lr=lr, min_lr=min_lr)
    expected = min_lr + 0.5 * (lr - min_lr)
    assert result == pytest.approx(expected, rel=1e-5)


def test_get_lr_monotone_decay() -> None:
    """LR must be monotonically non-increasing after warmup ends."""
    warmup = 100
    max_steps = 1000
    lr = 1e-3

    values = [
        get_lr(step, warmup_steps=warmup, max_steps=max_steps, lr=lr)
        for step in range(warmup, max_steps + 1)
    ]

    for i in range(len(values) - 1):
        assert values[i] >= values[i + 1] - 1e-12, (
            f"LR increased at step {warmup + i}: {values[i]} -> {values[i + 1]}"
        )


def test_get_lr_warmup_linear() -> None:
    """During warmup, LR must increase linearly from 0 to lr."""
    warmup = 100
    lr = 1e-3

    for step in range(warmup + 1):
        result = get_lr(step, warmup_steps=warmup, max_steps=1000, lr=lr)
        expected = lr * step / warmup
        assert result == pytest.approx(expected, rel=1e-6 if step > 0 else abs(result) + 1e-10)


def test_get_lr_exactly_at_max() -> None:
    """At step == max_steps, return min_lr."""
    result = get_lr(1000, warmup_steps=100, max_steps=1000, lr=1e-3, min_lr=5e-5)
    assert result == pytest.approx(5e-5)


# --- CosineWithWarmup tests ---

def _make_optimizer() -> tuple[torch.nn.Module, optim.Optimizer]:
    """Create a minimal model + optimizer for scheduler tests."""
    model = torch.nn.Linear(4, 4, bias=False)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt


def test_cosine_with_warmup_optimizer_lr() -> None:
    """CosineWithWarmup.step() must update optimizer param_groups lr."""
    _, opt = _make_optimizer()
    sched = CosineWithWarmup(opt, warmup_steps=10, max_steps=100, lr=1e-3)

    # Step 0 → warmup → lr near 0
    lr_0 = sched.step(0)
    assert lr_0 == 0.0
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0)

    # Step 10 → end of warmup → peak lr
    lr_10 = sched.step(10)
    assert lr_10 == pytest.approx(1e-3)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_cosine_with_warmup_returns_lr_value() -> None:
    """step() must return the new LR value."""
    _, opt = _make_optimizer()
    sched = CosineWithWarmup(opt, warmup_steps=10, max_steps=100, lr=1e-3)

    for step in range(100):
        returned = sched.step(step)
        expected = get_lr(step, warmup_steps=10, max_steps=100, lr=1e-3)
        assert returned == pytest.approx(expected, rel=1e-6)


def test_cosine_with_warmup_multiple_param_groups() -> None:
    """Scheduler must update all optimizer param_groups."""
    model_a = torch.nn.Linear(4, 4, bias=False)
    model_b = torch.nn.Linear(4, 4, bias=False)
    opt = optim.AdamW(
        [
            {"params": model_a.parameters(), "lr": 1e-3},
            {"params": model_b.parameters(), "lr": 1e-3},
        ]
    )
    sched = CosineWithWarmup(opt, warmup_steps=5, max_steps=50, lr=1e-3)
    sched.step(5)  # peak lr

    for group in opt.param_groups:
        assert group["lr"] == pytest.approx(1e-3)
