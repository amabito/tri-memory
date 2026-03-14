"""Learning-rate schedule: linear warmup + cosine decay."""
from __future__ import annotations
import math
import torch.optim as optim


def get_lr(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    lr: float,
    min_lr: float = 0.0,
) -> float:
    """Compute LR at a given step.

    Linear warmup for steps 0..warmup_steps, then cosine decay
    to min_lr over warmup_steps..max_steps.
    After max_steps: returns min_lr.

    Args:
        step: current training step (0-indexed)
        warmup_steps: number of warmup steps
        max_steps: total training steps
        lr: peak learning rate
        min_lr: floor LR (default 0.0 = full decay to zero)
    """
    if step < warmup_steps:
        # Linear warmup: 0 -> lr
        return lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


class CosineWithWarmup:
    """Stateful LR scheduler compatible with optimizer param_groups.

    Usage:
        sched = CosineWithWarmup(optimizer, warmup_steps=100, max_steps=10000, lr=3e-4)
        for step in range(max_steps):
            sched.step(step)
            optimizer.step()
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        lr: float,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.lr = lr
        self.min_lr = min_lr

    def step(self, step: int) -> float:
        """Update optimizer LR and return the new value."""
        new_lr = get_lr(
            step,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            lr=self.lr,
            min_lr=self.min_lr,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr
