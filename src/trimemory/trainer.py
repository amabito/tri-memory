"""Training loop for TRN language models."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import TRNConfig
from .data import PackedDataset, build_dataloader
from .model import TRNModel
from .scheduler import CosineWithWarmup


@dataclass
class TrainConfig:
    max_steps: int = 10_000
    warmup_steps: int = 500
    lr: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 8
    grad_accum: int = 1
    log_interval: int = 50
    save_interval: int = 1_000
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"


class Trainer:
    """Training loop for TRNModel.

    Handles optimizer setup, cosine LR schedule, gradient accumulation,
    gradient clipping, and periodic checkpointing.
    """

    def __init__(
        self,
        model: TRNModel,
        train_dataset: PackedDataset,
        val_dataset: Optional[PackedDataset] = None,
        cfg: TrainConfig = TrainConfig(),
    ) -> None:
        self.model = model.to(cfg.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.step = 0
        self.loss_history: list[float] = []

        param_groups = model.configure_optimizer_param_groups(cfg.weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95))
        self.scheduler = CosineWithWarmup(
            self.optimizer,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            lr=cfg.lr,
            min_lr=cfg.lr_min,
        )

    def train(self) -> list[float]:
        """Run training loop. Returns per-step loss history."""
        cfg = self.cfg
        device = cfg.device
        loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        data_iter = cycle(loader)

        self.model.train()
        self.optimizer.zero_grad()

        while self.step < cfg.max_steps:
            # Apply LR schedule before the optimizer step
            current_lr = self.scheduler.step(self.step)

            # Gradient accumulation loop
            accum_loss = 0.0
            for _ in range(cfg.grad_accum):
                batch = next(data_iter)
                input_ids = batch["input_ids"].to(device)
                # model.forward applies causal shift internally (shift_labels = labels[:,1:]).
                # PackedDataset.labels = chunk[1:], so shift_labels would be chunk[2:] — wrong.
                # Pass input_ids so shift_labels = input_ids[:,1:] = chunk[1:seq_len-1] — correct.
                out = self.model(input_ids, labels=input_ids)
                loss = out["loss"] / cfg.grad_accum
                loss.backward()
                accum_loss += loss.item()

            # Clip gradients and update
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.loss_history.append(accum_loss)

            if self.step % cfg.log_interval == 0:
                recent = self.loss_history[-min(10, len(self.loss_history)):]
                avg = sum(recent) / len(recent)
                print(f"step {self.step:6d}: loss={avg:.4f}  lr={current_lr:.2e}")

            if (
                cfg.save_interval > 0
                and self.step > 0
                and self.step % cfg.save_interval == 0
            ):
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    step=self.step,
                    loss=accum_loss,
                    checkpoint_dir=cfg.checkpoint_dir,
                    tag=f"step_{self.step:06d}",
                )

            self.step += 1

        return self.loss_history


class SimpleTrainer:
    """High-level training loop that accepts TRNConfig + hyper-params directly.

    Supports both real data (PackedDataset binary file) and synthetic random data.
    """

    def __init__(
        self,
        cfg: TRNConfig,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        log_every: int = 10,
        save_every: int = 500,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every = log_every
        self.save_every = save_every
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.model = TRNModel(cfg).to(device)
        param_groups = self.model.configure_optimizer_param_groups(weight_decay=weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr)
        # max_steps will be updated before each run
        self.scheduler = CosineWithWarmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            max_steps=10_000,
            lr=lr,
            min_lr=lr * 0.1,
        )

    def _train_step(self, input_ids: torch.Tensor) -> float:
        """Run one forward + backward + optimizer step. Returns loss value."""
        self.model.train()
        self.optimizer.zero_grad()
        # model.forward applies causal shift internally (shift_labels = labels[:,1:]).
        # Pass input_ids as labels so targets = input_ids[:,1:] (correct GPT-style).
        ids = input_ids.to(self.device)
        out = self.model(ids, labels=ids)
        loss = out["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def train_synthetic(
        self,
        n_steps: int,
        batch_size: int,
        seq_len: Optional[int] = None,
    ) -> list[float]:
        """Train on synthetic random data for n_steps. Returns list of losses."""
        seq_len = seq_len or self.cfg.max_seq_len
        self.scheduler.max_steps = n_steps
        losses: list[float] = []

        for step in range(n_steps):
            self.scheduler.step(step)
            input_ids = torch.randint(
                0, self.cfg.vocab_size, (batch_size, seq_len), device=self.device
            )
            loss_val = self._train_step(input_ids)
            losses.append(loss_val)

            if (step + 1) % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"step {step + 1:5d}/{n_steps} | loss {loss_val:.4f} | lr {lr:.2e}")

            if self.save_every > 0 and (step + 1) % self.save_every == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    step=step,
                    loss=loss_val,
                    checkpoint_dir=self.checkpoint_dir,
                )

        return losses

    def train(
        self,
        data_path: str,
        n_steps: int,
        batch_size: int,
    ) -> list[float]:
        """Train on real packed binary data for n_steps. Returns list of losses."""
        loader = build_dataloader(
            data_path,
            seq_len=self.cfg.max_seq_len,
            batch_size=batch_size,
            shuffle=True,
        )
        self.scheduler.max_steps = n_steps
        losses: list[float] = []
        loader_iter = iter(loader)

        for step in range(n_steps):
            self.scheduler.step(step)
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            loss_val = self._train_step(batch["input_ids"])
            losses.append(loss_val)

            if (step + 1) % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"step {step + 1:5d}/{n_steps} | loss {loss_val:.4f} | lr {lr:.2e}")

            if self.save_every > 0 and (step + 1) % self.save_every == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    step=step,
                    loss=loss_val,
                    checkpoint_dir=self.checkpoint_dir,
                )

        return losses
