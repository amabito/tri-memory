"""Perplexity evaluation for TRNModel."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import PackedDataset
from .model import TRNModel


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> float:
    """Compute perplexity on a dataset.

    Args:
        model: TRNModel (or any model with forward(input_ids, labels) -> dict with 'loss')
        dataloader: yields dicts with 'input_ids' and 'labels'
        device: device to run on
        max_batches: limit number of batches (None = all)

    Returns:
        perplexity = exp(mean cross-entropy loss)
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)

        # model.forward applies causal shift internally (labels[:,1:]).
        # Pass input_ids as labels — not pre-shifted PackedDataset labels.
        out = model(input_ids, labels=input_ids)
        total_loss += out["loss"].item()
        total_batches += 1

    if total_batches == 0:
        return float("inf")

    mean_loss = total_loss / total_batches
    return math.exp(mean_loss)


def compute_perplexity(
    model: TRNModel,
    dataset: PackedDataset,
    batch_size: int = 8,
    device: str = "cpu",
) -> float:
    """Compute perplexity on a dataset.

    Returns exp(mean cross-entropy loss) averaged over all batches.
    Uses inference_mode for speed.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_batches = 0

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            # model.forward does the causal shift internally (labels[:,1:]),
            # so pass input_ids as labels — not the pre-shifted PackedDataset labels.
            out = model(input_ids, labels=input_ids)
            total_loss += out["loss"].item()
            total_batches += 1

    if total_batches == 0:
        return float("inf")

    mean_loss = total_loss / total_batches
    return math.exp(mean_loss)


def evaluate(
    model: TRNModel,
    dataset: PackedDataset,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict:
    """Full evaluation returning loss + perplexity."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    n = 0

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            # model.forward does causal shift internally — pass input_ids as labels.
            out = model(input_ids, labels=input_ids)
            total_loss += out["loss"].item()
            n += 1

    mean_loss = total_loss / max(n, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "n_batches": n,
    }
