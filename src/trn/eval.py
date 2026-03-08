"""Perplexity evaluation for TRN language models."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import PackedDataset


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataset_or_loader,
    batch_size: int = 8,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> dict:
    """Evaluate model perplexity on a dataset or dataloader.

    Args:
        model: any model with forward(input_ids, labels) -> dict with 'loss'
        dataset_or_loader: PackedDataset or DataLoader
        batch_size: batch size (ignored if DataLoader provided)
        device: device to run on
        max_batches: limit number of batches (None = all)

    Returns:
        dict with 'loss', 'perplexity', 'n_batches'
    """
    model.eval()

    if isinstance(dataset_or_loader, DataLoader):
        loader = dataset_or_loader
    else:
        loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    n = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        out = model(input_ids, labels=input_ids)
        total_loss += out["loss"].item()
        n += 1

    if n == 0:
        return {"loss": float("inf"), "perplexity": float("inf"), "n_batches": 0}

    mean_loss = total_loss / n
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "n_batches": n,
    }


# Backward-compatible aliases
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> float:
    """Compute perplexity. Thin wrapper around evaluate()."""
    result = evaluate(model, dataloader, device=device, max_batches=max_batches)
    return result["perplexity"]


def compute_perplexity(
    model: nn.Module,
    dataset: PackedDataset,
    batch_size: int = 8,
    device: str = "cpu",
) -> float:
    """Compute perplexity. Thin wrapper around evaluate()."""
    return evaluate(model, dataset, batch_size=batch_size, device=device)["perplexity"]
