"""Streaming packed-sequence dataset for causal language model training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class PackedDataset(Dataset):
    """Memory-mapped packed dataset — no padding.

    Stores token ids as uint16 numpy memmap. Each example is a
    seq_len+1 slice used for input[i] and target[i+1].

    Buffer shifts by seq_len (NOT seq_len+1) between consecutive examples
    so the LAST token of one example == FIRST token of the next.
    This is how GPT-style datasets work — fully packed, no wasted tokens.

    Args:
        path: Path to binary file (uint16, raw token ids)
        seq_len: Sequence length for training
    """

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.seq_len = seq_len
        data = np.memmap(str(path), dtype=np.uint16, mode="r")
        # Store as int32 for safety (torch doesn't have uint16)
        self._data = torch.from_numpy(data.astype(np.int32))

    def __len__(self) -> int:
        # Each example needs seq_len+1 tokens; stride is seq_len
        return max(0, (len(self._data) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        start = idx * self.seq_len
        chunk = self._data[start : start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1].long(),
            "labels": chunk[1:].long(),
        }


def build_dataloader(
    path: str | Path,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Convenience function to build a DataLoader from a packed dataset."""
    ds = PackedDataset(path, seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
