"""Tests for PackedDataset and build_dataloader."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from trn.data import PackedDataset, build_dataloader


@pytest.fixture
def tmp_bin(tmp_path):
    """Create a temp binary file with 1000 uint16 tokens (values 0..999)."""
    path = tmp_path / "train.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)
    return path


def test_packed_dataset_length(tmp_bin):
    """__len__ == (N-1) // seq_len."""
    seq_len = 64
    ds = PackedDataset(tmp_bin, seq_len)
    expected = (1000 - 1) // seq_len  # 15
    assert len(ds) == expected


def test_packed_dataset_getitem_shape(tmp_bin):
    """input_ids and labels both have shape (seq_len,)."""
    seq_len = 64
    ds = PackedDataset(tmp_bin, seq_len)
    item = ds[0]
    assert item["input_ids"].shape == (seq_len,)
    assert item["labels"].shape == (seq_len,)


def test_packed_dataset_labels_offset(tmp_bin):
    """labels[i] == input_ids[i+1] within an example (consecutive token shift)."""
    seq_len = 16
    ds = PackedDataset(tmp_bin, seq_len)
    item = ds[0]
    # input_ids[1:] should equal labels[:-1]
    assert torch.equal(item["input_ids"][1:], item["labels"][:-1])


def test_packed_dataset_no_padding(tmp_bin):
    """All tokens in input_ids are valid — no artificial zero-padding."""
    seq_len = 32
    ds = PackedDataset(tmp_bin, seq_len)
    # The data is np.arange(1000), so tokens 0..31 are 0..31.
    # We just verify that the last example also has full seq_len tokens.
    last_idx = len(ds) - 1
    item = ds[last_idx]
    assert item["input_ids"].shape == (seq_len,)
    assert item["labels"].shape == (seq_len,)


def test_build_dataloader_smoke(tmp_bin):
    """build_dataloader returns a DataLoader; first batch has correct shapes."""
    seq_len = 16
    batch_size = 4
    dl = build_dataloader(tmp_bin, seq_len=seq_len, batch_size=batch_size, shuffle=False)
    batch = next(iter(dl))
    assert batch["input_ids"].shape == (batch_size, seq_len)
    assert batch["labels"].shape == (batch_size, seq_len)


def test_packed_dataset_empty_file(tmp_path):
    """File with fewer than seq_len+1 tokens → __len__ == 0."""
    path = tmp_path / "tiny.bin"
    # Only 8 tokens, seq_len=16 → can't form even one example
    np.arange(8, dtype=np.uint16).tofile(path)
    ds = PackedDataset(path, seq_len=16)
    assert len(ds) == 0


def test_packed_dataset_adversarial_single_example(tmp_path):
    """Exactly seq_len+1 tokens → __len__ == 1."""
    seq_len = 32
    path = tmp_path / "exact.bin"
    np.arange(seq_len + 1, dtype=np.uint16).tofile(path)
    ds = PackedDataset(path, seq_len=seq_len)
    assert len(ds) == 1
    item = ds[0]
    assert item["input_ids"].shape == (seq_len,)
    assert item["labels"].shape == (seq_len,)
