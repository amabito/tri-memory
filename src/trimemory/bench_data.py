"""Benchmark datasets: synthetic tasks, tiny corpus, and seed utilities."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import CharTokenizer


def seed_everything(seed: int = 42) -> None:
    """Seed all RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Synthetic task 1: Next-Token Copy ────────────────────────────────────────

class NextTokenCopyDataset(Dataset):
    """Fixed periodic sequence. Model must learn the period and reach near-zero loss.

    Sequence: pattern * N, where pattern has `period` distinct tokens.
    Both TRN and Transformer should perfectly memorize this; equal footing.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        seq_len: int = 64,
        vocab_size: int = 32,
        period: int = 8,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        # pattern uses tokens 4..vocab_size-1 (avoids special token IDs 0-3)
        pattern = rng.integers(4, vocab_size, size=period).tolist()
        full_len = seq_len + 1
        full = (pattern * ((full_len + period - 1) // period))[:full_len]
        self.data = torch.tensor(full, dtype=torch.long)
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.period = period

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        # All windows are identical for the periodic pattern
        chunk = self.data[:self.seq_len + 1]
        return {"input_ids": chunk[:-1].clone(), "labels": chunk[:-1].clone()}


# ── Synthetic task 2: Selective Copy ─────────────────────────────────────────

class SelectiveCopyDataset(Dataset):
    """Associative recall task requiring sequential memory.

    Sequence format (total length = n_vals + 3):
        [v0, v1, ..., v_{K-1}, SEP, query_idx, answer]

    Model must predict answer = v_{query_idx} given the prefix.
    This tests whether the model can store and recall specific earlier tokens.

    vocab_size must be >= n_vals + 5 (values 4..vocab_size-3, SEP=vocab_size-2,
    query indices 1..n_vals are re-used token IDs).
    """

    SEP_OFFSET = 2  # SEP = vocab_size - SEP_OFFSET

    def __init__(
        self,
        n_samples: int = 2000,
        n_vals: int = 8,
        vocab_size: int = 32,
        seed: int = 42,
    ) -> None:
        assert vocab_size >= n_vals + 6, "vocab_size too small for SelectiveCopy"
        self.n_samples = n_samples
        self.n_vals = n_vals
        self.vocab_size = vocab_size
        self.sep_id = vocab_size - self.SEP_OFFSET
        self.seq_len = n_vals + 2  # prefix length (without answer)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        # values in [4, sep_id - 1]
        vals = self._rng.integers(4, self.sep_id, size=self.n_vals).tolist()
        query = int(self._rng.integers(0, self.n_vals))
        answer = vals[query]
        query_token = query + 1  # 1-indexed to avoid 0

        # Full sequence including answer (for training)
        full = vals + [self.sep_id, query_token, answer]
        ids = torch.tensor(full, dtype=torch.long)
        # Input = prefix + query; model predicts at last position
        return {"input_ids": ids[:-1].clone(), "labels": ids[:-1].clone()}


# ── Real-text tiny corpus ─────────────────────────────────────────────────────

TINY_CORPUS = """\
The quick brown fox jumps over the lazy dog.
To be or not to be, that is the question.
Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune.
All that glitters is not gold.
A journey of a thousand miles begins with a single step.
In the beginning was the Word, and the Word was with God, and the Word was God.
It was the best of times, it was the worst of times, it was the age of wisdom.
It was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity.
Call me Ishmael. Some years ago, never mind how long precisely.
It is a truth universally acknowledged that a single man in possession of a good fortune.
Ask not what your country can do for you, ask what you can do for your country.
We hold these truths to be self-evident, that all men are created equal.
The only way to do great work is to love what you do.
Stay hungry, stay foolish.
In three words I can sum up everything I have learned about life: it goes on.
Two roads diverged in a yellow wood, and sorry I could not travel both.
Not all those who wander are lost.
Elementary, my dear Watson.
May the Force be with you.
To infinity and beyond.
"""


class TinyCorpusDataset(Dataset):
    """Character-level dataset from a small text corpus.

    Splits 80% train / 20% val by character position.
    Uses CharTokenizer; vocab_size is determined by the corpus.
    """

    def __init__(
        self,
        seq_len: int = 64,
        split: str = "train",
        corpus: str = TINY_CORPUS,
        tokenizer: Optional[CharTokenizer] = None,
    ) -> None:
        tok = tokenizer if tokenizer is not None else CharTokenizer().fit(corpus)
        self.tokenizer = tok
        all_ids = torch.tensor(tok.encode(corpus), dtype=torch.long)
        split_pos = int(len(all_ids) * 0.8)
        self.data = all_ids[:split_pos] if split == "train" else all_ids[split_pos:]
        self.seq_len = seq_len

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return {"input_ids": chunk[:-1].clone(), "labels": chunk[:-1].clone()}


def make_loaders(
    dataset_cls,
    dataset_kwargs: dict,
    batch_size: int = 32,
    val_seed_offset: int = 1000,
) -> tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders for a synthetic dataset."""
    train_ds = dataset_cls(**dataset_kwargs)
    val_kwargs = {**dataset_kwargs, "seed": dataset_kwargs.get("seed", 42) + val_seed_offset}
    val_ds = dataset_cls(**val_kwargs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


# ── Synthetic task 3: Counting ────────────────────────────────────────────────

class CountingDataset(Dataset):
    """Counting task: at each step, predict how many tokens have been seen so far.

    Input: random sequence of tokens from vocab [4, vocab_size).
    Label at position t = t+1 (count of tokens seen up to and including position t).
    Tests whether the model can maintain an incrementing counter in its state.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        seq_len: int = 16,
        n_examples: int = 1000,
        seed: int = 42,
    ) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        tokens = torch.randint(4, vocab_size, (n_examples, seq_len), generator=rng)
        # labels[i, t] = t+1, clamped to [0, vocab_size-1]
        labels = torch.arange(1, seq_len + 1).unsqueeze(0).expand(n_examples, -1)
        labels = labels.clamp(0, vocab_size - 1)
        self.input_ids = tokens
        self.labels = labels.clone()

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


# ── Synthetic task 4: Reverse ─────────────────────────────────────────────────

class ReverseDataset(Dataset):
    """Reversal task: predict the reversed suffix given the source prefix.

    Sequence format (total length = seq_len):
        input_ids = [x0, x1, ..., x_{H-1}, x_{H-1}, x_{H-2}, ..., x_0]
    where H = seq_len // 2.

    Labels are masked to 0 for the source half; only the reversed suffix matters.
    Tests whether the model can store and replay tokens in reverse order.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        seq_len: int = 16,
        n_examples: int = 1000,
        seed: int = 42,
    ) -> None:
        assert seq_len % 2 == 0, "seq_len must be even for ReverseDataset"
        half = seq_len // 2
        rng = torch.Generator()
        rng.manual_seed(seed)
        src = torch.randint(4, vocab_size, (n_examples, half), generator=rng)
        rev = src.flip(dims=[1])
        # input: [src | rev], shape (n_examples, seq_len)
        input_ids = torch.cat([src, rev], dim=1)
        # labels: 0-padded for first half, rev tokens for second half
        labels = torch.cat([
            torch.zeros(n_examples, half, dtype=torch.long),
            rev,
        ], dim=1)
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


# ── Synthetic task 5: Induction Head ─────────────────────────────────────────

class InductionHeadDataset(Dataset):
    """Classic induction head task: [A, B, ..., A] -> predict B.

    A bigram (A, B) is placed at positions (0, 1). The sequence ends with A
    at position seq_len-1 as a query; the model must predict B at the last step.

    Labels are 0 everywhere except the last position, which is B.
    Tests in-context pattern completion (two-hop retrieval).
    """

    def __init__(
        self,
        vocab_size: int = 64,
        seq_len: int = 32,
        n_examples: int = 1000,
        seed: int = 42,
    ) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        noise = torch.randint(4, vocab_size, (n_examples, seq_len), generator=rng)
        A = torch.randint(4, vocab_size, (n_examples,), generator=rng)
        B = torch.randint(4, vocab_size, (n_examples,), generator=rng)
        input_ids = noise.clone()
        input_ids[:, 0] = A
        input_ids[:, 1] = B
        input_ids[:, -1] = A  # query
        labels = torch.zeros_like(input_ids)
        labels[:, -1] = B     # only last position matters
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


# ── Synthetic task 6: Associative Recall ──────────────────────────────────────

class AssociativeRecallDataset(Dataset):
    """K key-value pairs followed by a query key; predict the associated value.

    Sequence format:
        [k0, v0, k1, v1, ..., k_{K-1}, v_{K-1}, SEP, query_k, answer_v, <pad>...]

    Keys are drawn from [4, vocab_size//2); values from [vocab_size//2, vocab_size).
    SEP token = 3. Labels are 0 everywhere except the answer position.

    Tests whether the model can maintain a key-value store in its recurrent state.
    """

    SEP_TOKEN: int = 3

    def __init__(
        self,
        vocab_size: int = 64,
        seq_len: int = 32,
        K: int = 4,
        n_examples: int = 1000,
        seed: int = 42,
    ) -> None:
        content_len = 2 * K + 3  # kv pairs + SEP + query + answer
        assert content_len <= seq_len, (
            f"seq_len={seq_len} too small for K={K} (need {content_len})"
        )
        rng = torch.Generator()
        rng.manual_seed(seed)
        key_hi = max(vocab_size // 2, 5)
        val_lo = key_hi
        val_hi = vocab_size

        keys = torch.randint(4, key_hi, (n_examples, K), generator=rng)
        vals = torch.randint(val_lo, val_hi, (n_examples, K), generator=rng)
        q_idx = torch.randint(0, K, (n_examples,), generator=rng)

        seqs: list[list[int]] = []
        lbls: list[list[int]] = []

        for i in range(n_examples):
            kv: list[int] = []
            for j in range(K):
                kv.extend([keys[i, j].item(), vals[i, j].item()])
            answer_val = vals[i, q_idx[i]].item()
            content = kv + [self.SEP_TOKEN, keys[i, q_idx[i]].item(), answer_val]
            pad_len = seq_len - len(content)
            seq = content + [0] * pad_len
            lbl = [0] * (len(content) - 1) + [answer_val] + [0] * pad_len
            seqs.append(seq[:seq_len])
            lbls.append(lbl[:seq_len])

        self.input_ids = torch.tensor(seqs, dtype=torch.long)
        self.labels = torch.tensor(lbls, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}
