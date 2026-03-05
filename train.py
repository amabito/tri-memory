#!/usr/bin/env python3
"""TRN training entry point.

Usage:
  # Toy run (synthetic data, no real dataset needed):
  python train.py --synthetic --steps 200 --model-size toy

  # Real data (pre-tokenized .txt file, char-level tokenization):
  python train.py --data path/to/text.txt --steps 10000 --model-size trn_100m
"""
from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trn.config import TRNConfig
from trn.data import PackedDataset
from trn.eval import compute_perplexity
from trn.model import TRNModel
from trn.trainer import TrainConfig, Trainer


def _make_synthetic_dataset(
    tmp_dir: Path, n_tokens: int, seq_len: int, vocab_size: int
) -> PackedDataset:
    """Create a temporary random token binary file and return a PackedDataset."""
    path = tmp_dir / "synthetic.bin"
    rng = np.random.default_rng(seed=0)
    data = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    data.tofile(str(path))
    return PackedDataset(path, seq_len)


def _tokenize_chars(text: str, tmp_dir: Path, seq_len: int) -> PackedDataset:
    """Char-level tokenization: map characters to uint16 token ids."""
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    tokens = np.array([char_to_id[c] for c in text], dtype=np.uint16)
    path = tmp_dir / "tokens.bin"
    tokens.tofile(str(path))
    return PackedDataset(path, seq_len)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a Temporal Resonance Network")
    parser.add_argument(
        "--model-size",
        default="toy",
        choices=["toy", "trn_100m", "trn_400m", "trn_1b"],
    )
    parser.add_argument("--data", default=None, help="Path to .txt file (char tokenized)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic random data (for testing)",
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override seq_len (default: from model config)",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save checkpoint every N steps (0=disabled)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-at-end", action="store_true", default=True)
    args = parser.parse_args(argv)

    # Build model config
    cfg_factory = {
        "toy": TRNConfig.toy,
        "trn_100m": TRNConfig.trn_100m,
        "trn_400m": TRNConfig.trn_400m,
        "trn_1b": TRNConfig.trn_1b,
    }[args.model_size]
    model_cfg = cfg_factory()
    seq_len = args.seq_len or model_cfg.max_seq_len

    print(
        f"Model: {args.model_size} | d_model={model_cfg.d_model} "
        f"layers={model_cfg.n_layers} K={model_cfg.n_oscillators}"
    )

    # Build dataset (synthetic or from text file)
    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)

        if args.synthetic:
            print("Using synthetic random dataset")
            n_tokens = max(seq_len * 200, 50_000)
            dataset = _make_synthetic_dataset(tmp_dir, n_tokens, seq_len, model_cfg.vocab_size)
        elif args.data:
            text = Path(args.data).read_text(encoding="utf-8")
            dataset = _tokenize_chars(text, tmp_dir, seq_len)
            print(f"Loaded text -> {len(dataset)} sequences of length {seq_len}")
        else:
            parser.error("Provide --data or --synthetic")

        print(f"Dataset: {len(dataset)} sequences")

        model = TRNModel(model_cfg)
        n_params = model.num_parameters()
        print(f"Parameters: {n_params:,} (non-embedding)")

        train_cfg = TrainConfig(
            max_steps=args.steps,
            warmup_steps=args.warmup,
            lr=args.lr,
            batch_size=args.batch_size,
            save_interval=args.save_every,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
        )

        trainer = Trainer(model, dataset, cfg=train_cfg)
        losses = trainer.train()

    if losses:
        n10 = min(10, len(losses))
        first10 = sum(losses[:n10]) / n10
        last10 = sum(losses[-n10:]) / n10
        direction = "improved" if last10 < first10 else "did not improve"
        print(
            f"\nTraining complete. Loss: {first10:.4f} -> {last10:.4f} ({direction})"
        )

    if args.eval_at_end:
        # Rebuild dataset outside temp dir for final eval — just report from loss history
        if losses:
            import math
            final_loss = sum(losses[-min(10, len(losses)):]) / min(10, len(losses))
            print(f"Approx final perplexity (train): {math.exp(final_loss):.2f}")


if __name__ == "__main__":
    main()
