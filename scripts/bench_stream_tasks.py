#!/usr/bin/env python3
"""Streaming sequential task benchmark: TRN vs Transformer vs Hybrid.

Tasks:
  timeseries   -- Next-step prediction of quantized sine+noise
  smoothing    -- Moving average prediction on random walk
  char_lm      -- Character-level LM on synthetic text
  running_mean -- Predict running mean of input tokens

Models compared:
  TRN    -- pure TRN (TemporalResonanceLayer blocks)
  TF     -- pure Transformer (CausalSelfAttention blocks)
  Hybrid -- interleaved TRN + Attention (trn_ratio=0.5 by default)

Output:
  Per-task: results/stream/{task}_curves.csv (step, model, train_loss, val_loss)
  Summary:  results/bench_stream_tasks.csv   (task, TRN, TF, Hybrid, best)

Usage:
    python scripts/bench_stream_tasks.py
    python scripts/bench_stream_tasks.py --tasks timeseries --steps 50 --no-csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from trn.baseline import TransformerModel
from trn.bench_data import seed_everything
from trn.config import TRNConfig
from trn.hybrid_model import HybridModel
from trn.model import TRNModel
from trn.scheduler import CosineWithWarmup

# ---------------------------------------------------------------------------
# Training constants (matching bench_sequence_tasks.py)
# ---------------------------------------------------------------------------

LR = 3e-4
LR_MIN = 3e-5
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 100

DEFAULT_D_MODEL = 128
DEFAULT_N_LAYERS = 4
DEFAULT_N_OSC = 64
DEFAULT_D_FF = 512
DEFAULT_SEQ_LEN = 64
DEFAULT_N_TRAIN = 2000
DEFAULT_N_VAL = 500

TASK_NAMES = ["timeseries", "smoothing", "char_lm", "running_mean"]
LOG_STEPS = [50, 100, 200, 500, 1000]

VOCAB_OFFSET = 4   # tokens 0-3 reserved; data starts at 4
N_BINS = 64        # quantization bins for numeric tasks; vocab = N_BINS + VOCAB_OFFSET = 68


# ---------------------------------------------------------------------------
# Dataset implementations
# ---------------------------------------------------------------------------

class _TensorPairDataset(Dataset):
    """Simple dataset wrapping pre-computed input_ids and labels tensors."""

    def __init__(self, ids: Tensor, labels: Tensor) -> None:
        assert ids.shape == labels.shape
        self.input_ids = ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def _quantize(values: np.ndarray, n_bins: int, offset: int) -> np.ndarray:
    """Quantize float array to integer bins with a fixed offset.

    Maps [min, max] -> [offset, offset + n_bins - 1].
    """
    v_min, v_max = float(values.min()), float(values.max())
    if abs(v_max - v_min) < 1e-8:
        return np.full_like(values, offset + n_bins // 2, dtype=np.int64)
    normed = (values - v_min) / (v_max - v_min)  # [0, 1]
    binned = (normed * (n_bins - 1)).clip(0, n_bins - 1).astype(np.int64)
    return binned + offset


def _make_timeseries_dataset(n_samples: int, seed: int) -> _TensorPairDataset:
    """Next-step prediction of quantized sine + noise.

    freq ~ U[0.5, 2.0], phase ~ U[0, 2*pi], noise_std ~ U[0, 0.2]
    Vocab size = 68 (64 bins + 4 offset).
    """
    rng = np.random.default_rng(seed)
    seq_len = DEFAULT_SEQ_LEN
    t = np.arange(seq_len + 1, dtype=np.float32)

    ids_list = []
    labels_list = []

    for _ in range(n_samples):
        freq = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0.0, 2 * math.pi)
        noise_std = rng.uniform(0.0, 0.2)
        signal = np.sin(freq * t + phase) + rng.normal(0.0, noise_std, size=len(t))
        tokens = _quantize(signal, N_BINS, VOCAB_OFFSET)
        ids_list.append(tokens[:-1])    # signal[0:T]
        labels_list.append(tokens[1:])  # signal[1:T+1]

    ids = torch.tensor(np.stack(ids_list), dtype=torch.long)
    labels = torch.tensor(np.stack(labels_list), dtype=torch.long)
    return _TensorPairDataset(ids, labels)


def _make_smoothing_dataset(n_samples: int, seed: int) -> _TensorPairDataset:
    """Predict moving average of random walk with window=4.

    Vocab size = 68 (64 bins + 4 offset), both input and label quantized.
    """
    rng = np.random.default_rng(seed)
    seq_len = DEFAULT_SEQ_LEN
    window = 4

    ids_list = []
    labels_list = []

    for _ in range(n_samples):
        # Random walk
        noise = rng.normal(0.0, 1.0, size=seq_len).astype(np.float32)
        signal = np.cumsum(noise)

        # Moving average label
        ma = np.convolve(signal, np.ones(window) / window, mode="full")[:seq_len]

        input_tokens = _quantize(signal, N_BINS, VOCAB_OFFSET)
        label_tokens = _quantize(ma, N_BINS, VOCAB_OFFSET)
        ids_list.append(input_tokens)
        labels_list.append(label_tokens)

    ids = torch.tensor(np.stack(ids_list), dtype=torch.long)
    labels = torch.tensor(np.stack(labels_list), dtype=torch.long)
    return _TensorPairDataset(ids, labels)


def _make_char_lm_dataset(n_samples: int, seed: int) -> tuple[_TensorPairDataset, int]:
    """Character-level LM on repeated synthetic corpus.

    Returns (dataset, vocab_size). Tokens = char_index + VOCAB_OFFSET.
    """
    corpus = "the quick brown fox jumps over the lazy dog. " * 200
    unique_chars = sorted(set(corpus))
    char2idx = {c: i + VOCAB_OFFSET for i, c in enumerate(unique_chars)}
    vocab_size = len(unique_chars) + VOCAB_OFFSET

    rng = np.random.default_rng(seed)
    all_ids = np.array([char2idx[c] for c in corpus], dtype=np.int64)
    seq_len = DEFAULT_SEQ_LEN

    ids_list = []
    labels_list = []

    max_start = len(all_ids) - seq_len - 1
    starts = rng.integers(0, max(1, max_start), size=n_samples)
    for s in starts:
        chunk = all_ids[s: s + seq_len + 1]
        ids_list.append(chunk[:-1])
        labels_list.append(chunk[1:])

    ids = torch.tensor(np.stack(ids_list), dtype=torch.long)
    labels = torch.tensor(np.stack(labels_list), dtype=torch.long)
    return _TensorPairDataset(ids, labels), vocab_size


def _make_running_mean_dataset(n_samples: int, seed: int) -> _TensorPairDataset:
    """Predict running mean of input token sequence.

    Input: random tokens in [VOCAB_OFFSET, VOCAB_OFFSET + N_BINS).
    Label at position t = floor(cumsum(tokens[0:t+1]) / (t+1)), clamped to valid range.
    Vocab size = 68.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    seq_len = DEFAULT_SEQ_LEN
    lo = VOCAB_OFFSET
    hi = VOCAB_OFFSET + N_BINS  # exclusive: 68

    input_ids = torch.randint(lo, hi, (n_samples, seq_len), generator=rng)
    positions = torch.arange(1, seq_len + 1, dtype=torch.float32).unsqueeze(0)  # (1, seq_len)
    cumsum = input_ids.float().cumsum(dim=1)
    running_mean = (cumsum / positions).floor().long()
    running_mean = running_mean.clamp(lo, hi - 1)

    return _TensorPairDataset(input_ids, running_mean)


def _make_datasets(
    task: str, seed: int
) -> tuple[_TensorPairDataset, _TensorPairDataset, int]:
    """Return (train_ds, val_ds, vocab_size) for the given task."""
    if task == "timeseries":
        train_ds = _make_timeseries_dataset(DEFAULT_N_TRAIN, seed)
        val_ds = _make_timeseries_dataset(DEFAULT_N_VAL, seed + 1000)
        return train_ds, val_ds, N_BINS + VOCAB_OFFSET

    if task == "smoothing":
        train_ds = _make_smoothing_dataset(DEFAULT_N_TRAIN, seed)
        val_ds = _make_smoothing_dataset(DEFAULT_N_VAL, seed + 1000)
        return train_ds, val_ds, N_BINS + VOCAB_OFFSET

    if task == "char_lm":
        train_ds, vocab_size = _make_char_lm_dataset(DEFAULT_N_TRAIN, seed)
        val_ds, _ = _make_char_lm_dataset(DEFAULT_N_VAL, seed + 1000)
        return train_ds, val_ds, vocab_size

    if task == "running_mean":
        train_ds = _make_running_mean_dataset(DEFAULT_N_TRAIN, seed)
        val_ds = _make_running_mean_dataset(DEFAULT_N_VAL, seed + 1000)
        return train_ds, val_ds, N_BINS + VOCAB_OFFSET

    raise ValueError(f"Unknown task: {task!r}")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_models(
    cfg: TRNConfig,
    trn_ratio: float,
    seed: int,
) -> list[tuple[str, torch.nn.Module]]:
    seed_everything(seed)
    trn = TRNModel(cfg)

    seed_everything(seed)
    tf = TransformerModel(cfg)

    seed_everything(seed)
    hybrid = HybridModel(cfg, trn_ratio=trn_ratio)

    return [("TRN", trn), ("TF", tf), (f"Hybrid({trn_ratio:.1f})", hybrid)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task: str
    model_name: str
    loss_at_steps: dict[int, float] = field(default_factory=dict)
    final_val_loss: float = float("inf")
    wall_time_s: float = 0.0


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int = 50,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        total += out["loss"].item()
        n += 1
    model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Training loop (matches bench_sequence_tasks.py pattern)
# ---------------------------------------------------------------------------

def _train_model(
    model: torch.nn.Module,
    model_name: str,
    task: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_steps: int,
    device: str,
    seed: int,
    log_steps: list[int],
) -> TaskResult:
    seed_everything(seed)
    model.to(device).train()

    param_groups = model.configure_optimizer_param_groups(WEIGHT_DECAY)
    opt = torch.optim.AdamW(param_groups, lr=LR, betas=BETAS, eps=EPS)
    sched = CosineWithWarmup(
        opt,
        warmup_steps=min(WARMUP_STEPS, n_steps // 4),
        max_steps=n_steps,
        lr=LR,
        min_lr=LR_MIN,
    )

    loader_iter = iter(train_loader)
    log_set = set(log_steps)
    loss_at_steps: dict[int, float] = {}
    train_loss_at_steps: dict[int, float] = {}
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        sched.step(step)
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        loss = out["loss"]

        if not torch.isfinite(loss):
            print(f"  [{model_name}] WARN: non-finite loss at step {step}, skipping")
            opt.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad()

        if step in log_set:
            train_loss = loss.item()
            val_loss = _evaluate(model, val_loader, device)
            elapsed = time.perf_counter() - t0
            loss_at_steps[step] = val_loss
            train_loss_at_steps[step] = train_loss
            print(
                f"  [{model_name:>18}] step {step:5d}/{n_steps}"
                f"  train={train_loss:.4f}  val={val_loss:.4f}"
                f"  ({elapsed:.1f}s)"
            )

    wall_time = time.perf_counter() - t0
    final_val = loss_at_steps.get(n_steps, float("inf"))

    result = TaskResult(
        task=task,
        model_name=model_name,
        loss_at_steps=loss_at_steps,
        final_val_loss=final_val,
        wall_time_s=wall_time,
    )
    # Attach train losses for CSV output
    result.train_loss_at_steps = train_loss_at_steps  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def run_task(
    task: str,
    n_steps: int,
    batch_size: int,
    trn_ratio: float,
    seed: int,
    device: str,
    out_dir: Optional[Path],
    log_steps: list[int],
) -> list[TaskResult]:
    print(f"\n{'='*70}")
    print(f"Task: {task}")
    print(f"{'='*70}")

    train_ds, val_ds, vocab_size = _make_datasets(task, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    cfg = TRNConfig(
        vocab_size=vocab_size,
        d_model=DEFAULT_D_MODEL,
        n_layers=DEFAULT_N_LAYERS,
        n_oscillators=DEFAULT_N_OSC,
        d_ff=DEFAULT_D_FF,
        max_seq_len=DEFAULT_SEQ_LEN + 8,
    )

    print(f"  vocab={vocab_size}  seq_len={DEFAULT_SEQ_LEN}  "
          f"d_model={DEFAULT_D_MODEL}  n_layers={DEFAULT_N_LAYERS}  "
          f"n_osc={DEFAULT_N_OSC}  trn_ratio={trn_ratio}")

    models = _make_models(cfg, trn_ratio, seed)
    results: list[TaskResult] = []

    for model_name, model in models:
        print(f"\n  -- Training {model_name} --")
        result = _train_model(
            model=model,
            model_name=model_name,
            task=task,
            train_loader=train_loader,
            val_loader=val_loader,
            n_steps=n_steps,
            device=device,
            seed=seed,
            log_steps=log_steps,
        )
        results.append(result)

    # Save per-task curves CSV
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{task}_curves.csv"
        all_steps = sorted({s for r in results for s in r.loss_at_steps})
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            # Header: step, model1_train, model1_val, model2_train, ...
            header = ["step"]
            for r in results:
                header.extend([f"{r.model_name}_train", f"{r.model_name}_val"])
            w.writerow(header)
            for s in all_steps:
                row: list = [s]
                for r in results:
                    train_loss = getattr(r, "train_loss_at_steps", {}).get(s, "")
                    val_loss = r.loss_at_steps.get(s, "")
                    row.append(f"{train_loss:.4f}" if isinstance(train_loss, float) else train_loss)
                    row.append(f"{val_loss:.4f}" if isinstance(val_loss, float) else val_loss)
                w.writerow(row)
        print(f"\n  Saved: {csv_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Streaming sequential task benchmark: TRN vs TF vs Hybrid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASK_NAMES),
        help=f"Comma-separated tasks. Options: {', '.join(TASK_NAMES)}",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--trn-ratio", type=float, default=0.5,
        help="Fraction of Hybrid layers that are TRN (0-1)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    default_log = LOG_STEPS
    log_steps = sorted({s for s in default_log if s <= args.steps} | {args.steps})

    project_root = Path(__file__).resolve().parent.parent
    stream_dir = None if args.no_csv else project_root / "results" / "stream"
    summary_csv = None if args.no_csv else project_root / "results" / "bench_stream_tasks.csv"

    all_results: list[list[TaskResult]] = []

    for task in tasks:
        task_results = run_task(
            task=task,
            n_steps=args.steps,
            batch_size=args.batch_size,
            trn_ratio=args.trn_ratio,
            seed=args.seed,
            device=args.device,
            out_dir=stream_dir,
            log_steps=log_steps,
        )
        all_results.append(task_results)

    # Summary table
    col_w = 20
    if all_results:
        model_names = [r.model_name for r in all_results[0]]
    else:
        model_names = []

    header_width = 14 + col_w * len(model_names) + 10
    print(f"\n{'='*header_width}")
    print(f"Summary  |  steps={args.steps}  d_model={DEFAULT_D_MODEL}  n_layers={DEFAULT_N_LAYERS}")
    print(f"{'='*header_width}")
    header_str = f"{'task':<14}" + "".join(f"{n:>{col_w}}" for n in model_names) + f"{'best':>8}"
    print(header_str)
    print("-" * header_width)

    summary_rows: list[list] = []
    for task_results in all_results:
        task_name = task_results[0].task if task_results else "?"
        losses = {r.model_name: r.final_val_loss for r in task_results}
        best = min(losses, key=lambda k: losses[k]) if losses else "-"
        row_str = f"{task_name:<14}"
        csv_row: list = [task_name]
        for name in model_names:
            v = losses.get(name, float("inf"))
            row_str += f"{v:>{col_w}.4f}"
            csv_row.append(f"{v:.4f}" if v != float("inf") else "")
        row_str += f"{best:>8}"
        csv_row.append(best)
        print(row_str)
        summary_rows.append(csv_row)

    print(f"{'='*header_width}\n")

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task"] + model_names + ["best"])
            w.writerows(summary_rows)
        print(f"Summary saved: {summary_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
