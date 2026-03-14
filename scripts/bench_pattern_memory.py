#!/usr/bin/env python3
"""Pattern memory benchmark: TRN vs Transformer vs DualMemoryEngine.

Measures pattern retention at varying distances from the probe position.
TRN should retain pattern/trend/frequency information even at distances
far beyond any KV window, while Transformer degrades.

Tasks:
  trend_shift         -- Detect trend direction change at distance D from end
  signal_continuation -- Predict next signal value after D filler tokens
  frequency_drift     -- Detect frequency regime at distance D
  amplitude_envelope  -- Detect amplitude level at distance D

Models compared:
  TRN      -- pure TRN (TemporalResonanceLayer blocks)
  TF       -- pure Transformer (CausalSelfAttention blocks)
  dual_wXX -- DualMemoryEngine (KV window W + TRN state)

Output:
  results/bench_pattern_memory.csv
  results/pattern_memory/{task}_curves.csv

Usage:
    python scripts/bench_pattern_memory.py --device cpu
    python scripts/bench_pattern_memory.py --tasks trend_shift,signal_continuation --steps 500
    python scripts/bench_pattern_memory.py --window-sizes 64,128,256
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
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trimemory.baseline import TransformerModel
from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.integrations.vllm_backend import DualMemoryEngine
from trimemory.model import TRNModel
from trimemory.scheduler import CosineWithWarmup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 128
N_OSC = 64
N_LAYERS = 4
D_FF = 512
MAX_SEQ = 2048

LR = 3e-4
LR_MIN = 3e-5
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.1

N_BINS = 64
BIN_OFFSET = 4
QUERY_TOKEN = 3

TASK_NAMES = ["trend_shift", "signal_continuation", "frequency_drift", "amplitude_envelope"]

# Trend shift classes
TREND_CLASSES = ["up_to_down", "up_to_flat", "down_to_up", "down_to_flat"]
N_TREND_CLASSES = len(TREND_CLASSES)

# Frequency classes
FREQ_CLASSES = [0.02, 0.05, 0.10, 0.20]
N_FREQ_CLASSES = len(FREQ_CLASSES)

# Amplitude levels
AMP_LEVELS = [0.2, 0.5, 0.8, 1.0]
N_AMP_LEVELS = len(AMP_LEVELS)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_cfg() -> TRNConfig:
    return TRNConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_oscillators=N_OSC,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
    )


def _make_models(
    backends: List[str],
    device: str,
    seed: int,
    window_sizes: Optional[List[int]] = None,
) -> List[Tuple[str, nn.Module]]:
    """Create models for the requested backends."""
    if window_sizes is None:
        window_sizes = [64]
    cfg = _make_cfg()
    models = []
    for backend in backends:
        seed_everything(seed)
        if backend == "trn":
            models.append(("trn", TRNModel(cfg).to(device)))
        elif backend == "tf":
            models.append(("tf", TransformerModel(cfg).to(device)))
        elif backend.startswith("dual_w"):
            w = int(backend.split("_w")[1])
            models.append((backend, DualMemoryEngine(cfg, window_size=w).to(device)))
        else:
            raise ValueError(f"Unknown backend: {backend}")
    return models


# ---------------------------------------------------------------------------
# Hidden-state capture hook (same as bench_needle_haystack.py)
# ---------------------------------------------------------------------------

def _register_hidden_hook(model: nn.Module) -> Tuple[list, any]:
    captured: list = []
    if isinstance(model, TRNModel):
        layer = model.norm_out
    elif isinstance(model, TransformerModel):
        layer = model.norm
    elif isinstance(model, DualMemoryEngine):
        layer = model.norm_out
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    def _hook(module, input, output):
        captured.clear()
        captured.append(output.detach())

    handle = layer.register_forward_hook(_hook)
    return captured, handle


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def _quantize(values: np.ndarray) -> np.ndarray:
    v_min, v_max = float(values.min()), float(values.max())
    if abs(v_max - v_min) < 1e-8:
        return np.full_like(values, BIN_OFFSET + N_BINS // 2, dtype=np.int64)
    normed = (values - v_min) / (v_max - v_min)
    binned = (normed * (N_BINS - 1)).clip(0, N_BINS - 1).astype(np.int64)
    return binned + BIN_OFFSET


# ---------------------------------------------------------------------------
# Task 1: Trend Shift
# ---------------------------------------------------------------------------

def _make_trend_shift_data(
    n_samples: int,
    distance: int,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    """Generate sequences where trend shifts at `distance` from the end.

    Sequence structure:
        [trend_before ... SHIFT_POINT ... trend_after ... QUERY_TOKEN]
        |<- pre_len ->|              |<-- distance --->|

    Labels: 4-class trend shift type.
    """
    rng = np.random.default_rng(seed)
    pre_len = max(32, distance)  # enough pre-shift context
    total_len = pre_len + distance + 1  # +1 for query token

    ids_list = []
    labels = []

    for _ in range(n_samples):
        cls = rng.integers(0, N_TREND_CLASSES)
        # Generate trend before shift
        if cls in (0, 1):  # was going up
            slope_before = rng.uniform(0.02, 0.08)
        else:  # was going down
            slope_before = rng.uniform(-0.08, -0.02)

        # Generate trend after shift
        if cls in (0, 3):  # shifts to down/flat
            slope_after = rng.uniform(-0.08, -0.01) if cls == 0 else 0.0
        else:  # shifts to up/flat
            slope_after = rng.uniform(0.01, 0.08) if cls == 2 else 0.0

        # Build signal
        t_before = np.arange(pre_len, dtype=np.float32)
        t_after = np.arange(distance, dtype=np.float32)
        noise_std = 0.05

        sig_before = slope_before * t_before + rng.normal(0, noise_std, pre_len)
        base_val = sig_before[-1]
        sig_after = base_val + slope_after * t_after + rng.normal(0, noise_std, distance)

        signal = np.concatenate([sig_before, sig_after]).astype(np.float32)
        tokens = _quantize(signal)
        seq = np.append(tokens, QUERY_TOKEN)

        ids_list.append(seq[:total_len])
        labels.append(cls)

    return (
        torch.tensor(np.stack(ids_list), dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Task 2: Signal Continuation
# ---------------------------------------------------------------------------

def _make_signal_continuation_data(
    n_samples: int,
    distance: int,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    """Generate sequences: periodic signal + filler gap + query.

    After the gap, the model should predict the next signal value.
    Labels: quantized bin of the next signal value (regression via classification).
    """
    rng = np.random.default_rng(seed)
    signal_len = 64
    total_len = signal_len + distance + 1  # +1 for query

    ids_list = []
    labels = []

    for _ in range(n_samples):
        freq = rng.uniform(0.05, 0.2)
        phase = rng.uniform(0, 2 * math.pi)
        amp = rng.uniform(0.5, 1.0)
        noise_std = 0.02

        # Signal portion
        t = np.arange(signal_len + 1, dtype=np.float32)
        signal = amp * np.sin(2 * math.pi * freq * t + phase)
        signal += rng.normal(0, noise_std, len(t))

        # The next signal value (target)
        next_val = signal[signal_len]
        # Quantize for label
        # Map [-1.5, 1.5] -> [0, N_BINS-1]
        next_bin = int(np.clip((next_val + 1.5) / 3.0 * (N_BINS - 1), 0, N_BINS - 1))
        next_bin += BIN_OFFSET

        # Quantize signal tokens
        sig_tokens = _quantize(signal[:signal_len])

        # Filler tokens (noise)
        filler = rng.integers(BIN_OFFSET, BIN_OFFSET + N_BINS, size=distance).astype(np.int64)

        # Build sequence
        seq = np.concatenate([sig_tokens, filler, [QUERY_TOKEN]])
        ids_list.append(seq[:total_len])
        labels.append(next_bin)

    return (
        torch.tensor(np.stack(ids_list), dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Task 3: Frequency Drift
# ---------------------------------------------------------------------------

def _make_frequency_drift_data(
    n_samples: int,
    distance: int,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    """Frequency regime changes at `distance` from end.

    Before shift: one frequency class. After shift: filler.
    Probe must detect which frequency was playing before the filler.
    """
    rng = np.random.default_rng(seed)
    freq_len = 64
    total_len = freq_len + distance + 1

    ids_list = []
    labels = []

    for _ in range(n_samples):
        cls = rng.integers(0, N_FREQ_CLASSES)
        freq = FREQ_CLASSES[cls]
        phase = rng.uniform(0, 2 * math.pi)
        noise_std = 0.05

        t = np.arange(freq_len, dtype=np.float32)
        signal = np.sin(2 * math.pi * freq * t + phase) + rng.normal(0, noise_std, freq_len)
        sig_tokens = _quantize(signal.astype(np.float32))

        filler = rng.integers(BIN_OFFSET, BIN_OFFSET + N_BINS, size=distance).astype(np.int64)
        seq = np.concatenate([sig_tokens, filler, [QUERY_TOKEN]])

        ids_list.append(seq[:total_len])
        labels.append(cls)

    return (
        torch.tensor(np.stack(ids_list), dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Task 4: Amplitude Envelope
# ---------------------------------------------------------------------------

def _make_amplitude_envelope_data(
    n_samples: int,
    distance: int,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    """Amplitude level of a signal, followed by filler at `distance` from end.

    Probe detects which amplitude level the signal had.
    """
    rng = np.random.default_rng(seed)
    sig_len = 64
    total_len = sig_len + distance + 1

    ids_list = []
    labels = []

    for _ in range(n_samples):
        cls = rng.integers(0, N_AMP_LEVELS)
        amp = AMP_LEVELS[cls]
        freq = 0.1
        phase = rng.uniform(0, 2 * math.pi)
        noise_std = 0.03

        t = np.arange(sig_len, dtype=np.float32)
        signal = amp * np.sin(2 * math.pi * freq * t + phase) + rng.normal(0, noise_std, sig_len)
        sig_tokens = _quantize(signal.astype(np.float32))

        filler = rng.integers(BIN_OFFSET, BIN_OFFSET + N_BINS, size=distance).astype(np.int64)
        seq = np.concatenate([sig_tokens, filler, [QUERY_TOKEN]])

        ids_list.append(seq[:total_len])
        labels.append(cls)

    return (
        torch.tensor(np.stack(ids_list), dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Data factory
# ---------------------------------------------------------------------------

_TASK_DATA_FN = {
    "trend_shift": _make_trend_shift_data,
    "signal_continuation": _make_signal_continuation_data,
    "frequency_drift": _make_frequency_drift_data,
    "amplitude_envelope": _make_amplitude_envelope_data,
}

_TASK_N_CLASSES = {
    "trend_shift": N_TREND_CLASSES,
    "signal_continuation": N_BINS + BIN_OFFSET,  # regression via classification
    "frequency_drift": N_FREQ_CLASSES,
    "amplitude_envelope": N_AMP_LEVELS,
}


# ---------------------------------------------------------------------------
# Training: backbone on next-token prediction
# ---------------------------------------------------------------------------

def _train_backbone(
    model: nn.Module,
    task: str,
    distance: int,
    steps: int,
    batch_size: int,
    device: str,
    seed: int,
) -> None:
    """Train model on next-token prediction for the task sequences."""
    model.train()
    param_groups = model.configure_optimizer_param_groups(WEIGHT_DECAY)
    opt = torch.optim.AdamW(param_groups, lr=LR, betas=BETAS)
    warmup = max(10, int(steps * WARMUP_FRAC))
    sched = CosineWithWarmup(opt, warmup_steps=warmup, max_steps=steps, lr=LR, min_lr=LR_MIN)

    data_fn = _TASK_DATA_FN[task]
    n_train = max(batch_size * 4, 500)

    for step in range(1, steps + 1):
        sched.step(step)
        # Generate fresh batch each step (different distances for generalization)
        # Use the target distance for training to learn patterns at that scale
        ids, _ = data_fn(batch_size, distance, seed + step)
        ids = ids.to(device)

        out = model(ids, labels=ids)
        loss = out["loss"]

        if not torch.isfinite(loss):
            opt.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        opt.zero_grad()


# ---------------------------------------------------------------------------
# Probe training and evaluation
# ---------------------------------------------------------------------------

def _train_and_eval_probe(
    model: nn.Module,
    task: str,
    distance: int,
    n_classes: int,
    probe_steps: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict:
    """Train linear probe on frozen backbone hidden states, then evaluate.

    Returns dict with accuracy and MSE (MSE only meaningful for signal_continuation).
    """
    model.eval()
    captured, hook = _register_hidden_hook(model)

    data_fn = _TASK_DATA_FN[task]
    is_regression = (task == "signal_continuation")

    # Probe: linear on final hidden state -> n_classes
    probe = nn.Linear(D_MODEL, n_classes).to(device)
    probe_opt = torch.optim.Adam(probe.parameters(), lr=3e-3)

    # Train probe
    probe.train()
    rng_offset = seed + 5000
    for ps in range(probe_steps):
        ids, labels = data_fn(batch_size, distance, rng_offset + ps)
        ids = ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            model(ids)
        hidden = captured[0]  # (B, T, d_model)
        probe_in = hidden[:, -1, :]  # last position (query)

        logits = probe(probe_in)
        loss = F.cross_entropy(logits, labels)
        probe_opt.zero_grad()
        loss.backward()
        probe_opt.step()

    # Evaluate probe
    probe.eval()
    correct = 0
    total_mse = 0.0
    n_eval = 512
    eval_batches = max(1, n_eval // batch_size)
    actual_total = 0

    rng_eval = seed + 9000
    with torch.no_grad():
        for eb in range(eval_batches):
            ids, labels = data_fn(batch_size, distance, rng_eval + eb)
            ids = ids.to(device)
            labels = labels.to(device)

            model(ids)
            hidden = captured[0]
            probe_in = hidden[:, -1, :]
            logits = probe(probe_in)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            actual_total += batch_size

            if is_regression:
                # MSE in bin space
                mse = ((preds.float() - labels.float()) ** 2).mean().item()
                total_mse += mse

    hook.remove()

    accuracy = correct / max(actual_total, 1)
    avg_mse = total_mse / max(eval_batches, 1) if is_regression else 0.0

    return {"accuracy": accuracy, "mse": avg_mse}


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

@dataclass
class PatternResult:
    task: str
    model: str
    backend: str
    distance: int
    accuracy: float
    mse: float
    tf_accuracy: float = 0.0
    tf_mse: float = 0.0


def run_task(
    task: str,
    backends: List[str],
    distances: List[int],
    steps: int,
    probe_steps: int,
    batch_size: int,
    device: str,
    seed: int,
    window_sizes: List[int],
) -> List[PatternResult]:
    """Run a single task across all backends and distances."""
    n_classes = _TASK_N_CLASSES[task]
    results: List[PatternResult] = []

    print(f"\n{'='*70}")
    print(f"Task: {task}  (n_classes={n_classes})")
    print(f"Distances: {distances}")
    print(f"{'='*70}")

    # Collect TF results first for comparison
    tf_metrics: dict[int, dict] = {}

    for dist in distances:
        for backend in backends:
            seed_everything(seed)
            models = _make_models([backend], device, seed, window_sizes)
            model_name, model = models[0]

            print(f"\n  [{model_name}] distance={dist} training backbone ({steps} steps)...",
                  end="", flush=True)
            t0 = time.time()
            _train_backbone(model, task, dist, steps, batch_size, device, seed)
            elapsed = time.time() - t0
            print(f" {elapsed:.1f}s", end="")

            print(f" -> probe ({probe_steps} steps)...", end="", flush=True)
            t1 = time.time()
            metrics = _train_and_eval_probe(
                model, task, dist, n_classes, probe_steps, batch_size, device, seed,
            )
            elapsed2 = time.time() - t1
            print(f" acc={metrics['accuracy']:.4f} mse={metrics['mse']:.4f} ({elapsed2:.1f}s)")

            if backend == "tf":
                tf_metrics[dist] = metrics

            results.append(PatternResult(
                task=task,
                model=model_name,
                backend=backend,
                distance=dist,
                accuracy=metrics["accuracy"],
                mse=metrics["mse"],
            ))

            del model
            torch.cuda.empty_cache() if device != "cpu" else None

    # Fill in tf_accuracy/tf_mse for comparison
    for r in results:
        tf_m = tf_metrics.get(r.distance, {"accuracy": 0.0, "mse": 0.0})
        r.tf_accuracy = tf_m["accuracy"]
        r.tf_mse = tf_m["mse"]

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "task", "model", "backend", "distance",
    "accuracy", "mse", "tf_accuracy", "tf_mse",
    "acc_gap_vs_tf", "mse_ratio_vs_tf",
]


def _write_csv(results: List[PatternResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in results:
            acc_gap = r.accuracy - r.tf_accuracy
            mse_ratio = r.mse / max(r.tf_mse, 1e-8) if r.tf_mse > 0 else float("nan")
            w.writerow([
                r.task, r.model, r.backend, r.distance,
                f"{r.accuracy:.4f}", f"{r.mse:.4f}",
                f"{r.tf_accuracy:.4f}", f"{r.tf_mse:.4f}",
                f"{acc_gap:.4f}", f"{mse_ratio:.4f}",
            ])
    print(f"\n  Saved: {path}")


def _write_summary(all_results: List[PatternResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in all_results:
            acc_gap = r.accuracy - r.tf_accuracy
            mse_ratio = r.mse / max(r.tf_mse, 1e-8) if r.tf_mse > 0 else float("nan")
            w.writerow([
                r.task, r.model, r.backend, r.distance,
                f"{r.accuracy:.4f}", f"{r.mse:.4f}",
                f"{r.tf_accuracy:.4f}", f"{r.tf_mse:.4f}",
                f"{acc_gap:.4f}", f"{mse_ratio:.4f}",
            ])


# ---------------------------------------------------------------------------
# W-sweep comparison output
# ---------------------------------------------------------------------------

def _write_w_sweep_csv(results: List[PatternResult], path: Path) -> None:
    """Write w_sweep_comparison.csv for T3-11 evaluation.

    Groups results by (task, distance) and compares window sizes.
    """
    # Only dual backends
    dual_results = [r for r in results if r.backend.startswith("dual_w")]
    if not dual_results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "distance", "backend", "window_size", "accuracy", "mse"])
        for r in sorted(dual_results, key=lambda x: (x.task, x.distance, x.backend)):
            ws = int(r.backend.split("_w")[1])
            w.writerow([r.task, r.distance, r.backend, ws,
                        f"{r.accuracy:.4f}", f"{r.mse:.4f}"])
    print(f"  W-sweep CSV saved: {path}")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def _print_summary(results: List[PatternResult]) -> None:
    print(f"\n{'='*90}")
    print(f"{'task':<20} {'model':<12} {'dist':>6} {'acc':>8} {'mse':>10} "
          f"{'tf_acc':>8} {'gap':>8}")
    print(f"{'-'*90}")

    for r in sorted(results, key=lambda x: (x.task, x.distance, x.model)):
        gap = r.accuracy - r.tf_accuracy
        print(f"{r.task:<20} {r.model:<12} {r.distance:>6} {r.accuracy:>8.4f} "
              f"{r.mse:>10.4f} {r.tf_accuracy:>8.4f} {gap:>+8.4f}")
    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pattern memory benchmark: TRN vs TF vs Dual",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tasks", type=str, default=",".join(TASK_NAMES),
        help=f"Comma-separated tasks. Options: {', '.join(TASK_NAMES)}",
    )
    parser.add_argument("--steps", type=int, default=1000,
                        help="Backbone training steps per (model, distance)")
    parser.add_argument("--probe-steps", type=int, default=200,
                        help="Linear probe training steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--backends", type=str, default="trn,tf,dual_w64",
        help="Comma-separated backends: trn, tf, dual_wXX",
    )
    parser.add_argument(
        "--window-sizes", type=str, default="64",
        help="Comma-separated window sizes for dual backends (for W-sweep)",
    )
    parser.add_argument(
        "--distances", type=str, default=None,
        help="Comma-separated distances. Default: [0, W//2, W, 2W, 4W]",
    )
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    window_sizes = [int(w) for w in args.window_sizes.split(",")]

    # Default distances based on window size
    W = window_sizes[0] if window_sizes else 64
    if args.distances:
        distances = [int(d) for d in args.distances.split(",")]
    else:
        distances = [0, W // 2, W, 2 * W, 4 * W]

    print(f"Pattern Memory Benchmark")
    print(f"  Tasks: {tasks}")
    print(f"  Backends: {backends}")
    print(f"  Distances: {distances}")
    print(f"  Window sizes: {window_sizes}")
    print(f"  Steps: {args.steps}, Probe steps: {args.probe_steps}")
    print(f"  Device: {args.device}")

    project_root = Path(__file__).resolve().parent.parent
    all_results: List[PatternResult] = []

    for task in tasks:
        task_results = run_task(
            task=task,
            backends=backends,
            distances=distances,
            steps=args.steps,
            probe_steps=args.probe_steps,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            window_sizes=window_sizes,
        )
        all_results.extend(task_results)

        # Per-task CSV
        if not args.no_csv:
            task_csv = project_root / "results" / "pattern_memory" / f"{task}_curves.csv"
            _write_csv(task_results, task_csv)

    _print_summary(all_results)

    if not args.no_csv and all_results:
        summary_csv = project_root / "results" / "bench_pattern_memory.csv"
        _write_summary(all_results, summary_csv)
        print(f"\nSummary saved: {summary_csv}")

        # W-sweep CSV
        w_sweep_csv = project_root / "results" / "w_sweep_comparison.csv"
        _write_w_sweep_csv(all_results, w_sweep_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
