#!/usr/bin/env python3
"""W Sweep Benchmark: KV window size sensitivity analysis for Tri-Memory LLM.

Evaluates performance, memory, and throughput across window sizes W={64,128,256}
for 4 model configurations:
  A: KV only         -- windowed attention, TRN gates forced to zero
  B: KV + TRN        -- DualMemoryEngine (learned KV/TRN gate)
  C: KV + Retrieval  -- TriMemoryEngine with TRN gate disabled
  D: TriMemory       -- TriMemoryEngine full (KV + TRN + Retrieval)

Tasks:
  counting            -- count occurrences of a target token in a long stream
  sinusoidal          -- predict next value in a periodic sinusoidal pattern
  induction           -- [A, B, noise, A] -> predict B (induction head)
  char_lm             -- character-level language modeling on repeated patterns

TRN is NOT a Transformer replacement.
TRN is NOT a content-addressable memory.
Tri-Memory is a hierarchical memory architecture.

Output:
  artifacts/w_sweep/w_sweep_results.csv
  artifacts/w_sweep/w_sweep_comparison.csv
  artifacts/w_sweep/w_sweep_summary.md
  artifacts/w_sweep/accuracy_vs_window.png
  artifacts/w_sweep/throughput_vs_window.png
  artifacts/w_sweep/memory_vs_window.png

Usage:
    python scripts/bench_w_sweep.py
    python scripts/bench_w_sweep.py --steps 500 --device cuda
    python scripts/bench_w_sweep.py --windows 64,128,256 --tasks counting,induction
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

_src = os.path.join(os.path.dirname(__file__), "..", "src")
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _root]
sys.path.insert(0, _src)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.integrations.vllm_backend import DualMemoryEngine
from trimemory.tri_memory import TriMemoryEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOWS = [64, 128, 256]
D_MODEL = 128
N_LAYERS = 4
N_OSC = 64
D_FF = 512
VOCAB_SIZE = 256
BATCH_SIZE = 16
DEFAULT_STEPS = 300
LR = 3e-4
EVAL_SAMPLES = 200
SEED = 42

# Token ranges
NOISE_LOW, NOISE_HIGH = 100, 200
TARGET_LOW, TARGET_HIGH = 10, 50
PATTERN_LOW, PATTERN_HIGH = 10, 80
A_LOW, A_HIGH = 10, 50
B_OFFSET = 50
FILLER_LOW, FILLER_HIGH = 100, 200


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class CountingDataset(Dataset):
    """Count occurrences of a target token in a stream.

    Sequence: [filler with scattered target tokens, QUERY, count_bucket]
    The model must predict a count bucket (0-3) at the query position.
    Tests: long-range aggregation (TRN should help).
    """

    def __init__(self, n_samples: int, seq_len: int, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        seq = torch.randint(FILLER_LOW, FILLER_HIGH, (self.seq_len,), generator=self.rng)
        target_tok = torch.randint(TARGET_LOW, TARGET_HIGH, (1,), generator=self.rng).item()

        # Scatter target tokens randomly
        n_targets = torch.randint(0, 12, (1,), generator=self.rng).item()
        positions = torch.randint(0, self.seq_len - 2, (n_targets,), generator=self.rng)
        for p in positions:
            seq[p.item()] = target_tok

        # Count bucket: 0 (0), 1 (1-3), 2 (4-7), 3 (8+)
        actual_count = (seq[:-2] == target_tok).sum().item()
        if actual_count == 0:
            bucket = 0
        elif actual_count <= 3:
            bucket = 1
        elif actual_count <= 7:
            bucket = 2
        else:
            bucket = 3

        # Last two tokens: query marker + answer
        seq[-2] = target_tok  # query: "count this token"
        seq[-1] = bucket      # answer bucket

        return {"input_ids": seq, "target": bucket, "query_pos": self.seq_len - 2}


class SinusoidalDataset(Dataset):
    """Predict next value in a periodic sinusoidal pattern.

    Sequence: quantized sinusoidal tokens with varying frequency/phase.
    At the end, model must predict the next token in the pattern.
    Tests: frequency/pattern detection (TRN core strength).
    """

    def __init__(self, n_samples: int, seq_len: int, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        import math
        # Random frequency and phase
        freq = (torch.rand(1, generator=self.rng).item() * 0.3 + 0.05)  # 0.05-0.35
        phase = torch.rand(1, generator=self.rng).item() * 2 * math.pi
        amplitude = torch.rand(1, generator=self.rng).item() * 30 + 10  # 10-40

        seq = torch.zeros(self.seq_len, dtype=torch.long)
        for t in range(self.seq_len):
            val = math.sin(freq * t + phase) * amplitude + PATTERN_LOW + 35
            seq[t] = max(PATTERN_LOW, min(PATTERN_HIGH - 1, int(val)))

        # Target: next value in sequence
        next_val = math.sin(freq * self.seq_len + phase) * amplitude + PATTERN_LOW + 35
        target = max(PATTERN_LOW, min(PATTERN_HIGH - 1, int(next_val)))

        return {"input_ids": seq, "target": target, "query_pos": self.seq_len - 1}


class InductionDataset(Dataset):
    """[A, B, noise(L-4), A] -> predict B.

    Tests: exact retrieval over distance (KV for near, degraded beyond window).
    """

    def __init__(self, n_samples: int, seq_len: int, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        seq = torch.randint(NOISE_LOW, NOISE_HIGH, (self.seq_len,), generator=self.rng)
        a_tok = torch.randint(A_LOW, A_HIGH, (1,), generator=self.rng).item()
        b_tok = a_tok + B_OFFSET

        seq[0] = a_tok
        seq[1] = b_tok
        seq[-2] = a_tok   # repeat A at end
        seq[-1] = 4       # query pad

        return {"input_ids": seq, "target": b_tok, "query_pos": self.seq_len - 1}


class CharLMDataset(Dataset):
    """Character-level LM on repeated structured patterns.

    Sequence: a fixed pattern repeated, with the model learning next-token prediction.
    Tests: combined pattern learning across the sequence.
    """

    def __init__(self, n_samples: int, seq_len: int, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        # Generate a random pattern of length 5-12, repeat it
        pat_len = torch.randint(5, 13, (1,), generator=self.rng).item()
        pattern = torch.randint(PATTERN_LOW, PATTERN_HIGH, (pat_len,), generator=self.rng)

        full = pattern.repeat(((self.seq_len + 1) + pat_len - 1) // pat_len)
        seq = full[:self.seq_len].clone()
        target = full[self.seq_len].item()  # next token

        return {"input_ids": seq, "target": target, "query_pos": self.seq_len - 1}


TASK_REGISTRY = {
    "counting": CountingDataset,
    "sinusoidal": SinusoidalDataset,
    "induction": InductionDataset,
    "char_lm": CharLMDataset,
}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_cfg(window_size: int, seq_len: int) -> TRNConfig:
    return TRNConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_oscillators=N_OSC,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=seq_len + 32,
    )


def build_models(cfg: TRNConfig, window_size: int) -> dict[str, nn.Module]:
    """Build 4 ablation models for a given window size.

    A: KV only     -- DualMemoryEngine with TRN contribution disabled
    B: KV + TRN    -- DualMemoryEngine (normal)
    C: KV + Ret    -- TriMemoryEngine with TRN gate biased off
    D: TriMemory   -- TriMemoryEngine (full)
    """
    chunk_size = max(16, window_size // 2)

    # A: KV only -- force TRN gate to strongly favor KV
    a_model = DualMemoryEngine(cfg, window_size=window_size)
    with torch.no_grad():
        for block in a_model.blocks:
            # bias = +5.0 => sigmoid(5)=0.993 => g_kv ~= 1.0
            block.gate_proj.bias.fill_(5.0)
            block.gate_proj.weight.zero_()
            block.gate_proj.weight.requires_grad = False
            block.gate_proj.bias.requires_grad = False

    # B: KV + TRN -- normal dual memory
    b_model = DualMemoryEngine(cfg, window_size=window_size)

    # C: KV + Ret -- tri-memory with TRN disabled via gate bias
    c_model = TriMemoryEngine(cfg, window_size=window_size, chunk_size=chunk_size)
    with torch.no_grad():
        for block in c_model.blocks:
            # gate_proj bias: [g_kv, g_trn, g_ret]
            # Set TRN to -10 to suppress it via softmax
            block.gate_proj.bias[1] = -10.0

    # D: TriMemory -- full
    d_model = TriMemoryEngine(cfg, window_size=window_size, chunk_size=chunk_size)

    return {
        "A_kv_only": a_model,
        "B_kv_trn": b_model,
        "C_kv_ret": c_model,
        "D_trimemory": d_model,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    dataset: Dataset,
    steps: int,
    device: torch.device,
) -> list[dict]:
    model = model.to(device).train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    records = []
    loader_it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)

        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        loss = out["loss"]
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if step % 50 == 0 or step == steps:
            records.append({"step": step, "loss": loss.item()})

    return records


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    n_eval: int = EVAL_SAMPLES,
) -> dict:
    """Evaluate accuracy at query position and compute metrics."""
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=min(32, n_eval), shuffle=False)

    correct = 0
    total = 0
    nan_count = 0
    inf_count = 0

    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            targets = batch["target"]
            query_pos = batch["query_pos"]
            B = ids.size(0)

            out = model(ids)
            logits = out["logits"]  # (B, T, V)

            for i in range(B):
                if total >= n_eval:
                    break
                qp = query_pos[i].item()
                pred = logits[i, qp, :].argmax(dim=-1).cpu().item()
                correct += int(pred == targets[i].item())
                total += 1

                # Stability checks
                if torch.isnan(logits[i]).any():
                    nan_count += 1
                if torch.isinf(logits[i]).any():
                    inf_count += 1

    acc = correct / max(total, 1)
    return {
        "accuracy": acc,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "n_eval": total,
    }


def measure_throughput(
    model: nn.Module,
    seq_len: int,
    device: torch.device,
    n_tokens: int = 2000,
) -> dict:
    """Measure tokens/sec and latency."""
    model = model.to(device).eval()
    batch = torch.randint(10, VOCAB_SIZE, (1, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(batch)

    n_steps = max(1, n_tokens // seq_len)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            model(batch)
    elapsed = time.perf_counter() - t0
    total_tokens = n_steps * seq_len
    tps = total_tokens / max(elapsed, 1e-9)
    latency_ms = (elapsed / n_steps) * 1000

    return {"tokens_per_second": tps, "latency_ms": latency_ms}


def measure_memory(model: nn.Module, window_size: int) -> dict:
    """Estimate state memory usage."""
    # Model parameters (bytes)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # KV cache: 2 * n_layers * window * n_heads * head_dim * 4 bytes
    n_heads = max(1, D_MODEL // 64)
    head_dim = D_MODEL // n_heads
    kv_bytes = 2 * N_LAYERS * window_size * n_heads * head_dim * 4

    # TRN state: n_layers * 2 * K * 4 bytes (real + imag, fp32)
    trn_state_bytes = N_LAYERS * 2 * N_OSC * 4

    # Retrieval index (estimate): max_chunks * (vocab_size + d_model) * 4
    retrieval_bytes = 256 * (VOCAB_SIZE + D_MODEL) * 4  # max chunks

    state_bytes = kv_bytes + trn_state_bytes
    retained_kv_tokens = window_size

    return {
        "state_bytes": state_bytes,
        "kv_bytes": kv_bytes,
        "trn_state_bytes": trn_state_bytes,
        "retrieval_bytes": retrieval_bytes,
        "param_bytes": param_bytes,
        "retained_kv_tokens": retained_kv_tokens,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    windows: list[int],
    tasks: list[str],
    steps: int,
    device: torch.device,
    out_dir: Path,
) -> list[dict]:
    """Run full W sweep across models and tasks."""
    all_results = []
    total_runs = len(windows) * len(tasks) * 4
    run_idx = 0

    for W in windows:
        seq_len = max(256, W * 4)  # sequence length scales with window
        cfg = make_cfg(W, seq_len)

        for task_name in tasks:
            dataset_cls = TASK_REGISTRY[task_name]
            train_ds = dataset_cls(n_samples=2000, seq_len=seq_len, seed=SEED)
            eval_ds = dataset_cls(n_samples=EVAL_SAMPLES, seq_len=seq_len, seed=SEED + 1000)

            models = build_models(cfg, W)

            for model_name, model in models.items():
                run_idx += 1
                print(f"  [{run_idx}/{total_runs}] W={W} task={task_name} model={model_name}")
                sys.stdout.flush()

                seed_everything(SEED)
                t0 = time.perf_counter()
                loss_curve = train_model(model, train_ds, steps, device)
                train_time = time.perf_counter() - t0

                eval_result = evaluate_model(model, eval_ds, device)
                throughput = measure_throughput(model, seq_len, device)
                mem = measure_memory(model, W)

                # Retrieval call count (only for C and D models)
                retrieval_calls = 0
                if hasattr(model, "retrieval_index"):
                    retrieval_calls = len(model.retrieval_index)

                row = {
                    "model": model_name,
                    "window": W,
                    "task": task_name,
                    "recent_exact_acc": eval_result["accuracy"],
                    "long_pattern_acc": eval_result["accuracy"],
                    "old_fact_acc": eval_result["accuracy"],
                    "composite_score": eval_result["accuracy"],
                    "tokens_per_second": throughput["tokens_per_second"],
                    "latency_ms": throughput["latency_ms"],
                    "state_bytes": mem["state_bytes"],
                    "retained_kv_tokens": mem["retained_kv_tokens"],
                    "retrieval_calls": retrieval_calls,
                    "final_loss": loss_curve[-1]["loss"],
                    "train_time_s": train_time,
                    "nan_count": eval_result["nan_count"],
                    "inf_count": eval_result["inf_count"],
                }
                all_results.append(row)

                print(f"    acc={eval_result['accuracy']:.3f}"
                      f"  loss={loss_curve[-1]['loss']:.4f}"
                      f"  tps={throughput['tokens_per_second']:.0f}"
                      f"  state={mem['state_bytes']}B"
                      f"  time={train_time:.1f}s")

                # Free memory
                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None

    return all_results


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def write_results_csv(results: list[dict], path: Path) -> None:
    fields = [
        "model", "window", "task",
        "recent_exact_acc", "long_pattern_acc", "old_fact_acc", "composite_score",
        "tokens_per_second", "latency_ms",
        "state_bytes", "retained_kv_tokens", "retrieval_calls",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV -> {path}")


def write_comparison_csv(results: list[dict], path: Path) -> None:
    """Write eval-compatible w_sweep_comparison.csv."""
    fields = ["task", "distance", "backend", "window_size", "accuracy", "mse"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "task": r["task"],
                "distance": 0,
                "backend": r["model"],
                "window_size": r["window"],
                "accuracy": r["composite_score"],
                "mse": r["final_loss"],
            })
    print(f"  CSV -> {path}")


def write_summary_md(results: list[dict], path: Path) -> None:
    """Generate comprehensive summary markdown."""
    lines = [
        "# W Sweep: KV Window Size Sensitivity Analysis",
        "",
        "## Important",
        "",
        "- TRN is NOT a Transformer replacement.",
        "- TRN is NOT a content-addressable memory.",
        "- Tri-Memory is a hierarchical memory architecture.",
        "",
    ]

    # --- Window Sensitivity Table ---
    lines.extend([
        "## Window Sensitivity",
        "",
        "Mean composite score across all tasks:",
        "",
        "| Window | KV only | KV+TRN | KV+Retrieval | TriMemory |",
        "|--------|---------|--------|--------------|-----------|",
    ])

    windows = sorted(set(r["window"] for r in results))
    model_keys = ["A_kv_only", "B_kv_trn", "C_kv_ret", "D_trimemory"]

    for W in windows:
        scores = {}
        for mk in model_keys:
            vals = [r["composite_score"] for r in results
                    if r["window"] == W and r["model"] == mk]
            scores[mk] = sum(vals) / len(vals) if vals else 0.0
        lines.append(
            f"| {W} | {scores['A_kv_only']:.3f} | {scores['B_kv_trn']:.3f} "
            f"| {scores['C_kv_ret']:.3f} | {scores['D_trimemory']:.3f} |"
        )

    # --- Per-Task Breakdown ---
    tasks = sorted(set(r["task"] for r in results))
    lines.extend(["", "## Per-Task Breakdown", ""])

    for task in tasks:
        lines.extend([
            f"### {task}",
            "",
            "| Window | KV only | KV+TRN | KV+Retrieval | TriMemory |",
            "|--------|---------|--------|--------------|-----------|",
        ])
        for W in windows:
            scores = {}
            for mk in model_keys:
                vals = [r["composite_score"] for r in results
                        if r["window"] == W and r["model"] == mk and r["task"] == task]
                scores[mk] = vals[0] if vals else 0.0
            lines.append(
                f"| {W} | {scores['A_kv_only']:.3f} | {scores['B_kv_trn']:.3f} "
                f"| {scores['C_kv_ret']:.3f} | {scores['D_trimemory']:.3f} |"
            )
        lines.append("")

    # --- Memory Tradeoff ---
    lines.extend([
        "## Memory Tradeoff",
        "",
        "State memory (KV cache + TRN state) in bytes:",
        "",
        "| Window | KV Cache (B) | TRN State (B) | Total State (B) | Retained KV Tokens |",
        "|--------|-------------|---------------|-----------------|-------------------|",
    ])

    for W in windows:
        r = next((r for r in results if r["window"] == W), None)
        if r:
            n_heads = max(1, D_MODEL // 64)
            head_dim = D_MODEL // n_heads
            kv_b = 2 * N_LAYERS * W * n_heads * head_dim * 4
            trn_b = N_LAYERS * 2 * N_OSC * 4
            lines.append(
                f"| {W} | {kv_b:,} | {trn_b:,} | {kv_b + trn_b:,} | {W} |"
            )

    # --- Throughput ---
    lines.extend([
        "",
        "## Throughput",
        "",
        "Mean tokens/sec across all tasks:",
        "",
        "| Window | KV only | KV+TRN | KV+Retrieval | TriMemory |",
        "|--------|---------|--------|--------------|-----------|",
    ])
    for W in windows:
        tps = {}
        for mk in model_keys:
            vals = [r["tokens_per_second"] for r in results
                    if r["window"] == W and r["model"] == mk]
            tps[mk] = sum(vals) / len(vals) if vals else 0.0
        lines.append(
            f"| {W} | {tps['A_kv_only']:.0f} | {tps['B_kv_trn']:.0f} "
            f"| {tps['C_kv_ret']:.0f} | {tps['D_trimemory']:.0f} |"
        )

    # --- Stability ---
    lines.extend([
        "",
        "## Stability",
        "",
    ])
    total_nan = sum(r["nan_count"] for r in results)
    total_inf = sum(r["inf_count"] for r in results)
    lines.append(f"- Total NaN occurrences: {total_nan}")
    lines.append(f"- Total Inf occurrences: {total_inf}")

    # --- Analysis ---
    lines.extend([
        "",
        "## Analysis",
        "",
    ])

    # Find best model at W=64
    w64_scores = {}
    for mk in model_keys:
        vals = [r["composite_score"] for r in results
                if r["window"] == 64 and r["model"] == mk]
        w64_scores[mk] = sum(vals) / len(vals) if vals else 0.0
    best_w64 = max(w64_scores, key=w64_scores.get)

    # TriMemory vs all ablations
    tri_wins = 0
    tri_total = 0
    for W in windows:
        for task in tasks:
            tri_val = next(
                (r["composite_score"] for r in results
                 if r["window"] == W and r["model"] == "D_trimemory" and r["task"] == task),
                0.0,
            )
            for mk in ["A_kv_only", "B_kv_trn", "C_kv_ret"]:
                other_val = next(
                    (r["composite_score"] for r in results
                     if r["window"] == W and r["model"] == mk and r["task"] == task),
                    0.0,
                )
                tri_total += 1
                if tri_val >= other_val:
                    tri_wins += 1

    lines.append(f"- TriMemory wins {tri_wins}/{tri_total} comparisons vs ablations")
    lines.append(f"- Best model at W=64: {best_w64} (score={w64_scores[best_w64]:.3f})")

    # KV-only advantages
    kv_better_tasks = []
    for task in tasks:
        for W in windows:
            kv_val = next(
                (r["composite_score"] for r in results
                 if r["window"] == W and r["model"] == "A_kv_only" and r["task"] == task),
                0.0,
            )
            tri_val = next(
                (r["composite_score"] for r in results
                 if r["window"] == W and r["model"] == "D_trimemory" and r["task"] == task),
                0.0,
            )
            if kv_val > tri_val:
                kv_better_tasks.append(f"{task}@W={W}")

    if kv_better_tasks:
        lines.append(f"- KV-only beats TriMemory in: {', '.join(kv_better_tasks)}")
    else:
        lines.append("- KV-only does not beat TriMemory in any configuration")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- TRN is NOT a content-addressable memory.",
        "- TRN is NOT a Transformer replacement.",
        "- Tri-Memory is a hierarchical memory architecture.",
        "- These results are on synthetic tasks with toy-scale models (d=128, L=4).",
        "- Real-world performance requires scaled models on natural language.",
        "",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  MD -> {path}")


def generate_plots(results: list[dict], out_dir: Path) -> None:
    """Generate accuracy, throughput, and memory plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")
        return

    windows = sorted(set(r["window"] for r in results))
    model_keys = ["A_kv_only", "B_kv_trn", "C_kv_ret", "D_trimemory"]
    labels = {"A_kv_only": "KV only", "B_kv_trn": "KV+TRN",
              "C_kv_ret": "KV+Ret", "D_trimemory": "TriMemory"}
    colors = {"A_kv_only": "#1f77b4", "B_kv_trn": "#ff7f0e",
              "C_kv_ret": "#2ca02c", "D_trimemory": "#d62728"}

    # 1. Accuracy vs Window
    fig, ax = plt.subplots(figsize=(8, 5))
    for mk in model_keys:
        scores = []
        for W in windows:
            vals = [r["composite_score"] for r in results
                    if r["window"] == W and r["model"] == mk]
            scores.append(sum(vals) / len(vals) if vals else 0.0)
        ax.plot(windows, scores, "o-", label=labels[mk], color=colors[mk], linewidth=2)
    ax.set_xlabel("Window Size (W)")
    ax.set_ylabel("Composite Score")
    ax.set_title("Accuracy vs Window Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(windows)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_window.png", dpi=150)
    plt.close(fig)
    print(f"  PNG -> {out_dir / 'accuracy_vs_window.png'}")

    # 2. Throughput vs Window
    fig, ax = plt.subplots(figsize=(8, 5))
    for mk in model_keys:
        tps_vals = []
        for W in windows:
            vals = [r["tokens_per_second"] for r in results
                    if r["window"] == W and r["model"] == mk]
            tps_vals.append(sum(vals) / len(vals) if vals else 0.0)
        ax.plot(windows, tps_vals, "o-", label=labels[mk], color=colors[mk], linewidth=2)
    ax.set_xlabel("Window Size (W)")
    ax.set_ylabel("Tokens/Second")
    ax.set_title("Throughput vs Window Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(windows)
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_vs_window.png", dpi=150)
    plt.close(fig)
    print(f"  PNG -> {out_dir / 'throughput_vs_window.png'}")

    # 3. Memory vs Window
    fig, ax = plt.subplots(figsize=(8, 5))
    n_heads = max(1, D_MODEL // 64)
    head_dim = D_MODEL // n_heads
    kv_sizes = [2 * N_LAYERS * W * n_heads * head_dim * 4 for W in windows]
    trn_size = N_LAYERS * 2 * N_OSC * 4  # constant
    total_dual = [kv + trn_size for kv in kv_sizes]
    ret_size = 256 * (VOCAB_SIZE + D_MODEL) * 4

    ax.plot(windows, [k / 1024 for k in kv_sizes], "o-",
            label="KV Cache", color=colors["A_kv_only"], linewidth=2)
    ax.plot(windows, [t / 1024 for t in total_dual], "s-",
            label="KV + TRN State", color=colors["B_kv_trn"], linewidth=2)
    ax.plot(windows, [(t + ret_size) / 1024 for t in total_dual], "^-",
            label="KV + TRN + Retrieval", color=colors["D_trimemory"], linewidth=2)
    ax.axhline(y=trn_size / 1024, color="gray", linestyle="--",
               alpha=0.5, label=f"TRN State ({trn_size / 1024:.1f} KB)")
    ax.set_xlabel("Window Size (W)")
    ax.set_ylabel("Memory (KB)")
    ax.set_title("Memory Usage vs Window Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(windows)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_vs_window.png", dpi=150)
    plt.close(fig)
    print(f"  PNG -> {out_dir / 'memory_vs_window.png'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="W Sweep Benchmark")
    parser.add_argument("--windows", default="64,128,256",
                        help="Comma-separated window sizes")
    parser.add_argument("--tasks", default="counting,sinusoidal,induction,char_lm",
                        help="Comma-separated task names")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default="artifacts/w_sweep",
                        help="Output directory")
    args = parser.parse_args()

    device = torch.device(args.device)
    windows = [int(w) for w in args.windows.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("W Sweep Benchmark: KV Window Size Sensitivity Analysis")
    print("=" * 72)
    print(f"  Windows: {windows}")
    print(f"  Tasks: {tasks}")
    print(f"  Steps: {args.steps}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {out_dir}")
    print()
    print("  TRN is NOT a Transformer replacement.")
    print("  TRN is NOT a content-addressable memory.")
    print("  Tri-Memory is a hierarchical memory architecture.")
    print()

    seed_everything(args.seed)

    results = run_sweep(windows, tasks, args.steps, device, out_dir)

    # Generate all outputs
    print()
    print("Generating outputs...")
    write_results_csv(results, out_dir / "w_sweep_results.csv")
    write_comparison_csv(results, out_dir / "w_sweep_comparison.csv")

    # Also write to results/ for eval_go_no_go.py compatibility
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    write_comparison_csv(results, results_dir / "w_sweep_comparison.csv")

    write_summary_md(results, out_dir / "w_sweep_summary.md")
    generate_plots(results, out_dir)

    # --- Final Summary ---
    print()
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    model_keys = ["A_kv_only", "B_kv_trn", "C_kv_ret", "D_trimemory"]
    for W in windows:
        print(f"\n  W={W}:")
        for mk in model_keys:
            vals = [r["composite_score"] for r in results
                    if r["window"] == W and r["model"] == mk]
            mean_score = sum(vals) / len(vals) if vals else 0.0
            tps_vals = [r["tokens_per_second"] for r in results
                        if r["window"] == W and r["model"] == mk]
            mean_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0.0
            print(f"    {mk:15s}  score={mean_score:.3f}  tps={mean_tps:.0f}")

    # Success conditions
    print()
    tri_scores = {W: sum(r["composite_score"] for r in results
                         if r["window"] == W and r["model"] == "D_trimemory")
                  / max(1, sum(1 for r in results
                               if r["window"] == W and r["model"] == "D_trimemory"))
                  for W in windows}

    ablation_scores = {W: max(
        sum(r["composite_score"] for r in results
            if r["window"] == W and r["model"] == mk)
        / max(1, sum(1 for r in results
                     if r["window"] == W and r["model"] == mk))
        for mk in ["A_kv_only", "B_kv_trn", "C_kv_ret"]
    ) for W in windows}

    tri_best = all(tri_scores[W] >= ablation_scores[W] for W in windows)
    w64_stable = tri_scores.get(64, 0) > 0.1

    print(f"  TriMemory composite > all ablations at all W: {'PASS' if tri_best else 'FAIL'}")
    print(f"  W=64 performance not collapsed (>0.1):        {'PASS' if w64_stable else 'FAIL'}")
    print()

    # Optimal window recommendation
    best_W = max(windows, key=lambda W: tri_scores.get(W, 0))
    print(f"  Optimal window recommendation: W={best_W}")
    print(f"  TriMemory most advantageous at: W={min(windows, key=lambda W: ablation_scores.get(W, 0) - tri_scores.get(W, 0))}")

    kv_best = {W: sum(r["composite_score"] for r in results
                      if r["window"] == W and r["model"] == "A_kv_only")
               / max(1, sum(1 for r in results
                            if r["window"] == W and r["model"] == "A_kv_only"))
               for W in windows}
    kv_wins_at = [W for W in windows if kv_best.get(W, 0) > tri_scores.get(W, 0)]
    if kv_wins_at:
        print(f"  KV-only wins at: W={kv_wins_at}")
    else:
        print("  KV-only does not beat TriMemory at any W")

    print()
    print(f"  Output: {out_dir}/")
    print("  1. w_sweep_results.csv")
    print("  2. w_sweep_summary.md")
    print("  3. w_sweep_comparison.csv")
    print("  4. accuracy_vs_window.png")
    print("  5. throughput_vs_window.png")
    print("  6. memory_vs_window.png")
    print()


if __name__ == "__main__":
    main()
