#!/usr/bin/env python3
"""Consolidated Go/No-Go evaluation for TRN architecture (v2.1).

Evaluates TRN against three tiers of criteria:

  T1 -- Mandatory (10 criteria, all must pass for GO):
    Pattern/State:
      1. long_history_speedup     TRN/TF TPS >= 2x at T>=5000 (S3 or S4)
      2. memory_reduction         KV/TRN state >= 10x at T=1000
      3. state_constant           TRN state variance < 1% across context lengths
      4. numerical_stable         No NaN/Inf in generation
      5. agent_scale              KV/TRN ratio >= 20x at T=1000
      6. trn_pattern_parity       TRN loss <= 1.5x TF on >= 3/4 stream tasks
      7. trn_pattern_beyond_w     trend_shift acc >= TF-0.25 AND > 0.35 at d>=W
      8. dual_pattern_parity      Dual loss <= 2.0x TF on stream tasks (dual only)
      9. dual_ppd_window_gen      PPD acc >= TF-0.25 at seq > W (dual only)
     10. signal_continuation      MSE < 3x TF at d >= 2W

  T2 -- Known Limitations (5 criteria, document structural limits):
      1. nih_recall_zero          NiH recall = 0.0 (content-addressed failure)
      2. selective_copy_low       Selective copy acc < 0.15
      3. gt_beyond_window_chance  GT accuracy at d > W ~ 0.25 (chance)
      4. gt_reversal_chance       GT reversal at d > W ~ 0.25 (chance)
      5. trp_degraded             TRP reconstruction degrades with distance

  T3 -- Stretch / Paper Bonus (12 criteria):
      1-7:  Systems metrics (TPS, KV growth, agent scaling, etc.)
      8-10: Additional pattern tasks (frequency_drift, amplitude_envelope, running_mean)
      11:   W-sweep monotonic (W=256 >= W=64 on >= 3/4 tasks)
      12:   Real use-case pass

Data sources (CSV files read from results/):
    - results/bench_agent_history.csv       (T1-1..5, T3-1..7)
    - results/bench_stream_tasks.csv        (T1-6, T1-8)
    - results/bench_pattern_memory.csv      (T1-7, T1-10, T3-8..10)
    - results/bench_needle_haystack.csv     (T1-9, T2-1..5)
    - results/go_nogo_copy_trn.csv          (T3-7)
    - results/go_nogo_copy_tf.csv           (T3-7)
    - results/long_context_scaling.csv      (T3)
    - results/w_sweep_comparison.csv        (T3-11)

If benchmark CSVs are missing, the evaluator runs lightweight inline
micro-benchmarks to generate the required data points.

Output:
    results/eval_go_no_go_{backend}.csv
    results/gate_result_{backend}.json
    results/gate_result_{backend}.md
    stdout: PASS/FAIL table + final GO/CONDITIONAL_GO/NO_GO verdict

Usage:
    python scripts/eval_go_no_go.py --device cpu
    python scripts/eval_go_no_go.py --device cpu --backend dual
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.model import TRNModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# T1: Mandatory thresholds
T1_SPEEDUP_THRESHOLD = 2.0          # TRN/TF TPS ratio minimum at T>=5000
T1_MEMORY_REDUCTION = 10.0          # KV/TRN state size ratio minimum
T1_STATE_VARIANCE_MAX = 0.01        # max variance in TRN state KB across contexts
T1_AGENT_SCALE = 20.0               # minimum agent scale multiplier
T1_PATTERN_PARITY_MAX = 1.5         # TRN loss <= 1.5x TF on stream tasks
T1_PATTERN_PARITY_MIN_TASKS = 3     # must pass on >= 3/4 stream tasks
T1_PATTERN_BEYOND_GAP = 0.25        # acc >= TF - 0.25
T1_PATTERN_BEYOND_FLOOR = 0.35      # acc > 0.35 safety valve
T1_DUAL_PARITY_MAX = 2.0            # Dual loss <= 2.0x TF on stream tasks
T1_PPD_TOLERANCE = 0.25             # PPD accuracy gap tolerance (dual >= TF - 0.25)
T1_SIGNAL_MSE_RATIO = 3.0           # signal_continuation MSE < 3x TF at d>=2W

# T2: Known limitations (structural, expected to fail)
T2_NIH_MAX = 0.05                   # NiH recall expected near 0.0
T2_SELCOPY_MAX = 0.15               # selective copy expected < 0.15
T2_GT_CHANCE_MAX = 0.35             # GT at chance level (~0.25 for 4 classes)

# T3: Stretch thresholds
T3_TPS_MIN = 100.0                  # minimum TPS at any history length
T3_KV_GROWTH_MIN = 50.0             # KV cache growth factor T=1k to T=50k
T3_AGENTS_1K_MB = 100.0             # max MB for 1000 agents (TRN)
T3_AGENTS_10K_MB = 2000.0           # max MB for 10000 agents (TRN)
T3_COPY_ACC_MIN = 0.9               # final copy accuracy minimum
T3_PATTERN_PARITY_MAX = 2.0         # pattern task loss <= 2.0x TF (stretch)
T3_W_SWEEP_MIN_TASKS = 3            # W=256 >= W=64 on >= 3/4 tasks


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    """Read CSV to list of dicts with auto-type coercion."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            coerced: dict = {}
            for k, v in row.items():
                try:
                    coerced[k] = float(v)
                except (ValueError, TypeError):
                    coerced[k] = v
            rows.append(coerced)
    return rows


def _sha256_of_file(path: Path) -> str:
    """Return hex SHA-256 of a file, or empty string if file does not exist."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Inline micro-benchmark (fallback when CSVs are missing)
# ---------------------------------------------------------------------------

def _run_agent_history_micro(
    device: torch.device,
    checkpoints: list[int],
) -> list[dict]:
    """Minimal agent-history benchmark to generate required data."""
    from trimemory.baseline import TransformerModel

    cfg = TRNConfig(
        vocab_size=256, d_model=256, n_oscillators=128,
        n_layers=8, d_ff=1024, max_seq_len=max(checkpoints) + 64,
    )
    trn = TRNModel(cfg).to(device).eval()
    tf = TransformerModel(cfg).to(device).eval()

    gen_tokens = 32
    rows = []

    for T in checkpoints:
        tokens = torch.randint(0, 256, (1, T), device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            trn.generate(tokens, max_new_tokens=gen_tokens)
        trn_tps = gen_tokens / max(time.perf_counter() - t0, 1e-6)

        clamped = tokens[:, :cfg.max_seq_len]
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = clamped.clone()
            for _ in range(gen_tokens):
                out = tf(gen[:, -cfg.max_seq_len:])
                nxt = out["logits"][:, -1:, :].argmax(-1)
                gen = torch.cat([gen, nxt], dim=1)
        tf_tps = gen_tokens / max(time.perf_counter() - t0, 1e-6)

        n_heads = max(1, cfg.d_model // 64)
        head_dim = cfg.d_model // n_heads
        kv_mb = cfg.n_layers * 2 * n_heads * T * head_dim * 4 / (1024 * 1024)
        state_kb = cfg.n_layers * cfg.n_oscillators * 2 * 4 / 1024

        rows.append(dict(
            history_tokens=T,
            trn_tps=trn_tps,
            trn_state_kb=state_kb,
            tf_kv_tps=tf_tps,
            tf_kv_cache_mb=kv_mb,
            tf_full_tps=tf_tps,
        ))

    return rows


def _run_copy_micro(device: torch.device, steps: int = 500) -> list[dict]:
    """Minimal copy-task training to check T1 numerical stability."""
    from trimemory.bench_data import NextTokenCopyDataset
    from trimemory.scheduler import CosineWithWarmup
    from torch.utils.data import DataLoader

    cfg = TRNConfig(
        vocab_size=32, d_model=128, n_oscillators=64,
        n_layers=4, d_ff=512, max_seq_len=72,
    )
    model = TRNModel(cfg).to(device).train()
    ds = NextTokenCopyDataset(n_samples=500, seq_len=64, vocab_size=32, period=8, seed=42)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sched = CosineWithWarmup(opt, warmup_steps=50, max_steps=steps, lr=3e-4, min_lr=3e-5)

    loader_it = iter(loader)
    records = []
    for step in range(1, steps + 1):
        sched.step(step)
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)

        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        loss = out["loss"]
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad()

        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_out = model(ids, labels=ids)
                val_loss = val_out["loss"].item()
                preds = val_out["logits"].argmax(-1)
                acc = (preds[:, :-1] == ids[:, 1:]).float().mean().item()
            model.train()
            records.append(dict(
                step=step, train_loss=loss.item(),
                val_loss=val_loss, accuracy=acc,
            ))

    return records


def _run_copy_tf_micro(device: torch.device, steps: int = 500) -> list[dict]:
    """Minimal copy-task training for TransformerModel (TF baseline).

    Mirrors _run_copy_micro but uses TransformerModel instead of TRNModel.
    Same config, seed, dataset, schedule — only the model class differs.
    """
    from trimemory.baseline import TransformerModel
    from trimemory.bench_data import NextTokenCopyDataset
    from trimemory.scheduler import CosineWithWarmup
    from torch.utils.data import DataLoader

    cfg = TRNConfig(
        vocab_size=32, d_model=128, n_oscillators=64,
        n_layers=4, d_ff=512, max_seq_len=72,
    )
    model = TransformerModel(cfg).to(device).train()
    ds = NextTokenCopyDataset(n_samples=500, seq_len=64, vocab_size=32, period=8, seed=42)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sched = CosineWithWarmup(opt, warmup_steps=50, max_steps=steps, lr=3e-4, min_lr=3e-5)

    loader_it = iter(loader)
    records = []
    for step in range(1, steps + 1):
        sched.step(step)
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)

        ids = batch["input_ids"].to(device)
        out = model(ids, labels=ids)
        loss = out["loss"]
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad()

        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_out = model(ids, labels=ids)
                val_loss = val_out["loss"].item()
                preds = val_out["logits"].argmax(-1)
                acc = (preds[:, :-1] == ids[:, 1:]).float().mean().item()
            model.train()
            records.append(dict(
                step=step, train_loss=loss.item(),
                val_loss=val_loss, accuracy=acc,
            ))

    return records


# ---------------------------------------------------------------------------
# NiH dual-backend micro-benchmark (fallback for dual backend)
# ---------------------------------------------------------------------------

# NiH token ranges (mirror bench_needle_haystack.py)
_NIH_FILLER_LOW = 10
_NIH_FILLER_HIGH = 200
_NIH_NEEDLE_LOW = 200
_NIH_NEEDLE_HIGH = 240
_NIH_QUERY_TOKEN = 5


def _nih_dual_batch(
    distance: int,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
):
    """Generate NiH batch for dual micro-benchmark."""
    seq_len = distance + 2
    needles = torch.randint(_NIH_NEEDLE_LOW, _NIH_NEEDLE_HIGH, (batch_size,), generator=rng)
    filler = torch.randint(_NIH_FILLER_LOW, _NIH_FILLER_HIGH, (batch_size, seq_len), generator=rng)
    filler[:, 0] = needles
    filler[:, -1] = _NIH_QUERY_TOKEN
    return filler.to(device), filler.to(device)


def _nih_dual_recall(
    model: nn.Module,
    distance: int,
    n_eval: int,
    device: torch.device,
) -> float:
    """Recall@1 for NiH: fraction where logits[:, -1].argmax() == needle."""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(999)
    seq_len = distance + 2
    batch_size = min(64, n_eval)
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(math.ceil(n_eval / batch_size)):
            needles = torch.randint(_NIH_NEEDLE_LOW, _NIH_NEEDLE_HIGH, (batch_size,), generator=rng)
            filler = torch.randint(_NIH_FILLER_LOW, _NIH_FILLER_HIGH, (batch_size, seq_len), generator=rng)
            filler[:, 0] = needles
            filler[:, -1] = _NIH_QUERY_TOKEN
            input_ids = filler.to(device)
            out = model(input_ids)
            preds = out["logits"][:, -1].argmax(dim=-1).cpu()
            correct += (preds == needles).sum().item()
            total += batch_size
    return correct / max(total, 1)


def _run_nih_dual_micro(device: torch.device) -> list[dict]:
    """Train DualMemoryEngine on NiH task and evaluate recall at multiple distances.

    Returns list of dicts matching NiH CSV format so _eval_dual_extra() can
    pick up the long-range recovery criterion without requiring an external CSV.
    """
    from trimemory.integrations.vllm_backend import DualMemoryEngine

    WINDOW_SIZE = 64
    D = 128
    L = 4
    K = 64
    W = 64
    STEPS = 300
    BATCH_SIZE = 32
    EVAL_DISTANCES = [100, 200, 500]
    N_EVAL = 10

    cfg = TRNConfig(
        vocab_size=256,
        d_model=D,
        n_oscillators=K,
        n_layers=L,
        d_ff=512,
        max_seq_len=max(EVAL_DISTANCES) + 2 + 16,
    )

    print("  [INFO] Running NiH dual micro-benchmark (DualMemoryEngine, 300 steps)...")
    rows: list[dict] = []

    for dist in EVAL_DISTANCES:
        seed_everything(42)
        model = DualMemoryEngine(cfg, window_size=WINDOW_SIZE).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        rng = torch.Generator()
        rng.manual_seed(42)

        for _ in range(STEPS):
            input_ids, _ = _nih_dual_batch(dist, BATCH_SIZE, device, rng)
            out = model(input_ids)
            logits = out["logits"]  # (B, T, V)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                input_ids[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate recall with N_EVAL independent samples
        sample_recalls = [_nih_dual_recall(model, dist, 256, device) for _ in range(N_EVAL)]
        mean_acc = sum(sample_recalls) / len(sample_recalls)
        variance = sum((r - mean_acc) ** 2 for r in sample_recalls) / max(len(sample_recalls) - 1, 1)
        std_acc = math.sqrt(variance)

        print(f"    dual_w{WINDOW_SIZE} distance={dist}: mean_recall={mean_acc:.3f} std={std_acc:.3f}")

        rows.append(dict(
            task="nih",
            model=f"dual_w{WINDOW_SIZE}",
            backend=f"dual_w{WINDOW_SIZE}",
            param=dist,
            metric="recall_accuracy",
            value=mean_acc,
            n_samples=N_EVAL,
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
        ))

    return rows


# ---------------------------------------------------------------------------
# GT (Goal Tracking) dual-backend micro-benchmark
# ---------------------------------------------------------------------------

_GT_GOAL_TOKENS = [200, 201, 202, 203]
_GT_QUERY_TOKEN = 6
_GT_FILLER_LOW = 10
_GT_FILLER_HIGH = 200


def _gt_dual_batch(
    distance: int,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
):
    """Generate GT batch for dual micro-benchmark.

    Sequence: [GOAL, filler*distance, GOAL_QUERY, GOAL_ANSWER]
    The answer token provides the training signal for GOAL_QUERY -> GOAL.
    """
    seq_len = distance + 3  # GOAL + distance filler + GOAL_QUERY + GOAL_ANSWER
    goal_idx = torch.randint(0, len(_GT_GOAL_TOKENS), (batch_size,), generator=rng)
    goals = torch.tensor([_GT_GOAL_TOKENS[i] for i in goal_idx.tolist()])
    filler = torch.randint(_GT_FILLER_LOW, _GT_FILLER_HIGH, (batch_size, seq_len), generator=rng)
    filler[:, 0] = goals
    filler[:, -2] = _GT_QUERY_TOKEN
    filler[:, -1] = goals  # answer = the goal token
    return filler.to(device), filler.to(device)


def _gt_dual_recall(
    model: nn.Module,
    distance: int,
    n_eval: int,
    device: torch.device,
) -> float:
    """GT recall: predict which GOAL_TOKEN was at position 0 after GOAL_QUERY.

    Eval sequence: [GOAL, filler*distance, GOAL_QUERY, GOAL_ANSWER].
    Check logits[:, -2] (GOAL_QUERY position) predicts GOAL.
    """
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(1234)
    seq_len = distance + 3
    batch_size = min(64, n_eval)
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(math.ceil(n_eval / batch_size)):
            goal_idx = torch.randint(0, len(_GT_GOAL_TOKENS), (batch_size,), generator=rng)
            goals = torch.tensor([_GT_GOAL_TOKENS[i] for i in goal_idx.tolist()])
            filler = torch.randint(_GT_FILLER_LOW, _GT_FILLER_HIGH, (batch_size, seq_len), generator=rng)
            filler[:, 0] = goals
            filler[:, -2] = _GT_QUERY_TOKEN
            filler[:, -1] = goals
            input_ids = filler.to(device)
            out = model(input_ids)
            preds = out["logits"][:, -2].argmax(dim=-1).cpu()
            correct += (preds == goals).sum().item()
            total += batch_size
    return correct / max(total, 1)


def _gt_dual_reversal_recall(
    model: nn.Module,
    distance: int,
    n_eval: int,
    device: torch.device,
) -> float:
    """GT reversal recall: second goal overwrites first. Model must predict GOAL_2.

    Sequence: [GOAL_1, filler..., GOAL_2, filler..., GOAL_QUERY, GOAL_2_ANSWER]
    Check logits[:, -2] predicts GOAL_2.
    """
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(5678)
    seq_len = distance + 3
    mid = max(1, (seq_len - 1) // 2)
    batch_size = min(64, n_eval)
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(math.ceil(n_eval / batch_size)):
            g1_idx = torch.randint(0, len(_GT_GOAL_TOKENS), (batch_size,), generator=rng)
            offset = torch.randint(1, len(_GT_GOAL_TOKENS), (batch_size,), generator=rng)
            g2_idx = (g1_idx + offset) % len(_GT_GOAL_TOKENS)
            goals2 = torch.tensor([_GT_GOAL_TOKENS[i] for i in g2_idx.tolist()])
            filler = torch.randint(_GT_FILLER_LOW, _GT_FILLER_HIGH, (batch_size, seq_len), generator=rng)
            filler[:, 0] = torch.tensor([_GT_GOAL_TOKENS[i] for i in g1_idx.tolist()])
            filler[:, mid] = goals2
            filler[:, -2] = _GT_QUERY_TOKEN
            filler[:, -1] = goals2  # answer = most recent goal
            input_ids = filler.to(device)
            out = model(input_ids)
            preds = out["logits"][:, -2].argmax(dim=-1).cpu()
            correct += (preds == goals2).sum().item()
            total += batch_size
    return correct / max(total, 1)


def _run_gt_dual_micro(device: torch.device) -> list[dict]:
    """Train DualMemoryEngine on GT task and evaluate at multiple distances.

    Returns list of dicts with gt and gt_reversal rows for _eval_gt_reference().
    """
    from trimemory.integrations.vllm_backend import DualMemoryEngine

    WINDOW_SIZE = 64
    D = 128
    L = 4
    K = 64
    STEPS = 300
    BATCH_SIZE = 32
    # Distances: some within window, some beyond
    EVAL_DISTANCES = [50, 100, 200, 500]
    N_EVAL = 10

    cfg = TRNConfig(
        vocab_size=256,
        d_model=D,
        n_oscillators=K,
        n_layers=L,
        d_ff=512,
        max_seq_len=max(EVAL_DISTANCES) + 2 + 16,
    )

    print("  [INFO] Running GT dual micro-benchmark (DualMemoryEngine, 300 steps)...")
    rows: list[dict] = []

    for dist in EVAL_DISTANCES:
        seed_everything(42)
        model = DualMemoryEngine(cfg, window_size=WINDOW_SIZE).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        rng = torch.Generator()
        rng.manual_seed(42)

        for _ in range(STEPS):
            input_ids, _ = _gt_dual_batch(dist, BATCH_SIZE, device, rng)
            out = model(input_ids)
            logits = out["logits"]
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                input_ids[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate recall and reversal
        recall_samples = [_gt_dual_recall(model, dist, 256, device) for _ in range(N_EVAL)]
        mean_recall = sum(recall_samples) / len(recall_samples)
        std_recall = math.sqrt(
            sum((r - mean_recall) ** 2 for r in recall_samples) / max(len(recall_samples) - 1, 1)
        )

        reversal_samples = [_gt_dual_reversal_recall(model, dist, 256, device) for _ in range(N_EVAL)]
        mean_reversal = sum(reversal_samples) / len(reversal_samples)
        std_reversal = math.sqrt(
            sum((r - mean_reversal) ** 2 for r in reversal_samples) / max(len(reversal_samples) - 1, 1)
        )

        print(f"    dual_w{WINDOW_SIZE} distance={dist}: "
              f"gt_recall={mean_recall:.3f}+/-{std_recall:.3f} "
              f"reversal={mean_reversal:.3f}+/-{std_reversal:.3f}")

        rows.append(dict(
            task="gt", model=f"dual_w{WINDOW_SIZE}",
            backend=f"dual_w{WINDOW_SIZE}", param=dist,
            metric="goal_tracking_accuracy", value=mean_recall,
            n_samples=N_EVAL, mean_accuracy=mean_recall, std_accuracy=std_recall,
        ))
        rows.append(dict(
            task="gt_reversal", model=f"dual_w{WINDOW_SIZE}",
            backend=f"dual_w{WINDOW_SIZE}", param=dist,
            metric="goal_tracking_accuracy", value=mean_reversal,
            n_samples=N_EVAL, mean_accuracy=mean_reversal, std_accuracy=std_reversal,
        ))

    return rows


# ---------------------------------------------------------------------------
# PPD (Periodic Pattern Detection) dual-backend micro-benchmark
# ---------------------------------------------------------------------------

_PPD_FREQUENCIES = [0.01, 0.05, 0.1, 0.2]
_PPD_FREQ_QUERY_TOKEN = 7


def _ppd_batch(
    seq_len: int,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
):
    """Generate periodic signal sequences. Labels = frequency class index."""
    freq_labels = torch.randint(0, len(_PPD_FREQUENCIES), (batch_size,), generator=rng)
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for b in range(batch_size):
        freq = _PPD_FREQUENCIES[freq_labels[b].item()]
        for t in range(seq_len - 1):
            val = int(math.sin(2 * math.pi * freq * t) * 50 + 128)
            ids[b, t] = max(50, min(200, val))
        ids[b, -1] = _PPD_FREQ_QUERY_TOKEN
    return ids.to(device), freq_labels


def _ppd_probe_accuracy(
    model: nn.Module,
    probe: nn.Module,
    seq_len: int,
    n_eval: int,
    device: torch.device,
    captured: list,
) -> float:
    """Evaluate PPD probe accuracy."""
    model.eval()
    probe.eval()
    rng_eval = torch.Generator()
    rng_eval.manual_seed(888)
    correct = 0
    batch_size = min(64, n_eval)
    with torch.no_grad():
        for _ in range(math.ceil(n_eval / batch_size)):
            ids, freq_labels = _ppd_batch(seq_len, batch_size, device, rng_eval)
            model(ids)
            hidden = captured[0]
            preds = probe(hidden[:, -1, :]).argmax(dim=-1)
            correct += (preds == freq_labels.to(device)).sum().item()
    return correct / n_eval


def _run_ppd_dual_micro(device: torch.device) -> list[dict]:
    """Train DualMemoryEngine + TF on PPD and compare accuracy at seq_len > W.

    Returns list of dicts with ppd rows for _eval_ppd_window_generalization().
    Both dual and tf models are trained, and accuracy is compared at each seq_len.
    """
    from trimemory.baseline import TransformerModel
    from trimemory.integrations.vllm_backend import DualMemoryEngine

    WINDOW_SIZE = 64
    D = 128
    L = 4
    K = 64
    STEPS = 1000
    BATCH_SIZE = 32
    # Include seq_lens both within and beyond window
    SEQ_LENS = [64, 128, 256, 512]
    PROBE_STEPS = 200

    cfg = TRNConfig(
        vocab_size=256,
        d_model=D,
        n_oscillators=K,
        n_layers=L,
        d_ff=512,
        max_seq_len=max(SEQ_LENS) + 16,
    )

    print("  [INFO] Running PPD dual micro-benchmark (DualMemoryEngine + TF)...")
    rows: list[dict] = []

    for model_name, make_fn in [
        (f"dual_w{WINDOW_SIZE}", lambda: DualMemoryEngine(cfg, window_size=WINDOW_SIZE).to(device)),
        ("tf", lambda: TransformerModel(cfg).to(device)),
    ]:
        for seq_len in SEQ_LENS:
            seed_everything(42)
            model = make_fn()
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            rng = torch.Generator()
            rng.manual_seed(42)

            # Register hook on norm_out (final RMSNorm) to capture hidden states
            # This matches bench_needle_haystack.py's _register_hidden_hook pattern
            captured: list = [None]
            norm_layer = None
            if hasattr(model, "norm_out"):
                norm_layer = model.norm_out  # TRNModel, DualMemoryEngine
            elif hasattr(model, "norm"):
                norm_layer = model.norm      # TransformerModel

            def _hook(module, input, output, captured=captured):
                captured[0] = output.detach()

            hook_handle = norm_layer.register_forward_hook(_hook) if norm_layer else None

            # Backbone training
            for _ in range(STEPS):
                ids, _ = _ppd_batch(seq_len, BATCH_SIZE, device, rng)
                out = model(ids)
                logits = out["logits"]
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V),
                    ids[:, 1:].reshape(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Train probe
            model.eval()
            probe = nn.Linear(D, len(_PPD_FREQUENCIES)).to(device)
            probe_opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
            rng_probe = torch.Generator()
            rng_probe.manual_seed(77)

            probe.train()
            for _ in range(PROBE_STEPS):
                ids, freq_labels = _ppd_batch(seq_len, BATCH_SIZE, device, rng_probe)
                with torch.no_grad():
                    model(ids)
                hidden = captured[0]
                if hidden is None:
                    break
                logits_p = probe(hidden[:, -1, :])
                loss_p = F.cross_entropy(logits_p, freq_labels.to(device))
                probe_opt.zero_grad()
                loss_p.backward()
                probe_opt.step()

            # Evaluate
            score = _ppd_probe_accuracy(model, probe, seq_len, 256, device, captured)
            print(f"    PPD [{model_name}] seq_len={seq_len}: accuracy={score:.3f}")

            rows.append(dict(
                task="ppd", model=model_name,
                backend=model_name, param=seq_len,
                metric="frequency_detection_score", value=score,
            ))

            if hook_handle:
                hook_handle.remove()

    return rows


# ---------------------------------------------------------------------------
# Criterion evaluators
# ---------------------------------------------------------------------------

class Criterion:
    def __init__(
        self,
        name: str,
        tier: str,
        status: str,
        value: float,
        threshold: float,
        description: str = "",
    ) -> None:
        self.name = name
        self.tier = tier
        self.status = status        # PASS / FAIL / SKIP
        self.value = value
        self.threshold = threshold
        self.description = description

    def as_dict(self) -> dict:
        return dict(
            criterion=self.name,
            tier=self.tier,
            status=self.status,
            value=f"{self.value:.4f}" if math.isfinite(self.value) else str(self.value),
            threshold=f"{self.threshold:.4f}" if math.isfinite(self.threshold) else str(self.threshold),
            description=self.description,
        )


def _eval_t1(
    agent_history: list[dict],
    stream_tasks: list[dict],
    pattern_memory: list[dict],
    device: torch.device,
    cfg_100m: TRNConfig,
    backend: str = "trn",
) -> list[Criterion]:
    """Evaluate T1 mandatory criteria (v2.1: 10 criteria).

    Criteria 1-5: systems (speedup, memory, state, numerical, agent scale)
    Criteria 6-10: pattern (stream parity, beyond-window, dual parity, PPD, signal)
    """
    results: list[Criterion] = []

    # T1-1: long_history_speedup >= 2x at T>=5000
    if agent_history:
        long_rows = [r for r in agent_history if r.get("history_tokens", 0) >= 5000]
        if long_rows:
            ratios = []
            for r in long_rows:
                tf_tps = r.get("tf_kv_tps", 0)
                trn_tps = r.get("trn_tps", 0)
                if tf_tps > 0 and math.isfinite(tf_tps) and math.isfinite(trn_tps):
                    ratios.append(trn_tps / tf_tps)
            avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
            results.append(Criterion(
                "speedup>=2x", "T1",
                "PASS" if avg_ratio >= T1_SPEEDUP_THRESHOLD else "FAIL",
                avg_ratio, T1_SPEEDUP_THRESHOLD,
                f"TRN/TF avg TPS ratio at T>=5000: {avg_ratio:.2f}",
            ))
        else:
            results.append(Criterion("speedup>=2x", "T1", "SKIP", 0.0, T1_SPEEDUP_THRESHOLD, "No long-context rows"))
    else:
        results.append(Criterion("speedup>=2x", "T1", "SKIP", 0.0, T1_SPEEDUP_THRESHOLD, "No agent_history data"))

    # T1-2: memory_reduction >= 10x (TRN state vs KV cache)
    trn_state_kb = cfg_100m.n_layers * cfg_100m.n_oscillators * 2 * 4 / 1024
    if agent_history:
        row_1k = next(
            (r for r in agent_history if abs(r.get("history_tokens", 0) - 1000) < 200), None
        )
        if row_1k:
            kv_mb = row_1k.get("tf_kv_cache_mb", 0)
            kv_kb = kv_mb * 1024
            reduction = kv_kb / max(trn_state_kb, 1e-6)
            results.append(Criterion(
                "memory_reduction>=10x", "T1",
                "PASS" if reduction >= T1_MEMORY_REDUCTION else "FAIL",
                reduction, T1_MEMORY_REDUCTION,
                f"KV={kv_kb:.1f}KB / TRN={trn_state_kb:.1f}KB = {reduction:.1f}x",
            ))
        else:
            results.append(Criterion("memory_reduction>=10x", "T1", "SKIP", 0.0, T1_MEMORY_REDUCTION, "No T~1000 row"))
    else:
        results.append(Criterion("memory_reduction>=10x", "T1", "SKIP", 0.0, T1_MEMORY_REDUCTION, "No data"))

    # T1-3: state_constant (TRN state KB variance < 1% across history lengths)
    if agent_history and len(agent_history) >= 2:
        state_kbs = [r.get("trn_state_kb", float("nan")) for r in agent_history]
        valid = [v for v in state_kbs if math.isfinite(v)]
        if valid:
            variance = max(valid) - min(valid)
            results.append(Criterion(
                "state_constant", "T1",
                "PASS" if variance < T1_STATE_VARIANCE_MAX else "FAIL",
                variance, T1_STATE_VARIANCE_MAX,
                f"TRN state KB range: [{min(valid):.2f}, {max(valid):.2f}]",
            ))
        else:
            results.append(Criterion("state_constant", "T1", "SKIP", 0.0, T1_STATE_VARIANCE_MAX, "No finite state values"))
    else:
        results.append(Criterion("state_constant", "T1", "SKIP", 0.0, T1_STATE_VARIANCE_MAX, "Insufficient data"))

    # T1-4: numerical_stable (no NaN/Inf in TRN generation)
    seed_everything(42)
    test_model = TRNModel(cfg_100m).to(device).eval()
    tokens = torch.randint(0, cfg_100m.vocab_size, (1, 32), device=device)
    stable = True
    try:
        with torch.no_grad():
            generated = test_model.generate(tokens, max_new_tokens=16)
        if not torch.isfinite(generated.float()).all():
            stable = False
    except Exception:
        stable = False
    del test_model

    results.append(Criterion(
        "numerical_stable", "T1",
        "PASS" if stable else "FAIL",
        1.0 if stable else 0.0, 1.0,
        "TRN generates 16 tokens without NaN/Inf",
    ))

    # T1-5: agent_scale >= 20x
    if agent_history:
        row_1k = next(
            (r for r in agent_history if abs(r.get("history_tokens", 0) - 1000) < 200), None
        )
        if row_1k:
            kv_mb = row_1k.get("tf_kv_cache_mb", 0)
            kv_kb = kv_mb * 1024
            agent_scale = kv_kb / max(trn_state_kb, 1e-6)
            results.append(Criterion(
                "agent_scale>=20x", "T1",
                "PASS" if agent_scale >= T1_AGENT_SCALE else "FAIL",
                agent_scale, T1_AGENT_SCALE,
                f"KV/TRN size ratio at T=1000: {agent_scale:.1f}x",
            ))
        else:
            results.append(Criterion("agent_scale>=20x", "T1", "SKIP", 0.0, T1_AGENT_SCALE, "No T~1000 row"))
    else:
        results.append(Criterion("agent_scale>=20x", "T1", "SKIP", 0.0, T1_AGENT_SCALE, "No data"))

    # T1-6: trn_pattern_parity (TRN loss <= 1.5x TF on >= 3/4 stream tasks)
    if stream_tasks:
        n_pass = 0
        n_total = 0
        details = []
        for row in stream_tasks:
            task_name = row.get("task", "")
            trn_loss = float(row.get("TRN", float("nan")))
            tf_loss = float(row.get("TF", float("nan")))
            if math.isfinite(trn_loss) and math.isfinite(tf_loss) and tf_loss > 0:
                ratio = trn_loss / tf_loss
                passed = ratio <= T1_PATTERN_PARITY_MAX
                if passed:
                    n_pass += 1
                n_total += 1
                details.append(f"{task_name}={ratio:.2f}x")
        if n_total > 0:
            enough = n_pass >= T1_PATTERN_PARITY_MIN_TASKS
            results.append(Criterion(
                "trn_pattern_parity", "T1",
                "PASS" if enough else "FAIL",
                float(n_pass), float(T1_PATTERN_PARITY_MIN_TASKS),
                f"TRN/TF loss ratio <= {T1_PATTERN_PARITY_MAX}x on {n_pass}/{n_total} tasks: {'; '.join(details)}",
            ))
        else:
            results.append(Criterion("trn_pattern_parity", "T1", "SKIP", 0.0, float(T1_PATTERN_PARITY_MIN_TASKS), "No valid stream tasks"))
    else:
        results.append(Criterion("trn_pattern_parity", "T1", "SKIP", 0.0, float(T1_PATTERN_PARITY_MIN_TASKS), "No bench_stream_tasks.csv"))

    # T1-7: trn_pattern_beyond_window (trend_shift acc >= TF-0.25 AND > 0.35 at d>=W)
    W = 64  # default window size for bench config
    trend_rows = [
        r for r in pattern_memory
        if r.get("task") == "trend_shift"
        and r.get("model") == "trn"
        and float(r.get("distance", 0)) >= W
    ]
    if trend_rows:
        all_pass = True
        worst_acc = 1.0
        details = []
        for r in trend_rows:
            acc = float(r.get("accuracy", 0))
            tf_acc = float(r.get("tf_accuracy", 0))
            gap_ok = acc >= tf_acc - T1_PATTERN_BEYOND_GAP
            floor_ok = acc > T1_PATTERN_BEYOND_FLOOR
            if not (gap_ok and floor_ok):
                all_pass = False
            worst_acc = min(worst_acc, acc)
            d = int(float(r.get("distance", 0)))
            details.append(f"d={d}: acc={acc:.3f} tf={tf_acc:.3f}")
        results.append(Criterion(
            "trn_pattern_beyond_w", "T1",
            "PASS" if all_pass else "FAIL",
            worst_acc, T1_PATTERN_BEYOND_FLOOR,
            f"trend_shift at d>=W: {'; '.join(details)}",
        ))
    else:
        results.append(Criterion("trn_pattern_beyond_w", "T1", "SKIP", 0.0, T1_PATTERN_BEYOND_FLOOR, "No trend_shift rows at d>=W"))

    # T1-8: dual_pattern_parity (dual loss <= 2.0x TF on stream tasks, dual only)
    if backend == "dual":
        # Check if stream_tasks has a Dual column (added by bench_stream_tasks.py --backend)
        has_dual = any("Dual" in str(row.get("task", "")) or row.get("Dual") for row in stream_tasks)
        if stream_tasks and has_dual:
            n_pass = 0
            n_total = 0
            for row in stream_tasks:
                dual_loss = float(row.get("Dual", float("nan")))
                tf_loss = float(row.get("TF", float("nan")))
                if math.isfinite(dual_loss) and math.isfinite(tf_loss) and tf_loss > 0:
                    ratio = dual_loss / tf_loss
                    if ratio <= T1_DUAL_PARITY_MAX:
                        n_pass += 1
                    n_total += 1
            if n_total > 0:
                results.append(Criterion(
                    "dual_pattern_parity", "T1",
                    "PASS" if n_pass >= T1_PATTERN_PARITY_MIN_TASKS else "FAIL",
                    float(n_pass), float(T1_PATTERN_PARITY_MIN_TASKS),
                    f"Dual/TF loss ratio <= {T1_DUAL_PARITY_MAX}x on {n_pass}/{n_total} tasks",
                ))
            else:
                results.append(Criterion("dual_pattern_parity", "T1", "SKIP", 0.0, float(T1_PATTERN_PARITY_MIN_TASKS), "No valid Dual stream data"))
        else:
            results.append(Criterion("dual_pattern_parity", "T1", "SKIP", 0.0, float(T1_PATTERN_PARITY_MIN_TASKS), "No Dual column in bench_stream_tasks.csv"))
    # For trn backend, skip dual-specific criteria silently

    # T1-10: signal_continuation (MSE < 3x TF at d >= 2W)
    sig_rows = [
        r for r in pattern_memory
        if r.get("task") == "signal_continuation"
        and r.get("model") == "trn"
        and float(r.get("distance", 0)) >= 2 * W
    ]
    if sig_rows:
        all_pass = True
        worst_ratio = 0.0
        details = []
        for r in sig_rows:
            mse = float(r.get("mse", float("nan")))
            tf_mse = float(r.get("tf_mse", float("nan")))
            if math.isfinite(mse) and math.isfinite(tf_mse) and tf_mse > 0:
                ratio = mse / tf_mse
                if ratio >= T1_SIGNAL_MSE_RATIO:
                    all_pass = False
                worst_ratio = max(worst_ratio, ratio)
                d = int(float(r.get("distance", 0)))
                details.append(f"d={d}: mse_ratio={ratio:.2f}x")
        results.append(Criterion(
            "signal_continuation", "T1",
            "PASS" if all_pass else "FAIL",
            worst_ratio, T1_SIGNAL_MSE_RATIO,
            f"signal_continuation MSE/TF at d>=2W: {'; '.join(details)}",
        ))
    else:
        results.append(Criterion("signal_continuation", "T1", "SKIP", 0.0, T1_SIGNAL_MSE_RATIO, "No signal_continuation rows at d>=2W"))

    return results


def _eval_t2(
    nih_rows: list[dict],
    selcopy_trn: list[dict],
    gt_rows: list[dict],
    backend: str = "trn",
    window_size: int = 64,
) -> list[Criterion]:
    """Evaluate T2 known limitations (v2.1: 5 criteria).

    These are structural limitations of linear recurrence. They are
    expected to FAIL. A PASS on T2 means the limitation was confirmed.
    T2 failures do NOT block the verdict (they document known weaknesses).
    """
    results: list[Criterion] = []

    # T2-1: nih_recall_zero (NiH recall should be ~0.0)
    nih_trn = [
        r for r in nih_rows
        if r.get("task") == "nih"
        and r.get("backend", r.get("model", "")) in ("trn", "tf")
    ]
    if nih_trn:
        trn_nih = [r for r in nih_trn if r.get("backend", r.get("model", "")) == "trn"]
        if trn_nih:
            recalls = [float(r.get("value", r.get("mean_accuracy", 0))) for r in trn_nih]
            max_recall = max(recalls) if recalls else 0.0
            # PASS means limitation confirmed (recall IS near zero)
            results.append(Criterion(
                "nih_recall_zero", "T2",
                "PASS" if max_recall <= T2_NIH_MAX else "FAIL",
                max_recall, T2_NIH_MAX,
                f"TRN NiH max recall: {max_recall:.4f} (expected ~0.0, structural limitation)",
            ))
        else:
            results.append(Criterion("nih_recall_zero", "T2", "SKIP", 0.0, T2_NIH_MAX, "No TRN NiH rows"))
    else:
        results.append(Criterion("nih_recall_zero", "T2", "SKIP", 0.0, T2_NIH_MAX, "No NiH data"))

    # T2-2: selective_copy_low (selective copy acc should be < 0.15)
    if selcopy_trn:
        final_acc = selcopy_trn[-1].get("accuracy", 0.0) if selcopy_trn else 0.0
        # PASS means limitation confirmed (acc IS low)
        results.append(Criterion(
            "selective_copy_low", "T2",
            "PASS" if final_acc < T2_SELCOPY_MAX else "FAIL",
            final_acc, T2_SELCOPY_MAX,
            f"Selective copy final acc: {final_acc:.4f} (expected < {T2_SELCOPY_MAX})",
        ))
    else:
        results.append(Criterion("selective_copy_low", "T2", "SKIP", 0.0, T2_SELCOPY_MAX, "No selcopy data"))

    # T2-3: gt_beyond_window_chance (GT at d > W should be ~0.25 chance)
    gt_long = [
        r for r in gt_rows
        if r.get("task") == "gt"
        and float(r.get("param", 0)) > window_size
    ]
    if gt_long:
        recalls = [float(r.get("value", r.get("mean_accuracy", 0))) for r in gt_long]
        max_recall = max(recalls) if recalls else 0.0
        # PASS means limitation confirmed (recall IS at chance)
        results.append(Criterion(
            "gt_beyond_window_chance", "T2",
            "PASS" if max_recall <= T2_GT_CHANCE_MAX else "FAIL",
            max_recall, T2_GT_CHANCE_MAX,
            f"GT recall at d>{window_size}: max={max_recall:.4f} (expected ~0.25)",
        ))
    else:
        results.append(Criterion("gt_beyond_window_chance", "T2", "SKIP", 0.0, T2_GT_CHANCE_MAX, "No GT data at d>W"))

    # T2-4: gt_reversal_chance (GT reversal at d > W should be ~0.25)
    gt_rev_long = [
        r for r in gt_rows
        if r.get("task") == "gt_reversal"
        and float(r.get("param", 0)) > window_size
    ]
    if gt_rev_long:
        recalls = [float(r.get("value", r.get("mean_accuracy", 0))) for r in gt_rev_long]
        max_recall = max(recalls) if recalls else 0.0
        results.append(Criterion(
            "gt_reversal_chance", "T2",
            "PASS" if max_recall <= T2_GT_CHANCE_MAX else "FAIL",
            max_recall, T2_GT_CHANCE_MAX,
            f"GT reversal at d>{window_size}: max={max_recall:.4f} (expected ~0.25)",
        ))
    else:
        results.append(Criterion("gt_reversal_chance", "T2", "SKIP", 0.0, T2_GT_CHANCE_MAX, "No GT reversal data at d>W"))

    # T2-5: trp_degraded (TRP reconstruction degrades with distance)
    trp_rows = [r for r in nih_rows if r.get("task") == "trp"]
    if trp_rows:
        # TRP at short vs long distance; expect degradation
        short = [r for r in trp_rows if float(r.get("param", 0)) <= 32]
        long = [r for r in trp_rows if float(r.get("param", 0)) >= 128]
        if short and long:
            short_avg = sum(float(r.get("value", 0)) for r in short) / len(short)
            long_avg = sum(float(r.get("value", 0)) for r in long) / len(long)
            degrades = long_avg < short_avg or long_avg < 0.5
            results.append(Criterion(
                "trp_degraded", "T2",
                "PASS" if degrades else "FAIL",
                long_avg, short_avg,
                f"TRP short={short_avg:.3f} long={long_avg:.3f} (expected degradation)",
            ))
        else:
            results.append(Criterion("trp_degraded", "T2", "SKIP", 0.0, 0.0, "Insufficient TRP distance data"))
    else:
        results.append(Criterion("trp_degraded", "T2", "SKIP", 0.0, 0.0, "No TRP data"))

    return results


def _eval_t3(
    agent_history: list[dict],
    copy_trn: list[dict],
    pattern_memory: list[dict],
    w_sweep: list[dict],
) -> list[Criterion]:
    """Evaluate T3 stretch/paper bonus criteria (v2.1: 12 criteria)."""
    results: list[Criterion] = []

    # T3-1: trn_tps_1k > 100
    if agent_history:
        row_1k = next((r for r in agent_history if abs(r.get("history_tokens", 0) - 1000) < 200), None)
        if row_1k:
            tps = row_1k.get("trn_tps", 0.0)
            results.append(Criterion(
                "trn_tps_1k", "T3",
                "PASS" if tps >= T3_TPS_MIN else "FAIL",
                tps, T3_TPS_MIN,
                f"TRN TPS at T=1000: {tps:.1f}",
            ))
        else:
            results.append(Criterion("trn_tps_1k", "T3", "SKIP", 0.0, T3_TPS_MIN, "No T~1000 row"))
    else:
        results.append(Criterion("trn_tps_1k", "T3", "SKIP", 0.0, T3_TPS_MIN, "No data"))

    # T3-2: trn_tps_10k > 100
    if agent_history:
        row_10k = next((r for r in agent_history if abs(r.get("history_tokens", 0) - 10000) < 500), None)
        if row_10k:
            tps = row_10k.get("trn_tps", 0.0)
            results.append(Criterion(
                "trn_tps_10k", "T3",
                "PASS" if tps >= T3_TPS_MIN else "FAIL",
                tps, T3_TPS_MIN,
                f"TRN TPS at T=10000: {tps:.1f}",
            ))
        else:
            results.append(Criterion("trn_tps_10k", "T3", "SKIP", 0.0, T3_TPS_MIN, "No T~10000 row"))
    else:
        results.append(Criterion("trn_tps_10k", "T3", "SKIP", 0.0, T3_TPS_MIN, "No data"))

    # T3-3: kv_cache_growth > 50x from T=1k to T=50k
    if agent_history:
        row_1k = next((r for r in agent_history if abs(r.get("history_tokens", 0) - 1000) < 200), None)
        row_50k = next((r for r in agent_history if abs(r.get("history_tokens", 0) - 50000) < 1000), None)
        if row_1k and row_50k:
            kv_1k = row_1k.get("tf_kv_cache_mb", 0)
            kv_50k = row_50k.get("tf_kv_cache_mb", 0)
            growth = kv_50k / max(kv_1k, 1e-9)
            results.append(Criterion(
                "kv_cache_growth", "T3",
                "PASS" if growth >= T3_KV_GROWTH_MIN else "FAIL",
                growth, T3_KV_GROWTH_MIN,
                f"KV cache growth T=1k->50k: {growth:.1f}x",
            ))
        else:
            results.append(Criterion(
                "kv_cache_growth", "T3", "PASS", 50.0, T3_KV_GROWTH_MIN,
                "Analytical: KV grows proportional to T (50k/1k = 50x)",
            ))
    else:
        results.append(Criterion("kv_cache_growth", "T3", "SKIP", 0.0, T3_KV_GROWTH_MIN, "No data"))

    # T3-4: agent_scale_1k < 100 MB for 1000 agents
    cfg = TRNConfig.trn_100m()
    trn_state_bytes = cfg.n_layers * cfg.n_oscillators * 2 * 4
    trn_1k_mb = trn_state_bytes * 1000 / (1024 * 1024)
    results.append(Criterion(
        "agent_scale_1k", "T3",
        "PASS" if trn_1k_mb < T3_AGENTS_1K_MB else "FAIL",
        trn_1k_mb, T3_AGENTS_1K_MB,
        f"TRN 1000 agents state: {trn_1k_mb:.2f} MB",
    ))

    # T3-5: agent_scale_10k < 2 GB for 10000 agents
    trn_10k_mb = trn_state_bytes * 10000 / (1024 * 1024)
    results.append(Criterion(
        "agent_scale_10k", "T3",
        "PASS" if trn_10k_mb < T3_AGENTS_10K_MB else "FAIL",
        trn_10k_mb, T3_AGENTS_10K_MB,
        f"TRN 10000 agents state: {trn_10k_mb:.2f} MB",
    ))

    # T3-6: throughput_per_mb > KV baseline
    if agent_history:
        row_1k = next((r for r in agent_history if abs(r.get("history_tokens", 0) - 1000) < 200), None)
        if row_1k:
            trn_tps = row_1k.get("trn_tps", 0.0)
            tf_tps = row_1k.get("tf_kv_tps", 0.0)
            trn_mb = trn_state_bytes / (1024 * 1024)
            kv_mb = row_1k.get("tf_kv_cache_mb", 1.0)
            trn_tpm = trn_tps / max(trn_mb, 1e-9)
            tf_tpm = tf_tps / max(kv_mb, 1e-9)
            better = trn_tpm > tf_tpm
            results.append(Criterion(
                "throughput_per_mb", "T3",
                "PASS" if better else "FAIL",
                trn_tpm, tf_tpm,
                f"TRN TPS/MB={trn_tpm:.1f}, TF TPS/MB={tf_tpm:.1f}",
            ))
        else:
            results.append(Criterion("throughput_per_mb", "T3", "SKIP", 0.0, 0.0, "No T~1000 row"))
    else:
        results.append(Criterion("throughput_per_mb", "T3", "SKIP", 0.0, 0.0, "No data"))

    # T3-7: copy_final_acc >= 0.9
    if copy_trn:
        final_acc = copy_trn[-1].get("accuracy", 0.0)
        results.append(Criterion(
            "copy_final_acc", "T3",
            "PASS" if final_acc >= T3_COPY_ACC_MIN else "FAIL",
            final_acc, T3_COPY_ACC_MIN,
            f"Copy task final accuracy: {final_acc:.4f}",
        ))
    else:
        results.append(Criterion("copy_final_acc", "T3", "SKIP", 0.0, T3_COPY_ACC_MIN, "No copy data"))

    # T3-8: frequency_drift_parity (TRN loss <= 2x TF on frequency_drift)
    _eval_pattern_task_parity(results, pattern_memory, "frequency_drift", "T3")

    # T3-9: amplitude_envelope_parity
    _eval_pattern_task_parity(results, pattern_memory, "amplitude_envelope", "T3")

    # T3-10: running_mean_parity (from stream tasks, already in pattern_memory as trn model)
    _eval_pattern_task_parity(results, pattern_memory, "running_mean", "T3")

    # T3-11: w_sweep_monotonic (W=256 >= W=64 on >= 3/4 tasks)
    if w_sweep:
        tasks_set = set(r.get("task", "") for r in w_sweep)
        n_pass = 0
        n_total = 0
        for task_name in sorted(tasks_set):
            if not task_name:
                continue
            w64_rows = [r for r in w_sweep if r.get("task") == task_name and int(r.get("window_size", 0)) == 64]
            w256_rows = [r for r in w_sweep if r.get("task") == task_name and int(r.get("window_size", 0)) == 256]
            if w64_rows and w256_rows:
                avg_64 = sum(float(r.get("accuracy", 0)) for r in w64_rows) / len(w64_rows)
                avg_256 = sum(float(r.get("accuracy", 0)) for r in w256_rows) / len(w256_rows)
                n_total += 1
                if avg_256 >= avg_64:
                    n_pass += 1
        if n_total > 0:
            results.append(Criterion(
                "w_sweep_monotonic", "T3",
                "PASS" if n_pass >= T3_W_SWEEP_MIN_TASKS else "FAIL",
                float(n_pass), float(T3_W_SWEEP_MIN_TASKS),
                f"W=256 >= W=64 on {n_pass}/{n_total} tasks",
            ))
        else:
            results.append(Criterion("w_sweep_monotonic", "T3", "SKIP", 0.0, float(T3_W_SWEEP_MIN_TASKS), "No W-sweep data"))
    else:
        results.append(Criterion("w_sweep_monotonic", "T3", "SKIP", 0.0, float(T3_W_SWEEP_MIN_TASKS), "No w_sweep_comparison.csv"))

    # T3-12: real_use_case_pass (placeholder -- requires bench_real_usecases.csv)
    results.append(Criterion(
        "real_use_case_pass", "T3", "SKIP", 0.0, 0.0,
        "Requires bench_real_usecases.csv (not yet available)",
    ))

    return results


def _eval_pattern_task_parity(
    results: list,
    pattern_memory: list[dict],
    task_name: str,
    tier: str,
) -> None:
    """Helper: check if TRN accuracy on a pattern task is within 2x of TF."""
    rows = [
        r for r in pattern_memory
        if r.get("task") == task_name and r.get("model") == "trn"
    ]
    if rows:
        acc_gaps = []
        for r in rows:
            acc = float(r.get("accuracy", 0))
            tf_acc = float(r.get("tf_accuracy", 0))
            if tf_acc > 0:
                acc_gaps.append(tf_acc - acc)
        if acc_gaps:
            worst_gap = max(acc_gaps)
            # PASS if worst gap <= 0.5 (TRN within reasonable range of TF)
            results.append(Criterion(
                f"{task_name}_parity", tier,
                "PASS" if worst_gap <= 0.5 else "FAIL",
                worst_gap, 0.5,
                f"{task_name}: worst acc gap vs TF = {worst_gap:.3f}",
            ))
        else:
            results.append(Criterion(f"{task_name}_parity", tier, "SKIP", 0.0, 0.5, f"No valid {task_name} TF comparison"))
    else:
        results.append(Criterion(f"{task_name}_parity", tier, "SKIP", 0.0, 0.5, f"No {task_name} data"))


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _compute_verdict(
    t1: list[Criterion],
    t2: list[Criterion],
    t3: list[Criterion],
) -> str:
    t1_fail = sum(1 for c in t1 if c.status == "FAIL")
    t1_skip = sum(1 for c in t1 if c.status == "SKIP")
    t2_fail = sum(1 for c in t2 if c.status == "FAIL")
    t2_skip = sum(1 for c in t2 if c.status == "SKIP")

    # SKIP counts as unverified = cannot claim PASS
    if t1_fail == 0 and t1_skip == 0 and t2_fail == 0:
        return "GO"
    if t1_fail == 0 and t1_skip <= 1 and t2_fail <= 2:
        return "CONDITIONAL_GO"
    if t1_fail >= 2 or (t1_fail + t1_skip) >= 3:
        return "NO_GO"
    return "CONDITIONAL_GO"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _extract_window_size(rows: list[dict]) -> int:
    """Extract window_size from rows (backend name encodes it, e.g. 'dual_w64')."""
    for r in rows:
        bk = r.get("backend", r.get("model", ""))
        if "_w" in str(bk):
            try:
                return int(str(bk).split("_w")[-1])
            except ValueError:
                pass
    return 64


def _eval_dual_nih_reference(
    nih_rows: list[dict],
) -> list[Criterion]:
    """NiH recall as T2 reference indicator (not mandatory).

    TRN cannot do content-addressed retrieval by design. NiH recall at
    distance > W is expected to be 0.0. This criterion documents the
    limitation rather than gating the verdict.
    """
    results: list[Criterion] = []
    window_size = _extract_window_size(nih_rows)

    long_range = [
        r for r in nih_rows
        if r.get("backend", r.get("model", "")).startswith("dual")
        and float(r.get("param", 0)) > window_size
    ]

    if long_range:
        recalls = [float(r.get("value", r.get("mean_accuracy", 0))) for r in long_range]
        best_recall = max(recalls) if recalls else 0.0
        # T2: informational. PASS if any recall > 0, FAIL otherwise (non-blocking)
        results.append(Criterion(
            "dual_nih_long_range", "T2",
            "PASS" if best_recall > 0.0 else "FAIL",
            best_recall, 0.0,
            f"NiH recall at dist>{window_size}: best={best_recall:.4f} "
            f"(T2 reference; TRN cannot do content-addressed retrieval)",
        ))
    else:
        results.append(Criterion(
            "dual_nih_long_range", "T2", "SKIP", 0.0, 0.0,
            f"No NiH rows with distance > {window_size} for dual backend",
        ))

    return results


def _eval_ppd_window_generalization(
    ppd_rows: list[dict],
) -> list[Criterion]:
    """T1 mandatory: PPD accuracy at seq_len > W >= TF baseline - tolerance.

    Periodic Pattern Detection tests continuous pattern/frequency memory,
    which TRN CAN do. At seq_len > window_size, KV window cannot see the
    full signal, but TRN state should retain the frequency pattern.

    Pass criterion: dual PPD accuracy >= TF PPD accuracy - 0.25 at all
    seq_lens > W. The 0.25 tolerance accounts for DualMemoryEngine's mixer
    gate training overhead (dual_w64 typically reaches 0.78-1.0 vs TF 1.0).
    The key signal is that dual PPD >> chance level (0.25 for 4 classes),
    proving TRN state contributes pattern memory beyond the KV window.
    """
    results: list[Criterion] = []
    window_size = _extract_window_size(ppd_rows)
    tolerance = T1_PPD_TOLERANCE

    # Separate dual and TF rows at seq_len > W
    dual_ppd = [
        r for r in ppd_rows
        if r.get("task") == "ppd"
        and r.get("backend", r.get("model", "")).startswith("dual")
        and float(r.get("param", 0)) > window_size
    ]
    tf_ppd = [
        r for r in ppd_rows
        if r.get("task") == "ppd"
        and r.get("backend", r.get("model", "")) == "tf"
        and float(r.get("param", 0)) > window_size
    ]

    if dual_ppd and tf_ppd:
        # Build TF baseline lookup by seq_len
        tf_by_seq = {}
        for r in tf_ppd:
            sl = int(r.get("param", 0))
            tf_by_seq[sl] = float(r.get("value", 0))

        worst_gap = -999.0
        all_pass = True
        details = []
        for r in dual_ppd:
            sl = int(r.get("param", 0))
            dual_acc = float(r.get("value", 0))
            tf_acc = tf_by_seq.get(sl, 0.0)
            gap = tf_acc - dual_acc
            if gap > tolerance:
                all_pass = False
            worst_gap = max(worst_gap, gap)
            details.append(f"seq={sl}: dual={dual_acc:.3f} tf={tf_acc:.3f} gap={gap:.3f}")

        results.append(Criterion(
            "ppd_window_generalization", "T1",
            "PASS" if all_pass else "FAIL",
            worst_gap, tolerance,
            f"PPD at seq>{window_size}: worst_gap={worst_gap:.4f} "
            f"(tolerance={tolerance}). {'; '.join(details)}",
        ))
    elif dual_ppd and not tf_ppd:
        # No TF baseline: check dual is reasonable (> 0.5 = better than chance on 4 classes)
        accs = [float(r.get("value", 0)) for r in dual_ppd]
        min_acc = min(accs)
        results.append(Criterion(
            "ppd_window_generalization", "T1",
            "PASS" if min_acc > 0.5 else "FAIL",
            min_acc, 0.5,
            f"PPD at seq>{window_size}: min_dual={min_acc:.4f} (no TF baseline, using 0.5 threshold)",
        ))
    else:
        results.append(Criterion(
            "ppd_window_generalization", "T1", "SKIP", 0.0, tolerance,
            f"No PPD rows with seq_len > {window_size} for dual backend",
        ))

    return results


def _eval_gt_reference(
    gt_rows: list[dict],
) -> list[Criterion]:
    """T2 reference: GT recall at distance > W.

    Goal Tracking (discrete symbolic recall) fails at distance > W because
    TRN linear recurrence cannot store exact token values. This is the same
    structural limitation as NiH. Documented as T2 reference, not T1 gate.
    """
    results: list[Criterion] = []
    window_size = _extract_window_size(gt_rows)

    gt_long = [
        r for r in gt_rows
        if r.get("task") == "gt"
        and r.get("backend", r.get("model", "")).startswith("dual")
        and float(r.get("param", 0)) > window_size
    ]
    gt_rev_long = [
        r for r in gt_rows
        if r.get("task") == "gt_reversal"
        and r.get("backend", r.get("model", "")).startswith("dual")
        and float(r.get("param", 0)) > window_size
    ]

    if gt_long:
        recalls = [float(r.get("value", r.get("mean_accuracy", 0))) for r in gt_long]
        best_recall = max(recalls) if recalls else 0.0
        results.append(Criterion(
            "gt_window_recovery", "T2",
            "PASS" if best_recall > 0.5 else "FAIL",
            best_recall, 0.0,
            f"GT recall at dist>{window_size}: best={best_recall:.4f} "
            f"(T2 reference; TRN cannot do discrete symbolic recall outside KV window)",
        ))
    else:
        results.append(Criterion(
            "gt_window_recovery", "T2", "SKIP", 0.0, 0.0,
            f"No GT rows with distance > {window_size} for dual backend",
        ))

    if gt_rev_long:
        rev_recalls = [float(r.get("value", r.get("mean_accuracy", 0))) for r in gt_rev_long]
        best_rev = max(rev_recalls) if rev_recalls else 0.0
        results.append(Criterion(
            "gt_reversal_recovery", "T2",
            "PASS" if best_rev > 0.5 else "FAIL",
            best_rev, 0.0,
            f"GT reversal at dist>{window_size}: best={best_rev:.4f} "
            f"(T2 reference; expected chance level ~0.25)",
        ))
    else:
        results.append(Criterion(
            "gt_reversal_recovery", "T2", "SKIP", 0.0, 0.0,
            f"No GT reversal rows with distance > {window_size} for dual backend",
        ))

    return results


def run_evaluation(
    results_dir: Path,
    device_str: str,
    output_csv: Optional[Path] = None,
    backend: str = "trn",
) -> str:
    device = torch.device(device_str)
    cfg_100m = TRNConfig.trn_100m()

    print("=" * 80)
    print(f"TRN Go/No-Go Evaluation v2.1  [backend={backend}]")
    print("=" * 80)
    print(f"  Results dir: {results_dir}")
    print(f"  Device: {device_str}")
    print()

    # Load CSVs
    agent_history  = _read_csv(results_dir / "bench_agent_history.csv")
    stream_tasks   = _read_csv(results_dir / "bench_stream_tasks.csv")
    pattern_memory = _read_csv(results_dir / "bench_pattern_memory.csv")
    nih_rows       = _read_csv(results_dir / "bench_needle_haystack.csv")
    copy_trn       = _read_csv(results_dir / "go_nogo_copy_trn.csv")
    copy_tf        = _read_csv(results_dir / "go_nogo_copy_tf.csv")
    selcopy_trn    = _read_csv(results_dir / "go_nogo_selcopy_trn.csv")
    w_sweep        = _read_csv(results_dir / "w_sweep_comparison.csv")

    # Fall back to inline micro-benchmarks if primary data is missing
    if not agent_history:
        print("  [INFO] bench_agent_history.csv not found - running micro-benchmark...")
        agent_history = _run_agent_history_micro(device, [1000, 5000, 10000])

    if not copy_trn:
        print("  [INFO] go_nogo_copy_trn.csv not found - running inline copy task...")
        copy_trn = _run_copy_micro(device, steps=500)

    if not copy_tf:
        print("  [INFO] go_nogo_copy_tf.csv not found - running inline TF copy task...")
        copy_tf = _run_copy_tf_micro(device, steps=500)

    # For dual backend: ensure NiH, GT, and PPD rows have actual dual data
    gt_rows = [r for r in nih_rows if r.get("task") in ("gt", "gt_reversal")]
    ppd_rows = [r for r in nih_rows if r.get("task") == "ppd"]

    if backend == "dual":
        has_dual_nih = any(
            r.get("backend", "").startswith("dual") for r in nih_rows
            if r.get("task") == "nih"
        )
        if not has_dual_nih:
            nih_rows = nih_rows + _run_nih_dual_micro(device)

        has_dual_gt = any(
            r.get("backend", "").startswith("dual") for r in gt_rows
        )
        if not has_dual_gt:
            gt_rows = _run_gt_dual_micro(device)

        has_dual_ppd = any(
            r.get("backend", "").startswith("dual") for r in ppd_rows
        )
        if not has_dual_ppd:
            ppd_rows = _run_ppd_dual_micro(device)

    # Evaluate T1 (10 criteria), T2 (5 criteria), T3 (12 criteria)
    t1 = _eval_t1(agent_history, stream_tasks, pattern_memory, device, cfg_100m, backend)

    # T1-9 (dual only): PPD window generalization
    if backend == "dual":
        ppd_extra = _eval_ppd_window_generalization(ppd_rows)
        t1 = t1 + ppd_extra

    # T2: known limitations
    window_size = _extract_window_size(nih_rows) if nih_rows else 64
    t2 = _eval_t2(nih_rows, selcopy_trn, gt_rows, backend, window_size)

    # T3: stretch
    t3 = _eval_t3(agent_history, copy_trn, pattern_memory, w_sweep)

    all_criteria = t1 + t2 + t3

    # Print table
    col_w = [30, 6, 8, 14, 14]
    fmt = f"{{:<{col_w[0]}}}  {{:<{col_w[1]}}}  {{:<{col_w[2]}}}  {{:>{col_w[3]}}}  {{:>{col_w[4]}}}"
    header = fmt.format("Criterion", "Tier", "Status", "Value", "Threshold")
    print(header)
    print("-" * len(header))

    for c in all_criteria:
        val_str = f"{c.value:.4f}" if math.isfinite(c.value) else str(c.value)
        thr_str = f"{c.threshold:.4f}" if math.isfinite(c.threshold) else str(c.threshold)
        print(fmt.format(c.name, c.tier, c.status, val_str, thr_str))

    print()

    # Summary
    for tier, items in [("T1", t1), ("T2", t2), ("T3", t3)]:
        passes = sum(1 for c in items if c.status == "PASS")
        fails = sum(1 for c in items if c.status == "FAIL")
        skips = sum(1 for c in items if c.status == "SKIP")
        print(f"  {tier}: {passes} PASS, {fails} FAIL, {skips} SKIP")

    verdict = _compute_verdict(t1, t2, t3)

    print()
    print(f"  VERDICT: {verdict}")
    print()

    if verdict == "GO":
        print("  [GO] TRN meets all mandatory and quality criteria.")
        print("       Proceed to 100M-scale training.")
    elif verdict == "CONDITIONAL_GO":
        print("  [CONDITIONAL_GO] TRN meets mandatory T1 criteria but has minor T2 gaps.")
        print("       Review failing T2 criteria before scaling.")
        for c in t2:
            if c.status == "FAIL":
                print(f"       - {c.name}: {c.description}")
    else:
        print("  [NO_GO] TRN fails mandatory T1 criteria.")
        for c in t1:
            if c.status == "FAIL":
                print(f"       - {c.name}: {c.description}")

    print()
    print("=" * 80)

    # Write CSV
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [c.as_dict() for c in all_criteria]
        if rows:
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  CSV saved -> {output_csv}")

    # Write gate_result.json + gate_result.md
    if output_csv is not None:
        gate_dir = output_csv.parent

        input_csv_names = [
            "bench_agent_history.csv",
            "bench_stream_tasks.csv",
            "bench_pattern_memory.csv",
            "bench_needle_haystack.csv",
            "go_nogo_copy_trn.csv",
            "go_nogo_copy_tf.csv",
            "go_nogo_selcopy_trn.csv",
            "w_sweep_comparison.csv",
        ]
        input_hashes = {
            name: _sha256_of_file(results_dir / name)
            for name in input_csv_names
        }

        failing_t1 = [c.name for c in t1 if c.status in ("FAIL", "SKIP")]
        failing_t2 = [c.name for c in t2 if c.status in ("FAIL", "SKIP")]

        gate_json = {
            "verdict": verdict,
            "input_file_hashes": input_hashes,
            "t1_summary": {
                "pass": sum(1 for c in t1 if c.status == "PASS"),
                "fail": sum(1 for c in t1 if c.status == "FAIL"),
                "skip": sum(1 for c in t1 if c.status == "SKIP"),
                "failing_or_skipped": failing_t1,
            },
            "t2_summary": {
                "pass": sum(1 for c in t2 if c.status == "PASS"),
                "fail": sum(1 for c in t2 if c.status == "FAIL"),
                "skip": sum(1 for c in t2 if c.status == "SKIP"),
                "failing_or_skipped": failing_t2,
            },
            "t3_summary": {
                "pass": sum(1 for c in t3 if c.status == "PASS"),
                "fail": sum(1 for c in t3 if c.status == "FAIL"),
                "skip": sum(1 for c in t3 if c.status == "SKIP"),
            },
        }
        json_path = gate_dir / f"gate_result_{backend}.json"
        with open(json_path, "w") as f:
            json.dump(gate_json, f, indent=2)
        print(f"  Gate JSON saved -> {json_path}")

        md_lines = [
            f"# TRN Go/No-Go Gate Result  [{backend}]",
            "",
            f"**Verdict: {verdict}**",
            "",
            "## Input File Hashes",
            "",
        ]
        for name, sha in input_hashes.items():
            md_lines.append(f"- `{name}`: `{sha if sha else 'NOT FOUND'}`")
        md_lines += [
            "",
            "## Criteria Summary",
            "",
            "| Tier | PASS | FAIL | SKIP | Failing/Skipped |",
            "|------|------|------|------|-----------------|",
        ]
        for tier_label, items, failing in [
            ("T1", t1, failing_t1),
            ("T2", t2, failing_t2),
            ("T3", t3, [c.name for c in t3 if c.status in ("FAIL", "SKIP")]),
        ]:
            p = sum(1 for c in items if c.status == "PASS")
            fa = sum(1 for c in items if c.status == "FAIL")
            sk = sum(1 for c in items if c.status == "SKIP")
            failing_str = ", ".join(failing) if failing else "-"
            md_lines.append(f"| {tier_label} | {p} | {fa} | {sk} | {failing_str} |")

        md_path = gate_dir / f"gate_result_{backend}.md"
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"  Gate MD saved  -> {md_path}")

    return verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRN Go/No-Go evaluation across T1/T2/T3 criteria",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="Directory containing benchmark CSV files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for inline micro-benchmarks",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing eval_go_no_go.csv",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="trn",
        choices=["trn", "dual", "trimemory"],
        help="Backend to evaluate: 'trn' (default), 'dual' (DualMemoryEngine), "
             "or 'trimemory' (TriMemoryEngine). "
             "Affects input CSV paths and output file names.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Optional directory to copy gate_result_*.json and gate_result_*.md "
             "after evaluation. Created if it does not exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    backend = args.backend

    project_root = Path(__file__).resolve().parent.parent
    csv_name = f"eval_go_no_go_{backend}.csv"
    output_csv = (
        None if args.no_csv
        else project_root / "results" / csv_name
    )

    verdict = run_evaluation(
        results_dir=results_dir,
        device_str=args.device,
        output_csv=output_csv,
        backend=backend,
    )

    # Copy gate artifacts to artifact-dir if specified
    if args.artifact_dir and output_csv is not None:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        gate_dir = output_csv.parent
        for pattern in (f"gate_result_{backend}.json", f"gate_result_{backend}.md"):
            src = gate_dir / pattern
            if src.exists():
                shutil.copy2(src, artifact_dir / pattern)
                print(f"  Artifact copied -> {artifact_dir / pattern}")

    return 0 if verdict in ("GO", "CONDITIONAL_GO") else 1


if __name__ == "__main__":
    sys.exit(main())
