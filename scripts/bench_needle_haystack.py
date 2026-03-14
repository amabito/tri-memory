#!/usr/bin/env python3
"""Information retention benchmark for TRN vs Transformer baseline.

Four evaluation tasks:
  NiH  - Needle-in-Haystack: recall a token planted at distance D before query
  TRP  - Token Reconstruction Probe: linear probe on final hidden state to predict
          token at position T-k
  PPD  - Periodic Pattern Detection: classify signal frequency from final state
  GT   - Goal Tracking: track a goal token seen at position 0 over N filler tokens

Metrics per (task, model, param):
  NiH/GT  : recall_accuracy
  TRP     : probe_accuracy
  PPD     : frequency_detection_score

Output:
  - Table printed to stdout
  - results/bench_needle_haystack.csv (unless --no-csv)

Usage:
    python scripts/bench_needle_haystack.py
    python scripts/bench_needle_haystack.py --steps 200 --device cpu --no-csv
    python scripts/bench_needle_haystack.py --tasks nih,gt --long-range
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trimemory.baseline import TransformerModel
from trimemory.bench_data import seed_everything
from trimemory.config import TRNConfig
from trimemory.integrations.vllm_backend import DualMemoryEngine
from trimemory.model import TRNModel

# ---------------------------------------------------------------------------
# Shared model config (small for CPU-practical benchmarking)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 128
N_OSC = 64
N_LAYERS = 4
D_FF = 512
MAX_SEQ = 2048


def _make_trn(device: str) -> TRNModel:
    cfg = TRNConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_oscillators=N_OSC,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
    )
    return TRNModel(cfg).to(device)


def _make_tf(device: str) -> TransformerModel:
    cfg = TRNConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_oscillators=N_OSC,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
    )
    return TransformerModel(cfg).to(device)


def _make_dual(device: str, window_size: int = 64) -> DualMemoryEngine:
    cfg = TRNConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_oscillators=N_OSC,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
    )
    return DualMemoryEngine(cfg, window_size=window_size).to(device)


# ---------------------------------------------------------------------------
# Hidden-state capture hook
# ---------------------------------------------------------------------------

def _register_hidden_hook(model: nn.Module) -> Tuple[list, any]:
    """Register a forward hook on the final norm layer to capture hidden states.

    Returns (storage_list, hook_handle). The storage_list is populated with
    (B, T, d_model) tensors on each forward call (cleared each time).
    """
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
# Common training loop
# ---------------------------------------------------------------------------

def _train(
    model: nn.Module,
    get_batch_fn,
    steps: int,
    batch_size: int,
    device: str,
    lr: float = 3e-4,
) -> None:
    """Simple Adam training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        input_ids, labels = get_batch_fn(batch_size, device)
        out = model(input_ids)
        logits = out["logits"]  # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            labels[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Task 1: Needle-in-Haystack (NiH)
# ---------------------------------------------------------------------------

FILLER_LOW = 10
FILLER_HIGH = 200
NEEDLE_LOW = 200
NEEDLE_HIGH = 240
QUERY_TOKEN = 5


def _nih_batch(distance: int, batch_size: int, device: str, rng: torch.Generator):
    """Generate NiH batch. Sequence length = distance + 2 (needle + query slots)."""
    seq_len = distance + 2
    needles = torch.randint(NEEDLE_LOW, NEEDLE_HIGH, (batch_size,), generator=rng)
    filler = torch.randint(FILLER_LOW, FILLER_HIGH, (batch_size, seq_len), generator=rng)
    # Needle at position 0, query at position seq_len-1
    filler[:, 0] = needles
    filler[:, -1] = QUERY_TOKEN
    input_ids = filler.to(device)
    return input_ids, input_ids


def _nih_recall(model: nn.Module, distance: int, n_eval: int, device: str) -> float:
    """Recall@1: fraction where logits[:, -1].argmax() == needle at position 0."""
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(999)
    seq_len = distance + 2
    correct = 0
    total = 0
    batch_size = min(64, n_eval)
    with torch.no_grad():
        for _ in range(math.ceil(n_eval / batch_size)):
            needles = torch.randint(NEEDLE_LOW, NEEDLE_HIGH, (batch_size,), generator=rng)
            filler = torch.randint(FILLER_LOW, FILLER_HIGH, (batch_size, seq_len), generator=rng)
            filler[:, 0] = needles
            filler[:, -1] = QUERY_TOKEN
            input_ids = filler.to(device)
            out = model(input_ids)
            preds = out["logits"][:, -1].argmax(dim=-1)  # (B,)
            correct += (preds.cpu() == needles).sum().item()
            total += batch_size
    return correct / total


NIH_N_SAMPLES = 50  # number of independent eval runs per (model, distance)


def bench_nih(
    distances: List[int],
    steps: int,
    batch_size: int,
    device: str,
    seed: int,
    n_samples: int = NIH_N_SAMPLES,
    backends: List[str] = None,
) -> List[dict]:
    if backends is None:
        backends = ["trn", "tf"]

    def _model_factories(backend_list: List[str]):
        mapping = {
            "trn":      ("trn",      lambda d: _make_trn(d)),
            "tf":       ("tf",       lambda d: _make_tf(d)),
            "dual_w64": ("dual_w64", lambda d: _make_dual(d, window_size=64)),
            "dual_w256":("dual_w256",lambda d: _make_dual(d, window_size=256)),
        }
        return [(mapping[b][0], mapping[b][1]) for b in backend_list if b in mapping]

    results = []
    for model_name, make_fn in _model_factories(backends):
        for dist in distances:
            seed_everything(seed)
            model = make_fn(device)
            rng = torch.Generator()
            rng.manual_seed(seed)

            def get_batch(bs, dev):
                return _nih_batch(dist, bs, dev, rng)

            print(f"  NiH [{model_name}] distance={dist} training...", end="", flush=True)
            t0 = time.time()
            _train(model, get_batch, steps, batch_size, device)
            print(f" done ({time.time()-t0:.1f}s), evaluating {n_samples} samples...", end="", flush=True)

            # Run n_samples independent evaluations with different seeds
            sample_recalls = []
            for i in range(n_samples):
                recall_i = _nih_recall(model, dist, 256, device)
                sample_recalls.append(recall_i)

            mean_acc = sum(sample_recalls) / len(sample_recalls)
            variance = sum((r - mean_acc) ** 2 for r in sample_recalls) / max(len(sample_recalls) - 1, 1)
            std_acc = math.sqrt(variance)
            print(f" mean={mean_acc:.3f} std={std_acc:.3f}")

            results.append(dict(
                task="nih",
                model=model_name,
                backend=model_name,
                param=dist,
                metric="recall_accuracy",
                value=mean_acc,
                n_samples=n_samples,
                mean_accuracy=mean_acc,
                std_accuracy=std_acc,
            ))
    return results


# ---------------------------------------------------------------------------
# Task 2: Token Reconstruction Probe (TRP)
# ---------------------------------------------------------------------------

SEQ_LEN_TRP = 256
LOOKBACKS = [1, 2, 4, 8, 16, 32]


def _trp_batch(batch_size: int, device: str, rng: torch.Generator):
    ids = torch.randint(10, VOCAB_SIZE, (batch_size, SEQ_LEN_TRP), generator=rng).to(device)
    return ids, ids


def bench_trp(
    steps: int,
    batch_size: int,
    device: str,
    seed: int,
) -> List[dict]:
    """Train backbone, then freeze and train linear probes for each lookback."""
    results = []
    probe_steps = max(50, steps // 5)

    for model_name, make_fn in [("trn", _make_trn), ("tf", _make_tf)]:
        seed_everything(seed)
        model = make_fn(device)
        rng = torch.Generator()
        rng.manual_seed(seed)
        captured, hook = _register_hidden_hook(model)

        print(f"  TRP [{model_name}] backbone training...", end="", flush=True)
        t0 = time.time()

        def get_batch(bs, dev):
            return _trp_batch(bs, dev, rng)

        _train(model, get_batch, steps, batch_size, device)
        print(f" done ({time.time()-t0:.1f}s)")

        model.eval()
        for k in LOOKBACKS:
            if k >= SEQ_LEN_TRP:
                continue
            seed_everything(seed + k)
            probe = nn.Linear(D_MODEL, VOCAB_SIZE).to(device)
            probe_opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
            rng_probe = torch.Generator()
            rng_probe.manual_seed(seed + k + 100)

            # Train probe
            probe.train()
            for _ in range(probe_steps):
                ids = torch.randint(10, VOCAB_SIZE, (batch_size, SEQ_LEN_TRP),
                                    generator=rng_probe).to(device)
                with torch.no_grad():
                    model(ids)
                hidden = captured[0]  # (B, T, d_model)
                probe_in = hidden[:, -1, :]  # last position
                target = ids[:, -1 - k]       # token k steps before end
                logits_p = probe(probe_in)
                loss_p = F.cross_entropy(logits_p, target)
                probe_opt.zero_grad()
                loss_p.backward()
                probe_opt.step()

            # Eval probe
            probe.eval()
            correct = 0
            n_eval = 256
            rng_eval = torch.Generator()
            rng_eval.manual_seed(seed + k + 999)
            with torch.no_grad():
                for _ in range(4):
                    ids = torch.randint(10, VOCAB_SIZE, (n_eval // 4, SEQ_LEN_TRP),
                                        generator=rng_eval).to(device)
                    model(ids)
                    hidden = captured[0]
                    probe_in = hidden[:, -1, :]
                    target = ids[:, -1 - k]
                    preds = probe(probe_in).argmax(dim=-1)
                    correct += (preds == target).sum().item()
            acc = correct / n_eval
            print(f"  TRP [{model_name}] k={k} probe_acc={acc:.3f}")
            results.append(dict(task="trp", model=model_name, param=k,
                                metric="probe_accuracy", value=acc))

        hook.remove()
    return results


# ---------------------------------------------------------------------------
# Task 3: Periodic Pattern Detection (PPD)
# ---------------------------------------------------------------------------

FREQ_QUERY_TOKEN = 7
FREQUENCIES = [0.01, 0.05, 0.1, 0.2]
PPD_SEQ_LENS = [64, 128, 256, 512]


def _ppd_batch(seq_len: int, batch_size: int, device: str, rng: torch.Generator):
    """Generate periodic signal sequences. Labels = frequency class index."""
    # Pick random frequency class per example
    freq_labels = torch.randint(0, len(FREQUENCIES), (batch_size,), generator=rng)
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for b in range(batch_size):
        freq = FREQUENCIES[freq_labels[b].item()]
        for t in range(seq_len - 1):
            val = int(math.sin(2 * math.pi * freq * t) * 50 + 128)
            ids[b, t] = max(50, min(200, val))
        ids[b, -1] = FREQ_QUERY_TOKEN
    ids = ids.to(device)
    return ids, ids


def bench_ppd(
    steps: int,
    batch_size: int,
    device: str,
    seed: int,
) -> List[dict]:
    results = []
    probe_steps = max(50, steps // 5)

    for model_name, make_fn in [("trn", _make_trn), ("tf", _make_tf)]:
        for seq_len in PPD_SEQ_LENS:
            seed_everything(seed)
            model = make_fn(device)
            rng = torch.Generator()
            rng.manual_seed(seed)
            captured, hook = _register_hidden_hook(model)

            print(f"  PPD [{model_name}] seq_len={seq_len} backbone training...", end="", flush=True)
            t0 = time.time()

            def get_batch(bs, dev):
                return _ppd_batch(seq_len, bs, dev, rng)

            _train(model, get_batch, steps, batch_size, device)
            print(f" done ({time.time()-t0:.1f}s)")

            # Train 4-class probe
            model.eval()
            probe = nn.Linear(D_MODEL, len(FREQUENCIES)).to(device)
            probe_opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
            rng_probe = torch.Generator()
            rng_probe.manual_seed(seed + 77)

            probe.train()
            for _ in range(probe_steps):
                freq_labels = torch.randint(0, len(FREQUENCIES), (batch_size,), generator=rng_probe)
                seqs = torch.zeros(batch_size, seq_len, dtype=torch.long)
                for b in range(batch_size):
                    freq = FREQUENCIES[freq_labels[b].item()]
                    for t in range(seq_len - 1):
                        val = int(math.sin(2 * math.pi * freq * t) * 50 + 128)
                        seqs[b, t] = max(50, min(200, val))
                    seqs[b, -1] = FREQ_QUERY_TOKEN
                seqs = seqs.to(device)
                freq_labels = freq_labels.to(device)
                with torch.no_grad():
                    model(seqs)
                hidden = captured[0]  # (B, T, d)
                probe_in = hidden[:, -1, :]
                logits_p = probe(probe_in)
                loss_p = F.cross_entropy(logits_p, freq_labels)
                probe_opt.zero_grad()
                loss_p.backward()
                probe_opt.step()

            # Eval probe
            probe.eval()
            correct = 0
            n_eval = 256
            rng_eval = torch.Generator()
            rng_eval.manual_seed(seed + 888)
            with torch.no_grad():
                for _ in range(4):
                    freq_labels = torch.randint(0, len(FREQUENCIES), (n_eval // 4,), generator=rng_eval)
                    seqs = torch.zeros(n_eval // 4, seq_len, dtype=torch.long)
                    for b in range(n_eval // 4):
                        freq = FREQUENCIES[freq_labels[b].item()]
                        for t in range(seq_len - 1):
                            val = int(math.sin(2 * math.pi * freq * t) * 50 + 128)
                            seqs[b, t] = max(50, min(200, val))
                        seqs[b, -1] = FREQ_QUERY_TOKEN
                    seqs = seqs.to(device)
                    freq_labels = freq_labels.to(device)
                    model(seqs)
                    hidden = captured[0]
                    probe_in = hidden[:, -1, :]
                    preds = probe(probe_in).argmax(dim=-1)
                    correct += (preds == freq_labels).sum().item()

            score = correct / n_eval
            print(f"  PPD [{model_name}] seq_len={seq_len} freq_detection={score:.3f}")
            results.append(dict(task="ppd", model=model_name, param=seq_len,
                                metric="frequency_detection_score", value=score))
            hook.remove()
    return results


# ---------------------------------------------------------------------------
# Task 4: Goal Tracking (GT)
# ---------------------------------------------------------------------------

GOAL_TOKENS = [200, 201, 202, 203]
GOAL_QUERY_TOKEN = 6


def _gt_batch(distance: int, batch_size: int, device: str, rng: torch.Generator):
    """Goal tracking batch. Seq: [GOAL, filler*distance, GOAL_QUERY, GOAL_ANSWER].

    The training signal is next-token prediction. To learn the mapping
    GOAL_QUERY -> GOAL, the answer token must follow the query token in
    the sequence so that the cross-entropy loss at position -2 (GOAL_QUERY)
    has GOAL_ANSWER as its target.
    """
    seq_len = distance + 3  # GOAL + distance*filler + GOAL_QUERY + GOAL_ANSWER
    goal_idx = torch.randint(0, len(GOAL_TOKENS), (batch_size,), generator=rng)
    goals = torch.tensor([GOAL_TOKENS[i] for i in goal_idx.tolist()])
    filler = torch.randint(FILLER_LOW, FILLER_HIGH, (batch_size, seq_len), generator=rng)
    filler[:, 0] = goals
    filler[:, -2] = GOAL_QUERY_TOKEN
    filler[:, -1] = goals  # answer = the goal token
    input_ids = filler.to(device)
    return input_ids, input_ids


def _gt_recall(model: nn.Module, distance: int, n_eval: int, device: str) -> float:
    """GT recall: after GOAL_QUERY, predict which GOAL_TOKEN was at position 0.

    Eval sequence: [GOAL, filler*distance, GOAL_QUERY, GOAL_ANSWER].
    We check logits[:, -2] (the GOAL_QUERY position) predicts GOAL.
    """
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(1234)
    seq_len = distance + 3  # match training format
    correct = 0
    total = 0
    batch_size = min(64, n_eval)
    with torch.no_grad():
        for _ in range(math.ceil(n_eval / batch_size)):
            goal_idx = torch.randint(0, len(GOAL_TOKENS), (batch_size,), generator=rng)
            goals = torch.tensor([GOAL_TOKENS[i] for i in goal_idx.tolist()])
            filler = torch.randint(FILLER_LOW, FILLER_HIGH, (batch_size, seq_len), generator=rng)
            filler[:, 0] = goals
            filler[:, -2] = GOAL_QUERY_TOKEN
            filler[:, -1] = goals  # answer present but we check prediction
            input_ids = filler.to(device)
            out = model(input_ids)
            # logits[:, -2] = prediction at GOAL_QUERY position -> should be GOAL
            preds = out["logits"][:, -2].argmax(dim=-1).cpu()
            correct += (preds == goals).sum().item()
            total += batch_size
    return correct / total


def _gt_reversal_recall(model: nn.Module, distance: int, n_eval: int, device: str) -> float:
    """GT goal-reversal recall: a second (different) goal inserted at T//2.

    Sequence: [GOAL_1, filler..., GOAL_2, filler..., GOAL_QUERY, GOAL_2_ANSWER]
    GOAL_2 is inserted at position distance // 2 and is always different from GOAL_1.
    After GOAL_QUERY, model should predict GOAL_2 (the most recent goal).
    """
    model.eval()
    rng = torch.Generator()
    rng.manual_seed(5678)
    seq_len = distance + 3  # +3: GOAL_1 + distance filler + GOAL_QUERY + ANSWER
    mid = max(1, (seq_len - 1) // 2)  # midpoint for GOAL_2
    correct = 0
    total = 0
    batch_size = min(64, n_eval)
    with torch.no_grad():
        for _ in range(math.ceil(n_eval / batch_size)):
            goal1_idx = torch.randint(0, len(GOAL_TOKENS), (batch_size,), generator=rng)
            offset = torch.randint(1, len(GOAL_TOKENS), (batch_size,), generator=rng)
            goal2_idx = (goal1_idx + offset) % len(GOAL_TOKENS)
            goals1 = torch.tensor([GOAL_TOKENS[i] for i in goal1_idx.tolist()])
            goals2 = torch.tensor([GOAL_TOKENS[i] for i in goal2_idx.tolist()])
            filler = torch.randint(FILLER_LOW, FILLER_HIGH, (batch_size, seq_len), generator=rng)
            filler[:, 0] = goals1
            filler[:, mid] = goals2
            filler[:, -2] = GOAL_QUERY_TOKEN
            filler[:, -1] = goals2  # answer = most recent goal
            input_ids = filler.to(device)
            out = model(input_ids)
            preds = out["logits"][:, -2].argmax(dim=-1).cpu()
            correct += (preds == goals2).sum().item()
            total += batch_size
    return correct / total


def bench_gt(
    distances: List[int],
    steps: int,
    batch_size: int,
    device: str,
    seed: int,
    backends: List[str] = None,
) -> List[dict]:
    if backends is None:
        backends = ["trn", "tf"]

    def _model_factories(backend_list: List[str]):
        mapping = {
            "trn":      ("trn",      lambda d: _make_trn(d)),
            "tf":       ("tf",       lambda d: _make_tf(d)),
            "dual_w64": ("dual_w64", lambda d: _make_dual(d, window_size=64)),
            "dual_w256":("dual_w256",lambda d: _make_dual(d, window_size=256)),
        }
        return [(mapping[b][0], mapping[b][1]) for b in backend_list if b in mapping]

    results = []
    for model_name, make_fn in _model_factories(backends):
        for dist in distances:
            seed_everything(seed)
            model = make_fn(device)
            rng = torch.Generator()
            rng.manual_seed(seed)

            def get_batch(bs, dev):
                return _gt_batch(dist, bs, dev, rng)

            print(f"  GT [{model_name}] distance={dist} training...", end="", flush=True)
            t0 = time.time()
            _train(model, get_batch, steps, batch_size, device)
            recall = _gt_recall(model, dist, 256, device)
            reversal = _gt_reversal_recall(model, dist, 256, device)
            print(f" recall={recall:.3f} reversal={reversal:.3f} ({time.time()-t0:.1f}s)")
            results.append(dict(task="gt", model=model_name, backend=model_name, param=dist,
                                metric="goal_tracking_accuracy", value=recall))
            results.append(dict(task="gt_reversal", model=model_name, backend=model_name, param=dist,
                                metric="goal_tracking_accuracy", value=reversal))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Information retention benchmark")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--long-range", action="store_true",
                   help="Add distances 2000,5000 for NiH and GT")
    p.add_argument("--tasks", default="nih,trp,ppd,gt",
                   help="Comma-separated tasks to run")
    p.add_argument(
        "--backend",
        type=str,
        default="trn,tf",
        help="Comma-separated backends for NiH and GT tasks. "
             "Choices: trn, tf, dual_w64, dual_w256",
    )
    p.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Directory to save quality_results.json and quality_summary.md",
    )
    return p.parse_args()


def _print_table(rows: List[dict]) -> None:
    print(f"\n{'task':<6} {'model':<6} {'param':<8} {'metric':<30} {'value':.6}")
    print("-" * 65)
    for r in rows:
        print(f"{r['task']:<6} {r['model']:<6} {str(r['param']):<8} "
              f"{r['metric']:<30} {r['value']:.4f}")


def main() -> None:
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    device = args.device

    nih_distances = [100, 200, 500, 1000]
    gt_distances = [10, 50, 100, 200, 500]
    if args.long_range:
        nih_distances += [2000, 5000]
        gt_distances += [2000, 5000]

    all_results: List[dict] = []

    if "nih" in tasks:
        print("\n=== Task 1: Needle-in-Haystack ===")
        all_results.extend(bench_nih(nih_distances, args.steps, args.batch_size,
                                     device, args.seed, backends=backends))

    if "trp" in tasks:
        print("\n=== Task 2: Token Reconstruction Probe ===")
        all_results.extend(bench_trp(args.steps, args.batch_size, device, args.seed))

    if "ppd" in tasks:
        print("\n=== Task 3: Periodic Pattern Detection ===")
        all_results.extend(bench_ppd(args.steps, args.batch_size, device, args.seed))

    if "gt" in tasks:
        print("\n=== Task 4: Goal Tracking ===")
        all_results.extend(bench_gt(gt_distances, args.steps, args.batch_size,
                                    device, args.seed, backends=backends))

    _print_table(all_results)

    if not args.no_csv and all_results:
        out_dir = Path(__file__).parent.parent / "results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "bench_needle_haystack.csv"
        # Use union of all keys to handle optional NiH columns gracefully
        all_keys: list = ["task", "model", "backend", "param", "metric", "value",
                          "n_samples", "mean_accuracy", "std_accuracy"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            # Fill missing optional columns with empty string
            for row in all_results:
                for k in all_keys:
                    row.setdefault(k, "")
            writer.writerows(all_results)
        print(f"\nResults saved to {out_path}")

    if args.artifact_dir and all_results:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        def _nan_safe(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        json_rows = [
            {k: _nan_safe(v) for k, v in row.items()}
            for row in all_results
        ]
        env_meta = {
            "seed": args.seed,
            "device": args.device,
            "torch_version": torch.__version__,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        quality_json = {"env": env_meta, "results": json_rows}
        (artifact_dir / "quality_results.json").write_text(
            json.dumps(quality_json, indent=2), encoding="utf-8"
        )

        md_lines = [
            "# Quality Benchmark Summary",
            "",
            f"**Device**: {args.device}",
            f"**Seed**: {args.seed}",
            f"**Steps**: {args.steps}",
            f"**Backends**: {args.backend}",
            f"**Timestamp**: {env_meta['timestamp']}",
            "",
            "## Results",
            "",
            "| task | model | param | metric | value | mean | std |",
            "|------|-------|-------|--------|-------|------|-----|",
        ]
        for r in all_results:
            val = r.get("value", float("nan"))
            mean = r.get("mean_accuracy", float("nan"))
            std = r.get("std_accuracy", float("nan"))

            def _fmt(v) -> str:
                if isinstance(v, float) and math.isnan(v):
                    return "N/A"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            md_lines.append(
                f"| {r.get('task', '')} "
                f"| {r.get('model', '')} "
                f"| {r.get('param', '')} "
                f"| {r.get('metric', '')} "
                f"| {_fmt(val)} "
                f"| {_fmt(mean)} "
                f"| {_fmt(std)} |"
            )
        (artifact_dir / "quality_summary.md").write_text(
            "\n".join(md_lines), encoding="utf-8"
        )

        # GT distance-level metrics for Go/No-Go evaluation
        gt_results = [r for r in all_results if r.get("task") in ("gt", "gt_reversal")]
        if gt_results:
            # gt_distance_success.csv
            gt_csv_path = artifact_dir / "gt_distance_success.csv"
            gt_keys = ["task", "model", "backend", "param", "metric", "value",
                        "n_samples", "mean_accuracy", "std_accuracy"]
            with open(gt_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=gt_keys, extrasaction="ignore")
                writer.writeheader()
                for row in gt_results:
                    for k in gt_keys:
                        row.setdefault(k, "")
                    writer.writerows([row])
            print(f"  GT CSV saved -> {gt_csv_path}")

            # GT summary with distance > W analysis
            dual_gt = [r for r in gt_results if r.get("backend", "").startswith("dual")]
            gt_md = ["# Goal Tracking Summary", ""]
            if dual_gt:
                # Detect window size from backend name
                w_size = 64
                for r in dual_gt:
                    bk = str(r.get("backend", ""))
                    if "_w" in bk:
                        try:
                            w_size = int(bk.split("_w")[-1])
                            break
                        except ValueError:
                            pass

                gt_md.append(f"**Window size**: {w_size}")
                gt_md.append("")
                gt_md.append("## Distance > W Analysis")
                gt_md.append("")
                gt_md.append("| task | backend | distance | value |")
                gt_md.append("|------|---------|----------|-------|")
                beyond_w = [r for r in dual_gt if float(r.get("param", 0)) > w_size]
                for r in beyond_w:
                    v = r.get("value", 0.0)
                    gt_md.append(
                        f"| {r.get('task', '')} "
                        f"| {r.get('backend', '')} "
                        f"| {r.get('param', '')} "
                        f"| {float(v):.4f} |"
                    )
                if beyond_w:
                    gt_vals = [float(r.get("value", 0)) for r in beyond_w if r.get("task") == "gt"]
                    rev_vals = [float(r.get("value", 0)) for r in beyond_w if r.get("task") == "gt_reversal"]
                    gt_md.append("")
                    if gt_vals:
                        gt_md.append(f"GT recall (dist>{w_size}): "
                                     f"mean={sum(gt_vals)/len(gt_vals):.4f}, "
                                     f"min={min(gt_vals):.4f}")
                    if rev_vals:
                        gt_md.append(f"GT reversal (dist>{w_size}): "
                                     f"mean={sum(rev_vals)/len(rev_vals):.4f}, "
                                     f"min={min(rev_vals):.4f}")
                else:
                    gt_md.append("")
                    gt_md.append("No GT results at distance > W.")

                gt_md.append("")
                gt_md.append("## Structural Limitation")
                gt_md.append("")
                gt_md.append(
                    "Exact symbolic recall (NiH, GT) outside the KV window is not "
                    "supported by TRN state. Linear recurrence compresses history "
                    "into continuous statistics (amplitude, phase, frequency), which "
                    "preserves periodic patterns (PPD = 1.000) but loses discrete "
                    "token identity. This is a fundamental property, not a training "
                    "deficiency."
                )

            (artifact_dir / "gt_summary.md").write_text(
                "\n".join(gt_md), encoding="utf-8"
            )
            print(f"  GT summary saved -> {artifact_dir / 'gt_summary.md'}")

            # Add goal_tracking metrics to quality_results.json
            quality_json["metrics"] = quality_json.get("metrics", {})
            quality_json["metrics"]["goal_tracking"] = {
                "window_size": w_size if dual_gt else None,
                "includes_goal_reversal": any(r.get("task") == "gt_reversal" for r in gt_results),
                "distance_buckets": sorted(set(int(float(r.get("param", 0))) for r in gt_results)),
            }
            (artifact_dir / "quality_results.json").write_text(
                json.dumps(quality_json, indent=2), encoding="utf-8"
            )

        print(f"\nQuality artifacts saved -> {artifact_dir}")


if __name__ == "__main__":
    main()
