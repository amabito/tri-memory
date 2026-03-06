#!/usr/bin/env python3
"""Tri-Memory V2 Evaluation: task & loss redesign for memory path differentiation.

Changes from V1:
  - Query-weighted loss (answer/fact/pattern positions weighted 4-8x)
  - Multi-token old facts (2-5 tokens, not 1)
  - Stronger pattern task (periodic + regime shift + delayed induction)
  - Salient event (rare token burst outside KV window)
  - Revised composite: 0.20*recent + 0.30*old_fact_span + 0.30*pattern_span + 0.20*salient
  - Role-specific metrics: span exact accuracy, token accuracy

Usage:
    python scripts/run_trimemory_v2_eval.py --steps 1000 --seeds 0 1
    python scripts/run_trimemory_v2_eval.py --steps 1000 --seeds 0 1 --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_src = os.path.join(os.path.dirname(__file__), "..", "src")
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _root]
sys.path.insert(0, _src)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from trn.config import TRNConfig
from trn.tri_memory import TriMemoryEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE = 64
CHUNK_SIZE = 32
D_MODEL = 128
N_LAYERS = 4
N_OSC = 64
D_FF = 512
VOCAB_SIZE = 256
SEQ_LEN = 256
BATCH_SIZE = 16
LR = 3e-4

# Token ranges
FILLER_LOW = 100
FILLER_HIGH = 200
RECENT_VALUE_LOW = 200
RECENT_VALUE_HIGH = 220
OLD_FACT_LOW = 220
OLD_FACT_HIGH = 240
PATTERN_TOKEN_LOW = 10
PATTERN_TOKEN_HIGH = 60
SALIENT_TOKEN_LOW = 240
SALIENT_TOKEN_HIGH = 256

# Query markers (outside filler range)
QUERY_OLD_FACT = 5
QUERY_RECENT = 6
QUERY_PATTERN = 7
QUERY_SALIENT = 8

# Loss weights
W_NORMAL = 1.0
W_QUERY = 4.0
W_ANSWER = 8.0
W_FACT_SPAN = 8.0
W_PATTERN_TARGET = 6.0
W_SALIENT = 8.0

# Old fact span length
OLD_FACT_SPAN_MIN = 3
OLD_FACT_SPAN_MAX = 5

# Pattern config
PATTERN_PERIOD = 5
PATTERN_BLOCK_START = 30
PATTERN_BLOCK_END = 80
PATTERN_REGIME_SHIFT_POS = 55  # regime shift midway through pattern block

# Salient event config
SALIENT_POS_START = 10
SALIENT_POS_END = 15  # 5-token salient burst near position 10

# Recent values
RECENT_POS = -16  # relative to end of sequence (inside window)


# ---------------------------------------------------------------------------
# V2 Dataset
# ---------------------------------------------------------------------------
class MixedMemoryDatasetV2(Dataset):
    """Redesigned dataset with stronger signals for each memory tier.

    Sequence layout (SEQ_LEN=256):
      [0..4]:      old_fact span (3-5 tokens from OLD_FACT range)
      [10..14]:    salient event (5 rare tokens)
      [30..79]:    pattern block with regime shift at 55
      [100..238]:  filler
      [240]:       recent value
      [250]:       QUERY_OLD_FACT
      [251..253]:  old_fact span answer (3 tokens min)
      [254]:       QUERY_RECENT  -> answer at [255] (truncated if needed)
      -- Actually let's pack queries at the end more carefully --

    Query region (last 12 tokens, positions 244-255):
      244: QUERY_OLD_FACT
      245-249: old_fact answer span (up to 5 tokens)
      249: QUERY_PATTERN
      250-254: pattern induction answer (next 5 tokens of pattern)
      -- This gets complex. Let me simplify. --

    Simplified query region (last 10 positions = 246-255):
      246: QUERY_OLD_FACT
      247: old_fact_token[0]
      248: old_fact_token[1]
      249: old_fact_token[2]  (3-token span answer)
      250: QUERY_RECENT
      251: recent_val
      252: QUERY_PATTERN
      253: pattern_next  (next token after pattern)
      254: QUERY_SALIENT
      255: salient_token[0]
    """

    QUERY_REGION_SIZE = 10

    def __init__(
        self,
        n_samples: int = 2000,
        seq_len: int = SEQ_LEN,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        # Fixed old fact span length for this dataset instance
        self.old_fact_span_len = 3

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        seq = torch.randint(FILLER_LOW, FILLER_HIGH, (self.seq_len,), generator=self.rng)

        # Build loss weight mask (all start at W_NORMAL)
        loss_weights = torch.full((self.seq_len,), W_NORMAL, dtype=torch.float32)

        # --- Old fact span at positions 0..2 (3 tokens) ---
        old_fact_tokens = torch.randint(
            OLD_FACT_LOW, OLD_FACT_HIGH, (self.old_fact_span_len,), generator=self.rng
        )
        for i in range(self.old_fact_span_len):
            seq[i] = old_fact_tokens[i]
            loss_weights[i] = W_FACT_SPAN

        # --- Salient event at positions 10..14 (5 rare tokens) ---
        salient_tokens = torch.randint(
            SALIENT_TOKEN_LOW, SALIENT_TOKEN_HIGH, (5,), generator=self.rng
        )
        for i in range(5):
            seq[SALIENT_POS_START + i] = salient_tokens[i]
            loss_weights[SALIENT_POS_START + i] = W_SALIENT

        # --- Pattern block at positions 30..79 ---
        # Phase 1: period-5 pattern from 30..54
        pattern_a = torch.randint(
            PATTERN_TOKEN_LOW, PATTERN_TOKEN_HIGH, (PATTERN_PERIOD,), generator=self.rng
        )
        for i in range(PATTERN_BLOCK_START, PATTERN_REGIME_SHIFT_POS):
            seq[i] = pattern_a[i % PATTERN_PERIOD]
            loss_weights[i] = W_PATTERN_TARGET

        # Phase 2: different period-5 pattern from 55..79 (regime shift)
        pattern_b = torch.randint(
            PATTERN_TOKEN_LOW, PATTERN_TOKEN_HIGH, (PATTERN_PERIOD,), generator=self.rng
        )
        for i in range(PATTERN_REGIME_SHIFT_POS, PATTERN_BLOCK_END):
            seq[i] = pattern_b[(i - PATTERN_REGIME_SHIFT_POS) % PATTERN_PERIOD]
            loss_weights[i] = W_PATTERN_TARGET

        # Pattern answer: next token in pattern_b cycle after position 79
        # The model should predict what comes next in the most recent pattern
        pattern_next_idx = (PATTERN_BLOCK_END - PATTERN_REGIME_SHIFT_POS) % PATTERN_PERIOD
        pattern_answer = pattern_b[pattern_next_idx].item()

        # --- Recent value at position seq_len - 16 (inside KV window) ---
        recent_pos = self.seq_len + RECENT_POS  # e.g., 240
        recent_val = torch.randint(
            RECENT_VALUE_LOW, RECENT_VALUE_HIGH, (1,), generator=self.rng
        ).item()
        seq[recent_pos] = recent_val

        # --- Query region (last 10 positions: 246-255) ---
        qstart = self.seq_len - self.QUERY_REGION_SIZE  # 246

        # Q1: old fact recall
        seq[qstart] = QUERY_OLD_FACT
        loss_weights[qstart] = W_QUERY
        for i in range(self.old_fact_span_len):
            seq[qstart + 1 + i] = old_fact_tokens[i]
            loss_weights[qstart + 1 + i] = W_ANSWER
        # qstart+0: QUERY_OLD_FACT
        # qstart+1..3: old_fact answers

        # Q2: recent recall
        seq[qstart + 4] = QUERY_RECENT
        loss_weights[qstart + 4] = W_QUERY
        seq[qstart + 5] = recent_val
        loss_weights[qstart + 5] = W_ANSWER

        # Q3: pattern induction
        seq[qstart + 6] = QUERY_PATTERN
        loss_weights[qstart + 6] = W_QUERY
        seq[qstart + 7] = pattern_answer
        loss_weights[qstart + 7] = W_ANSWER

        # Q4: salient event recall
        seq[qstart + 8] = QUERY_SALIENT
        loss_weights[qstart + 8] = W_QUERY
        seq[qstart + 9] = salient_tokens[0].item()
        loss_weights[qstart + 9] = W_ANSWER

        return {
            "input_ids": seq,
            "loss_weights": loss_weights,
            # Metadata for evaluation
            "old_fact_tokens": old_fact_tokens,  # (3,)
            "recent_val": recent_val,
            "pattern_answer": pattern_answer,
            "salient_answer": salient_tokens[0].item(),
        }


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "kv": {"enable_trn": False, "enable_retrieval": False},
    "kv_trn": {"enable_trn": True, "enable_retrieval": False},
    "kv_ret": {"enable_trn": False, "enable_retrieval": True},
    "trimemory": {"enable_trn": True, "enable_retrieval": True},
}


def make_cfg() -> TRNConfig:
    return TRNConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_oscillators=N_OSC,
        n_layers=N_LAYERS, d_ff=D_FF, max_seq_len=SEQ_LEN + 16,
    )


def build_model(cfg: TRNConfig, model_name: str) -> TriMemoryEngine:
    flags = MODEL_CONFIGS[model_name]
    return TriMemoryEngine(
        cfg, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE, **flags,
    )


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Weighted cross-entropy loss
# ---------------------------------------------------------------------------
def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy with per-position weights.

    Args:
        logits: (B, T, V) raw logits
        labels: (B, T) target token IDs
        weights: (B, T) per-position loss weights
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1].contiguous()  # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()   # (B, T-1)
    shift_weights = weights[:, 1:].contiguous()  # (B, T-1)

    B, T_minus_1, V = shift_logits.shape
    # Per-token CE
    ce = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
    )  # (B * (T-1),)
    ce = ce.view(B, T_minus_1)

    # Weighted mean
    weighted_ce = (ce * shift_weights).sum() / shift_weights.sum()
    return weighted_ce


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    model: TriMemoryEngine,
    dataset: MixedMemoryDatasetV2,
    steps: int,
    device: torch.device,
    lr: float = LR,
) -> tuple[list[dict], bool]:
    model = model.to(device).train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    records = []
    stable = True
    loader_it = iter(loader)

    for step in range(1, steps + 1):
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)

        ids = batch["input_ids"].to(device)
        wts = batch["loss_weights"].to(device)
        out = model(ids)  # no labels -- we compute loss ourselves
        logits = out["logits"]

        loss = weighted_cross_entropy(logits, ids, wts)

        if not torch.isfinite(loss):
            stable = False
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == steps:
            records.append({"step": step, "loss": loss.item()})
            if step % 200 == 0:
                print(f"      step {step}/{steps} loss={loss.item():.4f}", flush=True)

    return records, stable


# ---------------------------------------------------------------------------
# Evaluation with role-specific metrics
# ---------------------------------------------------------------------------
def evaluate_model(
    model: TriMemoryEngine,
    dataset: MixedMemoryDatasetV2,
    device: torch.device,
    n_eval: int = 400,
) -> dict:
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=min(32, n_eval), shuffle=False)

    # Counters
    recent_correct = 0
    old_fact_token_correct = 0
    old_fact_span_exact = 0
    pattern_correct = 0
    salient_correct = 0
    total = 0

    # Gate telemetry
    gate_kv_sum = 0.0
    gate_trn_sum = 0.0
    gate_ret_sum = 0.0
    n_batches = 0

    qstart = dataset.seq_len - dataset.QUERY_REGION_SIZE  # 246

    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            B = ids.size(0)
            out = model(ids)
            logits = out["logits"]  # (B, T, V)

            # Old fact span: logits at [qstart, qstart+1, qstart+2] predict old_fact_tokens
            old_fact_gt = batch["old_fact_tokens"]  # (B, 3)
            span_all_correct = torch.ones(B, dtype=torch.bool)
            for i in range(dataset.old_fact_span_len):
                pred_pos = qstart + i  # predict token at pos qstart+1+i using logits at pos qstart+i
                preds = logits[:, pred_pos, :].argmax(dim=-1).cpu()
                correct = (preds == old_fact_gt[:, i])
                old_fact_token_correct += correct.sum().item()
                span_all_correct &= correct
            old_fact_span_exact += span_all_correct.sum().item()

            # Recent: logits at qstart+4 predict recent_val
            recent_preds = logits[:, qstart + 4, :].argmax(dim=-1).cpu()
            recent_correct += (recent_preds == batch["recent_val"]).sum().item()

            # Pattern: logits at qstart+6 predict pattern_answer
            pattern_preds = logits[:, qstart + 6, :].argmax(dim=-1).cpu()
            pattern_correct += (pattern_preds == batch["pattern_answer"]).sum().item()

            # Salient: logits at qstart+8 predict salient_answer
            salient_preds = logits[:, qstart + 8, :].argmax(dim=-1).cpu()
            salient_correct += (salient_preds == batch["salient_answer"]).sum().item()

            total += B

            # Gate telemetry
            tel = model.collect_gate_telemetry()
            gate_kv_sum += tel["router_kv_ratio"]
            gate_trn_sum += tel["router_trn_ratio"]
            gate_ret_sum += tel["router_ret_ratio"]
            n_batches += 1

    n_batches = max(n_batches, 1)
    total = max(total, 1)
    span_len = dataset.old_fact_span_len

    return {
        "recent_exact_acc": recent_correct / total,
        "old_fact_token_acc": old_fact_token_correct / (total * span_len),
        "old_fact_span_exact_acc": old_fact_span_exact / total,
        "pattern_token_acc": pattern_correct / total,
        "salient_acc": salient_correct / total,
        "n_eval": total,
        "router_kv_ratio": gate_kv_sum / n_batches,
        "router_trn_ratio": gate_trn_sum / n_batches,
        "router_ret_ratio": gate_ret_sum / n_batches,
    }


# ---------------------------------------------------------------------------
# Composite score (V2)
# ---------------------------------------------------------------------------
def compute_composite(metrics: dict, stable: bool) -> float:
    """0.20*recent + 0.30*old_fact_span + 0.30*pattern + 0.20*salient."""
    stability_penalty = 0.0 if stable else 0.05
    return (
        0.20 * metrics["recent_exact_acc"]
        + 0.30 * metrics["old_fact_span_exact_acc"]
        + 0.30 * metrics["pattern_token_acc"]
        + 0.20 * metrics["salient_acc"]
        - stability_penalty
    )


# ---------------------------------------------------------------------------
# Gate judgment (V2)
# ---------------------------------------------------------------------------
def gate_judgment(all_results: list[dict]) -> dict:
    # Aggregate by model
    model_means: dict[str, dict] = {}
    for model_name in MODEL_CONFIGS:
        rows = [r for r in all_results if r["model"] == model_name]
        if not rows:
            continue
        n = len(rows)
        model_means[model_name] = {
            "composite": sum(r["composite_score"] for r in rows) / n,
            "recent": sum(r["recent_exact_acc"] for r in rows) / n,
            "old_fact_span": sum(r["old_fact_span_exact_acc"] for r in rows) / n,
            "old_fact_token": sum(r["old_fact_token_acc"] for r in rows) / n,
            "pattern": sum(r["pattern_token_acc"] for r in rows) / n,
            "salient": sum(r["salient_acc"] for r in rows) / n,
        }

    criteria = {}

    # 1. D composite > max(A, B, C)
    d = model_means.get("trimemory", {})
    others_max = max(
        model_means.get("kv", {}).get("composite", 0),
        model_means.get("kv_trn", {}).get("composite", 0),
        model_means.get("kv_ret", {}).get("composite", 0),
    )
    criteria["composite_D_gt_max_ABC"] = {
        "pass": d.get("composite", 0) > others_max,
        "D": d.get("composite", 0),
        "max_ABC": others_max,
    }

    # 2. B pattern > A pattern
    b_pat = model_means.get("kv_trn", {}).get("pattern", 0)
    a_pat = model_means.get("kv", {}).get("pattern", 0)
    criteria["pattern_B_gt_A"] = {
        "pass": b_pat > a_pat,
        "B": b_pat, "A": a_pat,
    }

    # 3. C old_fact_span > A old_fact_span
    c_old = model_means.get("kv_ret", {}).get("old_fact_span", 0)
    a_old = model_means.get("kv", {}).get("old_fact_span", 0)
    criteria["old_fact_C_gt_A"] = {
        "pass": c_old > a_old,
        "C": c_old, "A": a_old,
    }

    # 4. C salient > A salient
    c_sal = model_means.get("kv_ret", {}).get("salient", 0)
    a_sal = model_means.get("kv", {}).get("salient", 0)
    criteria["salient_C_gt_A"] = {
        "pass": c_sal > a_sal,
        "C": c_sal, "A": a_sal,
    }

    all_pass = all(c["pass"] for c in criteria.values())

    return {
        "verdict": "V2_GO" if all_pass else "V2_FAIL",
        "criteria": criteria,
        "model_means": model_means,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def generate_summary(
    all_results: list[dict],
    gate: dict,
    out_dir: Path,
    steps: int,
    seeds: list[int],
) -> None:
    models = list(MODEL_CONFIGS.keys())
    model_data = {m: [r for r in all_results if r["model"] == m] for m in models}
    labels = {"kv": "A:KV", "kv_trn": "B:KV+TRN", "kv_ret": "C:KV+Ret", "trimemory": "D:Full"}

    def stat(vals: list[dict], key: str) -> str:
        v = [r[key] for r in vals]
        mean = sum(v) / len(v) if v else 0
        if len(v) > 1:
            std = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5
        else:
            std = 0
        return f"{mean:.4f}+/-{std:.4f}"

    lines = [
        "# Tri-Memory V2 Evaluation Results",
        "",
        f"**Steps**: {steps}  |  **Seeds**: {seeds}  |  **Verdict**: {gate['verdict']}",
        "",
        "## Changes from V1",
        "- Weighted loss (answer=8x, query=4x, fact_span=8x, pattern=6x, salient=8x)",
        "- Multi-token old facts (3-token span)",
        "- Pattern with regime shift (2 phases)",
        "- Salient event recall (rare token burst)",
        "- Composite: 0.20*recent + 0.30*old_fact_span + 0.30*pattern + 0.20*salient",
        "",
        "## Accuracy Summary",
        "",
        "| Model | Recent | OldFact(span) | OldFact(tok) | Pattern | Salient | Composite |",
        "|-------|--------|---------------|--------------|---------|---------|-----------|",
    ]
    for m in models:
        vals = model_data[m]
        if not vals:
            continue
        lines.append(
            f"| {labels[m]} | {stat(vals, 'recent_exact_acc')} "
            f"| {stat(vals, 'old_fact_span_exact_acc')} "
            f"| {stat(vals, 'old_fact_token_acc')} "
            f"| {stat(vals, 'pattern_token_acc')} "
            f"| {stat(vals, 'salient_acc')} "
            f"| {stat(vals, 'composite_score')} |"
        )

    lines.extend([
        "",
        "## Router Gate Usage",
        "",
        "| Model | g_kv | g_trn | g_ret |",
        "|-------|------|-------|-------|",
    ])
    for m in models:
        vals = model_data[m]
        if not vals:
            continue
        n = len(vals)
        gk = sum(r["router_kv_ratio"] for r in vals) / n
        gt = sum(r["router_trn_ratio"] for r in vals) / n
        gr = sum(r["router_ret_ratio"] for r in vals) / n
        lines.append(f"| {labels[m]} | {gk:.4f} | {gt:.4f} | {gr:.4f} |")

    lines.extend(["", "## Gate Judgment", ""])
    for crit_name, crit in gate["criteria"].items():
        status = "PASS" if crit["pass"] else "FAIL"
        detail = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in crit.items() if k != "pass"
        )
        lines.append(f"- **{crit_name}**: {status} ({detail})")

    lines.extend(["", f"**Verdict: {gate['verdict']}**", ""])

    with open(out_dir / "v2_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def generate_plots(all_results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    models = list(MODEL_CONFIGS.keys())
    model_data = {m: [r for r in all_results if r["model"] == m] for m in models}
    x_labels = ["A:KV", "B:+TRN", "C:+Ret", "D:Full"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    # Accuracy comparison
    metrics = [
        ("recent_exact_acc", "Recent Exact"),
        ("old_fact_span_exact_acc", "Old Fact (span)"),
        ("old_fact_token_acc", "Old Fact (token)"),
        ("pattern_token_acc", "Pattern"),
        ("salient_acc", "Salient"),
        ("composite_score", "Composite"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (metric, title) in zip(axes.flat, metrics):
        means = []
        stds = []
        for m in models:
            vals = [r[metric] for r in model_data[m]]
            mean = sum(vals) / len(vals) if vals else 0
            means.append(mean)
            if len(vals) > 1:
                stds.append((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5)
            else:
                stds.append(0)
        bars = ax.bar(range(4), means, yerr=stds, capsize=4, color=colors, alpha=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, max(max(means) * 1.5, 0.05))
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(plots_dir / "v2_accuracy.png", dpi=150)
    plt.close(fig)

    # Gate usage
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    for i, (gate_key, label) in enumerate(
        [("router_kv_ratio", "KV"), ("router_trn_ratio", "TRN"), ("router_ret_ratio", "Ret")]
    ):
        vals = [sum(r[gate_key] for r in model_data[m]) / max(len(model_data[m]), 1) for m in models]
        ax.bar([xi + i * width for xi in range(4)], vals, width, label=label, alpha=0.8)
    ax.set_xticks([xi + width for xi in range(4)])
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Gate ratio")
    ax.set_title("Router Gate Usage by Model (V2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "v2_router.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Tri-Memory V2 Evaluation")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/trimemory_v2/")
    args = parser.parse_args()

    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Tri-Memory V2 Evaluation]")
    print(f"  models: {args.models}")
    print(f"  steps: {args.steps}, seeds: {args.seeds}")
    print(f"  device: {args.device}")
    print(f"  output: {out_dir}")
    print(f"  loss weights: normal={W_NORMAL} query={W_QUERY} answer={W_ANSWER} "
          f"fact_span={W_FACT_SPAN} pattern={W_PATTERN_TARGET} salient={W_SALIENT}")
    print(flush=True)

    cfg = make_cfg()
    all_results: list[dict] = []

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---", flush=True)

        for model_name in args.models:
            print(f"\n  [{model_name}] seed={seed}", flush=True)
            seed_everything(seed)
            dataset = MixedMemoryDatasetV2(n_samples=2000, seq_len=SEQ_LEN, seed=seed)

            seed_everything(seed)
            model = build_model(cfg, model_name)

            # Train
            t0 = time.perf_counter()
            loss_curve, stable = train_model(model, dataset, args.steps, device)
            train_time = time.perf_counter() - t0
            final_loss = loss_curve[-1]["loss"] if loss_curve else float("nan")
            print(f"    Train: {train_time:.1f}s, loss={final_loss:.4f}, stable={stable}", flush=True)

            # Evaluate
            eval_result = evaluate_model(model, dataset, device, n_eval=400)
            composite = compute_composite(eval_result, stable)

            row = {
                "model": model_name,
                "seed": seed,
                **eval_result,
                "composite_score": composite,
                "final_loss": final_loss,
                "train_time_s": train_time,
                "stable": stable,
            }
            all_results.append(row)

            print(f"    recent={eval_result['recent_exact_acc']:.3f} "
                  f"old_span={eval_result['old_fact_span_exact_acc']:.3f} "
                  f"old_tok={eval_result['old_fact_token_acc']:.3f} "
                  f"pattern={eval_result['pattern_token_acc']:.3f} "
                  f"salient={eval_result['salient_acc']:.3f} "
                  f"comp={composite:.4f}", flush=True)
            print(f"    gate: kv={eval_result['router_kv_ratio']:.3f} "
                  f"trn={eval_result['router_trn_ratio']:.3f} "
                  f"ret={eval_result['router_ret_ratio']:.3f}", flush=True)

            # Save per-seed data
            seed_file = out_dir / f"seed_{seed}_data.json"
            seed_data = [r for r in all_results if r["seed"] == seed]
            with open(seed_file, "w") as f:
                json.dump(seed_data, f, indent=2, default=str)

    # Gate judgment
    gate = gate_judgment(all_results)
    print(f"\n=== GATE VERDICT: {gate['verdict']} ===", flush=True)
    for crit_name, crit in gate["criteria"].items():
        status = "PASS" if crit["pass"] else "FAIL"
        print(f"  {crit_name}: {status}")

    # Save CSV
    csv_path = out_dir / "v2_results.csv"
    fieldnames = [
        "model", "seed", "recent_exact_acc", "old_fact_token_acc",
        "old_fact_span_exact_acc", "pattern_token_acc", "salient_acc",
        "composite_score", "router_kv_ratio", "router_trn_ratio",
        "router_ret_ratio", "final_loss", "train_time_s", "stable",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # Save gate result
    with open(out_dir / "v2_gate.json", "w") as f:
        json.dump(gate, f, indent=2, default=str)

    # Save all results JSON
    with open(out_dir / "v2_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate plots and summary
    generate_plots(all_results, out_dir)
    generate_summary(all_results, gate, out_dir, args.steps, args.seeds)

    # Print comparison table
    print(f"\n=== V2 COMPARISON TABLE (mean across seeds) ===")
    print(f"{'Model':<12} {'Recent':>8} {'OldSpan':>8} {'OldTok':>8} {'Pattern':>8} {'Salient':>8} {'Comp':>8}")
    print("-" * 68)
    for m in args.models:
        rows = [r for r in all_results if r["model"] == m]
        if not rows:
            continue
        n = len(rows)
        print(f"{m:<12} "
              f"{sum(r['recent_exact_acc'] for r in rows) / n:>8.4f} "
              f"{sum(r['old_fact_span_exact_acc'] for r in rows) / n:>8.4f} "
              f"{sum(r['old_fact_token_acc'] for r in rows) / n:>8.4f} "
              f"{sum(r['pattern_token_acc'] for r in rows) / n:>8.4f} "
              f"{sum(r['salient_acc'] for r in rows) / n:>8.4f} "
              f"{sum(r['composite_score'] for r in rows) / n:>8.4f}")

    print(f"\n[DONE] All results saved to {out_dir}/")


if __name__ == "__main__":
    main()
