#!/usr/bin/env python3
"""Tri-Memory V3 Evaluation: old_fact signal tuning for retrieval observability.

Changes from V2:
  - old_fact token range narrowed: 220-240 (20 types) -> 220-225 (5 types)
  - old_fact span len: 3 -> 2 (easier span exact)
  - fact span loss weight: 8.0 -> 10.0
  - Added old_fact_span_partial_acc metric
  - pattern / salient / recent unchanged from V2

V3-3000 extensions:
  - --checkpoint-at for mid-training evaluation (default: 1000)
  - Derived metrics: pattern_B_minus_A, old_fact_C_minus_A, D_minus_maxABC
  - V3_3000_GO / V3_3000_PLATEAU / INCONCLUSIVE verdict
  - Cross-step comparison table (checkpoint vs final)

Usage:
    python scripts/run_trimemory_v3_eval.py --steps 1000 --seeds 0 1
    python scripts/run_trimemory_v3_eval.py --steps 3000 --seeds 0 1 2 3 4 --checkpoint-at 1000
    python scripts/run_trimemory_v3_eval.py --steps 3000 --seeds 0 1 2 3 4 --device cuda
"""
from __future__ import annotations

import argparse
import copy
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
# Constants -- shared with V2 unless noted
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

# V3 CHANGE: old_fact range narrowed from 20 types to 5 types
OLD_FACT_LOW = 220
OLD_FACT_HIGH = 225

PATTERN_TOKEN_LOW = 10
PATTERN_TOKEN_HIGH = 60
SALIENT_TOKEN_LOW = 240
SALIENT_TOKEN_HIGH = 256

# Query markers
QUERY_OLD_FACT = 5
QUERY_RECENT = 6
QUERY_PATTERN = 7
QUERY_SALIENT = 8

# Loss weights
W_NORMAL = 1.0
W_QUERY = 4.0
W_ANSWER = 8.0
W_FACT_SPAN = 10.0
W_PATTERN_TARGET = 6.0
W_SALIENT = 8.0

# V3 CHANGE: old fact span length 3 -> 2
OLD_FACT_SPAN_LEN = 2

# Pattern config
PATTERN_PERIOD = 5
PATTERN_BLOCK_START = 30
PATTERN_BLOCK_END = 80
PATTERN_REGIME_SHIFT_POS = 55

# Salient event config
SALIENT_POS_START = 10
SALIENT_POS_END = 15

# Recent
RECENT_POS = -16


# ---------------------------------------------------------------------------
# V3 Dataset
# ---------------------------------------------------------------------------
class MixedMemoryDatasetV3(Dataset):
    """V3 dataset: narrower old_fact range, shorter span, otherwise same as V2.

    Sequence layout (SEQ_LEN=256):
      [0..1]:      old_fact span (2 tokens from 220-225, 5 types)
      [10..14]:    salient event (5 rare tokens, unchanged)
      [30..79]:    pattern block with regime shift at 55 (unchanged)
      [filler]:    positions 2-9, 15-29, 80-239
      [240]:       recent value (unchanged)

    Query region (last 9 positions = 247-255):
      247: QUERY_OLD_FACT
      248: old_fact_token[0]   (answer)
      249: old_fact_token[1]   (answer)
      250: QUERY_RECENT
      251: recent_val          (answer)
      252: QUERY_PATTERN
      253: pattern_answer      (answer)
      254: QUERY_SALIENT
      255: salient_token[0]    (answer)
    """

    QUERY_REGION_SIZE = 9

    def __init__(
        self,
        n_samples: int = 2000,
        seq_len: int = SEQ_LEN,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.old_fact_span_len = OLD_FACT_SPAN_LEN
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        seq = torch.randint(FILLER_LOW, FILLER_HIGH, (self.seq_len,), generator=self.rng)
        loss_weights = torch.full((self.seq_len,), W_NORMAL, dtype=torch.float32)

        # --- Old fact span at positions 0..1 (2 tokens, 5 types each) ---
        old_fact_tokens = torch.randint(
            OLD_FACT_LOW, OLD_FACT_HIGH, (self.old_fact_span_len,), generator=self.rng
        )
        for i in range(self.old_fact_span_len):
            seq[i] = old_fact_tokens[i]
            loss_weights[i] = W_FACT_SPAN

        # --- Salient event at positions 10..14 (unchanged from V2) ---
        salient_tokens = torch.randint(
            SALIENT_TOKEN_LOW, SALIENT_TOKEN_HIGH, (5,), generator=self.rng
        )
        for i in range(5):
            seq[SALIENT_POS_START + i] = salient_tokens[i]
            loss_weights[SALIENT_POS_START + i] = W_SALIENT

        # --- Pattern block at positions 30..79 (unchanged from V2) ---
        pattern_a = torch.randint(
            PATTERN_TOKEN_LOW, PATTERN_TOKEN_HIGH, (PATTERN_PERIOD,), generator=self.rng
        )
        for i in range(PATTERN_BLOCK_START, PATTERN_REGIME_SHIFT_POS):
            seq[i] = pattern_a[i % PATTERN_PERIOD]
            loss_weights[i] = W_PATTERN_TARGET

        pattern_b = torch.randint(
            PATTERN_TOKEN_LOW, PATTERN_TOKEN_HIGH, (PATTERN_PERIOD,), generator=self.rng
        )
        for i in range(PATTERN_REGIME_SHIFT_POS, PATTERN_BLOCK_END):
            seq[i] = pattern_b[(i - PATTERN_REGIME_SHIFT_POS) % PATTERN_PERIOD]
            loss_weights[i] = W_PATTERN_TARGET

        pattern_next_idx = (PATTERN_BLOCK_END - PATTERN_REGIME_SHIFT_POS) % PATTERN_PERIOD
        pattern_answer = pattern_b[pattern_next_idx].item()

        # --- Recent value (unchanged from V2) ---
        recent_pos = self.seq_len + RECENT_POS
        recent_val = torch.randint(
            RECENT_VALUE_LOW, RECENT_VALUE_HIGH, (1,), generator=self.rng
        ).item()
        seq[recent_pos] = recent_val

        # --- Query region (last 9 positions: 247-255) ---
        qstart = self.seq_len - self.QUERY_REGION_SIZE  # 247

        # Q1: old fact recall (2-token span)
        seq[qstart] = QUERY_OLD_FACT
        loss_weights[qstart] = W_QUERY
        for i in range(self.old_fact_span_len):
            seq[qstart + 1 + i] = old_fact_tokens[i]
            loss_weights[qstart + 1 + i] = W_ANSWER

        # Q2: recent recall
        seq[qstart + 3] = QUERY_RECENT
        loss_weights[qstart + 3] = W_QUERY
        seq[qstart + 4] = recent_val
        loss_weights[qstart + 4] = W_ANSWER

        # Q3: pattern induction
        seq[qstart + 5] = QUERY_PATTERN
        loss_weights[qstart + 5] = W_QUERY
        seq[qstart + 6] = pattern_answer
        loss_weights[qstart + 6] = W_ANSWER

        # Q4: salient event recall
        seq[qstart + 7] = QUERY_SALIENT
        loss_weights[qstart + 7] = W_QUERY
        seq[qstart + 8] = salient_tokens[0].item()
        loss_weights[qstart + 8] = W_ANSWER

        return {
            "input_ids": seq,
            "loss_weights": loss_weights,
            "old_fact_tokens": old_fact_tokens,  # (2,)
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
# Weighted cross-entropy
# ---------------------------------------------------------------------------
def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()
    B, T_minus_1, V = shift_logits.shape
    ce = F.cross_entropy(
        shift_logits.view(-1, V), shift_labels.view(-1), reduction="none",
    ).view(B, T_minus_1)
    return (ce * shift_weights).sum() / shift_weights.sum()


# ---------------------------------------------------------------------------
# Training with optional checkpoint
# ---------------------------------------------------------------------------
def train_model(
    model: TriMemoryEngine,
    dataset: MixedMemoryDatasetV3,
    steps: int,
    device: torch.device,
    lr: float = LR,
    checkpoint_at: int | None = None,
) -> tuple[list[dict], bool, dict | None]:
    """Train model and optionally save a checkpoint state dict.

    Returns:
        (loss_curve, stable, checkpoint_state_dict_or_None)
    """
    model = model.to(device).train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    records = []
    stable = True
    loader_it = iter(loader)
    checkpoint_state = None

    for step in range(1, steps + 1):
        try:
            batch = next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            batch = next(loader_it)

        ids = batch["input_ids"].to(device)
        wts = batch["loss_weights"].to(device)
        logits = model(ids)["logits"]
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

        # Checkpoint snapshot
        if checkpoint_at is not None and step == checkpoint_at:
            checkpoint_state = copy.deepcopy(model.state_dict())
            print(f"      [checkpoint saved at step {step}]", flush=True)

    return records, stable, checkpoint_state


# ---------------------------------------------------------------------------
# Evaluation with role-specific metrics + partial accuracy
# ---------------------------------------------------------------------------
def evaluate_model(
    model: TriMemoryEngine,
    dataset: MixedMemoryDatasetV3,
    device: torch.device,
    n_eval: int = 400,
) -> dict:
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=min(32, n_eval), shuffle=False)

    recent_correct = 0
    old_fact_token_correct = 0
    old_fact_span_exact = 0
    old_fact_span_partial = 0
    pattern_correct = 0
    salient_correct = 0
    total = 0

    gate_kv_sum = 0.0
    gate_trn_sum = 0.0
    gate_ret_sum = 0.0
    retrieval_used_count = 0
    total_chunks = 0
    n_batches = 0

    qstart = dataset.seq_len - dataset.QUERY_REGION_SIZE
    span_len = dataset.old_fact_span_len

    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            B = ids.size(0)
            logits = model(ids)["logits"]

            # Old fact span
            old_fact_gt = batch["old_fact_tokens"]  # (B, 2)
            span_correct_count = torch.zeros(B, dtype=torch.long)
            for i in range(span_len):
                preds = logits[:, qstart + i, :].argmax(dim=-1).cpu()
                correct = (preds == old_fact_gt[:, i])
                old_fact_token_correct += correct.sum().item()
                span_correct_count += correct.long()
            old_fact_span_exact += (span_correct_count == span_len).sum().item()
            old_fact_span_partial += (span_correct_count >= 1).sum().item()

            # Recent
            recent_preds = logits[:, qstart + 3, :].argmax(dim=-1).cpu()
            recent_correct += (recent_preds == batch["recent_val"]).sum().item()

            # Pattern
            pattern_preds = logits[:, qstart + 5, :].argmax(dim=-1).cpu()
            pattern_correct += (pattern_preds == batch["pattern_answer"]).sum().item()

            # Salient
            salient_preds = logits[:, qstart + 7, :].argmax(dim=-1).cpu()
            salient_correct += (salient_preds == batch["salient_answer"]).sum().item()

            total += B

            # Gate telemetry
            tel = model.collect_gate_telemetry()
            gate_kv_sum += tel["router_kv_ratio"]
            gate_trn_sum += tel["router_trn_ratio"]
            gate_ret_sum += tel["router_ret_ratio"]
            if tel.get("retrieval_used", False):
                retrieval_used_count += 1
            total_chunks += tel.get("archive_chunk_count", 0)
            n_batches += 1

    n_batches = max(n_batches, 1)
    total = max(total, 1)

    return {
        "recent_exact_acc": recent_correct / total,
        "old_fact_token_acc": old_fact_token_correct / (total * span_len),
        "old_fact_span_exact_acc": old_fact_span_exact / total,
        "old_fact_span_partial_acc": old_fact_span_partial / total,
        "pattern_token_acc": pattern_correct / total,
        "salient_event_acc": salient_correct / total,
        "n_eval": total,
        "router_kv_ratio": gate_kv_sum / n_batches,
        "router_trn_ratio": gate_trn_sum / n_batches,
        "router_ret_ratio": gate_ret_sum / n_batches,
        "retrieval_calls": n_batches,
        "retrieval_hit_rate": retrieval_used_count / max(n_batches, 1),
        "archive_chunk_count": total_chunks / n_batches,
    }


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
def compute_composite(m: dict, stable: bool) -> float:
    penalty = 0.0 if stable else 0.05
    return (
        0.20 * m["recent_exact_acc"]
        + 0.30 * m["old_fact_span_exact_acc"]
        + 0.30 * m["pattern_token_acc"]
        + 0.20 * m["salient_event_acc"]
        - penalty
    )


# ---------------------------------------------------------------------------
# Sanity checks (V3-specific)
# ---------------------------------------------------------------------------
def sanity_checks(dataset: MixedMemoryDatasetV3) -> dict:
    return {
        "old_fact_token_range_narrowed": (OLD_FACT_HIGH - OLD_FACT_LOW) <= 10,
        "fact_span_len_valid": dataset.old_fact_span_len in {2, 3},
        "fact_span_outside_kv": True,
        "query_weight_mask_nonzero": W_FACT_SPAN > 1.0 and W_ANSWER > 1.0,
        "old_fact_types": OLD_FACT_HIGH - OLD_FACT_LOW,
        "old_fact_random_baseline_per_token": 1.0 / (OLD_FACT_HIGH - OLD_FACT_LOW),
    }


# ---------------------------------------------------------------------------
# Derived metrics computation
# ---------------------------------------------------------------------------
METRIC_KEYS = [
    "composite_score", "recent_exact_acc", "old_fact_token_acc",
    "old_fact_span_exact_acc", "old_fact_span_partial_acc",
    "pattern_token_acc", "salient_event_acc",
]


def compute_model_means(results: list[dict]) -> dict[str, dict[str, float]]:
    """Compute per-model mean metrics from a list of result rows."""
    means: dict[str, dict[str, float]] = {}
    for model_name in MODEL_CONFIGS:
        rows = [r for r in results if r["model"] == model_name]
        if not rows:
            continue
        n = len(rows)
        means[model_name] = {k: sum(r[k] for r in rows) / n for k in METRIC_KEYS}
    return means


def compute_derived_metrics(model_means: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute cross-model derived metrics for V3-3000 judgment."""
    a = model_means.get("kv", {})
    b = model_means.get("kv_trn", {})
    c = model_means.get("kv_ret", {})
    d = model_means.get("trimemory", {})

    pattern_B_minus_A = b.get("pattern_token_acc", 0) - a.get("pattern_token_acc", 0)
    old_fact_C_minus_A = c.get("old_fact_token_acc", 0) - a.get("old_fact_token_acc", 0)
    old_fact_span_C_minus_A = c.get("old_fact_span_exact_acc", 0) - a.get("old_fact_span_exact_acc", 0)
    salient_C_minus_A = c.get("salient_event_acc", 0) - a.get("salient_event_acc", 0)

    max_abc_comp = max(
        a.get("composite_score", 0),
        b.get("composite_score", 0),
        c.get("composite_score", 0),
    )
    D_minus_maxABC = d.get("composite_score", 0) - max_abc_comp

    return {
        "pattern_B_minus_A": pattern_B_minus_A,
        "old_fact_C_minus_A": old_fact_C_minus_A,
        "old_fact_span_C_minus_A": old_fact_span_C_minus_A,
        "salient_C_minus_A": salient_C_minus_A,
        "D_minus_maxABC": D_minus_maxABC,
    }


# ---------------------------------------------------------------------------
# Gate judgment (V3 -- 1000-step verdict)
# ---------------------------------------------------------------------------
def gate_judgment(all_results: list[dict]) -> dict:
    model_means = compute_model_means(all_results)
    criteria = {}

    c_tok = model_means.get("kv_ret", {}).get("old_fact_token_acc", 0)
    a_tok = model_means.get("kv", {}).get("old_fact_token_acc", 0)
    criteria["old_fact_token_C_gt_A"] = {"pass": c_tok > a_tok, "C": c_tok, "A": a_tok}

    c_span = model_means.get("kv_ret", {}).get("old_fact_span_exact_acc", 0)
    a_span = model_means.get("kv", {}).get("old_fact_span_exact_acc", 0)
    criteria["old_fact_span_C_ge_A"] = {"pass": c_span >= a_span, "C": c_span, "A": a_span}

    c_sal = model_means.get("kv_ret", {}).get("salient_event_acc", 0)
    a_sal = model_means.get("kv", {}).get("salient_event_acc", 0)
    criteria["salient_C_ge_A"] = {"pass": c_sal >= a_sal, "C": c_sal, "A": a_sal}

    b_pat = model_means.get("kv_trn", {}).get("pattern_token_acc", 0)
    a_pat = model_means.get("kv", {}).get("pattern_token_acc", 0)
    criteria["pattern_B_ge_A"] = {"pass": b_pat >= a_pat, "B": b_pat, "A": a_pat}

    no_nan = all(r.get("stable", True) for r in all_results)
    criteria["no_nan_inf"] = {"pass": no_nan}

    all_pass = all(c["pass"] for c in criteria.values())

    d_comp = model_means.get("trimemory", {}).get("composite_score", 0)
    others_max = max(
        model_means.get("kv", {}).get("composite_score", 0),
        model_means.get("kv_trn", {}).get("composite_score", 0),
        model_means.get("kv_ret", {}).get("composite_score", 0),
    )
    criteria["supplementary_D_gt_max_ABC"] = {
        "pass": d_comp > others_max, "D": d_comp, "max_ABC": others_max,
    }

    return {
        "verdict": "V3_SIGNAL_GO" if all_pass else "V3_SIGNAL_FAIL",
        "criteria": criteria,
        "model_means": model_means,
    }


# ---------------------------------------------------------------------------
# V3-3000 verdict: learning vs design insufficiency
# ---------------------------------------------------------------------------
GROWTH_THRESHOLD = 0.02  # derived metric must grow by at least this
ABSOLUTE_THRESHOLD = 0.05  # derived metric must exceed this at 3000 steps


def v3_3000_verdict(
    derived_ckpt: dict[str, float],
    derived_final: dict[str, float],
) -> dict:
    """Compare checkpoint (1000) vs final (3000) derived metrics.

    Returns verdict:
      V3_3000_GO: signal grows with steps -- learning insufficiency confirmed
      V3_3000_PLATEAU: signal does not grow -- design insufficiency
      INCONCLUSIVE: mixed signals
    """
    checks = {}

    # 1. old_fact_C_minus_A growth
    of_ckpt = derived_ckpt["old_fact_C_minus_A"]
    of_final = derived_final["old_fact_C_minus_A"]
    of_growth = of_final - of_ckpt
    checks["old_fact_C_minus_A_grows"] = {
        "pass": of_growth > GROWTH_THRESHOLD and of_final > ABSOLUTE_THRESHOLD,
        "ckpt": of_ckpt, "final": of_final, "growth": of_growth,
    }

    # 2. pattern_B_minus_A growth
    pb_ckpt = derived_ckpt["pattern_B_minus_A"]
    pb_final = derived_final["pattern_B_minus_A"]
    pb_growth = pb_final - pb_ckpt
    checks["pattern_B_minus_A_grows"] = {
        "pass": pb_growth > GROWTH_THRESHOLD and pb_final > ABSOLUTE_THRESHOLD,
        "ckpt": pb_ckpt, "final": pb_final, "growth": pb_growth,
    }

    # 3. D_minus_maxABC improvement
    d_ckpt = derived_ckpt["D_minus_maxABC"]
    d_final = derived_final["D_minus_maxABC"]
    d_growth = d_final - d_ckpt
    checks["D_minus_maxABC_improves"] = {
        "pass": d_final > 0 and d_growth > 0,
        "ckpt": d_ckpt, "final": d_final, "growth": d_growth,
    }

    # 4. old_fact_span_C_minus_A growth
    os_ckpt = derived_ckpt["old_fact_span_C_minus_A"]
    os_final = derived_final["old_fact_span_C_minus_A"]
    os_growth = os_final - os_ckpt
    checks["old_fact_span_C_minus_A_grows"] = {
        "pass": os_growth > GROWTH_THRESHOLD / 2 and os_final > 0,
        "ckpt": os_ckpt, "final": os_final, "growth": os_growth,
    }

    n_pass = sum(1 for c in checks.values() if c["pass"])

    if n_pass >= 3:
        verdict = "V3_3000_GO"
    elif n_pass == 0:
        verdict = "V3_3000_PLATEAU"
    else:
        verdict = "INCONCLUSIVE"

    # Interpretation branch
    if verdict == "V3_3000_GO":
        interpretation = (
            "Signal grows with training steps. Learning insufficiency confirmed. "
            "Next: scale to 10K steps or increase model capacity."
        )
    elif verdict == "V3_3000_PLATEAU":
        interpretation = (
            "Signal plateaus between 1000 and 3000 steps. Design insufficiency. "
            "Next: redesign task (increase query signal density, retrieval-only tokens, "
            "or explicit retrieval supervision)."
        )
    else:
        interpretation = (
            "Mixed signals -- some metrics grow, others plateau. "
            "Next: analyze per-metric breakdown to identify which path underperforms."
        )

    return {
        "verdict": verdict,
        "checks": checks,
        "n_pass": n_pass,
        "n_total": len(checks),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def generate_summary(
    all_results: list[dict],
    gate: dict,
    sanity: dict,
    out_dir: Path,
    steps: int,
    seeds: list[int],
    checkpoint_results: list[dict] | None = None,
    checkpoint_step: int | None = None,
) -> None:
    models = list(MODEL_CONFIGS.keys())
    model_data = {m: [r for r in all_results if r["model"] == m] for m in models}
    labels = {"kv": "A:KV", "kv_trn": "B:KV+TRN", "kv_ret": "C:KV+Ret", "trimemory": "D:Full"}

    def stat(vals: list[dict], key: str) -> str:
        v = [r[key] for r in vals]
        mean = sum(v) / len(v) if v else 0
        std = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5 if len(v) > 1 else 0
        return f"{mean:.4f}+/-{std:.4f}"

    lines = [
        "# Tri-Memory V3 Evaluation -- old_fact Signal Tuning",
        "",
        f"**Steps**: {steps}  |  **Seeds**: {seeds}  |  **Verdict**: {gate['verdict']}",
        "",
    ]

    # Cross-step comparison if checkpoint available
    if checkpoint_results is not None and checkpoint_step is not None:
        ckpt_means = compute_model_means(checkpoint_results)
        final_means = compute_model_means(all_results)
        derived_ckpt = compute_derived_metrics(ckpt_means)
        derived_final = compute_derived_metrics(final_means)
        v3k = v3_3000_verdict(derived_ckpt, derived_final)

        lines.extend([
            f"## V3-3000 Verdict: {v3k['verdict']} ({v3k['n_pass']}/{v3k['n_total']} checks pass)",
            "",
            f"**Interpretation**: {v3k['interpretation']}",
            "",
            "### Cross-Step Derived Metrics",
            "",
            f"| Metric | step={checkpoint_step} | step={steps} | Growth |",
            "|--------|------------|-----------|--------|",
        ])
        for key in ["pattern_B_minus_A", "old_fact_C_minus_A", "old_fact_span_C_minus_A",
                     "salient_C_minus_A", "D_minus_maxABC"]:
            ck = derived_ckpt[key]
            fn = derived_final[key]
            gr = fn - ck
            lines.append(f"| {key} | {ck:+.4f} | {fn:+.4f} | {gr:+.4f} |")

        lines.extend([
            "",
            "### Cross-Step Accuracy Comparison",
            "",
            f"| Model | Metric | step={checkpoint_step} | step={steps} | Delta |",
            "|-------|--------|------------|-----------|-------|",
        ])
        for m in models:
            for metric in ["old_fact_token_acc", "old_fact_span_exact_acc",
                           "pattern_token_acc", "salient_event_acc", "composite_score"]:
                ck_val = ckpt_means.get(m, {}).get(metric, 0)
                fn_val = final_means.get(m, {}).get(metric, 0)
                delta = fn_val - ck_val
                lines.append(f"| {labels[m]} | {metric} | {ck_val:.4f} | {fn_val:.4f} | {delta:+.4f} |")
        lines.append("")

        # V3-3000 check details
        lines.extend(["### V3-3000 Check Details", ""])
        for check_name, check in v3k["checks"].items():
            status = "PASS" if check["pass"] else "FAIL"
            lines.append(
                f"- **{check_name}**: {status} "
                f"(ckpt={check['ckpt']:+.4f}, final={check['final']:+.4f}, "
                f"growth={check['growth']:+.4f})"
            )
        lines.append("")

    # V3 changes section
    lines.extend([
        "## 1. What Changed from V2",
        "",
        "- old_fact token range: V2=220-240 (20 types) -> V3=220-225 (5 types)",
        "- old_fact span len: V2=3 -> V3=2",
        "- fact span loss weight: V2=8.0 -> V3=10.0",
        "- Added old_fact_span_partial_acc metric",
        "- Random baseline per token: V2=5% -> V3=20%",
        "- Random baseline span exact: V2=0.05^3=0.01% -> V3=0.20^2=4%",
        "",
        "## 2. V3 Results (Final)",
        "",
        "### Accuracy Summary",
        "",
        "| Model | Recent | OldFact(tok) | OldFact(span) | OldFact(partial) | Pattern | Salient | Composite |",
        "|-------|--------|-------------|---------------|------------------|---------|---------|-----------|",
    ])
    for m in models:
        vals = model_data[m]
        if not vals:
            continue
        lines.append(
            f"| {labels[m]} "
            f"| {stat(vals, 'recent_exact_acc')} "
            f"| {stat(vals, 'old_fact_token_acc')} "
            f"| {stat(vals, 'old_fact_span_exact_acc')} "
            f"| {stat(vals, 'old_fact_span_partial_acc')} "
            f"| {stat(vals, 'pattern_token_acc')} "
            f"| {stat(vals, 'salient_event_acc')} "
            f"| {stat(vals, 'composite_score')} |"
        )

    lines.extend([
        "",
        "### Router Gate Usage",
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

    # Gate judgment section
    lines.extend(["", "## 3. Gate Judgment", ""])
    for crit_name, crit in gate["criteria"].items():
        status = "PASS" if crit["pass"] else "FAIL"
        detail = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in crit.items() if k != "pass"
        )
        lines.append(f"- **{crit_name}**: {status} ({detail})")
    lines.extend(["", f"**Verdict: {gate['verdict']}**", ""])

    # Derived metrics
    final_means = compute_model_means(all_results)
    derived = compute_derived_metrics(final_means)
    lines.extend(["## 4. Derived Metrics (Final)", ""])
    for k, v in derived.items():
        lines.append(f"- {k}: {v:+.4f}")
    lines.append("")

    # Sanity checks
    lines.extend(["## Sanity Checks", ""])
    for k, v in sanity.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # What this experiment can / cannot confirm
    if checkpoint_results is not None:
        lines.extend([
            "## What This Experiment Confirms",
            "",
            "- Whether V3 micro-differences at 1000 steps grow or plateau at 3000 steps",
            "- Whether the gap is learning insufficiency (grows) or design insufficiency (plateaus)",
            "- Per-path contribution: which of TRN/Retrieval shows clearer signal with more training",
            "",
            "## What This Experiment Cannot Confirm",
            "",
            "- Whether 10K+ steps would change the verdict (only 1K vs 3K compared)",
            "- Whether a different task design would perform better (only V3 config tested)",
            "- Statistical significance (5 seeds give rough trends, not p-values)",
            "",
        ])

    with open(out_dir / "internal_summary.md", "w", encoding="utf-8") as f:
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

    def _mean_std(data_list, key):
        vals = [r[key] for r in data_list]
        mean = sum(vals) / len(vals) if vals else 0
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0
        return mean, std

    # -- oldfact_token_vs_span.png --
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in [
        (ax1, "old_fact_token_acc", "Old Fact Token Acc"),
        (ax2, "old_fact_span_exact_acc", "Old Fact Span Exact"),
        (ax3, "old_fact_span_partial_acc", "Old Fact Span Partial"),
    ]:
        means, stds = zip(*[_mean_std(model_data[m], metric) for m in models])
        bars = ax.bar(range(4), means, yerr=stds, capsize=4, color=colors, alpha=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, max(max(means) * 1.5, 0.05))
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", fontsize=7)
    fig.suptitle("V3: Old Fact Metrics by Model", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "oldfact_token_vs_span.png", dpi=150)
    plt.close(fig)

    # -- role_specific_accuracy.png --
    metrics_role = [
        ("recent_exact_acc", "Recent"),
        ("old_fact_token_acc", "OldFact(tok)"),
        ("pattern_token_acc", "Pattern"),
        ("salient_event_acc", "Salient"),
        ("composite_score", "Composite"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (metric, title) in zip(axes, metrics_role):
        means, stds = zip(*[_mean_std(model_data[m], metric) for m in models])
        bars = ax.bar(range(4), means, yerr=stds, capsize=4, color=colors, alpha=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(max(means) * 1.5, 0.05))
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", fontsize=6)
    fig.suptitle("V3: Role-Specific Accuracy", fontsize=11)
    fig.tight_layout()
    fig.savefig(plots_dir / "role_specific_accuracy.png", dpi=150)
    plt.close(fig)

    # -- composite_score.png --
    fig, ax = plt.subplots(figsize=(6, 4))
    means, stds = zip(*[_mean_std(model_data[m], "composite_score") for m in models])
    bars = ax.bar(range(4), means, yerr=stds, capsize=4, color=colors, alpha=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(x_labels)
    ax.set_title("V3 Composite Score")
    ax.set_ylim(0, max(max(means) * 1.3, 0.3))
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "composite_score.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Tri-Memory V3 Evaluation")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/v3_oldfact_tuning/")
    parser.add_argument("--checkpoint-at", type=int, default=None,
                        help="Step to save mid-training checkpoint for cross-step comparison")
    args = parser.parse_args()

    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    use_checkpoint = args.checkpoint_at is not None and args.checkpoint_at < args.steps

    print("[Tri-Memory V3 Evaluation -- old_fact Signal Tuning]")
    print(f"  models: {args.models}")
    print(f"  steps: {args.steps}, seeds: {args.seeds}")
    if use_checkpoint:
        print(f"  checkpoint-at: {args.checkpoint_at} (cross-step comparison enabled)")
    print(f"  device: {args.device}")
    print(f"  output: {out_dir}")
    print(f"  V3 changes: old_fact range={OLD_FACT_LOW}-{OLD_FACT_HIGH} "
          f"({OLD_FACT_HIGH - OLD_FACT_LOW} types), "
          f"span_len={OLD_FACT_SPAN_LEN}, fact_weight={W_FACT_SPAN}")
    print(flush=True)

    cfg = make_cfg()
    all_results: list[dict] = []
    checkpoint_results: list[dict] = []

    # Sanity checks
    dummy_ds = MixedMemoryDatasetV3(n_samples=1, seed=0)
    sanity = sanity_checks(dummy_ds)
    print(f"  Sanity: {sanity}", flush=True)

    # Save configs
    dataset_config = {
        "old_fact_low": OLD_FACT_LOW,
        "old_fact_high": OLD_FACT_HIGH,
        "old_fact_types": OLD_FACT_HIGH - OLD_FACT_LOW,
        "old_fact_span_len": OLD_FACT_SPAN_LEN,
        "random_baseline_token": 1.0 / (OLD_FACT_HIGH - OLD_FACT_LOW),
        "random_baseline_span_exact": (1.0 / (OLD_FACT_HIGH - OLD_FACT_LOW)) ** OLD_FACT_SPAN_LEN,
        "pattern_period": PATTERN_PERIOD,
        "pattern_block": f"{PATTERN_BLOCK_START}-{PATTERN_BLOCK_END}",
        "pattern_regime_shift": PATTERN_REGIME_SHIFT_POS,
        "salient_pos": f"{SALIENT_POS_START}-{SALIENT_POS_END}",
        "salient_types": SALIENT_TOKEN_HIGH - SALIENT_TOKEN_LOW,
        "recent_range": f"{RECENT_VALUE_LOW}-{RECENT_VALUE_HIGH}",
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
    }
    with open(out_dir / "dataset_config.json", "w") as f:
        json.dump(dataset_config, f, indent=2)

    loss_config = {
        "W_NORMAL": W_NORMAL, "W_QUERY": W_QUERY, "W_ANSWER": W_ANSWER,
        "W_FACT_SPAN": W_FACT_SPAN, "W_PATTERN_TARGET": W_PATTERN_TARGET,
        "W_SALIENT": W_SALIENT,
    }
    with open(out_dir / "weighted_loss_config.json", "w") as f:
        json.dump(loss_config, f, indent=2)

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---", flush=True)

        for model_name in args.models:
            print(f"\n  [{model_name}] seed={seed}", flush=True)
            seed_everything(seed)
            dataset = MixedMemoryDatasetV3(n_samples=2000, seq_len=SEQ_LEN, seed=seed)

            seed_everything(seed)
            model = build_model(cfg, model_name)

            t0 = time.perf_counter()
            loss_curve, stable, ckpt_state = train_model(
                model, dataset, args.steps, device,
                checkpoint_at=args.checkpoint_at if use_checkpoint else None,
            )
            train_time = time.perf_counter() - t0
            final_loss = loss_curve[-1]["loss"] if loss_curve else float("nan")
            print(f"    Train: {train_time:.1f}s, loss={final_loss:.4f}, stable={stable}", flush=True)

            # Evaluate final model
            eval_result = evaluate_model(model, dataset, device, n_eval=400)
            composite = compute_composite(eval_result, stable)

            row = {
                "model": model_name, "seed": seed,
                "step": args.steps,
                **eval_result,
                "composite_score": composite,
                "final_loss": final_loss,
                "train_time_s": train_time,
                "stable": stable,
            }
            all_results.append(row)

            print(f"    recent={eval_result['recent_exact_acc']:.3f} "
                  f"old_tok={eval_result['old_fact_token_acc']:.3f} "
                  f"old_span={eval_result['old_fact_span_exact_acc']:.3f} "
                  f"old_part={eval_result['old_fact_span_partial_acc']:.3f} "
                  f"pat={eval_result['pattern_token_acc']:.3f} "
                  f"sal={eval_result['salient_event_acc']:.3f} "
                  f"comp={composite:.4f}", flush=True)
            print(f"    gate: kv={eval_result['router_kv_ratio']:.3f} "
                  f"trn={eval_result['router_trn_ratio']:.3f} "
                  f"ret={eval_result['router_ret_ratio']:.3f}", flush=True)

            # Evaluate checkpoint model if available
            if ckpt_state is not None:
                model.load_state_dict(ckpt_state)
                ckpt_eval = evaluate_model(model, dataset, device, n_eval=400)
                ckpt_composite = compute_composite(ckpt_eval, stable)
                ckpt_row = {
                    "model": model_name, "seed": seed,
                    "step": args.checkpoint_at,
                    **ckpt_eval,
                    "composite_score": ckpt_composite,
                    "final_loss": final_loss,
                    "train_time_s": train_time,
                    "stable": stable,
                }
                checkpoint_results.append(ckpt_row)
                print(f"    [ckpt@{args.checkpoint_at}] "
                      f"old_tok={ckpt_eval['old_fact_token_acc']:.3f} "
                      f"old_span={ckpt_eval['old_fact_span_exact_acc']:.3f} "
                      f"pat={ckpt_eval['pattern_token_acc']:.3f} "
                      f"sal={ckpt_eval['salient_event_acc']:.3f} "
                      f"comp={ckpt_composite:.4f}", flush=True)

            # Per-seed save
            seed_file = out_dir / f"seed_{seed}_data.json"
            seed_data = [r for r in all_results if r["seed"] == seed]
            with open(seed_file, "w") as f:
                json.dump(seed_data, f, indent=2, default=str)

    # Gate judgment (on final results)
    gate = gate_judgment(all_results)
    print(f"\n=== GATE VERDICT: {gate['verdict']} ===", flush=True)
    for crit_name, crit in gate["criteria"].items():
        status = "PASS" if crit["pass"] else "FAIL"
        print(f"  {crit_name}: {status}")

    # Derived metrics
    final_means = compute_model_means(all_results)
    derived_final = compute_derived_metrics(final_means)
    print(f"\n=== DERIVED METRICS (step={args.steps}) ===")
    for k, v in derived_final.items():
        print(f"  {k}: {v:+.4f}")

    # V3-3000 verdict if checkpoint available
    if checkpoint_results:
        ckpt_means = compute_model_means(checkpoint_results)
        derived_ckpt = compute_derived_metrics(ckpt_means)

        print(f"\n=== DERIVED METRICS (step={args.checkpoint_at}) ===")
        for k, v in derived_ckpt.items():
            print(f"  {k}: {v:+.4f}")

        v3k = v3_3000_verdict(derived_ckpt, derived_final)
        print(f"\n=== V3-3000 VERDICT: {v3k['verdict']} ({v3k['n_pass']}/{v3k['n_total']} pass) ===")
        for check_name, check in v3k["checks"].items():
            status = "PASS" if check["pass"] else "FAIL"
            print(f"  {check_name}: {status} "
                  f"(ckpt={check['ckpt']:+.4f}, final={check['final']:+.4f}, "
                  f"growth={check['growth']:+.4f})")
        print(f"\n  Interpretation: {v3k['interpretation']}")

        # Save V3-3000 verdict
        with open(out_dir / "v3_3000_verdict.json", "w") as f:
            json.dump(v3k, f, indent=2, default=str)

        # Save checkpoint results
        with open(out_dir / "checkpoint_results.json", "w") as f:
            json.dump(checkpoint_results, f, indent=2, default=str)

    # Save CSV
    csv_path = out_dir / "internal_results.csv"
    fieldnames = [
        "model", "seed", "step", "recent_exact_acc",
        "old_fact_token_acc", "old_fact_span_exact_acc", "old_fact_span_partial_acc",
        "pattern_token_acc", "salient_event_acc",
        "composite_score",
        "router_kv_ratio", "router_trn_ratio", "router_ret_ratio",
        "retrieval_calls", "archive_chunk_count",
        "final_loss", "train_time_s", "stable",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # Save gate
    with open(out_dir / "internal_gate.json", "w") as f:
        json.dump(gate, f, indent=2, default=str)

    # Save all results
    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save derived metrics
    with open(out_dir / "derived_metrics.json", "w") as f:
        json.dump({
            "final": derived_final,
            "checkpoint": compute_derived_metrics(compute_model_means(checkpoint_results))
            if checkpoint_results else None,
        }, f, indent=2)

    # Plots and summary
    generate_plots(all_results, out_dir)
    generate_summary(
        all_results, gate, sanity, out_dir, args.steps, args.seeds,
        checkpoint_results=checkpoint_results if checkpoint_results else None,
        checkpoint_step=args.checkpoint_at if checkpoint_results else None,
    )

    # Print cross-step comparison table
    if checkpoint_results:
        print(f"\n=== CROSS-STEP COMPARISON (step {args.checkpoint_at} vs {args.steps}) ===")
        ckpt_means = compute_model_means(checkpoint_results)
        header = f"{'Model':<12} {'Metric':<25} {'ckpt':>7} {'final':>7} {'delta':>7}"
        print(header)
        print("-" * len(header))
        for m in args.models:
            for metric in ["old_fact_token_acc", "old_fact_span_exact_acc",
                           "pattern_token_acc", "salient_event_acc", "composite_score"]:
                ck = ckpt_means.get(m, {}).get(metric, 0)
                fn = final_means.get(m, {}).get(metric, 0)
                delta = fn - ck
                print(f"{m:<12} {metric:<25} {ck:>7.4f} {fn:>7.4f} {delta:>+7.4f}")

    # Print final table
    print(f"\n=== V3 COMPARISON TABLE (mean across seeds, step={args.steps}) ===")
    header = f"{'Model':<12} {'Recent':>7} {'OldTok':>7} {'OldSpan':>8} {'OldPart':>8} {'Pattern':>8} {'Salient':>8} {'Comp':>7}"
    print(header)
    print("-" * len(header))
    for m in args.models:
        rows = [r for r in all_results if r["model"] == m]
        if not rows:
            continue
        n = len(rows)
        print(f"{m:<12} "
              f"{sum(r['recent_exact_acc'] for r in rows) / n:>7.4f} "
              f"{sum(r['old_fact_token_acc'] for r in rows) / n:>7.4f} "
              f"{sum(r['old_fact_span_exact_acc'] for r in rows) / n:>8.4f} "
              f"{sum(r['old_fact_span_partial_acc'] for r in rows) / n:>8.4f} "
              f"{sum(r['pattern_token_acc'] for r in rows) / n:>8.4f} "
              f"{sum(r['salient_event_acc'] for r in rows) / n:>8.4f} "
              f"{sum(r['composite_score'] for r in rows) / n:>7.4f}")

    print(f"\n[DONE] All results saved to {out_dir}/")


if __name__ == "__main__":
    main()
