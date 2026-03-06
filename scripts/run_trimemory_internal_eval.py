#!/usr/bin/env python3
"""Tri-Memory Internal Validation: 4-way path ablation across seeds.

Validates each memory tier's role:
  KV        = recent exact memory
  TRN       = long-range pattern memory
  Retrieval = old factual memory

All models use TriMemoryEngine with paths enabled/disabled.
No external baselines -- internal comparison only.

Usage:
    python scripts/run_trimemory_internal_eval.py \
      --models kv kv_trn kv_ret trimemory \
      --steps 3000 --seeds 0 1 2 \
      --output artifacts/trimemory_internal/
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
# Constants (match bench_trimemory_mixed.py)
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

RECENT_VALUE_LOW = 200
RECENT_VALUE_HIGH = 220
OLD_FACT_LOW = 220
OLD_FACT_HIGH = 240
PATTERN_TOKEN_LOW = 10
PATTERN_TOKEN_HIGH = 100
FILLER_LOW = 100
FILLER_HIGH = 200
QUERY_RECENT = 5
QUERY_OLD_FACT = 6
QUERY_PATTERN = 7


# ---------------------------------------------------------------------------
# Dataset (identical to bench_trimemory_mixed.py)
# ---------------------------------------------------------------------------
class MixedMemoryDataset(Dataset):
    def __init__(
        self, n_samples: int = 2000, seq_len: int = 256,
        old_fact_distance: int = 150, pattern_period: int = 8,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.pattern_period = pattern_period
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        seq = torch.randint(FILLER_LOW, FILLER_HIGH, (self.seq_len,), generator=self.rng)

        old_fact = torch.randint(OLD_FACT_LOW, OLD_FACT_HIGH, (1,), generator=self.rng).item()
        seq[0] = old_fact

        pattern_len = self.pattern_period
        pattern = torch.randint(PATTERN_TOKEN_LOW, PATTERN_TOKEN_HIGH, (pattern_len,), generator=self.rng)
        for i in range(30, 90):
            seq[i] = pattern[i % pattern_len]
        pattern_answer = pattern[0].item()

        recent_val = torch.randint(RECENT_VALUE_LOW, RECENT_VALUE_HIGH, (1,), generator=self.rng).item()
        seq[-20] = recent_val

        query_start = self.seq_len - 6
        seq[query_start] = QUERY_OLD_FACT
        seq[query_start + 1] = old_fact
        seq[query_start + 2] = QUERY_RECENT
        seq[query_start + 3] = recent_val
        seq[query_start + 4] = QUERY_PATTERN
        seq[query_start + 5] = pattern_answer

        return {
            "input_ids": seq,
            "old_fact": old_fact,
            "recent_val": recent_val,
            "pattern_answer": pattern_answer,
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


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    model: TriMemoryEngine, dataset: MixedMemoryDataset,
    steps: int, device: torch.device, lr: float = LR,
) -> tuple[list[dict], bool]:
    """Train model. Returns (loss_curve, stable)."""
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
        out = model(ids, labels=ids)
        loss = out["loss"]

        if not torch.isfinite(loss):
            stable = False
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == steps:
            records.append({"step": step, "loss": loss.item()})
            if step % 500 == 0:
                print(f"      step {step}/{steps} loss={loss.item():.4f}", flush=True)

    return records, stable


# ---------------------------------------------------------------------------
# Evaluation with telemetry
# ---------------------------------------------------------------------------
def evaluate_model(
    model: TriMemoryEngine, dataset: MixedMemoryDataset,
    device: torch.device, n_eval: int = 400,
) -> dict:
    """Evaluate accuracy + collect gate/retrieval telemetry."""
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=min(32, n_eval), shuffle=False)

    recent_correct = 0
    old_fact_correct = 0
    pattern_correct = 0
    total = 0

    # Telemetry accumulators
    gate_kv_sum = 0.0
    gate_trn_sum = 0.0
    gate_ret_sum = 0.0
    retrieval_calls = 0
    retrieval_used_count = 0
    total_chunks = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            if total >= n_eval:
                break
            ids = batch["input_ids"].to(device)
            B = ids.size(0)
            out = model(ids)
            logits = out["logits"]

            query_start = dataset.seq_len - 6

            old_preds = logits[:, query_start, :].argmax(dim=-1).cpu()
            old_fact_correct += (old_preds == batch["old_fact"]).sum().item()

            recent_preds = logits[:, query_start + 2, :].argmax(dim=-1).cpu()
            recent_correct += (recent_preds == batch["recent_val"]).sum().item()

            pattern_preds = logits[:, query_start + 4, :].argmax(dim=-1).cpu()
            pattern_correct += (pattern_preds == batch["pattern_answer"]).sum().item()

            total += B

            # Collect telemetry
            tel = model.collect_gate_telemetry()
            gate_kv_sum += tel["router_kv_ratio"]
            gate_trn_sum += tel["router_trn_ratio"]
            gate_ret_sum += tel["router_ret_ratio"]
            if tel["retrieval_used"]:
                retrieval_used_count += 1
            retrieval_calls += 1
            total_chunks += tel["archive_chunk_count"]
            n_batches += 1

    n_batches = max(n_batches, 1)
    recent_acc = recent_correct / max(total, 1)
    old_fact_acc = old_fact_correct / max(total, 1)
    pattern_acc = pattern_correct / max(total, 1)

    return {
        "recent_exact_acc": recent_acc,
        "old_fact_acc": old_fact_acc,
        "long_pattern_acc": pattern_acc,
        "n_eval": total,
        "router_kv_ratio": gate_kv_sum / n_batches,
        "router_trn_ratio": gate_trn_sum / n_batches,
        "router_ret_ratio": gate_ret_sum / n_batches,
        "retrieval_calls": retrieval_calls,
        "retrieval_hit_rate": retrieval_used_count / max(retrieval_calls, 1),
        "archive_chunk_count": total_chunks / n_batches,
        "state_bytes": model.state_memory_bytes,
        "retained_kv_tokens": model.window_size,
    }


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------
def compute_composite(recent: float, old_fact: float, pattern: float, stable: bool) -> float:
    stability = 1.0 if stable else 0.0
    return 0.30 * recent + 0.30 * old_fact + 0.30 * pattern + 0.10 * stability


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_check(model_name: str, result: dict) -> tuple[bool, list[str]]:
    """Returns (valid, list_of_failures)."""
    failures = []
    has_ret = MODEL_CONFIGS[model_name]["enable_retrieval"]

    if has_ret:
        if result["archive_chunk_count"] <= 0:
            failures.append("archive_chunk_count <= 0")
        if result["retrieval_calls"] <= 0:
            failures.append("retrieval_calls <= 0")
        if result["router_ret_ratio"] <= 0.05:
            failures.append(f"router_ret_ratio={result['router_ret_ratio']:.4f} <= 0.05")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Gate judgment
# ---------------------------------------------------------------------------
def gate_judgment(all_results: dict) -> dict:
    """Evaluate Go/No-Go criteria across all seeds."""
    # Aggregate by model (mean across seeds)
    model_means = {}
    for model_name in MODEL_CONFIGS:
        seeds_data = [r for r in all_results if r["model"] == model_name]
        if not seeds_data:
            continue
        model_means[model_name] = {
            "composite": sum(r["composite_score"] for r in seeds_data) / len(seeds_data),
            "recent": sum(r["recent_exact_acc"] for r in seeds_data) / len(seeds_data),
            "old_fact": sum(r["old_fact_acc"] for r in seeds_data) / len(seeds_data),
            "pattern": sum(r["long_pattern_acc"] for r in seeds_data) / len(seeds_data),
        }

    criteria = {}

    # D composite > max(A, B, C)
    d_comp = model_means.get("trimemory", {}).get("composite", 0.0)
    others_max = max(
        model_means.get("kv", {}).get("composite", 0.0),
        model_means.get("kv_trn", {}).get("composite", 0.0),
        model_means.get("kv_ret", {}).get("composite", 0.0),
    )
    criteria["composite_D_gt_max_ABC"] = {
        "pass": d_comp > others_max,
        "D": d_comp,
        "max_ABC": others_max,
    }

    # old_fact(C) > old_fact(A)
    c_old = model_means.get("kv_ret", {}).get("old_fact", 0.0)
    a_old = model_means.get("kv", {}).get("old_fact", 0.0)
    criteria["old_fact_C_gt_A"] = {
        "pass": c_old > a_old,
        "C": c_old,
        "A": a_old,
    }

    # pattern(B) > pattern(A)
    b_pat = model_means.get("kv_trn", {}).get("pattern", 0.0)
    a_pat = model_means.get("kv", {}).get("pattern", 0.0)
    criteria["pattern_B_gt_A"] = {
        "pass": b_pat > a_pat,
        "B": b_pat,
        "A": a_pat,
    }

    all_pass = all(c["pass"] for c in criteria.values())

    return {
        "verdict": "INTERNAL_GO" if all_pass else "INTERNAL_FAIL",
        "criteria": criteria,
        "model_means": model_means,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_plots(all_results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available, skipping plots")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Aggregate by model
    models = list(MODEL_CONFIGS.keys())
    model_data = {m: [r for r in all_results if r["model"] == m] for m in models}

    # -- accuracy_by_model.png --
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = ["recent_exact_acc", "old_fact_acc", "long_pattern_acc", "composite_score"]
    titles = ["Recent Exact", "Old Fact", "Long Pattern", "Composite"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for ax, metric, title in zip(axes, metrics, titles):
        means = []
        stds = []
        for m in models:
            vals = [r[metric] for r in model_data[m]]
            means.append(sum(vals) / len(vals) if vals else 0)
            if len(vals) > 1:
                mean = means[-1]
                stds.append((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5)
            else:
                stds.append(0)
        bars = ax.bar(range(len(models)), means, yerr=stds, capsize=4,
                      color=colors[:len(models)], alpha=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(["A:KV", "B:+TRN", "C:+Ret", "D:Full"], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, max(max(means) * 1.5, 0.05))
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "accuracy_by_model.png", dpi=150)
    plt.close(fig)

    # -- router_usage.png --
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(models))
    width = 0.25
    for i, (gate, label) in enumerate(
        [("router_kv_ratio", "KV"), ("router_trn_ratio", "TRN"), ("router_ret_ratio", "Ret")]
    ):
        vals = [sum(r[gate] for r in model_data[m]) / max(len(model_data[m]), 1) for m in models]
        ax.bar([xi + i * width for xi in x], vals, width, label=label, alpha=0.8)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(["A:KV", "B:+TRN", "C:+Ret", "D:Full"])
    ax.set_ylabel("Gate ratio")
    ax.set_title("Router Gate Usage by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "router_usage.png", dpi=150)
    plt.close(fig)

    # -- memory_usage.png --
    fig, ax = plt.subplots(figsize=(6, 4))
    state_bytes = [model_data[m][0]["state_bytes"] if model_data[m] else 0 for m in models]
    kv_tokens = [model_data[m][0]["retained_kv_tokens"] if model_data[m] else 0 for m in models]
    archive = [sum(r["archive_chunk_count"] for r in model_data[m]) / max(len(model_data[m]), 1)
               for m in models]

    ax.bar(range(len(models)), [s / 1024 for s in state_bytes], label="State (KB)", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(len(models)), archive, "ro-", label="Archive chunks")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["A:KV", "B:+TRN", "C:+Ret", "D:Full"])
    ax.set_ylabel("State (KB)")
    ax2.set_ylabel("Archive chunks")
    ax.set_title("Memory Usage")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plots_dir / "memory_usage.png", dpi=150)
    plt.close(fig)

    # -- retrieval_stats.png --
    fig, ax = plt.subplots(figsize=(6, 4))
    hit_rates = [sum(r["retrieval_hit_rate"] for r in model_data[m]) / max(len(model_data[m]), 1)
                 for m in models]
    calls = [sum(r["retrieval_calls"] for r in model_data[m]) / max(len(model_data[m]), 1)
             for m in models]
    ax.bar(range(len(models)), hit_rates, alpha=0.7, label="Hit rate")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["A:KV", "B:+TRN", "C:+Ret", "D:Full"])
    ax.set_ylabel("Retrieval hit rate")
    ax.set_title("Retrieval Statistics")
    for i, (hr, c) in enumerate(zip(hit_rates, calls)):
        ax.text(i, hr + 0.02, f"calls={c:.0f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "retrieval_stats.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}/")


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def generate_summary(
    all_results: list[dict], gate: dict, sanity_failures: dict,
    out_dir: Path, steps: int, seeds: list[int],
) -> None:
    models = list(MODEL_CONFIGS.keys())
    model_data = {m: [r for r in all_results if r["model"] == m] for m in models}

    lines = [
        "# Tri-Memory Internal Validation Results",
        "",
        f"**Steps**: {steps}  |  **Seeds**: {seeds}  |  **Verdict**: {gate['verdict']}",
        "",
        "## Accuracy Summary (mean +/- std across seeds)",
        "",
        "| Model | Recent | OldFact | Pattern | Composite |",
        "|-------|--------|---------|---------|-----------|",
    ]
    for m in models:
        vals = model_data[m]
        if not vals:
            continue
        for metric, label in [
            ("recent_exact_acc", "recent"), ("old_fact_acc", "old"),
            ("long_pattern_acc", "pat"), ("composite_score", "comp"),
        ]:
            pass

        def stat(key):
            v = [r[key] for r in vals]
            mean = sum(v) / len(v)
            if len(v) > 1:
                std = (sum((x - mean) ** 2 for x in v) / len(v)) ** 0.5
            else:
                std = 0
            return f"{mean:.4f}+/-{std:.4f}"

        label = {"kv": "A:KV", "kv_trn": "B:KV+TRN", "kv_ret": "C:KV+Ret", "trimemory": "D:Full"}[m]
        lines.append(
            f"| {label} | {stat('recent_exact_acc')} | {stat('old_fact_acc')} "
            f"| {stat('long_pattern_acc')} | {stat('composite_score')} |"
        )

    lines.extend([
        "",
        "## Router Gate Usage (mean across seeds)",
        "",
        "| Model | g_kv | g_trn | g_ret |",
        "|-------|------|-------|-------|",
    ])
    for m in models:
        vals = model_data[m]
        if not vals:
            continue
        label = {"kv": "A:KV", "kv_trn": "B:KV+TRN", "kv_ret": "C:KV+Ret", "trimemory": "D:Full"}[m]
        gk = sum(r["router_kv_ratio"] for r in vals) / len(vals)
        gt = sum(r["router_trn_ratio"] for r in vals) / len(vals)
        gr = sum(r["router_ret_ratio"] for r in vals) / len(vals)
        lines.append(f"| {label} | {gk:.4f} | {gt:.4f} | {gr:.4f} |")

    lines.extend([
        "",
        "## Gate Judgment",
        "",
    ])
    for crit_name, crit in gate["criteria"].items():
        status = "PASS" if crit["pass"] else "FAIL"
        detail = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in crit.items() if k != "pass")
        lines.append(f"- **{crit_name}**: {status} ({detail})")

    lines.extend(["", f"**Verdict: {gate['verdict']}**", ""])

    # Key findings
    lines.extend(["## Key Findings", ""])
    mm = gate["model_means"]

    # TRN contribution
    b_pat = mm.get("kv_trn", {}).get("pattern", 0)
    a_pat = mm.get("kv", {}).get("pattern", 0)
    lines.append(f"### TRN Contribution")
    lines.append(f"- Pattern: B({b_pat:.4f}) vs A({a_pat:.4f}) -- "
                 f"{'TRN helps' if b_pat > a_pat else 'TRN does not help'}")
    lines.append("")

    # Retrieval contribution
    c_old = mm.get("kv_ret", {}).get("old_fact", 0)
    a_old = mm.get("kv", {}).get("old_fact", 0)
    lines.append(f"### Retrieval Contribution")
    lines.append(f"- OldFact: C({c_old:.4f}) vs A({a_old:.4f}) -- "
                 f"{'Retrieval helps' if c_old > a_old else 'Retrieval does not help'}")
    lines.append("")

    # Full synergy
    d_comp = mm.get("trimemory", {}).get("composite", 0)
    others_max_val = max(
        mm.get("kv", {}).get("composite", 0),
        mm.get("kv_trn", {}).get("composite", 0),
        mm.get("kv_ret", {}).get("composite", 0),
    )
    lines.append(f"### Full Tri-Memory Synergy")
    lines.append(f"- Composite: D({d_comp:.4f}) vs max(A,B,C)({others_max_val:.4f}) -- "
                 f"{'Synergy confirmed' if d_comp > others_max_val else 'No synergy'}")
    lines.append("")

    # Sanity check results
    if sanity_failures:
        lines.extend(["## Sanity Check Failures", ""])
        for key, fails in sanity_failures.items():
            lines.append(f"- **{key}**: {', '.join(fails)}")
        lines.append("")

    # Failure analysis
    if gate["verdict"] == "INTERNAL_FAIL":
        lines.extend(["## Failure Analysis", ""])
        if not gate["criteria"]["composite_D_gt_max_ABC"]["pass"]:
            lines.append("- **Router collapse**: D does not outperform all ablations on composite.")
            d_ret = mm.get("trimemory", {})
            if d_ret.get("old_fact", 0) < mm.get("kv_ret", {}).get("old_fact", 0):
                lines.append("  - Retrieval underperforming in full model -- gate may be suppressing it.")
            if d_ret.get("pattern", 0) < mm.get("kv_trn", {}).get("pattern", 0):
                lines.append("  - TRN underperforming in full model -- interference with retrieval path.")
        if not gate["criteria"]["old_fact_C_gt_A"]["pass"]:
            lines.append("- **Retrieval useless**: C does not beat A on old_fact.")
            lines.append("  - Retrieval search may not be finding relevant chunks.")
        if not gate["criteria"]["pattern_B_gt_A"]["pass"]:
            lines.append("- **TRN underfit**: B does not beat A on pattern.")
            lines.append("  - TRN may need more training steps or the pattern task is too hard.")
        lines.append("")

    # Next step
    lines.extend([
        "## Next Step",
        "",
    ])
    if gate["verdict"] == "INTERNAL_GO":
        lines.append("-> Proceed to streaming evaluation (forward_with_memory)")
    else:
        lines.append("-> Router / retrieval / dataset redesign needed")
    lines.append("")

    with open(out_dir / "internal_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Tri-Memory Internal Validation")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/trimemory_internal/")
    args = parser.parse_args()

    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Tri-Memory Internal Validation]")
    print(f"  models: {args.models}")
    print(f"  steps: {args.steps}, seeds: {args.seeds}")
    print(f"  device: {args.device}")
    print(f"  output: {out_dir}")
    print(flush=True)

    cfg = make_cfg()
    all_results: list[dict] = []
    sanity_failures: dict = {}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---", flush=True)

        for model_name in args.models:
            print(f"\n  [{model_name}] seed={seed}", flush=True)
            seed_everything(seed)
            dataset = MixedMemoryDataset(n_samples=2000, seq_len=SEQ_LEN, seed=seed)

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
            composite = compute_composite(
                eval_result["recent_exact_acc"],
                eval_result["old_fact_acc"],
                eval_result["long_pattern_acc"],
                stable,
            )

            row = {
                "model": model_name,
                "seed": seed,
                "recent_exact_acc": eval_result["recent_exact_acc"],
                "old_fact_acc": eval_result["old_fact_acc"],
                "long_pattern_acc": eval_result["long_pattern_acc"],
                "composite_score": composite,
                "router_kv_ratio": eval_result["router_kv_ratio"],
                "router_trn_ratio": eval_result["router_trn_ratio"],
                "router_ret_ratio": eval_result["router_ret_ratio"],
                "retrieval_calls": eval_result["retrieval_calls"],
                "retrieval_hit_rate": eval_result["retrieval_hit_rate"],
                "archive_chunk_count": eval_result["archive_chunk_count"],
                "state_bytes": eval_result["state_bytes"],
                "retained_kv_tokens": eval_result["retained_kv_tokens"],
                "final_loss": final_loss,
                "train_time_s": train_time,
                "stable": stable,
            }
            all_results.append(row)

            print(f"    recent={row['recent_exact_acc']:.3f} old={row['old_fact_acc']:.3f} "
                  f"pat={row['long_pattern_acc']:.3f} comp={composite:.4f}", flush=True)
            print(f"    gate: kv={row['router_kv_ratio']:.3f} trn={row['router_trn_ratio']:.3f} "
                  f"ret={row['router_ret_ratio']:.3f}", flush=True)

            # Sanity check
            valid, fails = sanity_check(model_name, row)
            if not valid:
                key = f"{model_name}_seed{seed}"
                sanity_failures[key] = fails
                print(f"    [SANITY FAIL] {fails}", flush=True)

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
    csv_path = out_dir / "internal_results.csv"
    fieldnames = [
        "model", "seed", "recent_exact_acc", "old_fact_acc", "long_pattern_acc",
        "composite_score", "router_kv_ratio", "router_trn_ratio", "router_ret_ratio",
        "retrieval_calls", "retrieval_hit_rate", "archive_chunk_count",
        "state_bytes", "retained_kv_tokens",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # Save gate result
    with open(out_dir / "internal_gate.json", "w") as f:
        json.dump(gate, f, indent=2, default=str)

    # Generate plots
    generate_plots(all_results, out_dir)

    # Generate summary
    generate_summary(all_results, gate, sanity_failures, out_dir, args.steps, args.seeds)

    print(f"\n[DONE] All results saved to {out_dir}/")

    # Print comparison table
    print(f"\n=== COMPARISON TABLE (mean across seeds) ===")
    print(f"{'Model':<12} {'Recent':>8} {'OldFact':>8} {'Pattern':>8} {'Composite':>10}")
    print("-" * 52)
    for m in args.models:
        rows = [r for r in all_results if r["model"] == m]
        if not rows:
            continue
        n = len(rows)
        recent = sum(r["recent_exact_acc"] for r in rows) / n
        old = sum(r["old_fact_acc"] for r in rows) / n
        pat = sum(r["long_pattern_acc"] for r in rows) / n
        comp = sum(r["composite_score"] for r in rows) / n
        print(f"{m:<12} {recent:>8.4f} {old:>8.4f} {pat:>8.4f} {comp:>10.4f}")


if __name__ == "__main__":
    main()
