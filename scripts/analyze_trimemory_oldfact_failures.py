#!/usr/bin/env python3
"""Error analysis for Tri-Memory old_fact failures.

Classifies failures into:
  Type A: retrieval_search_failure -- top-k does not contain gold
  Type B: retrieval_found_but_gate_ignored -- gold found, gate ret ratio low
  Type C: retrieval_found_and_gate_used_but_decode_failed -- gold found, gate used, still wrong
  Type D: no_retrieval_call -- retrieval not triggered
  Type E: ambiguous_or_other

Usage:
    python scripts/analyze_trimemory_oldfact_failures.py \
        --input artifacts/trimemory_error_analysis/<ts>/all_episode_data.json \
        --failure-cases artifacts/trimemory_error_analysis/<ts>/oldfact_failure_cases.jsonl \
        --top-k 5 \
        --router-ret-threshold 0.15 \
        --output-dir artifacts/trimemory_error_analysis/<ts>/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Failure case data
# ---------------------------------------------------------------------------
@dataclass
class OldFactFailureCase:
    seed: int
    episode_id: int
    query_tokens: list[int]
    gold_answer_token: int
    pred_answer_token: int
    correct: bool
    retrieval_called: bool
    retrieval_topk_chunks: list[list[int]]
    retrieval_topk_scores: list[float]
    retrieval_topk_contains_gold: bool
    retrieval_top1_contains_gold: bool
    router_kv_ratio: float
    router_trn_ratio: float
    router_ret_ratio: float
    retrieval_context_length: int
    archive_chunk_count: int
    fact_span_positions: list[int]
    retrieved_chunk_positions: list[list[int]]
    decoder_topk_tokens: list[int]
    gate_logits_raw: list[float]
    gate_probs: list[float]
    failure_type: str = ""
    search_mode: str = "bag"
    search_w_hidden: float = 0.7
    search_w_bag: float = 0.3
    retrieval_topk_score_breakdown: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------
def classify_failure(
    case: OldFactFailureCase,
    router_ret_low: float,
) -> str:
    """Classify a failure case into Type A-E."""
    if case.correct:
        return "correct"  # not a failure

    if not case.retrieval_called:
        return "D_no_retrieval_call"

    if not case.retrieval_topk_contains_gold:
        return "A_retrieval_search_failure"

    if case.router_ret_ratio < router_ret_low:
        return "B_retrieval_found_but_gate_ignored"

    return "C_retrieval_found_gate_used_decode_failed"


def contains_gold_token(chunk_tokens: list[int], gold_token: int) -> bool:
    """Check if gold answer token appears in the chunk's token list."""
    return gold_token in chunk_tokens


# ---------------------------------------------------------------------------
# Load and process
# ---------------------------------------------------------------------------
def load_failure_cases(jsonl_path: str) -> list[OldFactFailureCase]:
    cases = []
    # Get valid field names for the dataclass
    valid_fields = {f.name for f in OldFactFailureCase.__dataclass_fields__.values()}
    with open(jsonl_path, "r") as f:
        for line in f:
            d = json.loads(line.strip())
            # Filter to known fields only (backward compat with older JSONLs)
            filtered = {k: v for k, v in d.items() if k in valid_fields}
            cases.append(OldFactFailureCase(**filtered))
    return cases


def compute_summaries(
    cases: list[OldFactFailureCase],
    router_ret_low: float,
) -> dict:
    """Compute all aggregate statistics."""
    total = len(cases)
    failures = [c for c in cases if not c.correct]
    successes = [c for c in cases if c.correct]
    n_fail = len(failures)
    n_success = len(successes)

    # Classify failures
    for c in failures:
        c.failure_type = classify_failure(c, router_ret_low)

    type_counts = Counter(c.failure_type for c in failures)

    # Retrieval quality
    gold_found_cases = [c for c in cases if c.retrieval_topk_contains_gold]
    gold_not_found_cases = [c for c in cases if c.retrieval_called and not c.retrieval_topk_contains_gold]
    top1_gold_cases = [c for c in cases if c.retrieval_top1_contains_gold]

    topk_gold_rate = len(gold_found_cases) / max(total, 1)
    top1_gold_rate = len(top1_gold_cases) / max(total, 1)

    scores_when_gold_found = [
        c.retrieval_topk_scores[0] if c.retrieval_topk_scores else 0.0
        for c in gold_found_cases
    ]
    scores_when_gold_not_found = [
        c.retrieval_topk_scores[0] if c.retrieval_topk_scores else 0.0
        for c in gold_not_found_cases
    ]

    # Gate usage
    ret_ratios_gold_found = [c.router_ret_ratio for c in gold_found_cases]
    ret_ratios_gold_not_found = [c.router_ret_ratio for c in gold_not_found_cases]

    # Decode success when gold found + gate used
    gold_found_gate_used = [
        c for c in cases
        if c.retrieval_topk_contains_gold and c.router_ret_ratio >= router_ret_low
    ]
    decode_success_rate = (
        sum(1 for c in gold_found_gate_used if c.correct) / max(len(gold_found_gate_used), 1)
    )

    return {
        "total_episodes": total,
        "total_correct": n_success,
        "total_failures": n_fail,
        "failure_rate": n_fail / max(total, 1),
        "type_counts": dict(type_counts),
        "type_rates": {k: v / max(n_fail, 1) for k, v in type_counts.items()},
        "retrieval_quality": {
            "topk_contains_gold_rate": topk_gold_rate,
            "top1_contains_gold_rate": top1_gold_rate,
            "mean_score_when_gold_found": float(np.mean(scores_when_gold_found)) if scores_when_gold_found else 0.0,
            "mean_score_when_gold_not_found": float(np.mean(scores_when_gold_not_found)) if scores_when_gold_not_found else 0.0,
        },
        "gate_usage": {
            "mean_ret_ratio_when_gold_found": float(np.mean(ret_ratios_gold_found)) if ret_ratios_gold_found else 0.0,
            "mean_ret_ratio_when_gold_not_found": float(np.mean(ret_ratios_gold_not_found)) if ret_ratios_gold_not_found else 0.0,
        },
        "decode_summary": {
            "gold_found_and_gate_used_count": len(gold_found_gate_used),
            "decode_success_rate": decode_success_rate,
        },
    }


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def write_failure_type_csv(summary: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["failure_type", "count", "rate"])
        for ftype in ["A_retrieval_search_failure", "B_retrieval_found_but_gate_ignored",
                       "C_retrieval_found_gate_used_decode_failed", "D_no_retrieval_call",
                       "E_ambiguous_or_other"]:
            count = summary["type_counts"].get(ftype, 0)
            rate = summary["type_rates"].get(ftype, 0.0)
            w.writerow([ftype, count, f"{rate:.4f}"])


def write_retrieval_quality_csv(summary: dict, path: str) -> None:
    rq = summary["retrieval_quality"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in rq.items():
            w.writerow([k, f"{v:.4f}"])


def write_gate_usage_csv(summary: dict, path: str) -> None:
    gu = summary["gate_usage"]
    ds = summary["decode_summary"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in gu.items():
            w.writerow([k, f"{v:.4f}"])
        for k, v in ds.items():
            w.writerow([k, f"{v:.4f}" if isinstance(v, float) else str(v)])


def write_top_fail_examples(cases: list[OldFactFailureCase], path: str, n: int = 20) -> None:
    failures = [c for c in cases if not c.correct][:n]
    # Detect search mode from first case
    mode_info = ""
    if failures and failures[0].search_mode:
        c0 = failures[0]
        mode_info = f" (search_mode={c0.search_mode}, w_hidden={c0.search_w_hidden}, w_bag={c0.search_w_bag})"
    lines = [f"# Top {len(failures)} Old-Fact Failure Examples{mode_info}\n"]
    for i, c in enumerate(failures):
        lines.append(f"## Case {i+1} [seed={c.seed} / episode={c.episode_id}]\n")
        lines.append(f"- **query tokens**: {c.query_tokens}")
        lines.append(f"- **gold**: {c.gold_answer_token}")
        lines.append(f"- **pred**: {c.pred_answer_token}")
        lines.append(f"- **search mode**: {c.search_mode}")
        lines.append(f"- **retrieval called**: {c.retrieval_called}")
        lines.append(f"- **router ratios**: kv={c.router_kv_ratio:.3f} trn={c.router_trn_ratio:.3f} ret={c.router_ret_ratio:.3f}")
        if c.retrieval_topk_scores:
            lines.append(f"- **top-k scores**: {[f'{s:.4f}' for s in c.retrieval_topk_scores]}")
        if c.retrieval_topk_score_breakdown:
            for j, sd in enumerate(c.retrieval_topk_score_breakdown[:3]):
                lines.append(f"  - chunk {j}: hidden={sd.get('hidden_score', 0):.4f} bag={sd.get('bag_score', 0):.4f} combined={sd.get('combined_score', 0):.4f}")
        if c.retrieval_topk_chunks:
            for j, chunk in enumerate(c.retrieval_topk_chunks[:3]):
                gold_in = contains_gold_token(chunk, c.gold_answer_token)
                lines.append(f"- **chunk {j}** (gold_in={gold_in}): {chunk[:20]}...")
        lines.append(f"- **gold in top-k**: {c.retrieval_topk_contains_gold}")
        lines.append(f"- **gold in top-1**: {c.retrieval_top1_contains_gold}")
        lines.append(f"- **decoder top-k tokens**: {c.decoder_topk_tokens[:10]}")
        lines.append(f"- **failure type**: **{c.failure_type}**")
        lines.append(f"- **archive chunks**: {c.archive_chunk_count}")
        lines.append(f"- **fact positions**: {c.fact_span_positions}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_summary_md(summary: dict, cases: list[OldFactFailureCase], path: str) -> None:
    tc = summary["type_counts"]
    rq = summary["retrieval_quality"]
    gu = summary["gate_usage"]
    ds = summary["decode_summary"]

    # Determine dominant failure
    sorted_types = sorted(tc.items(), key=lambda x: x[1], reverse=True)
    dominant = sorted_types[0] if sorted_types else ("none", 0)

    # Search mode info from cases
    search_mode_str = "bag (v1 default)"
    if cases:
        sm = cases[0].search_mode
        wh = cases[0].search_w_hidden
        wb = cases[0].search_w_bag
        search_mode_str = f"{sm} (w_hidden={wh}, w_bag={wb})"

    lines = [
        "# Old-Fact Failure Error Analysis\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Search mode: {search_mode_str}",
        f"Total episodes: {summary['total_episodes']}",
        f"Total correct: {summary['total_correct']}",
        f"Total failures: {summary['total_failures']}",
        f"Failure rate: {summary['failure_rate']:.3f}\n",
        "## 1. Main Diagnosis\n",
    ]
    for i, (ftype, count) in enumerate(sorted_types, 1):
        rate = summary["type_rates"].get(ftype, 0.0)
        lines.append(f"{i}. **{ftype}**: {count} ({rate:.1%})")
    lines.append("")

    lines.extend([
        "## 2. Evidence\n",
        f"- top-k gold containment rate: {rq['topk_contains_gold_rate']:.3f}",
        f"- top-1 gold containment rate: {rq['top1_contains_gold_rate']:.3f}",
        f"- mean retrieval score when gold found: {rq['mean_score_when_gold_found']:.4f}",
        f"- mean retrieval score when gold NOT found: {rq['mean_score_when_gold_not_found']:.4f}",
        f"- mean gate ret_ratio when gold found: {gu['mean_ret_ratio_when_gold_found']:.4f}",
        f"- mean gate ret_ratio when gold NOT found: {gu['mean_ret_ratio_when_gold_not_found']:.4f}",
        f"- gold found + gate used ({ds['gold_found_and_gate_used_count']} cases): decode success rate = {ds['decode_success_rate']:.3f}",
        "",
    ])

    lines.append("## 3. What This Means\n")
    if dominant[0] == "A_retrieval_search_failure":
        lines.append("Retrieval search is the primary bottleneck. The archive contains the old fact chunk "
                      "but bag-of-token cosine similarity fails to rank it in the top-k. "
                      "The query chunk at old_fact query time does not sufficiently overlap with the "
                      "archived chunk's token distribution.")
    elif dominant[0] == "B_retrieval_found_but_gate_ignored":
        lines.append("Retrieval search works -- it finds the gold fact. But the gate (router) assigns "
                      "too little weight to the retrieval path. The gate was not trained to recognize "
                      "that retrieval output is valuable for this query type.")
    elif dominant[0] == "C_retrieval_found_gate_used_decode_failed":
        lines.append("Both retrieval and gate are working, but the decoder/mixer cannot convert the "
                      "retrieved evidence into the correct answer token. The ret_proj or mixer "
                      "representation is insufficient.")
    elif dominant[0] == "D_no_retrieval_call":
        lines.append("Retrieval is not being called at old_fact query time. This means the archive "
                      "is empty or the router is not triggering search.")
    else:
        lines.append("Failure mode is ambiguous. Manual inspection required.")
    lines.append("")

    lines.append("## 4. Recommended Next Step\n")
    if dominant[0] == "A_retrieval_search_failure":
        lines.append("**Improve retrieval search.** The bag-of-token cosine search does not surface "
                      "the relevant chunk. Options: (a) switch to hidden-state cosine, (b) include "
                      "the gold token in the query chunk's bag more prominently, (c) use the "
                      "query_hidden for search, (d) index chunks with their original position metadata.")
    elif dominant[0] == "B_retrieval_found_but_gate_ignored":
        lines.append("**Tune gate to use retrieval.** Options: (a) initialize gate bias toward retrieval, "
                      "(b) add retrieval confidence as gate input, (c) add retrieval-aware supervision loss.")
    elif dominant[0] == "C_retrieval_found_gate_used_decode_failed":
        lines.append("**Improve ret_proj / mixer.** The retrieved hidden is not informative enough to "
                      "reconstruct the answer. Options: (a) replace mean-pooled hidden with token-level "
                      "cross-attention, (b) add retrieval-specific loss on the answer position.")
    elif dominant[0] == "D_no_retrieval_call":
        lines.append("**Ensure retrieval is triggered.** Check saliency threshold and eviction logic.")
    else:
        lines.append("**Manual investigation required.** Examine top_fail_examples.md for patterns.")
    lines.append("")

    lines.extend([
        "## 5. Failure Type Counts\n",
        "| Type | Count | Rate |",
        "|------|-------|------|",
    ])
    for ftype, count in sorted_types:
        rate = summary["type_rates"].get(ftype, 0.0)
        lines.append(f"| {ftype} | {count} | {rate:.1%} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_plots(cases: list[OldFactFailureCase], plot_dir: str) -> None:
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")
        return

    os.makedirs(plot_dir, exist_ok=True)
    failures = [c for c in cases if not c.correct]

    # 1. Failure type pie chart
    type_counts = Counter(c.failure_type for c in failures)
    if type_counts:
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = ["#e74c3c", "#f39c12", "#3498db", "#95a5a6", "#2ecc71"][:len(labels)]
        ax.pie(sizes, labels=[l.split("_", 1)[1] if "_" in l else l for l in labels],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title("Old-Fact Failure Types")
        fig.savefig(os.path.join(plot_dir, "failure_type_pie.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2. Router ret ratio histogram
    ret_ratios = [c.router_ret_ratio for c in cases]
    if ret_ratios:
        fig, ax = plt.subplots(figsize=(8, 5))
        correct_ratios = [c.router_ret_ratio for c in cases if c.correct]
        wrong_ratios = [c.router_ret_ratio for c in cases if not c.correct]
        ax.hist([correct_ratios, wrong_ratios], bins=20, label=["correct", "wrong"],
                alpha=0.7, color=["#2ecc71", "#e74c3c"])
        ax.set_xlabel("router_ret_ratio")
        ax.set_ylabel("count")
        ax.set_title("Router Retrieval Ratio: Correct vs Wrong")
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "router_ret_ratio_hist.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Retrieval score histogram
    all_scores = [c.retrieval_topk_scores[0] for c in cases if c.retrieval_topk_scores]
    if all_scores:
        fig, ax = plt.subplots(figsize=(8, 5))
        found_scores = [c.retrieval_topk_scores[0] for c in cases
                        if c.retrieval_topk_scores and c.retrieval_topk_contains_gold]
        not_found_scores = [c.retrieval_topk_scores[0] for c in cases
                            if c.retrieval_topk_scores and not c.retrieval_topk_contains_gold]
        ax.hist([found_scores, not_found_scores], bins=20,
                label=["gold in top-k", "gold NOT in top-k"],
                alpha=0.7, color=["#2ecc71", "#e74c3c"])
        ax.set_xlabel("top-1 retrieval score")
        ax.set_ylabel("count")
        ax.set_title("Retrieval Score Distribution")
        ax.legend()
        fig.savefig(os.path.join(plot_dir, "retrieval_score_hist.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Found vs not found summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    gold_found = sum(1 for c in cases if c.retrieval_topk_contains_gold)
    gold_not_found = sum(1 for c in cases if c.retrieval_called and not c.retrieval_topk_contains_gold)
    no_retrieval = sum(1 for c in cases if not c.retrieval_called)
    bars = ["gold in top-k", "gold NOT in top-k", "no retrieval call"]
    vals = [gold_found, gold_not_found, no_retrieval]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    ax.bar(bars, vals, color=colors)
    ax.set_ylabel("count")
    ax.set_title("Retrieval Gold Containment")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, str(v), ha="center")
    fig.savefig(os.path.join(plot_dir, "found_vs_not_found.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {plot_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Old-fact failure error analysis")
    parser.add_argument("--input", required=True, help="Path to oldfact_failure_cases.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--router-ret-threshold", type=float, default=0.15)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading failure cases from {args.input}")
    cases = load_failure_cases(args.input)
    print(f"  Loaded {len(cases)} cases")

    # Classify
    for c in cases:
        if not c.correct:
            c.failure_type = classify_failure(c, args.router_ret_threshold)
        else:
            c.failure_type = "correct"

    summary = compute_summaries(cases, args.router_ret_threshold)

    # Write outputs
    write_failure_type_csv(summary, os.path.join(args.output_dir, "failure_type_counts.csv"))
    write_retrieval_quality_csv(summary, os.path.join(args.output_dir, "retrieval_quality.csv"))
    write_gate_usage_csv(summary, os.path.join(args.output_dir, "gate_usage_summary.csv"))
    write_top_fail_examples(cases, os.path.join(args.output_dir, "top_fail_examples.md"), n=20)
    write_summary_md(summary, cases, os.path.join(args.output_dir, "oldfact_failure_summary.md"))

    # Plots
    print("  Generating plots...")
    generate_plots(cases, os.path.join(args.output_dir, "plots"))

    # Console summary
    print(f"\n{'='*60}")
    print(f"  Old-Fact Error Analysis Complete")
    print(f"  Total cases: {summary['total_episodes']}")
    print(f"  Correct: {summary['total_correct']}")
    print(f"  Failures: {summary['total_failures']}")
    print(f"  Failure rate: {summary['failure_rate']:.3f}")
    print(f"\n  Failure type breakdown:")
    for ftype, count in sorted(summary["type_counts"].items(), key=lambda x: x[1], reverse=True):
        rate = summary["type_rates"].get(ftype, 0.0)
        print(f"    {ftype}: {count} ({rate:.1%})")
    rq = summary["retrieval_quality"]
    print(f"\n  Retrieval quality:")
    print(f"    top-k gold rate: {rq['topk_contains_gold_rate']:.3f}")
    print(f"    top-1 gold rate: {rq['top1_contains_gold_rate']:.3f}")
    gu = summary["gate_usage"]
    print(f"\n  Gate usage:")
    print(f"    ret_ratio when gold found: {gu['mean_ret_ratio_when_gold_found']:.4f}")
    print(f"    ret_ratio when gold NOT found: {gu['mean_ret_ratio_when_gold_not_found']:.4f}")
    ds = summary["decode_summary"]
    print(f"\n  Decode: gold found + gate used -> success rate: {ds['decode_success_rate']:.3f}")
    print(f"\n  Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
