#!/usr/bin/env python3
"""Test config independence in V5 benchmark.

Runs Seed2 D config under 3 conditions:
  1. D-only (no prior configs)
  2. A->B->C->D (standard benchmark order)
  3. D-first (D run as first config)

Compares final eval metrics to detect config-order dependence.
"""
from __future__ import annotations

import argparse
import copy
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

from trimemory.config import TRNConfig
from trimemory.tri_memory import TriMemoryEngine

from run_trimemory_v3_eval import (
    MODEL_CONFIGS,
    WINDOW_SIZE, CHUNK_SIZE, D_MODEL, N_LAYERS, N_OSC, D_FF,
    VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, LR,
    compute_composite,
    seed_everything,
)
from run_trimemory_v5_trn_reeval import (
    RetrievalOnlyDataset,
    QUERY_MARKER_POS, MARKER_TEMPERATURE, QUERY_REGION_SIZE,
    H6_OLD_FACT_SPAN_LEN, W_RET_AUX,
    train_model, evaluate_h6, collect_retrieval_stats,
    weighted_cross_entropy_with_labels, compute_retrieval_aux_loss,
)


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


def run_single_config(
    cfg: TRNConfig,
    config_name: str,
    model_key: str,
    query_mode: str,
    decoder_mode: str,
    use_aux: bool,
    use_copy: bool,
    seed: int,
    steps: int,
    device: torch.device,
) -> dict:
    """Run a single config from scratch, return eval metrics."""
    seed_everything(seed)
    dataset = RetrievalOnlyDataset(n_samples=2000, seq_len=SEQ_LEN, seed=seed)
    seed_everything(seed)
    model = build_model(cfg, model_key)

    t0 = time.perf_counter()
    loss_curve, stable, _ = train_model(
        model, dataset, steps, device,
        query_mode=query_mode, decoder_mode=decoder_mode,
        use_aux_loss=use_aux, use_copy_head=use_copy,
        checkpoint_at=None,
    )
    train_time = time.perf_counter() - t0

    final_eval = evaluate_h6(
        model, dataset, device,
        query_mode=query_mode, decoder_mode=decoder_mode,
        eval_copy_head=use_copy,
    )
    final_comp = compute_composite({
        "recent_exact_acc": final_eval["recent"],
        "old_fact_span_exact_acc": final_eval["old_span"],
        "pattern_token_acc": final_eval["pattern"],
        "salient_event_acc": final_eval["salient"],
    }, stable)
    final_eval["composite"] = final_comp
    final_eval["stable"] = stable
    final_eval["train_time_s"] = train_time

    # Check for NaN in loss curve
    final_eval["has_nan_loss"] = any(
        not torch.tensor(r["loss"]).isfinite().item()
        for r in loss_curve
    ) if loss_curve else False
    final_eval["final_loss"] = loss_curve[-1]["loss"] if loss_curve else float("nan")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return final_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed independence test")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="artifacts/seed_independence_test/")
    args = parser.parse_args()

    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    steps = args.steps

    configs = [
        {"name": "A", "model": "kv",        "query_mode": "mean",   "decoder": "pooled",   "aux": False, "copy_head": False},
        {"name": "B", "model": "kv_trn",    "query_mode": "mean",   "decoder": "pooled",   "aux": False, "copy_head": False},
        {"name": "C", "model": "kv_ret",    "query_mode": "marker", "decoder": "copy_mix", "aux": True,  "copy_head": True},
        {"name": "D", "model": "trimemory", "query_mode": "marker", "decoder": "copy_mix", "aux": True,  "copy_head": True},
    ]
    d_config = configs[3]

    results = {}

    # =====================================================================
    # Condition 1: D-only (fresh CUDA context)
    # =====================================================================
    print("=" * 60)
    print(f"CONDITION 1: D-only (seed={seed}, steps={steps})")
    print("=" * 60, flush=True)

    cfg1 = make_cfg()
    r1 = run_single_config(
        cfg1, "D", d_config["model"],
        d_config["query_mode"], d_config["decoder"],
        d_config["aux"], d_config["copy_head"],
        seed, steps, device,
    )
    results["D_only"] = r1
    print(f"  D-only: pat={r1['pattern']:.4f} rec={r1['recent']:.4f} "
          f"comp={r1['composite']:.4f} stable={r1['stable']} "
          f"has_nan={r1['has_nan_loss']} final_loss={r1['final_loss']:.4f}")
    print(f"  time={r1['train_time_s']:.1f}s", flush=True)

    # =====================================================================
    # Condition 2: A->B->C->D (standard order)
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"CONDITION 2: A->B->C->D (seed={seed}, steps={steps})")
    print("=" * 60, flush=True)

    for exp in configs:
        name = exp["name"]
        print(f"\n  [{name}] model={exp['model']}", flush=True)
        cfg2 = make_cfg()
        r = run_single_config(
            cfg2, name, exp["model"],
            exp["query_mode"], exp["decoder"],
            exp["aux"], exp["copy_head"],
            seed, steps, device,
        )
        results[f"ABCD_{name}"] = r
        print(f"    {name}: pat={r['pattern']:.4f} rec={r['recent']:.4f} "
              f"comp={r['composite']:.4f} stable={r['stable']} "
              f"has_nan={r['has_nan_loss']} final_loss={r['final_loss']:.4f}")

    # =====================================================================
    # Condition 3: D-first (D before anything else after ABCD)
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"CONDITION 3: D-first after full ABCD run (seed={seed}, steps={steps})")
    print("=" * 60, flush=True)

    cfg3 = make_cfg()
    r3 = run_single_config(
        cfg3, "D", d_config["model"],
        d_config["query_mode"], d_config["decoder"],
        d_config["aux"], d_config["copy_head"],
        seed, steps, device,
    )
    results["D_after_ABCD"] = r3
    print(f"  D-after-ABCD: pat={r3['pattern']:.4f} rec={r3['recent']:.4f} "
          f"comp={r3['composite']:.4f} stable={r3['stable']} "
          f"has_nan={r3['has_nan_loss']} final_loss={r3['final_loss']:.4f}")
    print(f"  time={r3['train_time_s']:.1f}s", flush=True)

    # =====================================================================
    # Condition 4: Fresh process D-only (to rule out CUDA context)
    # We can't do fresh process here, but we can do another D-only after
    # resetting CUDA state more aggressively
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"CONDITION 4: D-only with aggressive CUDA reset (seed={seed}, steps={steps})")
    print("=" * 60, flush=True)

    # Aggressive CUDA cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    cfg4 = make_cfg()
    r4 = run_single_config(
        cfg4, "D", d_config["model"],
        d_config["query_mode"], d_config["decoder"],
        d_config["aux"], d_config["copy_head"],
        seed, steps, device,
    )
    results["D_only_reset"] = r4
    print(f"  D-only-reset: pat={r4['pattern']:.4f} rec={r4['recent']:.4f} "
          f"comp={r4['composite']:.4f} stable={r4['stable']} "
          f"has_nan={r4['has_nan_loss']} final_loss={r4['final_loss']:.4f}")
    print(f"  time={r4['train_time_s']:.1f}s", flush=True)

    # =====================================================================
    # Summary comparison
    # =====================================================================
    print("\n" + "=" * 60)
    print("SUMMARY: Seed2 D config independence test")
    print("=" * 60)

    header = f"{'Condition':<20} {'pattern':>8} {'recent':>8} {'old_tok':>8} {'comp':>8} {'stable':>7} {'NaN':>5} {'loss':>10}"
    print(header)
    print("-" * len(header))

    for label, key in [
        ("D-only (fresh)", "D_only"),
        ("ABCD -> D (seq)", "ABCD_D"),
        ("D after ABCD", "D_after_ABCD"),
        ("D-only (reset)", "D_only_reset"),
    ]:
        r = results[key]
        print(f"{label:<20} "
              f"{r['pattern']:>8.4f} "
              f"{r['recent']:>8.4f} "
              f"{r['old_tok']:>8.4f} "
              f"{r['composite']:>8.4f} "
              f"{str(r['stable']):>7} "
              f"{str(r['has_nan_loss']):>5} "
              f"{r['final_loss']:>10.4f}")

    # Check if results are identical (bitwise reproducibility)
    d_only = results["D_only"]
    d_seq = results["ABCD_D"]
    d_after = results["D_after_ABCD"]
    d_reset = results["D_only_reset"]

    print("\n--- Reproducibility Check ---")
    for metric in ["pattern", "recent", "old_tok", "composite", "final_loss"]:
        v1 = d_only[metric]
        v2 = d_seq[metric]
        v3 = d_after[metric]
        v4 = d_reset[metric]
        match_12 = abs(v1 - v2) < 1e-6
        match_13 = abs(v1 - v3) < 1e-6
        match_14 = abs(v1 - v4) < 1e-6
        match_all = match_12 and match_13 and match_14
        status = "IDENTICAL" if match_all else "DIVERGED"
        print(f"  {metric}: D-only={v1:.6f} ABCD-D={v2:.6f} "
              f"D-after={v3:.6f} D-reset={v4:.6f} -> {status}")

    # Verdict
    print("\n--- INDEPENDENCE VERDICT ---")
    nan_in_seq = d_seq.get("has_nan_loss", False) or not d_seq["stable"]
    nan_in_solo = d_only.get("has_nan_loss", False) or not d_only["stable"]

    if nan_in_seq and not nan_in_solo:
        print("  [NG] SEQUENTIAL EXECUTION CAUSES D INSTABILITY")
        print("  -> Prior config runs contaminate D execution")
        print("  -> Root cause: likely CUDA allocator state or RNG leakage")
    elif nan_in_seq and nan_in_solo:
        print("  [INFO] D is inherently unstable at this seed")
        print("  -> Not a benchmark independence issue")
    elif not nan_in_seq and not nan_in_solo:
        print("  [OK] D is stable in all conditions")
    else:
        print("  [ANOMALY] D stable in sequence but unstable solo (unlikely)")

    # Save
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[DONE] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
