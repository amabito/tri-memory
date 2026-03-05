"""Audit: random-target baseline sanity check.

Both TRN and Transformer should converge to log(vocab_size) when trained
with random (uniform) targets — they should NOT learn anything.

Usage:
    cd scripts
    python audit_random_targets.py

Output:
    scripts/results/audit_random_targets.csv
    scripts/results/audit_random_targets.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trn.config import TRNConfig
from trn.model import TRNModel
from trn.baseline import TransformerModel

# reuse data loading from train_lm_realdata
sys.path.insert(0, str(Path(__file__).parent))
from train_lm_realdata import load_or_build_cache, sample_batch

RESULTS_DIR = ROOT / "scripts" / "results"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STEPS = 200
BATCH_SIZE = 8
SEQ_LEN = 256
SIZE = "small"
DEVICE = "cpu"
LOG_EVERY = 50
TOLERANCE = 0.5  # nats


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
SIZE_CONFIGS: dict[str, dict] = {
    "small": dict(d_model=128, n_layers=4, n_oscillators=64, d_ff=512),
}


def build_model(model_type: str, vocab_size: int) -> torch.nn.Module:
    sc = SIZE_CONFIGS[SIZE]
    cfg = TRNConfig(
        vocab_size=vocab_size,
        d_model=sc["d_model"],
        n_oscillators=sc["n_oscillators"],
        n_layers=sc["n_layers"],
        d_ff=sc["d_ff"],
        max_seq_len=SEQ_LEN,
        dropout=0.0,
        use_parallel_scan=True,
        tie_weights=True,
    )
    if model_type == "trn":
        return TRNModel(cfg)
    return TransformerModel(cfg)


# ---------------------------------------------------------------------------
# Training with random targets
# ---------------------------------------------------------------------------
def train_random_targets(
    model_type: str,
    train_ids: np.ndarray,
    vocab_size: int,
) -> tuple[list[tuple[int, float]], float]:
    """Train with uniformly random targets. Returns (step_losses, final_loss)."""
    torch.manual_seed(42)
    model = build_model(model_type, vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{model_type.upper()}] params={n_params:,}  vocab={vocab_size}")

    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.95))

    rng = np.random.default_rng(42)
    step_losses: list[tuple[int, float]] = []
    loss_acc = 0.0
    final_loss = float("nan")

    for step in range(1, STEPS + 1):
        batch = sample_batch(train_ids, SEQ_LEN, BATCH_SIZE, rng, DEVICE)

        optimizer.zero_grad()

        # Forward pass to get logits
        out = model(batch)
        logits = out["logits"]  # (B, T, V)
        B, T, V = logits.shape

        # Random targets: uniform draw from vocab
        random_targets = torch.randint(0, V, (B, T), device=DEVICE)

        loss = F.cross_entropy(logits.reshape(-1, V), random_targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_acc += loss.item()
        final_loss = loss.item()

        if step % LOG_EVERY == 0 or step == STEPS:
            avg = loss_acc / LOG_EVERY
            step_losses.append((step, avg))
            print(f"  [{model_type.upper()}] step={step:4d}/{STEPS}  loss={avg:.4f}")
            loss_acc = 0.0

    return step_losses, final_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[audit_random_targets] Loading data...")
    train_ids, _val_ids, tok = load_or_build_cache()
    vocab_size = tok.vocab_size

    expected_loss = math.log(vocab_size)
    print(f"\n[config] vocab_size={vocab_size}  expected_random_loss={expected_loss:.4f} nats")
    print(f"[config] steps={STEPS}  batch_size={BATCH_SIZE}  seq_len={SEQ_LEN}  tolerance={TOLERANCE}")

    print("\n=== Training TRN with random targets ===")
    trn_steps, trn_final = train_random_targets("trn", train_ids, vocab_size)

    print("\n=== Training Transformer with random targets ===")
    tf_steps, tf_final = train_random_targets("tf", train_ids, vocab_size)

    # Merge step logs into unified CSV rows
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "audit_random_targets.csv"

    step_map_trn = {s: l for s, l in trn_steps}
    step_map_tf = {s: l for s, l in tf_steps}
    all_steps = sorted(set(step_map_trn) | set(step_map_tf))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "trn_loss", "tf_loss"])
        writer.writeheader()
        for step in all_steps:
            writer.writerow({
                "step": step,
                "trn_loss": f"{step_map_trn.get(step, float('nan')):.6f}",
                "tf_loss": f"{step_map_tf.get(step, float('nan')):.6f}",
            })

    # Evaluate PASS/FAIL
    trn_diff = abs(trn_final - expected_loss)
    tf_diff = abs(tf_final - expected_loss)
    trn_pass = trn_diff <= TOLERANCE
    tf_pass = tf_diff <= TOLERANCE
    overall = "PASS" if (trn_pass and tf_pass) else "FAIL"

    result = {
        "audit": "random_targets",
        "status": overall,
        "expected_random_loss": round(expected_loss, 6),
        "tolerance_nats": TOLERANCE,
        "vocab_size": vocab_size,
        "steps": STEPS,
        "trn": {
            "final_loss": round(trn_final, 6),
            "diff_from_expected": round(trn_diff, 6),
            "pass": trn_pass,
        },
        "tf": {
            "final_loss": round(tf_final, 6),
            "diff_from_expected": round(tf_diff, 6),
            "pass": tf_pass,
        },
    }

    json_path = RESULTS_DIR / "audit_random_targets.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print(f"  expected_random_loss = {expected_loss:.4f} nats  (log({vocab_size}))")
    print(f"  TRN  final_loss={trn_final:.4f}  diff={trn_diff:.4f}  {'PASS' if trn_pass else 'FAIL'}")
    print(f"  TF   final_loss={tf_final:.4f}  diff={tf_diff:.4f}  {'PASS' if tf_pass else 'FAIL'}")
    print(f"  OVERALL: {overall}")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 60)

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
