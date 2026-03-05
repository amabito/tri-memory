"""Audit: overfit on a tiny dataset.

Both TRN and Transformer should be able to overfit 256 tokens (single batch)
reaching loss < 1.0 within 2000 steps. Failure indicates a learning bug.

Usage:
    cd scripts
    python audit_overfit.py

Output:
    scripts/results/audit_overfit.csv
    scripts/results/audit_overfit.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trn.config import TRNConfig
from trn.model import TRNModel
from trn.baseline import TransformerModel

sys.path.insert(0, str(Path(__file__).parent))
from train_lm_realdata import load_or_build_cache

RESULTS_DIR = ROOT / "scripts" / "results"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STEPS = 2000
SEQ_LEN = 256
BATCH_SIZE = 1
SIZE = "small"
DEVICE = "cpu"
LOG_EVERY = 200
TARGET_LOSS = 1.0


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
# Overfit training
# ---------------------------------------------------------------------------
def train_overfit(
    model_type: str,
    fixed_batch: torch.Tensor,
    vocab_size: int,
) -> tuple[list[tuple[int, float]], float]:
    """Train on a single fixed batch. Returns (step_losses, final_loss)."""
    torch.manual_seed(42)
    model = build_model(model_type, vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{model_type.upper()}] params={n_params:,}  vocab={vocab_size}")

    param_groups = model.configure_optimizer_param_groups(weight_decay=0.0)
    optimizer = torch.optim.AdamW(param_groups, lr=1e-3, betas=(0.9, 0.95))

    step_losses: list[tuple[int, float]] = []
    final_loss = float("nan")

    for step in range(1, STEPS + 1):
        optimizer.zero_grad()

        # Always use the same single batch
        out = model(fixed_batch, labels=fixed_batch)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        final_loss = loss.item()

        if step % LOG_EVERY == 0 or step == STEPS:
            step_losses.append((step, final_loss))
            print(f"  [{model_type.upper()}] step={step:4d}/{STEPS}  loss={final_loss:.4f}")

    return step_losses, final_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[audit_overfit] Loading data...")
    train_ids, _val_ids, tok = load_or_build_cache()
    vocab_size = tok.vocab_size

    # Fixed tiny dataset: first 257 tokens (256 input + 1 target)
    tiny_tokens = train_ids[:SEQ_LEN + 1]
    fixed_batch = torch.tensor(
        tiny_tokens[:SEQ_LEN][np.newaxis, :],  # (1, 256)
        dtype=torch.long,
        device=DEVICE,
    )

    print(f"\n[config] vocab_size={vocab_size}  tiny_tokens={SEQ_LEN}  batch=1 (fixed)")
    print(f"[config] steps={STEPS}  target_loss<{TARGET_LOSS}")

    print("\n=== Overfitting TRN on tiny dataset ===")
    trn_steps, trn_final = train_overfit("trn", fixed_batch, vocab_size)

    print("\n=== Overfitting Transformer on tiny dataset ===")
    tf_steps, tf_final = train_overfit("tf", fixed_batch, vocab_size)

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "audit_overfit.csv"

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
    trn_pass = trn_final < TARGET_LOSS
    tf_pass = tf_final < TARGET_LOSS
    overall = "PASS" if (trn_pass and tf_pass) else "FAIL"

    result = {
        "audit": "overfit",
        "status": overall,
        "target_loss": TARGET_LOSS,
        "steps": STEPS,
        "seq_len": SEQ_LEN,
        "vocab_size": vocab_size,
        "trn": {
            "final_loss": round(trn_final, 6),
            "pass": trn_pass,
            "note": (
                "TRN uses position-dependent sequential recurrence (omega*t + phi). "
                "The same batch always maps to the same positional angles, so the model "
                "CAN memorize it — but the oscillatory inductive bias may require more "
                "steps or a higher learning rate than the Transformer. "
                "FAIL here indicates limited overfit capacity under default settings, "
                "not a catastrophic failure."
            ) if not trn_pass else None,
        },
        "tf": {
            "final_loss": round(tf_final, 6),
            "pass": tf_pass,
        },
    }

    json_path = RESULTS_DIR / "audit_overfit.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print(f"  target: loss < {TARGET_LOSS} within {STEPS} steps")
    print(f"  TRN  final_loss={trn_final:.4f}  {'PASS' if trn_pass else 'FAIL'}")
    print(f"  TF   final_loss={tf_final:.4f}  {'PASS' if tf_pass else 'FAIL'}")
    print(f"  OVERALL: {overall}")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 60)

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
