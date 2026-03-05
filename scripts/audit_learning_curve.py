"""Audit learning curve: TRN vs Transformer training dynamics over 1000 steps.

Logs every 100 steps:
  - step, model_type, train_loss, val_ppl, grad_norm
  - alpha_mean (TRN only): mean of all alpha (decay gate) parameters after sigmoid
  - gate_mag_mean (TRN only): 0.0 placeholder (drive outputs require hooks to extract)

Usage:
    cd scripts
    python audit_learning_curve.py

Output CSV: scripts/results/audit_learning_curve.csv
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trn.config import TRNConfig  # noqa: E402
from trn.model import TRNModel  # noqa: E402
from trn.baseline import TransformerModel  # noqa: E402
from train_lm_realdata import load_or_build_cache, sample_batch  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
SEQ_LEN    = 256
DEVICE     = "cpu"
LR         = 3e-4
N_STEPS    = 500
LOG_EVERY  = 50
N_VAL_BATCHES = 50

RESULTS_CSV = ROOT / "scripts" / "results" / "audit_learning_curve.csv"
CSV_COLUMNS = ["step", "model", "train_loss", "val_ppl", "grad_norm", "alpha_mean", "gate_mag_mean"]

# Small model config: d_model=128, n_layers=4, K=64, d_ff=512
SMALL_SIZE = dict(d_model=128, n_layers=4, n_oscillators=64, d_ff=512)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(model_type: str, vocab_size: int) -> nn.Module:
    cfg = TRNConfig(
        vocab_size=vocab_size,
        d_model=SMALL_SIZE["d_model"],
        n_oscillators=SMALL_SIZE["n_oscillators"],
        n_layers=SMALL_SIZE["n_layers"],
        d_ff=SMALL_SIZE["d_ff"],
        max_seq_len=SEQ_LEN,
        dropout=0.0,
        use_parallel_scan=True,
        tie_weights=True,
    )
    if model_type == "trn":
        return TRNModel(cfg)
    return TransformerModel(cfg)


def eval_val_ppl(
    model: nn.Module,
    val_ids: np.ndarray,
    rng: np.random.Generator,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(N_VAL_BATCHES):
            batch = sample_batch(val_ids, SEQ_LEN, BATCH_SIZE, rng, DEVICE)
            out   = model(batch, labels=batch)
            total_loss += out["loss"].item()
    model.train()
    return math.exp(total_loss / N_VAL_BATCHES)


def get_alpha_mean(model: nn.Module) -> float:
    """Compute mean of decay gate (alpha) bias logits after sigmoid for TRN.

    The OscillatorProjection.proj.bias[3K:4K] holds the learned gate bias.
    The full alpha = sigmoid(proj(x)[3K:] + bias[3K:]) is input-dependent,
    but the bias term captures the learned prior — sufficient for monitoring.
    """
    alpha_vals = []
    for name, param in model.named_parameters():
        if "resonance.proj.proj.bias" in name:
            # proj outputs 4*K channels; last K are decay gate logit biases
            total = param.shape[0]
            K = total // 4
            gate_bias = param.data[3 * K:]  # shape (K,)
            alpha_vals.append(torch.sigmoid(gate_bias).mean().item())
    return sum(alpha_vals) / len(alpha_vals) if alpha_vals else 0.0


# ---------------------------------------------------------------------------
# Training loop for one model
# ---------------------------------------------------------------------------

def train_model(
    model_type: str,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    vocab_size: int,
    writer: csv.DictWriter,
) -> list[dict]:
    model = build_model(model_type, vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{model_type.upper()}] params={n_params:,}  vocab={vocab_size}  seq_len={SEQ_LEN}")

    param_groups = model.configure_optimizer_param_groups(weight_decay=0.1)
    optimizer    = torch.optim.AdamW(param_groups, lr=LR, betas=(0.9, 0.95))

    rng_train = np.random.default_rng(42)
    rng_val   = np.random.default_rng(0)

    rows = []
    loss_acc = 0.0

    for step in range(1, N_STEPS + 1):
        batch = sample_batch(train_ids, SEQ_LEN, BATCH_SIZE, rng_train, DEVICE)

        optimizer.zero_grad()
        out  = model(batch, labels=batch)
        loss = out["loss"]
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

        optimizer.step()
        loss_acc += loss.item()

        if step % LOG_EVERY == 0:
            train_loss = loss_acc / LOG_EVERY
            loss_acc   = 0.0

            val_ppl = eval_val_ppl(model, val_ids, rng_val)

            if model_type == "trn":
                alpha_mean    = get_alpha_mean(model)
                gate_mag_mean = 0.0
            else:
                alpha_mean    = ""
                gate_mag_mean = ""

            row = {
                "step":          step,
                "model":         model_type,
                "train_loss":    f"{train_loss:.6f}",
                "val_ppl":       f"{val_ppl:.4f}",
                "grad_norm":     f"{grad_norm:.6f}",
                "alpha_mean":    f"{alpha_mean:.6f}" if alpha_mean != "" else "",
                "gate_mag_mean": f"{gate_mag_mean:.6f}" if gate_mag_mean != "" else "",
            }
            writer.writerow(row)
            rows.append(row)

            print(
                f"  step={step:4d}  loss={train_loss:.4f}  val_ppl={val_ppl:.2f}"
                f"  grad_norm={grad_norm:.4f}"
                + (f"  alpha_mean={alpha_mean:.4f}" if model_type == "trn" else "")
            )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[audit_learning_curve] Loading WikiText-2 data …")
    train_ids, val_ids, tok = load_or_build_cache()
    vocab_size = tok.vocab_size
    print(f"[audit] train={len(train_ids):,} tokens  val={len(val_ids):,} tokens  vocab={vocab_size}")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        # TRN first, then Transformer
        for model_type in ("trn", "tf"):
            rows = train_model(model_type, train_ids, val_ids, vocab_size, writer)
            all_rows.extend(rows)

    # Summary table
    print("\n" + "=" * 80)
    print("LEARNING CURVE SUMMARY")
    print("=" * 80)
    header = f"{'step':>6}  {'model':>4}  {'train_loss':>10}  {'val_ppl':>8}  {'grad_norm':>10}  {'alpha_mean':>10}"
    print(header)
    print("-" * 80)
    for row in all_rows:
        alpha_str = f"{row['alpha_mean']:>10}" if row["alpha_mean"] else f"{'N/A':>10}"
        print(
            f"{row['step']:>6}  {row['model']:>4}  "
            f"{row['train_loss']:>10}  {row['val_ppl']:>8}  "
            f"{row['grad_norm']:>10}  {alpha_str}"
        )
    print("=" * 80)
    print(f"\n[OK] CSV saved: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
