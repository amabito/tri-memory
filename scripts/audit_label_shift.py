"""Audit label shift: verify TRNModel and TransformerModel shift targets exactly once.

Usage:
    cd scripts
    python audit_label_shift.py

Output: scripts/results/audit_label_shift.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trn.config import TRNConfig   # noqa: E402
from trn.model import TRNModel     # noqa: E402
from trn.baseline import TransformerModel  # noqa: E402

RESULTS = ROOT / "scripts" / "results" / "audit_label_shift.json"

# Small config that keeps computation fast
SMALL_CFG = TRNConfig(
    vocab_size=10,
    d_model=32,
    n_oscillators=8,
    n_layers=1,
    d_ff=64,
    max_seq_len=16,
    dropout=0.0,
    use_parallel_scan=True,
    tie_weights=True,
)

INPUT_SEQ = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def audit_model(model_cls, cfg: TRNConfig, model_name: str) -> dict:
    """Verify label shift for a given model class. Returns per-model result dict."""
    torch.manual_seed(0)
    model = model_cls(cfg)
    model.eval()

    input_ids = torch.tensor([INPUT_SEQ], dtype=torch.long)  # (1, 10)
    labels    = input_ids.clone()

    with torch.no_grad():
        out = model(input_ids, labels=labels)

    logits = out["logits"]   # (1, 10, vocab)
    loss   = out["loss"]

    # --- Shape check ---
    # After shift: logits[:, :-1] -> (1, 9, vocab), targets -> [1..9]
    expected_logit_shape = (1, 9, cfg.vocab_size)
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    logit_shape_ok = tuple(shift_logits.shape) == expected_logit_shape
    target_seq_ok  = shift_labels.tolist() == [INPUT_SEQ[1:]]  # [[1,2,...,9]]

    # --- Manual CE loss with single shift ---
    manual_loss = F.cross_entropy(
        shift_logits.view(-1, cfg.vocab_size),
        shift_labels.view(-1),
    )
    loss_match = torch.isclose(loss, manual_loss, atol=1e-5).item()

    # --- Verify no double-shift ---
    # If double-shift happened: model would use labels[:, 2:] as targets (only 8 positions)
    # or logits[:, :-2] shape (1, 8, vocab). We check that loss != CE with double-shift.
    if logits.shape[1] >= 3:
        double_shift_logits = logits[:, :-2].contiguous()
        double_shift_labels = labels[:, 2:].contiguous()
        double_loss = F.cross_entropy(
            double_shift_logits.view(-1, cfg.vocab_size),
            double_shift_labels.view(-1),
        )
        no_double_shift = not torch.isclose(loss, double_loss, atol=1e-5).item()
    else:
        no_double_shift = True  # can't check, but shape check already failed

    passed = logit_shape_ok and target_seq_ok and loss_match and no_double_shift

    return {
        "model":              model_name,
        "logit_shape":        list(shift_logits.shape),
        "expected_shape":     list(expected_logit_shape),
        "logit_shape_ok":     logit_shape_ok,
        "target_seq_ok":      target_seq_ok,
        "model_loss":         round(loss.item(), 6),
        "manual_loss":        round(manual_loss.item(), 6),
        "loss_match":         loss_match,
        "no_double_shift":    no_double_shift,
        "passed":             passed,
    }


def main() -> None:
    results = {}

    for model_name, model_cls in [("TRNModel", TRNModel), ("TransformerModel", TransformerModel)]:
        print(f"[audit] Testing {model_name} …")
        res = audit_model(model_cls, SMALL_CFG, model_name)
        results[model_name] = res

        status = "PASS" if res["passed"] else "FAIL"
        print(f"  logit_shape_ok  : {res['logit_shape_ok']}  {res['logit_shape']} == {res['expected_shape']}")
        print(f"  target_seq_ok   : {res['target_seq_ok']}")
        print(f"  loss_match      : {res['loss_match']}  model={res['model_loss']:.6f}  manual={res['manual_loss']:.6f}")
        print(f"  no_double_shift : {res['no_double_shift']}")
        print(f"  -> {status}")
        print()

    overall_pass = all(r["passed"] for r in results.values())
    summary = {
        "status":  "PASS" if overall_pass else "FAIL",
        "models":  results,
        "input_seq": INPUT_SEQ,
        "expected_targets": INPUT_SEQ[1:],
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Overall -> {'PASS' if overall_pass else 'FAIL'}")
    print(f"Saved: {RESULTS}")


if __name__ == "__main__":
    main()
