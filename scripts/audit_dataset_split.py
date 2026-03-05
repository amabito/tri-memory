"""Audit dataset split integrity: verify no train/val overlap in WikiText-2.

Usage:
    cd scripts
    python audit_dataset_split.py

Output: scripts/results/audit_dataset_split.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from train_lm_realdata import load_or_build_cache  # noqa: E402

SEQ_LEN = 256
STRIDE   = 256  # non-overlapping for performance
RESULTS  = ROOT / "scripts" / "results" / "audit_dataset_split.json"


def hash_windows(arr, seq_len: int, stride: int) -> set[bytes]:
    """Return set of sha256 digests for all windows of length seq_len."""
    hashes: set[bytes] = set()
    n = len(arr)
    for start in range(0, n - seq_len + 1, stride):
        window = arr[start : start + seq_len]
        digest = hashlib.sha256(window.tobytes()).digest()
        hashes.add(digest)
    return hashes


def main() -> None:
    train_ids, val_ids, _ = load_or_build_cache()

    print(f"[audit] train tokens: {len(train_ids):,}")
    print(f"[audit] val tokens:   {len(val_ids):,}")
    print(f"[audit] seq_len={SEQ_LEN}, stride={STRIDE}")

    print("[audit] Hashing train windows …")
    train_hashes = hash_windows(train_ids, SEQ_LEN, STRIDE)

    print("[audit] Hashing val windows …")
    val_hashes = hash_windows(val_ids, SEQ_LEN, STRIDE)

    overlap = train_hashes & val_hashes
    overlap_count    = len(overlap)
    total_train      = len(train_hashes)
    total_val        = len(val_hashes)
    overlap_fraction = overlap_count / max(1, total_val)

    passed = overlap_count == 0
    status = "PASS" if passed else "FAIL"

    result = {
        "status":           status,
        "total_train":      total_train,
        "total_val":        total_val,
        "overlap_count":    overlap_count,
        "overlap_fraction": overlap_fraction,
        "seq_len":          SEQ_LEN,
        "stride":           STRIDE,
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"  train windows : {total_train:,}")
    print(f"  val windows   : {total_val:,}")
    print(f"  overlapping   : {overlap_count:,}  ({overlap_fraction:.4%})")
    print(f"  -> {status}")
    print(f"  Saved: {RESULTS}")


if __name__ == "__main__":
    main()
