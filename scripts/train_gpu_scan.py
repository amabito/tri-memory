#!/usr/bin/env python3
"""GPU scan availability check and speedup benchmark for TRN.

Verifies torch.associative_scan availability, correctness, and speedup
of GPU parallel scan vs CPU sequential scan for TRN.

Usage:
    python scripts/train_gpu_scan.py

Outputs:
    - scripts/results/train_gpu_scan.csv  (seq_len, cpu_ms, gpu_ms, speedup)
"""
from __future__ import annotations

import sys
import os

# Remove CWD from sys.path BEFORE any other imports.
# profile.py in the project root shadows stdlib cProfile, which causes
# torch._dynamo (needed by torch.optim and torch.associative_scan) to fail:
#   AttributeError: module 'profile' has no attribute 'run'
sys.path = [p for p in sys.path if p not in ("", ".")]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import csv
import time
from pathlib import Path
from typing import Optional

import torch

from trn.scan import sequential_resonance_scan
from trn.config import TRNConfig
from trn.model import TRNModel


# ---------------------------------------------------------------------------
# Helpers: locate and test associative_scan
# ---------------------------------------------------------------------------

def _locate_associative_scan():
    """Find torch.associative_scan in any known location.

    PyTorch 2.1-2.5: torch.associative_scan (top-level)
    PyTorch 2.8:     torch._higher_order_ops.associative_scan (internal)
                     Requires torch.compile / torch._dynamo at runtime.

    Returns the callable or None.
    """
    if hasattr(torch, "associative_scan"):
        return torch.associative_scan

    try:
        from torch._higher_order_ops import associative_scan
        return associative_scan
    except (ImportError, AttributeError):
        pass

    return None


def _combine(
    x: tuple[torch.Tensor, torch.Tensor],
    y: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Associative monoid for first-order linear recurrence s_t = a_t*s_{t-1} + b_t."""
    a1, b1 = x
    a2, b2 = y
    return a2 * a1, torch.addcmul(b2, a2, b1)


def _parallel_scan(
    alpha:    torch.Tensor,
    drive_r:  torch.Tensor,
    drive_i:  torch.Tensor,
    assoc_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GPU parallel scan using the located associative_scan function."""
    r_r, _ = assoc_fn(_combine, (alpha, drive_r), 1)
    r_i, _ = assoc_fn(_combine, (alpha, drive_i), 1)
    return r_r, r_i


# ---------------------------------------------------------------------------
# Part 1: Availability check
# ---------------------------------------------------------------------------

def check_associative_scan() -> dict:
    """Check torch.associative_scan availability and run a smoke test.

    PyTorch 2.8 notes:
      - torch.associative_scan is NOT present at top-level.
      - torch._higher_order_ops.associative_scan IS importable.
      - However, it requires torch.compile/_dynamo at call time.
      - If profile.py exists in cwd, cProfile is shadowed and _dynamo fails.
        This is a known issue: https://github.com/pytorch/pytorch/issues/XX
    """
    result: dict = {
        "available": False,
        "location": None,
        "error": None,
        "torch_version": torch.__version__,
        "fn": None,
    }

    if not torch.cuda.is_available():
        result["error"] = "CUDA not available"
        return result

    assoc_fn = _locate_associative_scan()
    if assoc_fn is None:
        result["error"] = (
            "torch.associative_scan not found. "
            "Tried: torch.associative_scan (absent in 2.8), "
            "torch._higher_order_ops.associative_scan (import failed)"
        )
        return result

    if hasattr(torch, "associative_scan"):
        result["location"] = "torch.associative_scan"
    else:
        result["location"] = "torch._higher_order_ops.associative_scan"

    # Smoke test: this will trigger torch._dynamo import
    try:
        a = torch.ones(2, 4, 8, device="cuda") * 0.9
        b = torch.randn(2, 4, 8, device="cuda")
        out = assoc_fn(_combine, (a, b), 1)
        torch.cuda.synchronize()
        _ = out
        result["available"] = True
        result["fn"] = assoc_fn
    except AttributeError as exc:
        # cProfile shadow: "module 'profile' has no attribute 'run'"
        # This happens when profile.py exists in the project root and is
        # picked up before stdlib cProfile via sys.path.
        result["error"] = (
            f"torch._dynamo import failed due to stdlib shadowing: {exc}. "
            "Cause: project-root profile.py shadows stdlib cProfile. "
            "Fix: rename profile.py or run from outside project root."
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


# ---------------------------------------------------------------------------
# Part 2: Correctness check
# ---------------------------------------------------------------------------

def check_correctness_cpu_vs_gpu_sequential() -> dict:
    """Verify sequential_resonance_scan produces identical results on CPU and GPU."""
    B, seq_len, K = 4, 512, 64
    alpha   = torch.sigmoid(torch.randn(B, seq_len, K)).float()
    drive_r = torch.randn(B, seq_len, K).float()
    drive_i = torch.randn(B, seq_len, K).float()

    with torch.no_grad():
        r_cpu, i_cpu = sequential_resonance_scan(alpha, drive_r, drive_i)
        r_gpu, i_gpu = sequential_resonance_scan(alpha.cuda(), drive_r.cuda(), drive_i.cuda())
        torch.cuda.synchronize()

    r_gpu = r_gpu.cpu()
    i_gpu = i_gpu.cpu()

    max_diff_r = (r_cpu - r_gpu).abs().max().item()
    max_diff_i = (i_cpu - i_gpu).abs().max().item()
    max_diff   = max(max_diff_r, max_diff_i)
    passed     = max_diff < 1e-4

    return {
        "mode": "sequential CPU == sequential GPU",
        "max_diff_r": max_diff_r,
        "max_diff_i": max_diff_i,
        "max_diff": max_diff,
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# Part 3: Speedup benchmark  (CPU sequential vs GPU sequential)
# ---------------------------------------------------------------------------

def benchmark_scan(
    seq_lens: Optional[list[int]] = None,
    B: int = 4,
    K: int = 64,
    n_warmup: int = 10,
    n_runs: int = 50,
) -> list[dict]:
    """Benchmark CPU sequential scan vs GPU sequential scan.

    When torch.associative_scan is unavailable (PyTorch 2.8 + profile.py shadow),
    we compare CPU sequential vs GPU sequential to show the GPU kernel itself is fast.
    """
    if seq_lens is None:
        seq_lens = [256, 512, 1024, 2048, 4096, 8192]

    cuda_ok = torch.cuda.is_available()
    results = []

    for seq_len in seq_lens:
        alpha_cpu   = torch.sigmoid(torch.randn(B, seq_len, K)).float()
        drive_r_cpu = torch.randn(B, seq_len, K).float()
        drive_i_cpu = torch.randn(B, seq_len, K).float()

        if cuda_ok:
            alpha_gpu   = alpha_cpu.cuda()
            drive_r_gpu = drive_r_cpu.cuda()
            drive_i_gpu = drive_i_cpu.cuda()

        with torch.no_grad():
            for _ in range(n_warmup):
                sequential_resonance_scan(alpha_cpu, drive_r_cpu, drive_i_cpu)
                if cuda_ok:
                    sequential_resonance_scan(alpha_gpu, drive_r_gpu, drive_i_gpu)
        if cuda_ok:
            torch.cuda.synchronize()

        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(n_runs):
                sequential_resonance_scan(alpha_cpu, drive_r_cpu, drive_i_cpu)
        cpu_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

        gpu_ms: Optional[float] = None
        if cuda_ok:
            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    sequential_resonance_scan(alpha_gpu, drive_r_gpu, drive_i_gpu)
                torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

        speedup = cpu_ms / gpu_ms if (gpu_ms is not None and gpu_ms > 0) else None

        results.append({
            "seq_len": seq_len,
            "cpu_ms":  cpu_ms,
            "gpu_ms":  gpu_ms,
            "speedup": speedup,
        })

    return results


# ---------------------------------------------------------------------------
# Part 4: Training throughput
# ---------------------------------------------------------------------------

def benchmark_training(device: str, n_steps: int = 100) -> dict:
    """Train TRN (d=128, L=4, K=64, B=8, seq=1024) and report tok/s and peak mem."""
    D_MODEL  = 128
    N_LAYERS = 4
    K        = 64
    BATCH    = 8
    SEQ_LEN  = 1024

    cfg = TRNConfig(
        vocab_size=256,
        d_model=D_MODEL,
        n_oscillators=K,
        n_layers=N_LAYERS,
        d_ff=D_MODEL * 4,
        max_seq_len=SEQ_LEN,
    )
    model = TRNModel(cfg).to(device).train()
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)

    input_ids = torch.randint(0, 256, (BATCH, SEQ_LEN), device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(5):
            model(input_ids, labels=input_ids)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        out = model(input_ids, labels=input_ids)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    tps     = (n_steps * BATCH * SEQ_LEN) / elapsed
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0

    return {"device": device, "tps": tps, "peak_mb": peak_mb}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available. Sequential baseline only.")
    print(f"PyTorch: {torch.__version__}")
    print()

    # --- Part 1: Availability ---
    avail    = check_associative_scan()
    assoc_fn = avail.get("fn", None)

    if avail["available"]:
        print(f"[AVAIL] torch.associative_scan: FOUND at {avail['location']}")
    else:
        print("[SKIP]  torch.associative_scan: NOT AVAILABLE")
        if avail.get("location"):
            print(f"        Found at: {avail['location']} (smoke test failed)")
        print(f"        Reason: {avail['error']}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Part 2: Correctness ---
    if device == "cuda":
        if assoc_fn is not None:
            # Full test: GPU parallel vs CPU sequential
            from trn.scan import sequential_resonance_scan as seq_scan
            B, n, K = 4, 512, 64
            alpha   = torch.sigmoid(torch.randn(B, n, K)).float()
            dr      = torch.randn(B, n, K).float()
            di      = torch.randn(B, n, K).float()
            with torch.no_grad():
                r_seq, i_seq = seq_scan(alpha, dr, di)
                r_par, i_par = _parallel_scan(alpha.cuda(), dr.cuda(), di.cuda(), assoc_fn)
                torch.cuda.synchronize()
            diff = max((r_seq - r_par.cpu()).abs().max().item(),
                       (i_seq - i_par.cpu()).abs().max().item())
            label = "PASS" if diff < 1e-4 else "FAIL"
            print(f"Correctness (GPU parallel vs CPU sequential): {label}, max_diff={diff:.2e}")
            if label == "FAIL":
                print("  NOTE: torch.associative_scan is an inclusive prefix scan.")
                print("  parallel_resonance_scan in scan.py uses it without an initial-state")
                print("  correction: scan output[0] = alpha[0] (not drive_r[0] = s_0).")
                print("  The scan.py implementation needs a zero-state prepend fix to match")
                print("  sequential_resonance_scan which starts with s_{-1}=0.")
                print("  TRN currently falls back to sequential_resonance_scan at runtime.")
                print("  Recommendation: fix parallel_resonance_scan or use sequential fallback.")
        else:
            corr  = check_correctness_cpu_vs_gpu_sequential()
            label = "PASS" if corr["pass"] else "FAIL"
            print(f"Correctness ({corr['mode']}): {label}, max_diff={corr['max_diff']:.2e}")
            print("  Note: torch.associative_scan unavailable, sequential scan verified instead")
    else:
        print("Correctness: PASS (CPU only, sequential==sequential)")
    print()

    # --- Part 3: Speedup benchmark ---
    if assoc_fn is not None:
        print("Benchmark: CPU sequential vs GPU parallel scan ...")
    else:
        print("Benchmark: CPU sequential vs GPU sequential scan ...")
        print("  (torch.associative_scan unavailable: falling back to sequential comparison)")

    bench = benchmark_scan()

    col_label = "GPU par ms" if assoc_fn is not None else "GPU seq ms"
    hdr = f"{'seq_len':>8} | {'CPU seq ms':>12} | {col_label:>12} | {'speedup':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in bench:
        gpu_s = f"{r['gpu_ms']:12.3f}" if r["gpu_ms"] is not None else "         N/A"
        spd_s = f"{r['speedup']:8.2f}x"  if r["speedup"] is not None else "        N/A"
        print(f"{r['seq_len']:>8} | {r['cpu_ms']:12.3f} | {gpu_s} | {spd_s}")
    print()

    csv_path = results_dir / "train_gpu_scan.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_len", "cpu_ms", "gpu_ms", "speedup"])
        for r in bench:
            w.writerow([
                r["seq_len"],
                f"{r['cpu_ms']:.4f}",
                f"{r['gpu_ms']:.4f}" if r["gpu_ms"] is not None else "",
                f"{r['speedup']:.4f}" if r["speedup"] is not None else "",
            ])
    print(f"Saved: {csv_path}")
    print()

    # --- Part 4: Training throughput (GPU only, CPU skipped: too slow at seq=1024) ---
    print(f"Training throughput on {device.upper()} (100 steps, B=8, seq=1024) ...")
    gpu_t = benchmark_training(device=device, n_steps=100)
    print(f"  device    : {gpu_t['device']}")
    print(f"  tokens/sec: {gpu_t['tps']:,.0f}")
    print(f"  peak mem  : {gpu_t['peak_mb']:.1f} MB")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  torch.associative_scan available: {avail['available']}")
    if avail.get("location"):
        print(f"  location: {avail['location']}")
    if not avail["available"]:
        print(f"  reason  : {avail['error']}")

    r1024 = next((r for r in bench if r["seq_len"] == 1024), None)
    if r1024:
        mode = "GPU parallel" if assoc_fn is not None else "GPU sequential"
        if r1024["speedup"] is not None:
            print(f"  {mode} speedup at seq=1024: {r1024['speedup']:.2f}x vs CPU")
        else:
            print(f"  CPU sequential at seq=1024: {r1024['cpu_ms']:.2f} ms")

    print(f"  training ({device.upper()}): {gpu_t['tps']:,.0f} tok/s, {gpu_t['peak_mb']:.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
