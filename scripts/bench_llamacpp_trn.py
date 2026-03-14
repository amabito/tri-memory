"""bench_llamacpp_trn.py — llama.cpp-style benchmark for the C TRN resonance layer.

Modes:
  --correctness-only   : verify C output matches Python step_single (max_abs_error < 1e-5)
  (default)            : throughput benchmark (tokens/s, ms/token) across batch sizes

Usage:
    # Correctness check only (no .bin required — creates a random one internally)
    python scripts/bench_llamacpp_trn.py --correctness-only

    # Full benchmark
    python scripts/bench_llamacpp_trn.py --weight-file path/to/layer.bin

    # Specify device, toy config, etc.
    python scripts/bench_llamacpp_trn.py --device cpu --config toy --n-tokens 200
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Ensure trn package is importable when running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trimemory.config import TRNConfig
from trimemory.resonance import TemporalResonanceLayer
from trimemory.integrations.llamacpp.export_weights import export_layer
from trimemory.integrations.llamacpp.ctypes_wrapper import TRNResonance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(name: str) -> TRNConfig:
    configs = {
        "toy":    TRNConfig.toy(),
        "100m":   TRNConfig.trn_100m(),
        "400m":   TRNConfig.trn_400m(),
        "1b":     TRNConfig.trn_1b(),
    }
    if name not in configs:
        raise ValueError(f"Unknown config '{name}'. Choose from: {list(configs)}")
    return configs[name]


def build_layer(cfg: TRNConfig, device: str = "cpu") -> TemporalResonanceLayer:
    layer = TemporalResonanceLayer(
        d_model=cfg.d_model,
        K=cfg.n_oscillators,
        use_parallel_scan=False,
        log_phase=True,
        state_norm=cfg.state_norm,
        amplitude_max=cfg.amplitude_max,
        res_scale_init=cfg.res_scale_init,
        gate_bias_init=cfg.gate_bias_init,
        phase_mode=cfg.phase_mode,
    ).to(device).eval()
    return layer


def export_and_load(layer: TemporalResonanceLayer, tmp_dir: Path) -> TRNResonance:
    """Export Python layer weights and load via ctypes."""
    # Build a minimal state_dict with the block.0.resonance prefix
    sd: dict[str, torch.Tensor] = {}
    sd["blocks.0.resonance.proj.proj.weight"] = layer.proj.proj.weight.detach()
    sd["blocks.0.resonance.proj.proj.bias"]   = layer.proj.proj.bias.detach()
    sd["blocks.0.resonance.proj.omega_base"]  = layer.proj.omega_base.detach()
    sd["blocks.0.resonance.W_res.weight"]     = layer.W_res.weight.detach()
    sd["blocks.0.resonance.res_scale"]        = layer.res_scale.detach()

    bin_path = tmp_dir / "resonance_layer_000.bin"
    export_layer(
        sd,
        layer_idx=0,
        output_path=bin_path,
        phase_mode="log" if (layer.log_phase or layer.phase_mode == "log") else "linear",
        state_norm=layer.state_norm_enabled,
        amplitude_max=layer.proj.amplitude_max,
    )
    return TRNResonance.load(bin_path)


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def run_correctness(
    cfg: TRNConfig,
    device: str,
    n_tokens: int,
    batch_size: int,
    rtol: float,
    atol: float,
) -> bool:
    print(f"\n=== Correctness check: d_model={cfg.d_model} K={cfg.n_oscillators} "
          f"n_tokens={n_tokens} batch={batch_size} ===")

    torch.manual_seed(42)
    py_layer = build_layer(cfg, device)

    with tempfile.TemporaryDirectory() as tmp:
        c_layer = export_and_load(py_layer, Path(tmp))

    print(f"  C layer loaded: d_model={c_layer.d_model} K={c_layer.K} "
          f"phase={c_layer.phase_mode} state_norm={c_layer.state_norm} "
          f"amplitude_max={c_layer.amplitude_max:.3f} res_scale={c_layer.res_scale:.6f}")

    # Allocate states
    py_r_real = torch.zeros(batch_size, cfg.n_oscillators, dtype=torch.float32, device=device)
    py_r_imag = torch.zeros(batch_size, cfg.n_oscillators, dtype=torch.float32, device=device)
    c_r_real, c_r_imag = c_layer.make_state(batch_size)

    max_abs_err = 0.0
    max_rel_err = 0.0
    worst_pos   = -1

    np.random.seed(7)
    all_pass = True

    for pos in range(n_tokens):
        # Random input embedding
        x_np  = np.random.randn(batch_size, cfg.d_model).astype(np.float32)
        x_pt  = torch.from_numpy(x_np).to(device)

        # Python step
        py_out, py_r_real, py_r_imag = py_layer.step_single(
            x_pt, py_r_real, py_r_imag, position=pos
        )
        py_out_np = py_out.cpu().float().numpy()

        # C step (modifies r_real/r_imag in-place)
        c_out, c_r_real, c_r_imag = c_layer.step(x_np, c_r_real, c_r_imag, position=pos)

        # Compare outputs
        diff = np.abs(py_out_np - c_out)
        ref  = np.abs(py_out_np) + 1e-8
        abs_err = float(diff.max())
        rel_err = float((diff / ref).max())

        if abs_err > max_abs_err:
            max_abs_err = abs_err
            worst_pos   = pos
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        if abs_err > atol:
            print(f"  FAIL at position {pos}: max_abs_err={abs_err:.2e} (atol={atol:.1e})")
            all_pass = False
            if pos > 5:
                break  # stop early after first failure

    # Compare final states
    py_r_real_np = py_r_real.cpu().float().numpy()
    py_r_imag_np = py_r_imag.cpu().float().numpy()
    state_err_r  = float(np.abs(py_r_real_np - c_r_real).max())
    state_err_i  = float(np.abs(py_r_imag_np - c_r_imag).max())

    print(f"  Output — max_abs_err={max_abs_err:.2e}  max_rel_err={max_rel_err:.2e}  "
          f"worst_pos={worst_pos}")
    print(f"  State  — r_real_err={state_err_r:.2e}  r_imag_err={state_err_i:.2e}")

    if all_pass and max_abs_err < atol:
        print(f"  [PASS] max_abs_err {max_abs_err:.2e} < {atol:.1e}")
    else:
        print(f"  [FAIL] max_abs_err {max_abs_err:.2e} >= {atol:.1e}")
        all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    cfg: TRNConfig,
    device: str,
    n_tokens: int,
    batch_sizes: list[int],
    warmup: int,
    c_layer: TRNResonance | None = None,
) -> None:
    print(f"\n=== Throughput benchmark: d_model={cfg.d_model} K={cfg.n_oscillators} "
          f"n_tokens={n_tokens} ===")

    if c_layer is None:
        torch.manual_seed(42)
        py_layer = build_layer(cfg, device)
        with tempfile.TemporaryDirectory() as tmp:
            c_layer = export_and_load(py_layer, Path(tmp))
        del py_layer

    print(f"{'batch':>6} | {'tokens/s':>12} | {'ms/token':>10} | {'total_ms':>10}")
    print("-" * 48)

    np.random.seed(99)

    for batch in batch_sizes:
        r_real, r_imag = c_layer.make_state(batch)

        # Pre-generate all inputs to exclude generation time from timing
        inputs = [
            np.random.randn(batch, cfg.d_model).astype(np.float32)
            for _ in range(warmup + n_tokens)
        ]

        # Warmup
        for i in range(warmup):
            c_layer.step(inputs[i], r_real, r_imag, position=i)

        # Reset state
        r_real, r_imag = c_layer.make_state(batch)

        t0 = time.perf_counter()
        for i in range(n_tokens):
            c_layer.step(inputs[warmup + i], r_real, r_imag, position=i)
        elapsed_s = time.perf_counter() - t0

        total_tokens = batch * n_tokens
        tokens_per_s = total_tokens / elapsed_s
        ms_per_token = elapsed_s * 1000.0 / n_tokens

        print(f"{batch:>6} | {tokens_per_s:>12,.1f} | {ms_per_token:>10.3f} | "
              f"{elapsed_s * 1000:>10.1f}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="llama.cpp-style benchmark for C TRN resonance layer."
    )
    parser.add_argument("--device",    default="cpu",
                        help="PyTorch device for Python reference (default: cpu).")
    parser.add_argument("--config",    default="toy",
                        choices=["toy", "100m", "400m", "1b"],
                        help="Model config preset (default: toy).")
    parser.add_argument("--n-tokens",  type=int, default=100,
                        help="Number of tokens to process (default: 100).")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 4, 8, 16],
                        help="Batch sizes for throughput benchmark.")
    parser.add_argument("--warmup",    type=int, default=10,
                        help="Warmup steps before timing (default: 10).")
    parser.add_argument("--weight-file", default=None,
                        help="Pre-exported .bin weight file (skips export step).")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Run correctness check only (no throughput benchmark).")
    parser.add_argument("--atol",      type=float, default=1e-5,
                        help="Absolute tolerance for correctness (default: 1e-5).")
    parser.add_argument("--rtol",      type=float, default=1e-4,
                        help="Relative tolerance for correctness (default: 1e-4).")
    parser.add_argument("--correctness-batch", type=int, default=2,
                        help="Batch size for correctness check (default: 2).")

    args = parser.parse_args()
    cfg  = make_config(args.config)

    success = True

    if args.correctness_only:
        ok = run_correctness(
            cfg,
            device=args.device,
            n_tokens=args.n_tokens,
            batch_size=args.correctness_batch,
            rtol=args.rtol,
            atol=args.atol,
        )
        return 0 if ok else 1

    # Full run: correctness + benchmark
    ok = run_correctness(
        cfg,
        device=args.device,
        n_tokens=100,
        batch_size=2,
        rtol=args.rtol,
        atol=args.atol,
    )
    success = success and ok

    # Load C layer for benchmark
    if args.weight_file:
        c_layer = TRNResonance.load(args.weight_file)
    else:
        torch.manual_seed(42)
        py_layer = build_layer(cfg, args.device)
        with tempfile.TemporaryDirectory() as tmp:
            c_layer = export_and_load(py_layer, Path(tmp))
        del py_layer

    run_benchmark(
        cfg,
        device=args.device,
        n_tokens=args.n_tokens,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        c_layer=c_layer,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
