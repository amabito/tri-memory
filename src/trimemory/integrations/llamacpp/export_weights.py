"""export_weights.py — Export TRN resonance layer weights to the binary format.

Binary weight format (all float32, little-endian):
  Header (32 bytes = 8 x 4-byte words):
    magic         : uint32  = 0x54524E31
    version       : uint32  = 1
    d_model       : int32
    K             : int32
    phase_mode    : int32   (0=log, 1=linear)
    state_norm    : int32   (0=off, 1=on)
    amplitude_max : float32
    res_scale     : float32
  Data (float32 row-major):
    proj_weight   [4K * d_model]
    proj_bias     [4K]
    omega_base    [K]
    w_res_weight  [d_model * K]

Weight keys per layer index i (from TRNModel state_dict):
    blocks.{i}.resonance.proj.proj.weight  -> [4K, d_model]
    blocks.{i}.resonance.proj.proj.bias    -> [4K]
    blocks.{i}.resonance.proj.omega_base   -> [K]
    blocks.{i}.resonance.W_res.weight      -> [d_model, K]
    blocks.{i}.resonance.res_scale         -> scalar
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch


TRN_MAGIC   = 0x54524E31
TRN_VERSION = 1

PHASE_MODE_LOG    = 0
PHASE_MODE_LINEAR = 1


def export_layer(
    state_dict: dict,
    layer_idx: int,
    output_path: str | Path,
    phase_mode: str = "log",
    state_norm: bool = True,
    amplitude_max: float = 3.0,
) -> None:
    """Export a single resonance layer to a binary weight file.

    Args:
        state_dict:    Full model state_dict (from model.state_dict()).
        layer_idx:     Block index (0-based).
        output_path:   Destination .bin file path.
        phase_mode:    "log" or "linear" (must match model config).
        state_norm:    Whether state normalisation is enabled.
        amplitude_max: Softplus clamp ceiling.
    """
    prefix = f"blocks.{layer_idx}.resonance"

    proj_weight = state_dict[f"{prefix}.proj.proj.weight"].float().cpu()   # [4K, d_model]
    proj_bias   = state_dict[f"{prefix}.proj.proj.bias"].float().cpu()     # [4K]
    omega_base  = state_dict[f"{prefix}.proj.omega_base"].float().cpu()    # [K]
    w_res_weight = state_dict[f"{prefix}.W_res.weight"].float().cpu()      # [d_model, K]
    res_scale_t  = state_dict[f"{prefix}.res_scale"].float().cpu()

    K      = omega_base.shape[0]
    d_model = proj_weight.shape[1]

    # Sanity checks
    assert proj_weight.shape == (4 * K, d_model), \
        f"proj_weight shape mismatch: {proj_weight.shape}"
    assert proj_bias.shape == (4 * K,), \
        f"proj_bias shape mismatch: {proj_bias.shape}"
    assert w_res_weight.shape == (d_model, K), \
        f"w_res_weight shape mismatch: {w_res_weight.shape}"

    res_scale    = float(res_scale_t.item())
    phase_mode_i = PHASE_MODE_LOG if phase_mode == "log" else PHASE_MODE_LINEAR
    state_norm_i = 1 if state_norm else 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # Header: 8 x 4-byte fields
        f.write(struct.pack("<II", TRN_MAGIC, TRN_VERSION))
        f.write(struct.pack("<iiii", d_model, K, phase_mode_i, state_norm_i))
        f.write(struct.pack("<ff", amplitude_max, res_scale))

        # Data tensors in order
        f.write(proj_weight.contiguous().numpy().tobytes())
        f.write(proj_bias.contiguous().numpy().tobytes())
        f.write(omega_base.contiguous().numpy().tobytes())
        f.write(w_res_weight.contiguous().numpy().tobytes())

    expected_size = 32 + (4 * K * d_model + 4 * K + K + d_model * K) * 4
    actual_size   = output_path.stat().st_size
    assert actual_size == expected_size, \
        f"Written size {actual_size} != expected {expected_size}"


def export_model(
    model_or_state_dict,
    output_dir: str | Path,
    *,
    phase_mode: str = "log",
    state_norm: bool = True,
    amplitude_max: float = 3.0,
    layers: list[int] | None = None,
) -> list[Path]:
    """Export all (or selected) resonance layers from a TRNModel.

    Args:
        model_or_state_dict: TRNModel instance or state_dict.
        output_dir:          Directory to write *.bin files.
        phase_mode:          "log" or "linear".
        state_norm:          Whether state normalisation is enabled.
        amplitude_max:       Softplus clamp ceiling.
        layers:              List of layer indices; None = export all.

    Returns:
        List of paths to the written files.
    """
    if hasattr(model_or_state_dict, "state_dict"):
        state_dict = model_or_state_dict.state_dict()
    else:
        state_dict = model_or_state_dict

    # Auto-detect number of layers
    n_layers = max(
        int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")
    ) + 1

    if layers is None:
        layers = list(range(n_layers))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i in layers:
        out_path = output_dir / f"resonance_layer_{i:03d}.bin"
        export_layer(
            state_dict,
            layer_idx=i,
            output_path=out_path,
            phase_mode=phase_mode,
            state_norm=state_norm,
            amplitude_max=amplitude_max,
        )
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export TRN resonance layer weights to binary format."
    )
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint (.pt) file.")
    parser.add_argument("output_dir", help="Directory to write .bin files.")
    parser.add_argument(
        "--layers", nargs="*", type=int, default=None,
        help="Layer indices to export (default: all)."
    )
    parser.add_argument(
        "--phase-mode", choices=["log", "linear"], default="log",
        help="Phase mode (must match model config)."
    )
    parser.add_argument("--no-state-norm", action="store_true",
                        help="Disable state normalisation in header.")
    parser.add_argument("--amplitude-max", type=float, default=3.0,
                        help="amplitude_max value (default: 3.0).")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt)

    paths = export_model(
        state_dict,
        args.output_dir,
        phase_mode=args.phase_mode,
        state_norm=not args.no_state_norm,
        amplitude_max=args.amplitude_max,
        layers=args.layers,
    )
    for p in paths:
        print(f"[OK] {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
