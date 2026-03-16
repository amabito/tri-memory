"""CUDA-accelerated TRN scan via torch.utils.cpp_extension.load (JIT compile).

Usage:
    from csrc.trn_scan import cuda_resonance_scan
    r_r, r_i = cuda_resonance_scan(alpha, drive_r, drive_i)

Requires: CUDA toolkit (nvcc) accessible. On WSL2:
    export PATH=/usr/local/cuda-12.8/bin:$PATH
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor

_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    from torch.utils.cpp_extension import load

    csrc_dir = Path(__file__).parent
    _MODULE = load(
        name="trn_scan_cuda",
        sources=[str(csrc_dir / "trn_scan_kernel.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _MODULE


def cuda_resonance_scan(
    alpha: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
) -> tuple[Tensor, Tensor]:
    """CUDA-accelerated scan: r_t = alpha_t * r_{t-1} + drive_t.

    Parallel across B*K channels, sequential across T positions.
    No Python loop. Single kernel launch per real/imag.
    """
    mod = _load_module()
    alpha = alpha.float().contiguous()
    drive_r = drive_r.float().contiguous()
    drive_i = drive_i.float().contiguous()
    r_r, r_i = mod.trn_scan_forward(alpha, drive_r, drive_i)
    return r_r, r_i
