"""fla_scan -- FLA-accelerated resonance scan with Triton / torch.compile fallbacks.

NOTE: This module is NOT used by TemporalResonanceLayer by default.
The default CUDA path uses Kogge-Stone parallel prefix scan in scan.py.
This module is an opt-in alternative for environments where Triton/FLA
are available (Linux + supported GPU). Import fla_resonance_scan directly
if you want to use it.

Provides a single public function::

    fla_resonance_scan(alpha, drive_r, drive_i) -> (out_r, out_i)

Backend selection order (highest priority first):

  A. flash-linear-attention (fla) -- if installed and on CUDA
  B. Triton kernel -- if triton is installed and on CUDA
  C. torch.compile on chunked scan -- if torch.compile is available (PyTorch >= 2.0)
  D. Plain chunked_resonance_scan -- always available (final fallback)

All backends produce numerically equivalent results within atol=1e-5 (fp32).
"""

from __future__ import annotations

import functools
import logging
from typing import Callable

import torch
from torch import Tensor

from .scan import chunked_resonance_scan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend availability detection
# ---------------------------------------------------------------------------

def _try_import_fla() -> bool:
    """Return True if flash-linear-attention ops are importable."""
    try:
        # fla exposes simple_gla or chunk_simple_gla under fla.ops
        import fla  # noqa: F401
        from fla.ops import chunk_simple_gla  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _try_import_triton() -> bool:
    """Return True if triton is importable and functional."""
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_FLA_AVAILABLE: bool = _try_import_fla()
_TRITON_AVAILABLE: bool = _try_import_triton()

# ---------------------------------------------------------------------------
# Option A: FLA backend
# ---------------------------------------------------------------------------

def _fla_resonance_scan(
    alpha: Tensor,    # (B, n, K) fp32
    drive_r: Tensor,  # (B, n, K) fp32
    drive_i: Tensor,  # (B, n, K) fp32
) -> tuple[Tensor, Tensor]:
    """Resonance scan via flash-linear-attention chunk_simple_gla.

    chunk_simple_gla expects inputs shaped (B, H, n, d) where H is heads and
    d is the per-head dimension. We map our (B, n, K) tensors as:
      B -> B, H=1, n -> n, d=K

    chunk_simple_gla signature (simplified):
      chunk_simple_gla(q, k, v, g, scale=1.0, ...) -> o
    where the recurrence is: h_t = g_t * h_{t-1} + k_t * v_t

    We exploit the identity: if q_t = 1 and k_t = 1, then
      o_t = h_t = g_t * h_{t-1} + v_t
    which matches our recurrence r_t = alpha_t * r_{t-1} + drive_t.

    So: g = alpha, v = drive, q = k = ones.
    """
    from fla.ops import chunk_simple_gla

    B, n, K = alpha.shape

    # Reshape to (B, H=1, n, K)
    g = alpha.unsqueeze(1)       # (B, 1, n, K)
    v_r = drive_r.unsqueeze(1)   # (B, 1, n, K)
    v_i = drive_i.unsqueeze(1)   # (B, 1, n, K)

    ones = torch.ones_like(g)    # q = k = 1 to extract h directly

    # FLA returns output tensor; some versions also return final state
    out_r = chunk_simple_gla(q=ones, k=ones, v=v_r, g=g, scale=1.0)
    out_i = chunk_simple_gla(q=ones, k=ones, v=v_i, g=g, scale=1.0)

    # strip head dimension: (B, 1, n, K) -> (B, n, K)
    if out_r.dim() == 4:
        out_r = out_r.squeeze(1)
    if out_i.dim() == 4:
        out_i = out_i.squeeze(1)

    return out_r, out_i


# ---------------------------------------------------------------------------
# Option B: Triton kernel backend
# ---------------------------------------------------------------------------

def _build_triton_kernel():
    """Build and return the Triton scan kernel (called lazily once)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _linear_scan_fwd_kernel(
        alpha_ptr,
        drive_ptr,
        output_ptr,
        B: tl.constexpr,  # noqa: N803 -- uppercase matches convention
        T: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """First-order linear scan: h_t = alpha_t * h_{t-1} + drive_t.

        Grid: (B, ceil(K / BLOCK_K))
        Each program owns one (batch, channel-block) pair and iterates
        sequentially over T timesteps. The parallelism is across B*K, not
        across T -- the T loop is intentionally sequential because the
        recurrence is causal.
        """
        pid_b = tl.program_id(0)   # batch index
        pid_k = tl.program_id(1)   # channel-block index

        k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Accumulate hidden state in registers (fp32)
        h = tl.zeros([BLOCK_K], dtype=tl.float32)

        for t in range(T):
            # Flat index: batch * T * K + t * K + channel
            base = pid_b * T * K + t * K + k_offsets

            a = tl.load(alpha_ptr + base, mask=k_mask, other=1.0)
            d = tl.load(drive_ptr + base, mask=k_mask, other=0.0)

            h = a * h + d

            tl.store(output_ptr + base, h, mask=k_mask)

    return _linear_scan_fwd_kernel


# Lazy-loaded kernel; None until first CUDA call
_triton_kernel = None


def _triton_resonance_scan(
    alpha: Tensor,    # (B, n, K) fp32 contiguous
    drive_r: Tensor,  # (B, n, K) fp32 contiguous
    drive_i: Tensor,  # (B, n, K) fp32 contiguous
) -> tuple[Tensor, Tensor]:
    """Launch the Triton linear-scan kernel for real and imaginary drives."""
    global _triton_kernel
    if _triton_kernel is None:
        _triton_kernel = _build_triton_kernel()

    import triton
    import math

    B, T, K = alpha.shape

    # Ensure contiguous fp32
    alpha_c = alpha.contiguous()
    drive_r_c = drive_r.contiguous()
    drive_i_c = drive_i.contiguous()

    out_r = torch.empty_like(drive_r_c)
    out_i = torch.empty_like(drive_i_c)

    # Choose BLOCK_K as a power of two, at least 16, at most 128
    BLOCK_K = min(128, max(16, 1 << math.floor(math.log2(K))))

    grid = (B, triton.cdiv(K, BLOCK_K))

    # Real part scan
    _triton_kernel[grid](
        alpha_c, drive_r_c, out_r,
        B=B, T=T, K=K, BLOCK_K=BLOCK_K,
    )

    # Imaginary part scan (same alpha, different drive)
    _triton_kernel[grid](
        alpha_c, drive_i_c, out_i,
        B=B, T=T, K=K, BLOCK_K=BLOCK_K,
    )

    return out_r, out_i


# ---------------------------------------------------------------------------
# Option C: torch.compile fallback
# ---------------------------------------------------------------------------

_compiled_chunked: Callable | None = None


def _compiled_resonance_scan(
    alpha: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
) -> tuple[Tensor, Tensor]:
    """chunked_resonance_scan wrapped with torch.compile (JIT once, reuse).

    If torch.compile is unavailable or fails (e.g. no C++ compiler on Windows
    outside a VS Developer Prompt), falls back transparently to the plain
    chunked scan so the caller sees no crash.
    """
    global _compiled_chunked
    if _compiled_chunked is None:
        try:
            _compiled_chunked = torch.compile(
                functools.partial(chunked_resonance_scan, chunk_size=16),
                fullgraph=False,
            )
        except Exception:
            # torch.compile unavailable or compiler missing -- use plain chunked
            _compiled_chunked = functools.partial(chunked_resonance_scan, chunk_size=16)

    try:
        return _compiled_chunked(alpha, drive_r, drive_i)
    except Exception:
        # Compilation succeeded but first invocation failed (e.g. missing cl.exe
        # on Windows, no MSVC in PATH). Replace with plain chunked for this session.
        _compiled_chunked = functools.partial(chunked_resonance_scan, chunk_size=16)
        return _compiled_chunked(alpha, drive_r, drive_i)


# ---------------------------------------------------------------------------
# Backend registry and auto-select
# ---------------------------------------------------------------------------

class _BackendTag:
    FLA = "fla"
    TRITON = "triton"
    COMPILE = "compile"
    CHUNKED = "chunked"


def _select_backend(device: torch.device) -> str:
    """Return the best available backend tag for the given device."""
    on_cuda = device.type == "cuda"

    if on_cuda and _FLA_AVAILABLE:
        return _BackendTag.FLA

    if on_cuda and _TRITON_AVAILABLE:
        return _BackendTag.TRITON

    # torch.compile is beneficial on both CPU and CUDA
    try:
        torch.compile  # available since PyTorch 2.0
        return _BackendTag.COMPILE
    except AttributeError:
        pass

    return _BackendTag.CHUNKED


_BACKEND_DISPATCH: dict[str, Callable] = {
    _BackendTag.FLA: _fla_resonance_scan,
    _BackendTag.TRITON: _triton_resonance_scan,
    _BackendTag.COMPILE: _compiled_resonance_scan,
    _BackendTag.CHUNKED: chunked_resonance_scan,
}

# Cache per device-type so we only probe once
_backend_cache: dict[str, str] = {}


def fla_resonance_scan(
    alpha: Tensor,    # (B, n, K) fp32 decay gates
    drive_r: Tensor,  # (B, n, K) fp32 real drive
    drive_i: Tensor,  # (B, n, K) fp32 imaginary drive
) -> tuple[Tensor, Tensor]:
    """FLA-accelerated resonance scan.

    Computes the first-order linear recurrence:

        r_t = alpha_t * r_{t-1} + drive_t,   r_0 = 0

    independently for the real (drive_r) and imaginary (drive_i) drives,
    sharing the same alpha tensor.

    Backend selection (in priority order):
      1. FLA (flash-linear-attention) -- if installed and on CUDA
      2. Triton kernel -- if triton is installed and on CUDA
      3. torch.compile wrapped chunked scan -- if torch.compile is available
      4. Plain chunked_resonance_scan -- always available

    All backends are numerically equivalent within atol=1e-5 (fp32).

    Args:
        alpha:   (B, n, K) fp32 decay gates, typically in [0, 1).
        drive_r: (B, n, K) fp32 real part drive signal.
        drive_i: (B, n, K) fp32 imaginary part drive signal.

    Returns:
        (out_r, out_i): each (B, n, K) fp32 -- the resonance state at every
        timestep for the real and imaginary parts respectively.
    """
    # Enforce fp32 -- callers may pass bf16 by accident
    if alpha.dtype != torch.float32:
        alpha = alpha.float()
    if drive_r.dtype != torch.float32:
        drive_r = drive_r.float()
    if drive_i.dtype != torch.float32:
        drive_i = drive_i.float()

    device_key = alpha.device.type
    if device_key not in _backend_cache:
        tag = _select_backend(alpha.device)
        _backend_cache[device_key] = tag
        logger.debug("fla_scan: selected backend '%s' for device '%s'", tag, device_key)

    tag = _backend_cache[device_key]
    backend_fn = _BACKEND_DISPATCH[tag]

    try:
        return backend_fn(alpha, drive_r, drive_i)
    except Exception as exc:
        # Degrade gracefully on first failure: remove this backend from cache
        # and retry with the plain chunked fallback.
        logger.warning(
            "fla_scan: backend '%s' raised %s -- falling back to chunked scan",
            tag, type(exc).__name__,
        )
        _backend_cache[device_key] = _BackendTag.CHUNKED
        return chunked_resonance_scan(alpha, drive_r, drive_i)


def get_active_backend(device: torch.device | str = "cpu") -> str:
    """Return the backend tag that would be used for the given device.

    Useful for test assertions and logging.
    """
    if isinstance(device, str):
        device = torch.device(device)
    device_key = device.type
    if device_key not in _backend_cache:
        tag = _select_backend(device)
        _backend_cache[device_key] = tag
    return _backend_cache[device_key]


def reset_backend_cache() -> None:
    """Clear the backend selection cache (forces re-probing on next call).

    Intended for tests that monkeypatch availability flags.
    """
    _backend_cache.clear()
    global _triton_kernel, _compiled_chunked
    _triton_kernel = None
    _compiled_chunked = None
