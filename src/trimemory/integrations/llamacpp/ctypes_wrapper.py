"""ctypes_wrapper.py — Python ctypes interface to trn_resonance.so/.dll.

Usage:
    from trimemory.integrations.llamacpp.ctypes_wrapper import TRNResonance

    layer = TRNResonance.load("resonance_layer_000.bin")
    r_real = np.zeros((batch, layer.K), dtype=np.float32)
    r_imag = np.zeros((batch, layer.K), dtype=np.float32)

    for pos, x_np in enumerate(token_embeddings):    # x_np: [batch, d_model]
        out, r_real, r_imag = layer.step(x_np, r_real, r_imag, position=pos)

Shared library search order:
    1. LIBTRN_RESONANCE_PATH env var
    2. Same directory as this file
    3. trn_resonance.so / trn_resonance.dll on LD_LIBRARY_PATH / PATH
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_library() -> ctypes.CDLL:
    """Locate and load the compiled shared library."""
    candidates: list[Path] = []

    # Env var override
    env_path = os.environ.get("LIBTRN_RESONANCE_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # Same directory as this module
    here = Path(__file__).parent
    for name in ("trn_resonance.so", "trn_resonance.dll", "libtrn_resonance.so"):
        candidates.append(here / name)

    for p in candidates:
        if p.exists():
            try:
                return ctypes.CDLL(str(p))
            except OSError:
                continue

    raise FileNotFoundError(
        "trn_resonance shared library not found. "
        "Build it with:\n"
        "  gcc -O2 -shared -fPIC -o trn_resonance.so trn_resonance.c -lm\n"
        "or set LIBTRN_RESONANCE_PATH to the .so/.dll path."
    )


def _bind_functions(lib: ctypes.CDLL) -> None:
    """Set argtypes / restype for all exported functions."""
    vp = ctypes.c_void_p
    fp = ctypes.c_float
    ip = ctypes.c_int
    sp = ctypes.c_char_p

    lib.trn_resonance_load.argtypes    = [sp]
    lib.trn_resonance_load.restype     = vp

    lib.trn_resonance_free.argtypes    = [vp]
    lib.trn_resonance_free.restype     = None

    lib.trn_resonance_state_alloc.argtypes = [vp, ip]
    lib.trn_resonance_state_alloc.restype  = ctypes.POINTER(ctypes.c_float)

    lib.trn_resonance_state_free.argtypes  = [ctypes.POINTER(ctypes.c_float)]
    lib.trn_resonance_state_free.restype   = None

    lib.trn_resonance_step.argtypes = [
        vp,                              # layer
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # r_real
        ctypes.POINTER(ctypes.c_float),  # r_imag
        ip,                              # position
        ctypes.POINTER(ctypes.c_float),  # out
        ip,                              # batch_size
    ]
    lib.trn_resonance_step.restype = ip

    lib.trn_resonance_d_model.argtypes     = [vp]
    lib.trn_resonance_d_model.restype      = ip

    lib.trn_resonance_K.argtypes           = [vp]
    lib.trn_resonance_K.restype            = ip

    lib.trn_resonance_phase_mode.argtypes  = [vp]
    lib.trn_resonance_phase_mode.restype   = ip

    lib.trn_resonance_state_norm.argtypes  = [vp]
    lib.trn_resonance_state_norm.restype   = ip

    lib.trn_resonance_amplitude_max.argtypes = [vp]
    lib.trn_resonance_amplitude_max.restype  = fp

    lib.trn_resonance_res_scale.argtypes   = [vp]
    lib.trn_resonance_res_scale.restype    = fp


# Module-level singleton (lazy load)
_lib: Optional[ctypes.CDLL] = None


def get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_library()
        _bind_functions(_lib)
    return _lib


# ---------------------------------------------------------------------------
# High-level Python class
# ---------------------------------------------------------------------------

class TRNResonance:
    """Python wrapper around a loaded TRN resonance layer.

    Attributes:
        d_model      (int)   : model width
        K            (int)   : number of oscillators
        phase_mode   (str)   : "log" or "linear"
        state_norm   (bool)  : whether state normalisation is active
        amplitude_max (float): softplus clamp ceiling
        res_scale     (float): learnable output scale
    """

    def __init__(self, handle: ctypes.c_void_p, lib: ctypes.CDLL) -> None:
        self._handle = handle
        self._lib    = lib
        self.d_model       = lib.trn_resonance_d_model(handle)
        self.K             = lib.trn_resonance_K(handle)
        self.phase_mode    = "log" if lib.trn_resonance_phase_mode(handle) == 0 else "linear"
        self.state_norm    = bool(lib.trn_resonance_state_norm(handle))
        self.amplitude_max = lib.trn_resonance_amplitude_max(handle)
        self.res_scale     = lib.trn_resonance_res_scale(handle)

    @classmethod
    def load(cls, path: str | Path) -> "TRNResonance":
        """Load a resonance layer from a binary .bin file."""
        lib = get_lib()
        handle = lib.trn_resonance_load(str(path).encode())
        if not handle:
            raise RuntimeError(f"Failed to load TRN resonance from '{path}'")
        return cls(handle, lib)

    def __del__(self) -> None:
        if self._handle and self._lib:
            self._lib.trn_resonance_free(self._handle)
            self._handle = None

    def step(
        self,
        x:        np.ndarray,
        r_real:   np.ndarray,
        r_imag:   np.ndarray,
        position: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute one autoregressive step.

        Args:
            x:        Input embeddings, shape (batch_size, d_model) float32.
            r_real:   Real resonance state, shape (batch_size, K) float32.
            r_imag:   Imaginary resonance state, shape (batch_size, K) float32.
            position: Absolute token index (0-based).

        Returns:
            (out, r_real, r_imag): output (batch, d_model), updated states.
        """
        x      = np.ascontiguousarray(x,      dtype=np.float32)
        r_real = np.ascontiguousarray(r_real,  dtype=np.float32)
        r_imag = np.ascontiguousarray(r_imag,  dtype=np.float32)

        if x.ndim == 1:
            x = x[np.newaxis, :]
        batch_size = x.shape[0]

        assert x.shape == (batch_size, self.d_model), \
            f"x shape {x.shape} != ({batch_size}, {self.d_model})"
        assert r_real.shape == (batch_size, self.K), \
            f"r_real shape {r_real.shape} != ({batch_size}, {self.K})"
        assert r_imag.shape == (batch_size, self.K), \
            f"r_imag shape {r_imag.shape} != ({batch_size}, {self.K})"

        out = np.empty((batch_size, self.d_model), dtype=np.float32)

        fp_p = ctypes.POINTER(ctypes.c_float)
        ret = self._lib.trn_resonance_step(
            self._handle,
            x.ctypes.data_as(fp_p),
            r_real.ctypes.data_as(fp_p),
            r_imag.ctypes.data_as(fp_p),
            int(position),
            out.ctypes.data_as(fp_p),
            int(batch_size),
        )
        if ret != 0:
            raise RuntimeError(f"trn_resonance_step returned error code {ret}")

        return out, r_real, r_imag

    def make_state(self, batch_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Create zero-initialised resonance state arrays.

        Returns:
            (r_real, r_imag): each shape (batch_size, K) float32.
        """
        r_real = np.zeros((batch_size, self.K), dtype=np.float32)
        r_imag = np.zeros((batch_size, self.K), dtype=np.float32)
        return r_real, r_imag
