from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

from .oscillator import OscillatorProjection
from .scan import chunked_resonance_scan, parallel_resonance_scan


class TemporalResonanceLayer(nn.Module):
    """Core TRN building block.

    Computes complex resonance states via a learnable oscillatory recurrence:

        v_t  = (1 - alpha_t) * A_t * exp(j * (omega_t * t + phi_t))
        r_t  = alpha_t * r_{t-1} + v_t        (first-order linear RNN, associative)
        y_t  = Re(r_t * exp(-j * (omega_t * t + phi_t)))   (demodulate)

    The recurrence is associative -> O(log n) parallel prefix scan on GPU.
    On CPU or when torch.associative_scan is unavailable, the sequential scan
    is used instead (O(n), state <= 2 KB for K=256 at fp32).

    Mixed-precision note:
        alpha and the resonance state are kept in fp32 at all times.
        bf16 cannot represent values close to 1.0 accurately (e.g. 0.99 rounds
        to 1.0), which would cause the decay gate to become a latch.
    """

    def __init__(
        self,
        d_model: int,
        K: int,
        use_parallel_scan: bool = True,
        clamp_resonance: bool = False,
        resonance_clamp_val: float = 10.0,
        amplitude_max: float = 3.0,
        state_norm: bool = True,
        res_scale_init: float = 0.05,
        gate_bias_init: float = 0.65,
        phase_mode: str = "log",
        scan_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.K = K
        self.use_parallel_scan = use_parallel_scan
        self.scan_chunk_size = scan_chunk_size
        self.clamp_resonance = clamp_resonance
        self.resonance_clamp_val = resonance_clamp_val
        self.state_norm_enabled = state_norm
        self.phase_mode = phase_mode

        self.proj = OscillatorProjection(
            d_model, K,
            amplitude_max=amplitude_max,
            gate_bias_init=gate_bias_init,
        )

        # P0-A: smaller W_res init + learnable output scale
        self.W_res = nn.Linear(K, d_model, bias=False)
        nn.init.normal_(self.W_res.weight, std=2e-3)

        # Learnable scalar that multiplies the resonance delta before residual add
        self.res_scale = nn.Parameter(torch.tensor(res_scale_init))

    def _apply_state_norm(self, r_r: Tensor, r_i: Tensor) -> tuple[Tensor, Tensor]:
        """Per-channel max-abs normalization: r /= max(max_abs(r), 1.0)."""
        max_abs = torch.maximum(r_r.abs(), r_i.abs())  # (B, n, K) or (B, K)
        scale = max_abs.clamp(min=1.0)
        return r_r / scale, r_i / scale

    def _compute_positions(self, n: int, device: torch.device) -> Tensor:
        """Compute position encoding based on phase_mode."""
        positions = torch.arange(n, device=device, dtype=torch.float32).view(1, n, 1)
        if self.phase_mode == "log":
            positions = torch.log1p(positions)
        return positions

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, n, d_model)
        Returns:
            out: (B, n, d_model)
        """
        B, n, _ = x.shape
        device = x.device

        A, omega, phi, alpha = self.proj(x)   # each (B, n, K)

        positions = self._compute_positions(n, device)
        angle = omega * positions + phi  # (B, n, K)

        one_m_a = 1.0 - alpha

        # Cast to fp32 before scan -- critical for mixed-precision training.
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        alpha_f  = alpha.float()
        drive_r  = (one_m_a * A * cos_angle).float()
        drive_i  = (one_m_a * A * sin_angle).float()

        # Disable AMP inside the scan to enforce fp32 arithmetic.
        with torch.amp.autocast("cuda", enabled=False):
            if self.use_parallel_scan and x.is_cuda:
                r_r, r_i = parallel_resonance_scan(alpha_f, drive_r, drive_i)
            else:
                r_r, r_i = chunked_resonance_scan(alpha_f, drive_r, drive_i, chunk_size=self.scan_chunk_size)

        # P0-D: Always-on per-channel state normalization (default ON)
        if self.state_norm_enabled:
            r_r, r_i = self._apply_state_norm(r_r, r_i)

        # Legacy L2 norm clamping (behind flag, default off)
        if self.clamp_resonance:
            state_norm = (r_r.pow(2) + r_i.pow(2)).sqrt().clamp(min=1e-8)
            scale = (state_norm / self.resonance_clamp_val).clamp(min=1.0)
            r_r = r_r / scale
            r_i = r_i / scale

        # Cast back to x dtype before the output projection.
        r_r = r_r.to(x.dtype)
        r_i = r_i.to(x.dtype)

        # Demodulate: project resonance onto the local carrier.
        cos_a = cos_angle.to(x.dtype)
        sin_a = sin_angle.to(x.dtype)
        rho   = r_r * cos_a + r_i * sin_a   # (B, n, K)

        # Debug: store rho for NaN tracing (detached, no graph impact)
        if getattr(self, "_debug_trace", False):
            self._debug_last_rho = rho.detach()
            self._debug_last_W_res_out = self.W_res(rho).detach()

        # P0-A: apply learnable res_scale before projection
        return self.res_scale * self.W_res(rho)  # (B, n, d_model)

    @torch.no_grad()
    def step_single(
        self,
        x_single: Tensor,   # (B, d_model) current token embedding
        r_real:   Tensor,   # (B, K) fp32 resonance state — real part
        r_imag:   Tensor,   # (B, K) fp32 resonance state — imag part
        position: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single-step inference (no scan, O(1) per token).

        Args:
            x_single: (B, d_model) embedding of the current token
            r_real:   (B, K)  real resonance state from previous step  (fp32)
            r_imag:   (B, K)  imag resonance state from previous step  (fp32)
            position: scalar int — absolute token index in the sequence

        Returns:
            out:    (B, d_model) output features
            r_real: (B, K)      updated real state  (fp32)
            r_imag: (B, K)      updated imag state  (fp32)
        """
        input_dtype = x_single.dtype
        # Cast to weight dtype so projection works under any AMP regime.
        proj_dtype = self.proj.proj.weight.dtype
        x_t = x_single.unsqueeze(1).to(proj_dtype)     # (B, 1, d_model)
        A_t, omega_t, phi_t, alpha_t = self.proj(x_t)  # each (B, 1, K)

        A_t     = A_t[:, 0]      # (B, K)
        omega_t = omega_t[:, 0]
        phi_t   = phi_t[:, 0]
        alpha_t = alpha_t[:, 0]

        pos = float(position)
        if self.phase_mode == "log":
            pos = math.log1p(pos)
        angle = omega_t * pos + phi_t  # (B, K)

        one_m_a = 1.0 - alpha_t
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        v_r = (one_m_a * A_t * cos_angle).float()
        v_i = (one_m_a * A_t * sin_angle).float()

        # State update in fp32.
        alpha_f = alpha_t.float()
        r_real  = alpha_f * r_real + v_r
        r_imag  = alpha_f * r_imag + v_i

        # P0-D: state normalization
        if self.state_norm_enabled:
            r_real, r_imag = self._apply_state_norm(r_real, r_imag)

        # Demodulate and project.
        cos_a = cos_angle.to(r_real.dtype)
        sin_a = sin_angle.to(r_real.dtype)
        rho   = r_real * cos_a + r_imag * sin_a   # (B, K) fp32

        # P0-A: apply learnable res_scale
        out = (self.res_scale * self.W_res(rho)).to(input_dtype)
        return out, r_real, r_imag
