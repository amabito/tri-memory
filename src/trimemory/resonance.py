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
        scan_chunk_size: int = 64,
        res_warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.K = K
        self.use_parallel_scan = use_parallel_scan
        self.scan_chunk_size = scan_chunk_size
        self.clamp_resonance = clamp_resonance
        self.resonance_clamp_val = resonance_clamp_val
        self.state_norm_enabled = state_norm
        self.phase_mode = phase_mode
        self._res_warmup_steps = res_warmup_steps

        self.proj = OscillatorProjection(
            d_model, K,
            amplitude_max=amplitude_max,
            gate_bias_init=gate_bias_init,
        )

        # W_res projects demodulated Re+Im (2K) to d_model.
        # Using both Re and Im doubles state information utilization.
        self.W_res = nn.Linear(2 * K, d_model, bias=False)
        nn.init.normal_(self.W_res.weight, std=2e-3)

        # Learnable scalar that multiplies the resonance delta before residual add
        self.res_scale = nn.Parameter(torch.tensor(res_scale_init))
        # Warmup counter: incremented each training forward pass (compile-friendly)
        self.register_buffer("_forward_count", torch.tensor(0, dtype=torch.long))

    def _apply_state_norm(self, r_r: Tensor, r_i: Tensor) -> tuple[Tensor, Tensor]:
        """Complex modulus normalization: r /= max(|r|, 1.0).

        Uses the true complex modulus sqrt(r_r^2 + r_i^2) rather than the
        old max-abs approximation. This preserves the phase of the complex
        state while preventing magnitude from exceeding 1.0.
        """
        modulus = (r_r.pow(2) + r_i.pow(2) + 1e-8).sqrt()
        scale = modulus.clamp(min=1.0)
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

        A, omega, phi, alpha, g_out, beta = self.proj(x)   # each (B, n, K)

        positions = self._compute_positions(n, device)
        angle = omega * positions + phi  # (B, n, K)

        # Move alpha to fp32 before computing one_m_a to prevent bf16 latch:
        # in bf16, values close to 1.0 (e.g. 0.99) round to 1.0, which causes
        # the decay gate to become a latch and kills gradient flow.
        alpha_f = alpha.float()
        one_m_a = 1.0 - alpha_f
        cos_angle = torch.cos(angle).float()
        sin_angle = torch.sin(angle).float()
        A_f = A.float()
        drive_r = one_m_a * A_f * cos_angle
        drive_i = one_m_a * A_f * sin_angle

        # Disable AMP inside the scan to enforce fp32 arithmetic.
        # device_type covers both cuda and cpu paths so autocast is a no-op on CPU
        # (avoids the "cuda autocast on CPU tensor" warning).
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            if self.use_parallel_scan and x.is_cuda:
                r_r, r_i = parallel_resonance_scan(alpha_f, drive_r, drive_i)
            else:
                r_r, r_i = chunked_resonance_scan(alpha_f, drive_r, drive_i, chunk_size=self.scan_chunk_size)

        # Delta rule (write-with-erase): subtract old content at current frequency
        # before the scan output is used. This is a post-scan approximation of
        # DeltaNet's targeted memory overwriting. The erase gate beta controls
        # how much of the current-frequency readout is removed from state.
        beta_f = beta.float()
        readout = r_r * cos_angle + r_i * sin_angle  # (B, n, K) -- cos/sin already fp32
        erase_r = beta_f * readout * cos_angle
        erase_i = beta_f * readout * sin_angle
        r_r = r_r - erase_r
        r_i = r_i - erase_i

        # P0-D: Always-on per-channel state normalization (default ON)
        if self.state_norm_enabled:
            r_r, r_i = self._apply_state_norm(r_r, r_i)

        # Legacy L2 norm clamping (behind flag, default off)
        # Deprecated: superseded by state_norm. Retained for ablation reproducibility.
        if self.clamp_resonance:
            state_norm = (r_r.pow(2) + r_i.pow(2)).sqrt().clamp(min=1e-8)
            scale = (state_norm / self.resonance_clamp_val).clamp(min=1.0)
            r_r = r_r / scale
            r_i = r_i / scale

        # Cast back to x dtype before the output projection.
        r_r = r_r.to(x.dtype)
        r_i = r_i.to(x.dtype)

        # Demodulate: extract both Re and Im of r * exp(-j*angle).
        cos_a = cos_angle.to(r_r.dtype)
        sin_a = sin_angle.to(r_r.dtype)
        rho_re = r_r * cos_a + r_i * sin_a         # (B, n, K)
        rho_im = -r_r * sin_a + r_i * cos_a        # (B, n, K)
        rho = torch.cat([rho_re, rho_im], dim=-1)  # (B, n, 2K)
        # Output gate (GLA-style): selective readout, repeated for Re+Im channels
        rho = g_out.to(rho.dtype).repeat(1, 1, 2) * rho

        # P0-A: apply learnable res_scale with smoothstep warmup before projection.
        # Warmup is training-only: during eval the full scale is applied immediately.
        # Compile-friendly: no .item() call, no Python branch on tensor value.
        # NOTE: _forward_count counts per-layer forward() calls, not optimizer steps.
        # For an N-layer model, res_warmup_steps=1000 means ~1000/N optimizer steps.
        # WARNING: _forward_count buffer mutation is incompatible with CUDA graphs.
        # Use torch.compile without CUDA graphs, or set res_warmup_steps=0.
        if self.training:
            self._forward_count += 1
            if self._res_warmup_steps > 0:
                warmup_t = (self._forward_count.float() / self._res_warmup_steps).clamp(max=1.0)
                warmup_factor: float | Tensor = warmup_t * warmup_t * (3.0 - 2.0 * warmup_t)
            else:
                warmup_factor = 1.0
        else:
            warmup_factor = 1.0
        return (self.res_scale * warmup_factor) * self.W_res(rho)  # (B, n, d_model)

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
        A_t, omega_t, phi_t, alpha_t, g_out_t, beta_t = self.proj(x_t)  # each (B, 1, K)

        A_t     = A_t[:, 0]      # (B, K)
        omega_t = omega_t[:, 0]
        phi_t   = phi_t[:, 0]
        alpha_t = alpha_t[:, 0]
        g_out_t = g_out_t[:, 0]
        beta_t  = beta_t[:, 0]

        pos = float(position)
        if self.phase_mode == "log":
            pos = math.log1p(pos)
        angle = omega_t * pos + phi_t  # (B, K)

        alpha_f = alpha_t.float()
        one_m_a = 1.0 - alpha_f
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        v_r = (one_m_a * A_t.float() * cos_angle.float())
        v_i = (one_m_a * A_t.float() * sin_angle.float())

        # State update in fp32.
        r_real  = alpha_f * r_real + v_r
        r_imag  = alpha_f * r_imag + v_i

        # Delta rule erase: remove old content at current frequency
        readout_t = r_real * cos_angle.float() + r_imag * sin_angle.float()
        beta_f = beta_t.float()
        r_real = r_real - beta_f * readout_t * cos_angle.float()
        r_imag = r_imag - beta_f * readout_t * sin_angle.float()

        # Demodulate and project.
        # state_norm is applied to a COPY used only for demodulation output.
        # The raw (un-normalized) r_real/r_imag are returned as the carried
        # state so that inference matches training: forward() runs the scan
        # without per-step normalization and normalizes the full output once
        # afterwards. Normalizing the carried state here would feed normalized
        # values back into the next recurrence step, making inference
        # mathematically different from training (max-abs norm is nonlinear).
        #
        # NOTE: Training applies _apply_state_norm per-element across the full
        # sequence (B, n, K), while inference applies it per-step (B, K).
        # The normalization scale differs: training uses per-position modulus,
        # inference also uses per-position (B, K) modulus. These ARE equivalent
        # when the norm is applied element-wise (which it is after the H2 fix
        # using complex modulus). No mismatch exists with per-element normalization.
        if self.state_norm_enabled:
            r_real_demod, r_imag_demod = self._apply_state_norm(r_real, r_imag)
        else:
            r_real_demod, r_imag_demod = r_real, r_imag

        cos_a = cos_angle.to(r_real.dtype)
        sin_a = sin_angle.to(r_real.dtype)
        # Demodulate: Re + Im
        rho_re = r_real_demod * cos_a + r_imag_demod * sin_a   # (B, K)
        rho_im = -r_real_demod * sin_a + r_imag_demod * cos_a  # (B, K)
        rho = torch.cat([rho_re, rho_im], dim=-1)              # (B, 2K)
        # Output gate (GLA-style): selective readout from state
        g_out_2k = g_out_t.to(rho.dtype).repeat(1, 2)
        rho = g_out_2k * rho

        # P0-A: apply learnable res_scale (no warmup during inference)
        out = (self.res_scale * self.W_res(rho)).to(input_dtype)
        return out, r_real, r_imag
