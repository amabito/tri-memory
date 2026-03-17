from __future__ import annotations

import math
from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OscillatorProjection(nn.Module):
    """Projects token embeddings to oscillator parameters (A, omega, phi, alpha, g_out, beta).

    Output layout (6*K channels total):
      [0 : K ]  amplitude logits   -> A     = softplus(.).clamp(max=amplitude_max)
      [K : 2K]  frequency offsets  -> omega = sigmoid(.) * pi + omega_base
      [2K: 3K]  phase offsets      -> phi   = tanh(.) * pi
      [3K: 4K]  decay gate logits  -> alpha = sigmoid(.)
      [4K: 5K]  output gate logits -> g_out = sigmoid(.)
      [5K: 6K]  erase gate logits  -> beta  = sigmoid(.)   (delta rule)
    """

    def __init__(
        self,
        d_model: int,
        K: int,
        amplitude_max: float = 3.0,
        gate_bias_init: float = 0.65,
    ) -> None:
        super().__init__()
        self.K = K
        self.amplitude_max = amplitude_max
        self.proj = nn.Linear(d_model, 6 * K, bias=True)

        # Learnable base frequencies: uniformly spread over (0.05*pi, 0.95*pi).
        self.omega_base = nn.Parameter(
            torch.linspace(0.05 * pi, 0.95 * pi, K)
        )

        self._init_weights(gate_bias_init)

    def _init_weights(self, gate_bias_init: float = 0.65) -> None:
        nn.init.normal_(self.proj.weight, std=self.proj.in_features ** -0.5)
        nn.init.zeros_(self.proj.bias)
        # Alpha (decay gate) logits at [3K:4K] -- multi-scale groups:
        #   Fast   (K//4):  alpha_center=0.30, ~1.4-step memory span
        #   Medium (K//2):  alpha_center=0.65, ~2.9-step memory span
        #   Slow   (K//4):  alpha_center=0.97, ~33-step memory span
        # Gate bias formula: sigmoid(b) = c  =>  b = log(c / (1 - c))
        K = self.K
        group_sizes = [K // 4, K // 2, K - K // 4 - K // 2]
        alpha_centers = [0.30, 0.65, 0.97]
        offset = 3 * K
        for n_k, alpha_c in zip(group_sizes, alpha_centers):
            alpha_c_clamped = max(0.01, min(0.99, alpha_c))
            bias_val = math.log(alpha_c_clamped / (1.0 - alpha_c_clamped))
            self.proj.bias.data[offset : offset + n_k].fill_(bias_val)
            offset += n_k
        # Output gate (g_out) at [4K:5K]: keep gate_bias_init default (0.65)
        gate_bias_init_clamped = max(0.01, min(0.99, gate_bias_init))
        g_out_bias = math.log(gate_bias_init_clamped / (1.0 - gate_bias_init_clamped))
        self.proj.bias.data[4 * K : 5 * K].fill_(g_out_bias)
        # Beta (erase gate) init: sigmoid(-2) ~ 0.12, weak erase at start
        self.proj.bias.data[5 * K :].fill_(-2.0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, n, d_model)
        Returns:
            A:     (B, n, K)  amplitude > 0
            omega: (B, n, K)  frequency in (0.05*pi, pi)
            phi:   (B, n, K)  phase in (-pi, pi)
            alpha: (B, n, K)  decay gate in (0, 1)
            g_out: (B, n, K)  output gate in (0, 1)
            beta:  (B, n, K)  erase gate in (0, 1)
        """
        out = self.proj(x)  # (B, n, 6K)
        A_r, Om_r, Ph_r, Ga_r, Go_r, Be_r = out.chunk(6, dim=-1)

        # softplus + clamp: avoids the double-saturation gradient vanishing that
        # occurred with the old softplus+tanh formulation. Maps to (0, amplitude_max).
        A = F.softplus(A_r).clamp(max=self.amplitude_max)
        # Clamp omega below the Nyquist limit (pi) to prevent aliasing in the
        # discrete-time recurrence. Without the clamp, sigmoid can push omega
        # arbitrarily close to pi, where cos/sin aliasing degrades scan quality.
        omega = (torch.sigmoid(Om_r) * pi + self.omega_base).clamp(min=1e-4, max=pi - 1e-4)
        phi   = torch.tanh(Ph_r) * pi
        # alpha in (0, 1) via sigmoid. Near-0 and near-1 saturation causes
        # gradient vanishing. Near-0 also triggers cumprod numerical issues
        # in scan.py. The gate_bias_init (default 0.65) keeps alpha centered
        # away from extremes at initialization.
        alpha = torch.sigmoid(Ga_r)
        g_out = torch.sigmoid(Go_r)
        beta  = torch.sigmoid(Be_r)

        return A, omega, phi, alpha, g_out, beta
