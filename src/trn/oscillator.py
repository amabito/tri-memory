from __future__ import annotations

from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OscillatorProjection(nn.Module):
    """Projects token embeddings to oscillator parameters (A, omega, phi, alpha).

    Output layout (4*K channels total):
      [0 : K ]  amplitude logits   -> A     = softplus(.).clamp(max=amplitude_max)
      [K : 2K]  frequency offsets  -> omega = sigmoid(.) * pi + omega_base
      [2K: 3K]  phase offsets      -> phi   = tanh(.) * pi
      [3K: 4K]  decay gate logits  -> alpha = sigmoid(.)
    """

    def __init__(
        self,
        d_model: int,
        K: int,
        amplitude_max: float = 3.0,
        gate_bias_init: float = 0.85,
    ) -> None:
        super().__init__()
        self.K = K
        self.amplitude_max = amplitude_max
        self.proj = nn.Linear(d_model, 4 * K, bias=True)

        # Learnable base frequencies: uniformly spread over (0.05*pi, 0.95*pi).
        self.omega_base = nn.Parameter(
            torch.linspace(0.05 * pi, 0.95 * pi, K)
        )

        self._init_weights(gate_bias_init)

    def _init_weights(self, gate_bias_init: float = 0.85) -> None:
        nn.init.normal_(self.proj.weight, std=self.proj.in_features ** -0.5)
        nn.init.zeros_(self.proj.bias)
        # Gate bias: sigmoid(b) = target  =>  b = log(target / (1 - target))
        # gate_bias_init=0.85 was the old default (sigmoid(1.73)~0.85).
        # P0 stabilization uses 0.65 => sigmoid(0.619)~0.65 for gentler decay.
        import math
        gate_bias_init_clamped = max(0.01, min(0.99, gate_bias_init))
        bias_val = math.log(gate_bias_init_clamped / (1.0 - gate_bias_init_clamped))
        self.proj.bias.data[3 * self.K :].fill_(bias_val)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, n, d_model)
        Returns:
            A:     (B, n, K)  amplitude > 0
            omega: (B, n, K)  frequency in (0.05*pi, 1.95*pi)
            phi:   (B, n, K)  phase in (-pi, pi)
            alpha: (B, n, K)  decay gate in (0, 1)
        """
        out = self.proj(x)  # (B, n, 4K)
        A_r, Om_r, Ph_r, Ga_r = out.chunk(4, dim=-1)

        # Soft clamp: tanh squash avoids gradient vanishing at the boundary.
        # Maps softplus output to (0, amplitude_max) with smooth gradient throughout.
        A_raw = F.softplus(A_r)
        A     = self.amplitude_max * torch.tanh(A_raw / self.amplitude_max)
        omega = torch.sigmoid(Om_r) * pi + self.omega_base
        phi   = torch.tanh(Ph_r) * pi
        # alpha in (0, 1) via sigmoid. Near-0 and near-1 saturation causes
        # gradient vanishing. Near-0 also triggers cumprod numerical issues
        # in scan.py. The gate_bias_init (default 0.65) keeps alpha centered
        # away from extremes at initialization.
        alpha = torch.sigmoid(Ga_r)

        return A, omega, phi, alpha
