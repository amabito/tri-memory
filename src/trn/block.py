from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TRNConfig
from .resonance import TemporalResonanceLayer
from .utils import build_rms_norm


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (LLaMA-style).

    Hidden dimension uses cfg.d_ff_hidden = (2/3) * d_ff (rounded to 256),
    so total parameter count matches a conventional FFN with expansion d_ff/d_model.
    """

    def __init__(self, d_model: int, d_ff_hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff_hidden, bias=False)
        self.up   = nn.Linear(d_model, d_ff_hidden, bias=False)
        self.down = nn.Linear(d_ff_hidden, d_model, bias=False)

        std_in  = d_model ** -0.5
        std_out = d_ff_hidden ** -0.5
        nn.init.normal_(self.gate.weight, std=std_in)
        nn.init.normal_(self.up.weight,   std=std_in)
        nn.init.normal_(self.down.weight, std=std_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TRNBlock(nn.Module):
    """One TRN layer: pre-norm resonance + pre-norm SwiGLU FFN."""

    def __init__(self, cfg: TRNConfig) -> None:
        super().__init__()
        self.norm1 = build_rms_norm(cfg.d_model)
        self.norm2 = build_rms_norm(cfg.d_model)
        self.resonance = TemporalResonanceLayer(
            d_model              = cfg.d_model,
            K                    = cfg.n_oscillators,
            use_parallel_scan    = cfg.use_parallel_scan,
            log_phase            = cfg.log_phase,
            clamp_resonance      = cfg.clamp_resonance,
            resonance_clamp_val  = cfg.resonance_clamp_val,
            amplitude_max        = cfg.amplitude_max,
            state_norm           = cfg.state_norm,
            res_scale_init       = cfg.res_scale_init,
            gate_bias_init       = cfg.gate_bias_init,
            phase_mode           = cfg.phase_mode,
        )
        self.ffn  = SwiGLU(cfg.d_model, cfg.d_ff_hidden)
        self.drop = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self.resonance(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x
