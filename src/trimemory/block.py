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

    gate and up projections are fused into a single Linear (gate_up) to halve
    the number of GEMMs on the input side. The fused weight is split after matmul.
    """

    def __init__(self, d_model: int, d_ff_hidden: int) -> None:
        super().__init__()
        self.d_ff_hidden = d_ff_hidden
        self.gate_up = nn.Linear(d_model, d_ff_hidden * 2, bias=False)
        self.down    = nn.Linear(d_ff_hidden, d_model, bias=False)

        std_in  = d_model ** -0.5
        std_out = d_ff_hidden ** -0.5
        nn.init.normal_(self.gate_up.weight, std=std_in)
        nn.init.normal_(self.down.weight, std=std_out)

    def forward(self, x: Tensor) -> Tensor:
        y = self.gate_up(x)
        gate, up = y.split(self.d_ff_hidden, dim=-1)
        return self.down(F.silu(gate) * up)


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
            clamp_resonance      = cfg.clamp_resonance,
            resonance_clamp_val  = cfg.resonance_clamp_val,
            amplitude_max        = cfg.amplitude_max,
            state_norm           = cfg.state_norm,
            res_scale_init       = cfg.res_scale_init,
            gate_bias_init       = cfg.gate_bias_init,
            phase_mode           = cfg.phase_mode,
            scan_chunk_size      = cfg.scan_chunk_size,
        )
        self.ffn  = SwiGLU(cfg.d_model, cfg.d_ff_hidden)
        self.drop = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self.resonance(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class CausalAttnBlock(nn.Module):
    """One causal self-attention layer with pre-norm + SwiGLU FFN.

    Uses PyTorch's scaled_dot_product_attention with is_causal=True.
    No sliding window -- full causal attention up to seq_len.
    """

    def __init__(self, cfg: TRNConfig, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = cfg.d_model // n_heads
        assert cfg.d_model % n_heads == 0

        self.norm1 = build_rms_norm(cfg.d_model)
        self.norm2 = build_rms_norm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff_hidden)
        self.drop = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )

        std = cfg.d_model ** -0.5
        nn.init.normal_(self.qkv.weight, std=std)
        nn.init.normal_(self.proj.weight, std=std)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        x = x + self.drop(self.proj(attn_out))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x
