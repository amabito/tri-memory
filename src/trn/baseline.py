"""GPT-style Transformer baseline for comparison with TRN."""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .config import TRNConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using F.scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.RMSNorm(d_model)
        # SwiGLU: gate * silu(gate)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        x = x + self.w3(F.silu(self.w1(h)) * self.w2(h))
        return x


class TransformerModel(nn.Module):
    """GPT-style language model with sinusoidal positional encoding.

    Has the same interface as TRNModel:
    - forward(input_ids, labels=None) -> dict with 'loss' and 'logits'
    - configure_optimizer_param_groups(weight_decay) -> list[dict]
    - num_parameters(non_embedding=True) -> int
    """

    def __init__(self, cfg: TRNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n_heads = max(1, cfg.d_model // 64)

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Sinusoidal PE as buffer (not learned)
        pe = self._build_sinusoidal_pe(cfg.max_seq_len, cfg.d_model)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, n_heads, cfg.d_ff)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> dict:
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pe[:T]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        result: dict = {"logits": logits}
        if labels is not None:
            # Causal shift: predict input_ids[1:] from input_ids[:-1]
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
            )
            result["loss"] = loss
        return result

    def configure_optimizer_param_groups(self, weight_decay: float) -> list[dict]:
        """Separate weight_decay params from no-decay (biases, norms, embeddings)."""
        decay: set[str] = set()
        no_decay: set[str] = set()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "norm" in name or "embed" in name or name.endswith(".bias"):
                no_decay.add(name)
            else:
                decay.add(name)
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        return [
            {"params": [params[n] for n in sorted(decay)], "weight_decay": weight_decay},
            {"params": [params[n] for n in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def num_parameters(self, non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed.weight.numel()
        return total
