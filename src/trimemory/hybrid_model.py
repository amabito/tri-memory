"""HybridModel: interleaved TRN and Transformer blocks.

Architecture:
    token_embedding
    -> N x HybridBlock (TRNBlock for even layers, TransformerBlock for odd)
    -> RMSNorm
    -> lm_head

The interleaving ratio is controlled by `trn_ratio` (default 0.5 = alternating).
Has the same interface as TRNModel and TransformerModel:
    forward(input_ids, labels=None) -> dict with 'loss' and 'logits'
    configure_optimizer_param_groups(weight_decay) -> list[dict]
    num_parameters(non_embedding=True) -> int
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .baseline import CausalSelfAttention
from .block import TRNBlock
from .config import TRNConfig
from .utils import build_rms_norm


class HybridBlock(nn.Module):
    """Single hybrid layer: either TRNBlock or Transformer attention + SwiGLU FFN.

    The block type is determined at construction time by `is_trn`.
    Both paths share the same pre-norm + residual pattern so the overall
    depth / parameter budget is comparable.
    """

    def __init__(self, cfg: TRNConfig, is_trn: bool) -> None:
        super().__init__()
        self.is_trn = is_trn

        if is_trn:
            self._block = TRNBlock(cfg)
        else:
            n_heads = max(1, cfg.d_model // 64)
            self.norm1 = build_rms_norm(cfg.d_model)
            self.attn = CausalSelfAttention(cfg.d_model, n_heads)
            self.norm2 = build_rms_norm(cfg.d_model)
            # SwiGLU FFN matching TRNBlock's d_ff_hidden
            self.w1 = nn.Linear(cfg.d_model, cfg.d_ff_hidden, bias=False)
            self.w2 = nn.Linear(cfg.d_model, cfg.d_ff_hidden, bias=False)
            self.w3 = nn.Linear(cfg.d_ff_hidden, cfg.d_model, bias=False)
            self.drop = (
                nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.is_trn:
            return self._block(x)
        # Transformer path: pre-norm attention + SwiGLU FFN
        x = x + self.drop(self.attn(self.norm1(x)))
        h = self.norm2(x)
        x = x + self.drop(self.w3(F.silu(self.w1(h)) * self.w2(h)))
        return x


class HybridModel(nn.Module):
    """Language model with interleaved TRN and Transformer blocks.

    Args:
        cfg: TRNConfig controlling model dimensions.
        trn_ratio: fraction of layers that are TRN (0.0 = pure Transformer,
                   1.0 = pure TRN, 0.5 = alternating). Applied greedily from
                   layer 0; the exact count is round(n_layers * trn_ratio).
                   If an int >= 1 is supplied, it is interpreted directly as
                   the number of TRN layers (convenient for small configs).

    Weight tying:
        lm_head.weight = embedding.weight when cfg.tie_weights is True.

    omega_base / res_scale parameters are excluded from weight decay; see
    configure_optimizer_param_groups for details.
    """

    def __init__(self, cfg: TRNConfig, trn_ratio: float = 0.5) -> None:
        super().__init__()
        self.cfg = cfg

        # Accept either a ratio in [0, 1] or a raw integer count of TRN layers.
        if isinstance(trn_ratio, int) or trn_ratio >= 1.0:
            n_trn = int(trn_ratio)
            assert 0 <= n_trn <= cfg.n_layers, (
                f"trn_ratio as int must be in [0, n_layers={cfg.n_layers}], got {n_trn}"
            )
            self.trn_ratio = n_trn / cfg.n_layers if cfg.n_layers > 0 else 0.0
        else:
            assert 0.0 <= trn_ratio <= 1.0, "trn_ratio must be in [0, 1]"
            self.trn_ratio = float(trn_ratio)
            n_trn = round(cfg.n_layers * trn_ratio)
        # Interleave: TRN layers spread as evenly as possible
        is_trn_layer = _interleave_flags(cfg.n_layers, n_trn)

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop_emb = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )
        self.blocks = nn.ModuleList([
            HybridBlock(cfg, is_trn=flag) for flag in is_trn_layer
        ])
        self.norm_out = build_rms_norm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=self.cfg.d_model ** -0.5)
        if not self.cfg.tie_weights:
            nn.init.normal_(self.lm_head.weight, std=self.cfg.d_model ** -0.5)

    def forward(
        self,
        input_ids: Tensor,              # (B, T)
        labels: Optional[Tensor] = None,  # (B, T)
    ) -> dict:
        x = self.drop_emb(self.embedding(input_ids))  # (B, T, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        result: dict = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
            )

        return result

    def configure_optimizer_param_groups(
        self,
        weight_decay: float = 0.1,
    ) -> list[dict]:
        """Split parameters into weight-decay and no-decay groups."""
        from .utils import configure_optimizer_param_groups
        return configure_optimizer_param_groups(self, weight_decay)

    def num_parameters(self, non_embedding: bool = True) -> int:
        """Count trainable parameters, optionally excluding the embedding table."""
        from .utils import num_parameters
        return num_parameters(self, non_embedding)

    def layer_type_summary(self) -> str:
        """Return a compact string showing layer type (T=TRN, A=Attention)."""
        parts = []
        for block in self.blocks:
            parts.append("T" if block.is_trn else "A")
        return "".join(parts)


def _interleave_flags(n_layers: int, n_trn: int) -> list[bool]:
    """Distribute n_trn TRN layers as evenly as possible across n_layers.

    Uses Bresenham-style even distribution so TRN and Attention blocks alternate
    rather than being grouped.

    Examples:
        n_layers=4, n_trn=2 -> [True, False, True, False]
        n_layers=4, n_trn=1 -> [True, False, False, False]
        n_layers=4, n_trn=3 -> [True, True, False, True]  (approx)
    """
    flags = [False] * n_layers
    if n_trn == 0:
        return flags
    if n_trn == n_layers:
        return [True] * n_layers

    # Bresenham: place n_trn True values spread over n_layers slots
    error = 0
    trn_placed = 0
    for i in range(n_layers):
        if trn_placed >= n_trn:
            break
        error += n_trn
        if error * 2 >= n_layers:
            flags[i] = True
            trn_placed += 1
            error -= n_layers

    return flags
