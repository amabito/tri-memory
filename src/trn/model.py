from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .block import TRNBlock
from .config import TRNConfig
from .utils import build_rms_norm


class TRNModel(nn.Module):
    """Causal Temporal Resonance Network language model.

    Architecture:
        token_embedding -> N x TRNBlock -> RMSNorm -> lm_head

    Weight tying:
        lm_head.weight = embedding.weight when cfg.tie_weights is True.

    Mixed-precision:
        Resonance state is always kept in fp32 inside TemporalResonanceLayer;
        the rest of the model participates in bf16/fp16 AMP normally.

    omega_base parameters are excluded from weight decay by convention —
    callers should pass them in a separate no-decay parameter group:
        no_decay = {n for n, _ in model.named_parameters() if "omega_base" in n}
    """

    def __init__(self, cfg: TRNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop_emb  = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )
        self.blocks   = nn.ModuleList([TRNBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = build_rms_norm(cfg.d_model)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=self.cfg.d_model ** -0.5)
        if not self.cfg.tie_weights:
            nn.init.normal_(self.lm_head.weight, std=self.cfg.d_model ** -0.5)

    def forward(
        self,
        input_ids: Tensor,                # (B, n)
        labels:    Optional[Tensor] = None,  # (B, n) — next-token targets
    ) -> dict:
        x = self.drop_emb(self.embedding(input_ids))  # (B, n, d_model)

        for block in self.blocks:
            x = block(x)

        x      = self.norm_out(x)
        logits = self.lm_head(x)          # (B, n, vocab_size)

        result: dict = {"logits": logits}

        if labels is not None:
            # Causal shift: predict token at position t using tokens 0..t-1.
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = nn.functional.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
            )

        return result

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids:     Tensor,   # (B, prompt_len)
        max_new_tokens: int   = 128,
        temperature:    float = 1.0,
        top_k:          int   = 50,
    ) -> Tensor:
        """Autoregressive generation using step_single (O(1) per step memory).

        Returns:
            generated tokens beyond the prompt, shape (B, max_new_tokens).
        """
        B           = prompt_ids.size(0)
        K           = self.cfg.n_oscillators
        prompt_len  = prompt_ids.size(1)
        device      = prompt_ids.device
        param_dtype = next(self.parameters()).dtype

        # Resonance states — always fp32 regardless of model dtype.
        states_r = [
            torch.zeros(B, K, device=device, dtype=torch.float32)
            for _ in range(self.cfg.n_layers)
        ]
        states_i = [
            torch.zeros(B, K, device=device, dtype=torch.float32)
            for _ in range(self.cfg.n_layers)
        ]

        generated = prompt_ids.clone()

        for pos in range(prompt_len, prompt_len + max_new_tokens):
            token = generated[:, -1]                    # (B,)
            x     = self.embedding(token).to(param_dtype)  # (B, d_model)
            x     = self.drop_emb(x)

            for layer_idx, block in enumerate(self.blocks):
                x_normed = block.norm1(x)
                res_out, states_r[layer_idx], states_i[layer_idx] = (
                    block.resonance.step_single(
                        x_normed,
                        states_r[layer_idx],
                        states_i[layer_idx],
                        pos,
                    )
                )
                x = x + res_out
                x = x + block.ffn(block.norm2(x))

            logit = self.lm_head(self.norm_out(x))      # (B, vocab_size)

            if temperature != 1.0:
                logit = logit / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logit, min(top_k, logit.size(-1)))
                logit[logit < top_vals[:, -1:]] = float("-inf")

            probs    = torch.softmax(logit, dim=-1)
            next_tok = torch.multinomial(probs, 1)       # (B, 1)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated[:, prompt_len:]

    def num_parameters(self, non_embedding: bool = True) -> int:
        """Count trainable parameters, optionally excluding the embedding table."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.embedding.weight.numel()
        return total

    def configure_optimizer_param_groups(
        self,
        weight_decay: float = 0.1,
    ) -> list[dict]:
        """Split parameters into decay / no-decay groups.

        omega_base and all bias / norm parameters are excluded from weight decay.
        """
        decay     = set()
        no_decay  = set()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (
                "omega_base" in name
                or "res_scale" in name
                or name.endswith(".bias")
                or "norm" in name.lower()
                or "embedding" in name
            ):
                no_decay.add(name)
            else:
                decay.add(name)

        params_by_name = {n: p for n, p in self.named_parameters()}
        return [
            {"params": [params_by_name[n] for n in sorted(decay)],
             "weight_decay": weight_decay},
            {"params": [params_by_name[n] for n in sorted(no_decay)],
             "weight_decay": 0.0},
        ]
