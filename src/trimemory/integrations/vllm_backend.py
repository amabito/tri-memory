"""DualMemoryEngine: windowed KV attention + TRN resonance per layer.

Each DualMemoryBlock applies both:
  - CausalSelfAttention with a fixed-size windowed KV cache (FIFO eviction)
  - TemporalResonanceLayer (constant-size state, updated on ALL tokens)
  - Mixer gate: g = sigmoid(W_gate * x), out = g * attn_out + (1-g) * trn_out

Training uses a banded causal mask (only last W tokens visible per query).
Generation uses per-layer WindowedKVCache + TRN step_single for O(1) decode.

Interface matches TRNModel / TransformerModel:
  forward(input_ids, labels=None) -> dict with logits + optional loss
  generate(prompt_ids, max_new_tokens, temperature=1.0) -> Tensor
  configure_optimizer_param_groups(weight_decay) -> list[dict]
  num_parameters(non_embedding=True) -> int
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trimemory.baseline import CausalSelfAttention
from trimemory.config import TRNConfig
from trimemory.resonance import TemporalResonanceLayer
from trimemory.utils import build_rms_norm


# ---------------------------------------------------------------------------
# KV window cache (FIFO eviction, used during generate)
# ---------------------------------------------------------------------------

@dataclass
class WindowedKVCache:
    """Fixed-size KV cache with FIFO eviction.

    Stores the last `window_size` keys and values for one attention layer.
    """
    k_cache: Tensor  # (B, n_heads, T_stored, head_dim)
    v_cache: Tensor  # (B, n_heads, T_stored, head_dim)
    window_size: int

    def append(self, k_new: Tensor, v_new: Tensor) -> "WindowedKVCache":
        """Append new K/V slice and evict oldest tokens if over window_size.

        Args:
            k_new: (B, n_heads, n_new, head_dim)
            v_new: (B, n_heads, n_new, head_dim)

        Returns:
            Updated cache (new instance).
        """
        k = torch.cat([self.k_cache, k_new], dim=2)
        v = torch.cat([self.v_cache, v_new], dim=2)
        if k.size(2) > self.window_size:
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]
        return WindowedKVCache(k_cache=k, v_cache=v, window_size=self.window_size)


# ---------------------------------------------------------------------------
# SwiGLU FFN (LLaMA-style, matches TRNBlock's d_ff_hidden)
# ---------------------------------------------------------------------------

class _SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff_hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff_hidden, bias=False)
        self.up   = nn.Linear(d_model, d_ff_hidden, bias=False)
        self.down = nn.Linear(d_ff_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# DualMemoryBlock: one layer with KV attention + TRN + gate
# ---------------------------------------------------------------------------

class DualMemoryBlock(nn.Module):
    """Single dual-memory layer.

    pre-norm attn/TRN in parallel -> mixer gate -> pre-norm FFN
    """

    def __init__(self, cfg: TRNConfig, window_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.window_size = window_size
        n_heads = max(1, cfg.d_model // 64)

        self.norm1 = build_rms_norm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg.d_model, n_heads)
        self.trn   = TemporalResonanceLayer(
            d_model             = cfg.d_model,
            K                   = cfg.n_oscillators,
            use_parallel_scan   = cfg.use_parallel_scan,
            clamp_resonance     = cfg.clamp_resonance,
            resonance_clamp_val = cfg.resonance_clamp_val,
            amplitude_max       = cfg.amplitude_max,
            state_norm          = cfg.state_norm,
            res_scale_init      = cfg.res_scale_init,
            gate_bias_init      = cfg.gate_bias_init,
            phase_mode          = cfg.phase_mode,
        )

        # Mixer gate: g = sigmoid(W_gate * x + b), out = g*attn + (1-g)*trn
        # bias=0.0 so sigmoid(0)=0.5 (equal initial weight)
        self.gate_proj = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self.norm2 = build_rms_norm(cfg.d_model)
        self.ffn   = _SwiGLUFFN(cfg.d_model, cfg.d_ff_hidden)
        self.drop  = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )

    _window_mask_cache: dict[tuple[int, str], Tensor] = {}

    @classmethod
    def _make_window_mask(cls, T: int, W: int, device: torch.device) -> Tensor:
        cache_key = (T, W, str(device))
        cached = cls._window_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        row = torch.arange(T, device=device).unsqueeze(1)
        col = torch.arange(T, device=device).unsqueeze(0)
        mask = torch.where(
            (col <= row) & (col >= row - W + 1),
            torch.tensor(0.0, device=device),
            torch.tensor(float("-inf"), device=device),
        )
        cls._window_mask_cache[cache_key] = mask
        return mask

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        h = self.norm1(x)

        # Windowed attention with banded mask
        mask = self._make_window_mask(T, self.window_size, x.device)  # (T, T)
        q, k, v = self.attn.qkv(h).split(C, dim=-1)
        n_heads  = self.attn.n_heads
        head_dim = self.attn.head_dim
        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.attn.proj(attn_out)

        # TRN parallel scan (full sequence)
        trn_out = self.trn(h)

        # Mixer gate per position: (B, T, 1)
        g = torch.sigmoid(self.gate_proj(h))
        mixed = g * attn_out + (1.0 - g) * trn_out

        x = x + self.drop(mixed)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# DualMemoryEngine: full language model
# ---------------------------------------------------------------------------

class DualMemoryEngine(nn.Module):
    """Dual-memory language model: windowed KV attention + TRN per layer.

    Interface matches TRNModel / TransformerModel:
      forward(input_ids, labels=None) -> dict
      generate(prompt_ids, max_new_tokens, temperature=1.0) -> Tensor
      configure_optimizer_param_groups(weight_decay) -> list[dict]
      num_parameters(non_embedding=True) -> int
    """

    def __init__(self, cfg: TRNConfig, window_size: int = 256) -> None:
        super().__init__()
        self.cfg = cfg
        self.window_size = window_size

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop_emb  = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )
        # Sinusoidal PE (same as TransformerModel)
        pe = self._build_sinusoidal_pe(cfg.max_seq_len, cfg.d_model)
        self.register_buffer("pe", pe)

        self.blocks   = nn.ModuleList([
            DualMemoryBlock(cfg, window_size) for _ in range(cfg.n_layers)
        ])
        self.norm_out = build_rms_norm(cfg.d_model)
        self.lm_head  = nn.Linear(cfg.vocab_size, cfg.d_model, bias=False)

        if cfg.tie_weights:
            # lm_head projects d_model -> vocab_size; weight is (vocab_size, d_model)
            # Redefine lm_head with correct dimensions and tie
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.embedding.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()

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

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=self.cfg.d_model ** -0.5)
        if not self.cfg.tie_weights:
            nn.init.normal_(self.lm_head.weight, std=self.cfg.d_model ** -0.5)

    def forward(
        self,
        input_ids: Tensor,                  # (B, T)
        labels:    Optional[Tensor] = None, # (B, T)
    ) -> dict:
        B, T = input_ids.shape
        x = self.drop_emb(self.embedding(input_ids)) + self.pe[:T]

        for block in self.blocks:
            x = block(x)

        x      = self.norm_out(x)
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

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids:     Tensor,  # (B, prompt_len)
        max_new_tokens: int   = 128,
        temperature:    float = 1.0,
        top_k:          int   = 50,
    ) -> Tensor:
        """Streaming autoregressive decode with KV window + TRN step_single.

        Prefill: build windowed KV cache + TRN states from prompt.
        Decode: per-token step using O(W) KV window and O(1) TRN state.

        Returns:
            generated tokens beyond the prompt, shape (B, max_new_tokens).
        """
        model_dtype = next(self.parameters()).dtype
        device      = prompt_ids.device
        B           = prompt_ids.size(0)
        K           = self.cfg.n_oscillators
        n_heads     = self.blocks[0].attn.n_heads
        head_dim    = self.blocks[0].attn.head_dim
        prompt_len  = prompt_ids.size(1)

        # --- Prefill: compute KV for last window_size tokens from prompt ---
        # Run forward on last min(prompt_len, max_seq_len) tokens to get initial KV.
        # We extract K/V by running attention qkv on the normed hidden states.
        C = self.cfg.d_model

        kv_caches: list[Optional[WindowedKVCache]] = [None] * self.cfg.n_layers

        # Prefill via forward pass with hooks on the internal normed hidden states.
        # We hook into each DualMemoryBlock.norm1 output to capture the normed x,
        # then compute K/V ourselves.
        prefill_input = prompt_ids
        if prompt_len > self.cfg.max_seq_len:
            # Use only last max_seq_len tokens for prefill forward
            prefill_input = prompt_ids[:, -self.cfg.max_seq_len:]

        # Collect normed hidden states per layer via hooks on norm1
        normed_states: list[Optional[Tensor]] = [None] * self.cfg.n_layers

        hooks = []
        for layer_idx, block in enumerate(self.blocks):
            def make_norm1_hook(idx: int):
                def hook(module, args, output):
                    # output is the normed x (B, T, C)
                    normed_states[idx] = output.detach()
                return hook
            hooks.append(block.norm1.register_forward_hook(make_norm1_hook(layer_idx)))

        with torch.inference_mode():
            self(prefill_input)

        for h in hooks:
            h.remove()

        # Build KV caches from captured normed states
        for layer_idx, block in enumerate(self.blocks):
            ns = normed_states[layer_idx]
            if ns is None:
                continue
            bt, tt, cc = ns.shape
            with torch.inference_mode():
                qkv = block.attn.qkv(ns)
            _, k_p, v_p = qkv.split(cc, dim=-1)
            k_p = k_p.view(bt, tt, n_heads, head_dim).transpose(1, 2)
            v_p = v_p.view(bt, tt, n_heads, head_dim).transpose(1, 2)
            if tt > self.window_size:
                k_p = k_p[:, :, -self.window_size:, :]
                v_p = v_p[:, :, -self.window_size:, :]
            kv_caches[layer_idx] = WindowedKVCache(
                k_cache=k_p, v_cache=v_p, window_size=self.window_size
            )

        # --- Build TRN states from prompt via step_single ---
        states_r = [
            torch.zeros(B, K, device=device, dtype=torch.float32)
            for _ in range(self.cfg.n_layers)
        ]
        states_i = [
            torch.zeros(B, K, device=device, dtype=torch.float32)
            for _ in range(self.cfg.n_layers)
        ]

        # Process prompt tokens through TRN step_single to build states
        for pos in range(prompt_len):
            tok = prompt_ids[:, pos]  # (B,)
            x   = self.embedding(tok).to(model_dtype)  # (B, d_model)
            x   = self.drop_emb(x) + self.pe[pos]

            for layer_idx, block in enumerate(self.blocks):
                x_normed = block.norm1(x)
                trn_out, states_r[layer_idx], states_i[layer_idx] = (
                    block.trn.step_single(
                        x_normed,
                        states_r[layer_idx],
                        states_i[layer_idx],
                        pos,
                    )
                )
                # We only update state, we don't need the output for prefill
                # (the full forward already ran above; we just re-drive TRN state)

        # --- Decode loop ---
        generated = prompt_ids.clone()
        # Compute gate bias once (from last prefill step, approximate)
        # Gate is applied per-token in decode below

        for step_idx in range(max_new_tokens):
            pos     = prompt_len + step_idx
            last_tok = generated[:, -1]  # (B,)
            x        = self.embedding(last_tok).to(model_dtype)  # (B, d_model)
            x        = self.drop_emb(x) + self.pe[min(pos, self.cfg.max_seq_len - 1)]

            for layer_idx, block in enumerate(self.blocks):
                h = block.norm1(x)

                # --- Windowed KV attention (single query token) ---
                cache = kv_caches[layer_idx]
                C = self.cfg.d_model
                qkv = block.attn.qkv(h.unsqueeze(1))  # (B, 1, 3*C)
                q, k_new, v_new = qkv.split(C, dim=-1)
                q     = q.view(B, 1, n_heads, head_dim).transpose(1, 2)
                k_new = k_new.view(B, 1, n_heads, head_dim).transpose(1, 2)
                v_new = v_new.view(B, 1, n_heads, head_dim).transpose(1, 2)

                # Append to cache and evict
                new_cache = cache.append(k_new, v_new)
                kv_caches[layer_idx] = new_cache

                k_full = new_cache.k_cache  # (B, n_heads, T_w, head_dim)
                v_full = new_cache.v_cache

                attn_out = F.scaled_dot_product_attention(
                    q, k_full, v_full, is_causal=False
                )  # (B, n_heads, 1, head_dim)
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, C)
                attn_out = block.attn.proj(attn_out)  # (B, C)

                # --- TRN step_single ---
                trn_out, states_r[layer_idx], states_i[layer_idx] = (
                    block.trn.step_single(
                        h,
                        states_r[layer_idx],
                        states_i[layer_idx],
                        pos,
                    )
                )

                # --- Mixer gate ---
                g = torch.sigmoid(block.gate_proj(h))  # (B, 1)
                mixed = g * attn_out + (1.0 - g) * trn_out  # (B, C)

                x = x + mixed
                x = x + block.ffn(block.norm2(x))

            logit = self.lm_head(self.norm_out(x))  # (B, vocab_size)

            if temperature != 1.0:
                logit = logit / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logit, min(top_k, logit.size(-1)))
                logit[logit < top_vals[:, -1:]] = float("-inf")

            probs    = torch.softmax(logit, dim=-1)
            next_tok = torch.multinomial(probs, 1)  # (B, 1)
            generated = torch.cat([generated, next_tok], dim=1)

        return generated[:, prompt_len:]

    def configure_optimizer_param_groups(
        self,
        weight_decay: float = 0.1,
    ) -> list[dict]:
        """Split parameters into weight-decay and no-decay groups."""
        decay:    set[str] = set()
        no_decay: set[str] = set()

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

        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        return [
            {"params": [params[n] for n in sorted(decay)],    "weight_decay": weight_decay},
            {"params": [params[n] for n in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def num_parameters(self, non_embedding: bool = True) -> int:
        """Count trainable parameters, optionally excluding the embedding table."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.embedding.weight.numel()
        return total

    @property
    def state_memory_bytes(self) -> int:
        """Constant TRN state memory: n_layers * K * 2 * 4 bytes (fp32 r+i)."""
        return self.cfg.n_layers * self.cfg.n_oscillators * 2 * 4

    def kv_window_bytes(self, dtype_bytes: int = 2) -> int:
        """KV window memory: n_layers * n_heads * W * head_dim * 2 * dtype_bytes."""
        n_heads  = self.blocks[0].attn.n_heads
        head_dim = self.blocks[0].attn.head_dim
        return self.cfg.n_layers * n_heads * self.window_size * head_dim * 2 * dtype_bytes
