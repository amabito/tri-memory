"""TriMemoryEngine: KV window + TRN state + Retrieval index.

Three-tier hierarchical memory on a Transformer backbone:
  - KV window (W=64): short-term exact memory via windowed attention
  - TRN state: long-range compressed pattern/state memory via resonance
  - Retrieval index: long-range exact sparse memory via archived chunks

Token lifecycle:
  1. Token enters Transformer, hidden/KV updated
  2. KV window evicts oldest chunk (C=32 tokens)
  3. Evicted chunk -> TRN state update (always)
  4. SaliencyArchiver scores the chunk
  5. High-saliency chunks -> RetrievalIndex

Router (rule-based v1):
  - exact fact / old tool output -> retrieval
  - very recent dependency -> KV
  - trend / pattern / state -> TRN
  - mixed -> weighted combination

StateTokenAdapter:
  TRN state (K,) -> Linear -> m state tokens (m, d_model)
  Injected into the mixer alongside KV output and retrieved chunks.

Mixer (gated sum):
  out = g_kv * kv_out + g_trn * state_out + g_ret * retrieval_out

Limitations:
  - TRN is NOT a content-addressable memory
  - TRN is NOT a Transformer replacement
  - Retrieval is NOT always-on (gated by router)
  - This is a hierarchical memory architecture, not a monolithic one
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trn.baseline import CausalSelfAttention
from trn.config import TRNConfig
from trn.resonance import TemporalResonanceLayer
from trn.retrieval import RetrievalIndex
from trn.utils import build_rms_norm


# ---------------------------------------------------------------------------
# SaliencyArchiver: scores evicted chunks for selective archival
# ---------------------------------------------------------------------------

class SaliencyArchiver:
    """Rule-based saliency scorer for evicted chunks.

    Score components:
      a * number_score      -- contains digits / hex / IDs
      b * entity_score      -- uppercase sequences (proper nouns)
      c * tool_boundary     -- chunk at tool output boundary
      d * high_token_var    -- high variance in token IDs (diverse content)
      e * rare_token_score  -- contains rare tokens (high ID)

    Chunks above threshold are archived to RetrievalIndex.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        vocab_size: int = 256,
        w_number: float = 0.3,
        w_entity: float = 0.2,
        w_tool: float = 0.25,
        w_variance: float = 0.15,
        w_rare: float = 0.1,
    ) -> None:
        self.threshold = threshold
        self.vocab_size = vocab_size
        self.w_number = w_number
        self.w_entity = w_entity
        self.w_tool = w_tool
        self.w_variance = w_variance
        self.w_rare = w_rare

    def score(
        self,
        token_ids: list[int],
        is_tool_boundary: bool = False,
        loss_values: Optional[list[float]] = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute saliency score for a chunk of token IDs.

        Returns:
            (total_score, component_dict) for logging/explainability
        """
        n = len(token_ids) if token_ids else 1

        # Number/ID score: fraction of tokens in "high" range (likely digits/special)
        high_range_count = sum(1 for t in token_ids if t >= self.vocab_size * 3 // 4)
        number_score = high_range_count / n

        # Entity score: consecutive high-value tokens (proxy for named entities)
        max_consecutive_high = 0
        current_run = 0
        for t in token_ids:
            if t >= self.vocab_size // 2:
                current_run += 1
                max_consecutive_high = max(max_consecutive_high, current_run)
            else:
                current_run = 0
        entity_score = min(1.0, max_consecutive_high / 4.0)

        # Tool boundary
        tool_score = 1.0 if is_tool_boundary else 0.0

        # Token variance (diverse content = more informative)
        if n > 1:
            t_tensor = torch.tensor(token_ids, dtype=torch.float32)
            variance_score = min(1.0, t_tensor.std().item() / (self.vocab_size / 4))
        else:
            variance_score = 0.0

        # Rare token score
        rare_threshold = self.vocab_size * 7 // 8
        rare_count = sum(1 for t in token_ids if t >= rare_threshold)
        rare_score = min(1.0, rare_count / max(n * 0.1, 1))

        total = (
            self.w_number * number_score
            + self.w_entity * entity_score
            + self.w_tool * tool_score
            + self.w_variance * variance_score
            + self.w_rare * rare_score
        )

        components = {
            "number": number_score,
            "entity": entity_score,
            "tool": tool_score,
            "variance": variance_score,
            "rare": rare_score,
            "total": total,
        }
        return total, components

    def should_archive(self, score: float) -> bool:
        return score >= self.threshold


# ---------------------------------------------------------------------------
# RuleBasedMemoryRouter: decides memory source weights
# ---------------------------------------------------------------------------

@dataclass
class RouterDecision:
    """Explainable routing decision."""
    g_kv: float
    g_trn: float
    g_ret: float
    reason: str


class RuleBasedMemoryRouter:
    """Rule-based router for KV / TRN / Retrieval gate weights.

    Features used:
      - recent token density (are query tokens likely in KV window?)
      - entity density (proper nouns -> likely retrieval target)
      - numeric density (numbers/IDs -> likely retrieval target)
      - is_tool_query (tool output boundary -> retrieval)
      - position (early positions -> KV dominant)

    Returns (g_kv, g_trn, g_ret) gate weights summing to 1.0.
    """

    def __init__(
        self,
        kv_window_size: int = 64,
        retrieval_entity_threshold: float = 0.3,
        retrieval_numeric_threshold: float = 0.3,
    ) -> None:
        self.kv_window_size = kv_window_size
        self.retrieval_entity_threshold = retrieval_entity_threshold
        self.retrieval_numeric_threshold = retrieval_numeric_threshold

    def route(
        self,
        position: int,
        query_token_ids: list[int],
        vocab_size: int = 256,
        is_tool_query: bool = False,
        has_retrieval_chunks: bool = False,
    ) -> RouterDecision:
        """Decide gate weights based on current context features."""

        # Default: mostly KV
        g_kv = 0.6
        g_trn = 0.3
        g_ret = 0.1

        n = len(query_token_ids) if query_token_ids else 1

        # Position-based: early positions are entirely KV
        if position < self.kv_window_size:
            return RouterDecision(
                g_kv=0.8, g_trn=0.15, g_ret=0.05,
                reason="within_kv_window"
            )

        # Numeric density: high numeric content -> retrieval for exact lookup
        high_range = sum(1 for t in query_token_ids if t >= vocab_size * 3 // 4)
        numeric_density = high_range / n

        # Entity density
        entity_range = sum(1 for t in query_token_ids if t >= vocab_size // 2)
        entity_density = entity_range / n

        # Tool query -> heavy retrieval
        if is_tool_query:
            g_kv = 0.2
            g_trn = 0.1
            g_ret = 0.7
            reason = "tool_query"
        elif numeric_density > self.retrieval_numeric_threshold:
            g_kv = 0.3
            g_trn = 0.2
            g_ret = 0.5
            reason = "numeric_lookup"
        elif entity_density > self.retrieval_entity_threshold:
            g_kv = 0.3
            g_trn = 0.2
            g_ret = 0.5
            reason = "entity_lookup"
        else:
            # Far from window: TRN becomes more important
            distance_factor = min(1.0, (position - self.kv_window_size) / 500.0)
            g_kv = 0.5 - 0.2 * distance_factor
            g_trn = 0.3 + 0.2 * distance_factor
            g_ret = 0.2
            reason = f"distance_blend(d={position})"

        # If no retrieval chunks available, redistribute to KV+TRN
        if not has_retrieval_chunks and g_ret > 0.05:
            redistribute = g_ret - 0.05
            g_kv += redistribute * 0.4
            g_trn += redistribute * 0.6
            g_ret = 0.05
            reason += "+no_ret_chunks"

        # Normalize
        total = g_kv + g_trn + g_ret
        g_kv /= total
        g_trn /= total
        g_ret /= total

        return RouterDecision(g_kv=g_kv, g_trn=g_trn, g_ret=g_ret, reason=reason)


# ---------------------------------------------------------------------------
# StateTokenAdapter: TRN state -> pseudo memory tokens
# ---------------------------------------------------------------------------

class StateTokenAdapter(nn.Module):
    """Projects TRN resonance state into m attention-compatible state tokens.

    TRN state (real + imag per layer) -> concat -> Linear -> reshape to (m, d_model)
    """

    def __init__(self, n_layers: int, K: int, d_model: int, m: int = 8) -> None:
        super().__init__()
        self.m = m
        self.d_model = d_model
        # Input: concatenated real + imag states from all layers
        input_dim = n_layers * K * 2
        self.proj = nn.Linear(input_dim, m * d_model, bias=True)
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        states_r: list[Tensor],  # list of (B, K) fp32
        states_i: list[Tensor],  # list of (B, K) fp32
    ) -> Tensor:
        """Convert TRN states to state tokens.

        Returns:
            state_tokens: (B, m, d_model)
        """
        # Concat all layer states: (B, n_layers * K * 2)
        all_states = []
        for sr, si in zip(states_r, states_i):
            all_states.append(sr)
            all_states.append(si)
        concat = torch.cat(all_states, dim=-1)  # (B, n_layers * K * 2)

        projected = self.proj(concat)  # (B, m * d_model)
        B = concat.size(0)
        return projected.view(B, self.m, self.d_model)  # (B, m, d_model)


# ---------------------------------------------------------------------------
# TriMemoryBlock: one layer with KV + TRN + retrieval fusion
# ---------------------------------------------------------------------------

class _SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff_hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff_hidden, bias=False)
        self.up = nn.Linear(d_model, d_ff_hidden, bias=False)
        self.down = nn.Linear(d_ff_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TriMemoryBlock(nn.Module):
    """Single tri-memory layer: KV attention + TRN + retrieval gate."""

    def __init__(
        self,
        cfg: TRNConfig,
        window_size: int,
        enable_trn: bool = True,
        enable_retrieval: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.window_size = window_size
        self.enable_trn = enable_trn
        self.enable_retrieval = enable_retrieval
        n_heads = max(1, cfg.d_model // 64)

        self.norm1 = build_rms_norm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, n_heads)
        self.trn = TemporalResonanceLayer(
            d_model=cfg.d_model,
            K=cfg.n_oscillators,
            use_parallel_scan=cfg.use_parallel_scan,
            log_phase=cfg.log_phase,
            clamp_resonance=cfg.clamp_resonance,
            resonance_clamp_val=cfg.resonance_clamp_val,
            amplitude_max=cfg.amplitude_max,
            state_norm=cfg.state_norm,
            res_scale_init=cfg.res_scale_init,
            gate_bias_init=cfg.gate_bias_init,
            phase_mode=cfg.phase_mode,
        )

        # 3-way gate: g = softmax([g_kv, g_trn, g_ret])
        self.gate_proj = nn.Linear(cfg.d_model, 3, bias=True)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        # Equal start -- let training decide the balance
        nn.init.zeros_(self.gate_proj.bias)

        # Retrieval projection: project retrieved chunk mean to d_model
        self.ret_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.normal_(self.ret_proj.weight, std=0.01)

        self.norm2 = build_rms_norm(cfg.d_model)
        self.ffn = _SwiGLUFFN(cfg.d_model, cfg.d_ff_hidden)
        self.drop = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )

    def _make_window_mask(self, T: int, device: torch.device) -> Tensor:
        W = self.window_size
        mask = torch.full((T, T), float("-inf"), device=device)
        for i in range(T):
            start = max(0, i - W + 1)
            mask[i, start: i + 1] = 0.0
        return mask

    def forward(
        self,
        x: Tensor,
        retrieval_context: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with optional retrieval context.

        Args:
            x: (B, T, d_model)
            retrieval_context: (B, d_model) mean of retrieved chunks, or None
        """
        B, T, C = x.shape
        h = self.norm1(x)

        # Windowed attention
        mask = self._make_window_mask(T, x.device)
        q, k, v = self.attn.qkv(h).split(C, dim=-1)
        n_heads = self.attn.n_heads
        head_dim = self.attn.head_dim
        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.attn.proj(attn_out)

        # TRN (zero if disabled)
        if self.enable_trn:
            trn_out = self.trn(h)
        else:
            trn_out = torch.zeros_like(attn_out)

        # Retrieval context (broadcast to all positions)
        if self.enable_retrieval and retrieval_context is not None:
            ret_out = self.ret_proj(retrieval_context)  # (B, d_model)
            ret_out = ret_out.unsqueeze(1).expand(B, T, C)  # (B, T, C)
        else:
            ret_out = torch.zeros_like(attn_out)

        # 3-way gated sum with masking for disabled paths
        gate_logits = self.gate_proj(h)  # (B, T, 3)
        if not self.enable_trn:
            gate_logits = gate_logits.clone()
            gate_logits[:, :, 1] = -1e9
        if not self.enable_retrieval:
            gate_logits = gate_logits.clone()
            gate_logits[:, :, 2] = -1e9
        gates = torch.softmax(gate_logits, dim=-1)  # (B, T, 3)
        g_kv = gates[:, :, 0:1]    # (B, T, 1)
        g_trn = gates[:, :, 1:2]
        g_ret = gates[:, :, 2:3]

        mixed = g_kv * attn_out + g_trn * trn_out + g_ret * ret_out

        x = x + self.drop(mixed)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# TriMemoryEngine: full language model
# ---------------------------------------------------------------------------

class TriMemoryEngine(nn.Module):
    """Tri-Memory language model: KV window + TRN state + Retrieval index.

    Interface matches TRNModel / TransformerModel / DualMemoryEngine:
      forward(input_ids, labels=None) -> dict
      configure_optimizer_param_groups(weight_decay) -> list[dict]
      num_parameters(non_embedding=True) -> int

    Tri-Memory is a hierarchical memory architecture:
      - recent exact memory (KV)
      - long-range state memory (TRN)
      - sparse exact archive (Retrieval)

    TRN is NOT a Transformer replacement.
    TRN is NOT a content-addressable memory.
    """

    def __init__(
        self,
        cfg: TRNConfig,
        window_size: int = 64,
        chunk_size: int = 32,
        retrieval_top_k: int = 4,
        state_tokens_m: int = 8,
        max_retrieval_chunks: int = 256,
        saliency_threshold: float = 0.3,
        enable_trn: bool = True,
        enable_retrieval: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.retrieval_top_k = retrieval_top_k
        self.state_tokens_m = state_tokens_m
        self.enable_trn = enable_trn
        self.enable_retrieval = enable_retrieval

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop_emb = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )

        # Sinusoidal PE
        pe = self._build_sinusoidal_pe(cfg.max_seq_len, cfg.d_model)
        self.register_buffer("pe", pe)

        self.blocks = nn.ModuleList([
            TriMemoryBlock(cfg, window_size, enable_trn=enable_trn,
                           enable_retrieval=enable_retrieval)
            for _ in range(cfg.n_layers)
        ])
        self.norm_out = build_rms_norm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.embedding.weight

        # StateTokenAdapter
        self.state_adapter = StateTokenAdapter(
            n_layers=cfg.n_layers,
            K=cfg.n_oscillators,
            d_model=cfg.d_model,
            m=state_tokens_m,
        )

        # Non-nn components (not part of model parameters)
        self.retrieval_index = RetrievalIndex(
            vocab_size=cfg.vocab_size,
            max_chunks=max_retrieval_chunks,
            d_model=cfg.d_model,
        )
        self.saliency_archiver = SaliencyArchiver(
            threshold=saliency_threshold,
            vocab_size=cfg.vocab_size,
        )
        self.router = RuleBasedMemoryRouter(kv_window_size=window_size)

        # Eviction buffer
        self._eviction_buffer: list[int] = []
        self._global_step: int = 0
        self._router_log: list[RouterDecision] = []

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
        return pe

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=self.cfg.d_model ** -0.5)
        if not self.cfg.tie_weights:
            nn.init.normal_(self.lm_head.weight, std=self.cfg.d_model ** -0.5)

    def _process_eviction(
        self,
        evicted_tokens: list[int],
        hidden_mean: Tensor,
        step: int,
        is_tool_boundary: bool = False,
    ) -> Optional[dict]:
        """Process an evicted chunk: score saliency, optionally archive.

        Returns saliency components dict if archived, None otherwise.
        """
        score, components = self.saliency_archiver.score(
            evicted_tokens,
            is_tool_boundary=is_tool_boundary,
        )
        if self.saliency_archiver.should_archive(score):
            self.retrieval_index.add_chunk(
                token_ids=evicted_tokens,
                hidden_mean=hidden_mean,
                step=step,
                saliency=score,
            )
            return components
        return None

    def _get_retrieval_context(
        self,
        query_tokens: list[int],
        device: torch.device,
    ) -> Optional[Tensor]:
        """Query retrieval index and return mean hidden of top-k results."""
        if len(self.retrieval_index) == 0:
            return None

        results = self.retrieval_index.search(
            query_token_ids=query_tokens,
            top_k=self.retrieval_top_k,
        )
        if not results:
            return None

        # Mean pool retrieved chunk hidden means
        hiddens = torch.stack([r.hidden_mean for r in results])  # (k, d_model)
        return hiddens.mean(dim=0).to(device)  # (d_model,)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> dict:
        """Forward pass with tri-memory integration.

        Retrieval simulation during training:
          Early tokens (outside KV window) are pooled per-chunk and
          the chunk with highest token overlap to the query region
          is selected as retrieval context. This teaches the retrieval
          gate to use archived information for old fact recall.

        The retrieval context embedding participates in the forward graph
        (through ret_proj) so the gate learns when to rely on it.
        """
        B, T = input_ids.shape
        pe_len = min(T, self.pe.size(0))
        x = self.drop_emb(self.embedding(input_ids))
        x[:, :pe_len] = x[:, :pe_len] + self.pe[:pe_len]

        # Build per-sample retrieval context from early tokens
        retrieval_context = self._build_train_retrieval(input_ids, x)

        for block in self.blocks:
            x = block(x, retrieval_context=retrieval_context)

        x = self.norm_out(x)
        logits = self.lm_head(x)

        result: dict = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
            )
        return result

    def _build_train_retrieval(
        self,
        input_ids: Tensor,
        x: Tensor,
    ) -> Optional[Tensor]:
        """Build retrieval context for training via embedding cosine
        similarity with soft attention over early chunks.

        Strategy:
          1. Partition early tokens (before KV window) into chunks
          2. Compute chunk embeddings from x (grad-enabled)
          3. Compute query embedding from the query region
          4. Score chunks by cosine similarity with query
          5. Soft-attention weighted combination (all chunks, differentiable)

        The soft attention allows gradients to flow through all chunks,
        teaching the model which chunks are relevant for query answering.

        Returns (B, d_model) or None if sequence too short.
        """
        B, T = input_ids.shape
        if not self.enable_retrieval:
            return None
        if T <= self.window_size + self.chunk_size:
            return None

        archive_end = T - self.window_size

        # Chunk boundaries
        chunk_starts = list(range(0, archive_end, self.chunk_size))
        n_chunks = len(chunk_starts)
        if n_chunks == 0:
            return None

        # Build chunk embeddings: (B, n_chunks, d_model)
        chunk_embs = []
        for cs in chunk_starts:
            ce = min(cs + self.chunk_size, archive_end)
            chunk_embs.append(x[:, cs:ce, :].mean(dim=1))  # (B, d_model)
        chunk_matrix = torch.stack(chunk_embs, dim=1)  # (B, n_chunks, d_model)

        # Query embedding: mean of query region
        query_emb = x[:, -self.window_size:, :].mean(dim=1)  # (B, d_model)

        # Cosine similarity: (B, n_chunks)
        q_norm = F.normalize(query_emb, dim=-1).unsqueeze(1)  # (B, 1, d_model)
        c_norm = F.normalize(chunk_matrix, dim=-1)  # (B, n_chunks, d_model)
        scores = (c_norm * q_norm).sum(dim=-1)  # (B, n_chunks)

        # Soft attention with temperature (sharper = more selective)
        weights = torch.softmax(scores * 5.0, dim=-1)  # (B, n_chunks)

        # Weighted combination of chunk embeddings
        context = (weights.unsqueeze(-1) * chunk_matrix).sum(dim=1)  # (B, d_model)
        return context

    def forward_with_memory(
        self,
        input_ids: Tensor,
        states_r: list[Tensor],
        states_i: list[Tensor],
        position: int,
        labels: Optional[Tensor] = None,
    ) -> tuple[dict, list[Tensor], list[Tensor]]:
        """Forward pass with explicit TRN state management and retrieval.

        Used for streaming/incremental processing where we maintain
        eviction buffer and retrieval index across calls.

        Returns:
            (result_dict, updated_states_r, updated_states_i)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Get retrieval context from current input tokens
        query_tokens = input_ids[0].tolist() if B > 0 else []
        retrieval_context = self._get_retrieval_context(query_tokens, device)

        # Build hidden states
        pe_start = min(position, self.pe.size(0) - T)
        pe_end = min(pe_start + T, self.pe.size(0))
        x = self.drop_emb(self.embedding(input_ids))
        actual_pe_len = pe_end - pe_start
        if actual_pe_len > 0:
            x[:, :actual_pe_len] = x[:, :actual_pe_len] + self.pe[pe_start:pe_end]

        # Process through blocks
        for block in self.blocks:
            x = block(x, retrieval_context=retrieval_context)

        x = self.norm_out(x)
        logits = self.lm_head(x)

        result: dict = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
            )

        # Update TRN states via step_single for each token
        param_dtype = next(self.parameters()).dtype
        for t_idx in range(T):
            tok = input_ids[:, t_idx]
            x_tok = self.embedding(tok).to(param_dtype)
            x_tok = self.drop_emb(x_tok)
            pos = position + t_idx
            if pos < self.pe.size(0):
                x_tok = x_tok + self.pe[pos]

            for layer_idx, block in enumerate(self.blocks):
                x_normed = block.norm1(x_tok)
                _, states_r[layer_idx], states_i[layer_idx] = (
                    block.trn.step_single(
                        x_normed,
                        states_r[layer_idx],
                        states_i[layer_idx],
                        pos,
                    )
                )

            # Eviction buffer
            self._eviction_buffer.append(tok[0].item())
            if len(self._eviction_buffer) >= self.chunk_size:
                evicted = self._eviction_buffer[:self.chunk_size]
                self._eviction_buffer = self._eviction_buffer[self.chunk_size:]
                # Compute hidden mean for evicted chunk
                with torch.no_grad():
                    evicted_tensor = torch.tensor(evicted, device=device).unsqueeze(0)
                    hidden = self.embedding(evicted_tensor).mean(dim=1).squeeze(0)
                self._process_eviction(evicted, hidden, self._global_step)
                self._global_step += 1

        return result, states_r, states_i

    def reset_memory(self) -> None:
        """Reset all non-parameter memory state."""
        self.retrieval_index.reset()
        self._eviction_buffer.clear()
        self._global_step = 0
        self._router_log.clear()

    def configure_optimizer_param_groups(
        self,
        weight_decay: float = 0.1,
    ) -> list[dict]:
        decay: set[str] = set()
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
            {"params": [params[n] for n in sorted(decay)], "weight_decay": weight_decay},
            {"params": [params[n] for n in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def num_parameters(self, non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.embedding.weight.numel()
        return total

    @property
    def state_memory_bytes(self) -> int:
        """TRN state memory: constant."""
        return self.cfg.n_layers * self.cfg.n_oscillators * 2 * 4

    def kv_window_bytes(self, dtype_bytes: int = 2) -> int:
        n_heads = self.blocks[0].attn.n_heads
        head_dim = self.blocks[0].attn.head_dim
        return self.cfg.n_layers * n_heads * self.window_size * head_dim * 2 * dtype_bytes

    def retrieval_memory_bytes(self) -> int:
        return self.retrieval_index.memory_bytes()

    def total_memory_bytes(self, dtype_bytes: int = 2) -> int:
        return (
            self.state_memory_bytes
            + self.kv_window_bytes(dtype_bytes)
            + self.retrieval_memory_bytes()
        )

    def memory_summary(self) -> dict:
        return {
            "trn_state_bytes": self.state_memory_bytes,
            "kv_window_bytes": self.kv_window_bytes(),
            "retrieval_bytes": self.retrieval_memory_bytes(),
            "total_bytes": self.total_memory_bytes(),
            "retrieval_chunks": len(self.retrieval_index),
            "eviction_buffer_len": len(self._eviction_buffer),
        }
