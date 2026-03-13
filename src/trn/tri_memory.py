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

from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trn.baseline import CausalSelfAttention
from trn.config import TRNConfig
from trn.resonance import TemporalResonanceLayer
from trn.retrieval import RetrievalIndex
from trn.router import RuleBasedMemoryRouter, RouterDecision
from trn.saliency import SaliencyArchiver
from trn.utils import build_rms_norm, build_sinusoidal_pe


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
            clamp_resonance=cfg.clamp_resonance,
            resonance_clamp_val=cfg.resonance_clamp_val,
            amplitude_max=cfg.amplitude_max,
            state_norm=cfg.state_norm,
            res_scale_init=cfg.res_scale_init,
            gate_bias_init=cfg.gate_bias_init,
            phase_mode=cfg.phase_mode,
            scan_chunk_size=cfg.scan_chunk_size,
        )
        self.trn_out_norm = build_rms_norm(cfg.d_model)

        # 3-way gate: g = softmax([g_kv, g_trn, g_ret])
        self.gate_proj = nn.Linear(cfg.d_model, 3, bias=True)
        nn.init.normal_(self.gate_proj.weight, std=0.01)
        # Gate bias: zeros (baseline default)
        # Fix B gate bias=[0,0.2,0] reverted -- ineffective in Fix C post-world.
        # Bimodality is addressed by Fix D (LR warmup 300 steps).
        nn.init.zeros_(self.gate_proj.bias)

        # Retrieval projection: project retrieved chunk mean to d_model
        self.ret_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.normal_(self.ret_proj.weight, std=0.01)

        self.norm2 = build_rms_norm(cfg.d_model)
        self.ffn = _SwiGLUFFN(cfg.d_model, cfg.d_ff_hidden)
        self.drop = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )
        self._window_mask_cache: dict[tuple[int, int, str], Tensor] = {}

    _MASK_CACHE_MAX = 16

    def _make_window_mask(self, T: int, W: int, device: torch.device) -> Tensor:
        cache_key = (T, W, str(device))
        cached = self._window_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        row = torch.arange(T, device=device).unsqueeze(1)
        col = torch.arange(T, device=device).unsqueeze(0)
        mask = torch.where(
            (col <= row) & (col >= row - W + 1),
            torch.tensor(0.0, device=device),
            torch.tensor(float("-inf"), device=device),
        )
        if len(self._window_mask_cache) >= self._MASK_CACHE_MAX:
            # Evict oldest entry
            oldest_key = next(iter(self._window_mask_cache))
            del self._window_mask_cache[oldest_key]
        self._window_mask_cache[cache_key] = mask
        return mask

    @staticmethod
    def _make_streaming_mask(
        T_q: int, T_kv: int, W: int, offset: int, device: torch.device,
    ) -> Tensor:
        """Build causal sliding-window mask for streaming attention.

        query positions: [offset, offset+T_q)
        key positions:   [offset+T_q-T_kv, offset+T_q)  (past + current)

        mask[i, j]: query i attends to key j iff:
          - causal: key_pos <= query_pos
          - window:  key_pos >= query_pos - W + 1
        """
        q_pos = torch.arange(T_q, device=device) + offset          # absolute query positions
        k_pos = torch.arange(T_kv, device=device) + (offset + T_q - T_kv)  # absolute key positions
        q_pos = q_pos.unsqueeze(1)  # (T_q, 1)
        k_pos = k_pos.unsqueeze(0)  # (1, T_kv)
        mask = torch.where(
            (k_pos <= q_pos) & (k_pos >= q_pos - W + 1),
            torch.tensor(0.0, device=device),
            torch.tensor(float("-inf"), device=device),
        )
        return mask  # (T_q, T_kv)

    def forward(
        self,
        x: Tensor,
        retrieval_context: Optional[Tensor] = None,
        retrieval_tokens: Optional[Tensor] = None,
        past_k: Optional[Tensor] = None,
        past_v: Optional[Tensor] = None,
        position_offset: int = 0,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Forward pass with optional retrieval context and KV cache.

        Args:
            x: (B, T, d_model)
            retrieval_context: (B, d_model) mean of retrieved chunks, or None
            retrieval_tokens: (B, R, d_model) prefix tokens for prefix mode, or None
            past_k: (B, n_heads, past_len, head_dim) cached keys, or None
            past_v: (B, n_heads, past_len, head_dim) cached values, or None
            position_offset: absolute position of first token in x

        Returns:
            (output, new_past_k, new_past_v) where new_past_* are truncated
            to window_size. When past_k/past_v are None (batch mode), returns
            (output, None, None) for backward compatibility.
        """
        B, T, C = x.shape
        h = self.norm1(x)
        use_cache = past_k is not None

        # Compute Q, K, V for current chunk
        q, k, v = self.attn.qkv(h).split(C, dim=-1)
        n_heads = self.attn.n_heads
        head_dim = self.attn.head_dim
        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

        # Prefix mode: compute K/V from retrieval tokens and prepend
        use_prefix = (
            self.enable_retrieval
            and retrieval_tokens is not None
            and retrieval_tokens.shape[1] > 0
        )
        if use_prefix:
            R = retrieval_tokens.shape[1]
            h_ret = self.norm1(retrieval_tokens)
            qkv_ret = self.attn.qkv(h_ret)
            _, k_ret, v_ret = qkv_ret.split(C, dim=-1)
            k_ret = k_ret.view(B, R, n_heads, head_dim).transpose(1, 2)
            v_ret = v_ret.view(B, R, n_heads, head_dim).transpose(1, 2)
        else:
            R = 0

        if use_cache:
            # Concat past KV with current KV
            k_full = torch.cat([past_k, k], dim=2)  # (B, n_heads, past+T, head_dim)
            v_full = torch.cat([past_v, v], dim=2)
            T_kv = k_full.shape[2]
            mask = self._make_streaming_mask(T, T_kv, self.window_size, position_offset, x.device)
            attn_out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)

            # Truncate cache to window_size
            keep = min(self.window_size, T_kv)
            new_past_k = k_full[:, :, -keep:, :].detach()
            new_past_v = v_full[:, :, -keep:, :].detach()
        else:
            # Batch mode: no cache, standard window mask
            if use_prefix:
                # Prepend retrieval K/V to sequence K/V
                k_full = torch.cat([k_ret, k], dim=2)  # (B, n_heads, R+T, head_dim)
                v_full = torch.cat([v_ret, v], dim=2)
                # Mask: prefix columns = 0 (always attend), sequence columns = window mask
                window_mask = self._make_window_mask(T, self.window_size, x.device)  # (T, T)
                prefix_mask = torch.zeros(T, R, device=x.device)  # all Q attend to all prefix
                full_mask = torch.cat([prefix_mask, window_mask], dim=1)  # (T, R+T)
                attn_out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=full_mask)
            else:
                mask = self._make_window_mask(T, self.window_size, x.device)
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            new_past_k = None
            new_past_v = None

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.attn.proj(attn_out)

        # TRN (zero if disabled)
        if self.enable_trn:
            trn_out = self.trn_out_norm(self.trn(h))
        else:
            trn_out = torch.zeros_like(attn_out)

        # Retrieval context (broadcast to all positions) -- pooled mode
        # In prefix mode, retrieval flows through attention, so ret_out = 0
        if self.enable_retrieval and retrieval_context is not None and not use_prefix:
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

        # Store gate values for telemetry (detached, no memory leak)
        self._last_gates = gates.detach().mean(dim=(0, 1))  # (3,)

        mixed = g_kv * attn_out + g_trn * trn_out + g_ret * ret_out

        x = x + self.drop(mixed)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, new_past_k, new_past_v


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
        search_mode: str = "hidden",
        search_w_hidden: float = 0.7,
        search_w_bag: float = 0.3,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.retrieval_top_k = retrieval_top_k
        self.state_tokens_m = state_tokens_m
        self.enable_trn = enable_trn
        self.enable_retrieval = enable_retrieval
        self.search_mode = search_mode
        self.search_w_hidden = search_w_hidden
        self.search_w_bag = search_w_bag

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop_emb = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        )

        # Sinusoidal PE
        pe = build_sinusoidal_pe(cfg.max_seq_len, cfg.d_model)
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

        # Retrieval query projection (marker mode)
        self.query_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.normal_(self.query_proj.weight, std=0.01)

        # Retrieval copy head: direct token prediction from retrieval context
        self.ret_copy_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        nn.init.normal_(self.ret_copy_head.weight, std=0.02)

        # Copy-mix alpha: learnable scalar for mixing copy logits into main logits
        self.copy_mix_alpha = nn.Parameter(torch.tensor(2.0))

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
        self._eviction_buffer: deque[int] = deque(maxlen=chunk_size * 2)
        self._global_step: int = 0
        _ROUTER_LOG_MAX = 1024
        self._router_log: deque[RouterDecision] = deque(maxlen=_ROUTER_LOG_MAX)

        self._init_weights()

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

        # Compute query_hidden from embedding mean for hidden/hybrid modes
        query_hidden = None
        if self.search_mode in ("hidden", "hybrid"):
            with torch.no_grad():
                q_ids = torch.tensor(query_tokens, device=device).unsqueeze(0)
                query_hidden = self.embedding(q_ids).mean(dim=1).squeeze(0)

        results = self.retrieval_index.search(
            query_token_ids=query_tokens,
            top_k=self.retrieval_top_k,
            query_hidden=query_hidden,
            mode=self.search_mode,
            w_hidden=self.search_w_hidden,
            w_bag=self.search_w_bag,
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
        retrieval_query_mode: str = "mean",
        retrieval_query_pos: int | None = None,
        retrieval_temperature: float = 5.0,
        retrieval_decoder_mode: str = "pooled",
        copy_mix_positions: Optional[list[tuple[int, int]]] = None,
    ) -> dict:
        """Forward pass with tri-memory integration.

        Args:
            retrieval_query_mode: "mean" (KV window mean) or "marker" (query_proj at pos)
            retrieval_query_pos: position of query marker token for "marker" mode
            retrieval_temperature: softmax temperature for retrieval attention
            retrieval_decoder_mode: "pooled", "prefix", or "copy_mix"
            copy_mix_positions: list of (seq_pos, chunk_token_idx) pairs for copy_mix mode.
                At each seq_pos in the logits, add alpha * copy_logits[:, chunk_token_idx, :].
        """
        B, T = input_ids.shape
        pe_len = min(T, self.pe.size(0))
        x = self.drop_emb(self.embedding(input_ids))
        x[:, :pe_len] = x[:, :pe_len] + self.pe[:pe_len]

        # Build per-sample retrieval context from early tokens
        retrieval_context, ret_weights, ret_chunk_starts, context_tokens = (
            self._build_train_retrieval(
                input_ids, x,
                retrieval_temperature=retrieval_temperature,
                query_mode=retrieval_query_mode,
                query_pos=retrieval_query_pos,
            )
        )

        # Copy head: token-level prediction from retrieval context
        # context_tokens: (B, chunk_size, d_model) -- per-token hidden states
        if context_tokens is not None:
            copy_logits = self.ret_copy_head(context_tokens)  # (B, chunk_size, vocab_size)
        else:
            copy_logits = None

        # Telemetry (single dict, cleared each forward)
        archive_end = T - self.window_size
        self._telemetry = {
            "ret_weights": ret_weights,
            "ret_chunk_starts": ret_chunk_starts,
            "retrieval_used": retrieval_context is not None,
            "n_chunks": max(0, archive_end // self.chunk_size) if T > self.window_size + self.chunk_size else 0,
        }

        # Select decoder mode
        use_prefix = retrieval_decoder_mode == "prefix"
        ret_ctx = None if use_prefix else retrieval_context
        ret_tok = context_tokens if use_prefix else None

        for block in self.blocks:
            x, _, _ = block(
                x,
                retrieval_context=ret_ctx,
                retrieval_tokens=ret_tok,
            )

        x = self.norm_out(x)
        logits = self.lm_head(x)

        # copy_mix: add copy_logits at specified sequence positions
        if (
            retrieval_decoder_mode == "copy_mix"
            and copy_logits is not None
            and copy_mix_positions is not None
        ):
            for seq_pos, chunk_tok_idx in copy_mix_positions:
                if seq_pos < logits.shape[1] and chunk_tok_idx < copy_logits.shape[1]:
                    logits[:, seq_pos, :] = (
                        logits[:, seq_pos, :]
                        + self.copy_mix_alpha * copy_logits[:, chunk_tok_idx, :]
                    )

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
        retrieval_temperature: float = 5.0,
        query_mode: str = "mean",
        query_pos: int | None = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor], Optional[list[int]], Optional[Tensor]]:
        """Build retrieval context for training via embedding cosine
        similarity with soft attention over early chunks.

        Args:
            query_mode: "mean" (KV window mean) or "marker" (query_proj at query_pos)
            query_pos: absolute position of the query marker token (required for "marker")

        Returns:
            (context, weights, chunk_starts, context_tokens) where:
              context: (B, d_model) or None -- pooled retrieval context
              weights: (B, n_chunks) soft attention weights or None
              chunk_starts: list of chunk start positions or None
              context_tokens: (B, chunk_size, d_model) or None -- prefix tokens
        """
        B, T = input_ids.shape
        C = x.shape[-1]
        if not self.enable_retrieval:
            return None, None, None, None
        if T <= self.window_size + self.chunk_size:
            return None, None, None, None

        archive_end = T - self.window_size

        # Chunk boundaries
        chunk_starts = list(range(0, archive_end, self.chunk_size))
        n_chunks = len(chunk_starts)
        if n_chunks == 0:
            return None, None, None, None

        # Build chunk embeddings: (B, n_chunks, d_model)
        # Also build per-token chunk matrix for prefix mode: (B, n_chunks, chunk_size, d_model)
        chunk_embs = []
        chunk_tokens_list = []
        for cs in chunk_starts:
            ce = min(cs + self.chunk_size, archive_end)
            chunk_hidden = x[:, cs:ce, :]  # (B, actual_len, d_model)
            chunk_embs.append(chunk_hidden.mean(dim=1))  # (B, d_model)
            # Pad to chunk_size if needed
            actual_len = ce - cs
            if actual_len < self.chunk_size:
                pad = torch.zeros(B, self.chunk_size - actual_len, C, device=x.device)
                chunk_hidden = torch.cat([chunk_hidden, pad], dim=1)
            chunk_tokens_list.append(chunk_hidden)  # (B, chunk_size, d_model)
        chunk_matrix = torch.stack(chunk_embs, dim=1)  # (B, n_chunks, d_model)
        chunk_tokens_matrix = torch.stack(chunk_tokens_list, dim=1)  # (B, n_chunks, chunk_size, d_model)

        # Query embedding
        if query_mode == "marker" and query_pos is not None and query_pos < T:
            query_emb = self.query_proj(x[:, query_pos, :])  # (B, d_model)
        else:
            query_emb = x[:, -self.window_size:, :].mean(dim=1)  # (B, d_model)

        # Cosine similarity: (B, n_chunks)
        q_norm = F.normalize(query_emb, dim=-1).unsqueeze(1)  # (B, 1, d_model)
        c_norm = F.normalize(chunk_matrix, dim=-1)  # (B, n_chunks, d_model)
        scores = (c_norm * q_norm).sum(dim=-1)  # (B, n_chunks)

        # Soft attention with temperature (sharper = more selective)
        weights = torch.softmax(scores * retrieval_temperature, dim=-1)  # (B, n_chunks)

        # Weighted combination of chunk embeddings (pooled mode)
        context = (weights.unsqueeze(-1) * chunk_matrix).sum(dim=1)  # (B, d_model)

        # Weighted combination of chunk tokens (prefix mode)
        # weights: (B, n_chunks) -> (B, n_chunks, 1, 1)
        context_tokens = (
            weights.unsqueeze(-1).unsqueeze(-1) * chunk_tokens_matrix
        ).sum(dim=1)  # (B, chunk_size, d_model)

        return context, weights, chunk_starts, context_tokens

    def forward_with_memory(
        self,
        input_ids: Tensor,
        states_r: list[Tensor],
        states_i: list[Tensor],
        position: int,
        labels: Optional[Tensor] = None,
        past_kv: Optional[list[tuple[Tensor, Tensor]]] = None,
    ) -> tuple[dict, list[Tensor], list[Tensor], list[tuple[Tensor, Tensor]]]:
        """Forward pass with explicit TRN state, retrieval, and KV cache.

        Used for streaming/incremental processing where we maintain
        eviction buffer, retrieval index, and KV cache across calls.

        Args:
            past_kv: list of (past_k, past_v) per layer, or None to initialize.

        Returns:
            (result_dict, updated_states_r, updated_states_i, updated_past_kv)
        """
        B, T = input_ids.shape
        device = input_ids.device
        n_layers = len(self.blocks)

        # Initialize KV cache if not provided
        if past_kv is None:
            n_heads = self.blocks[0].attn.n_heads
            head_dim = self.blocks[0].attn.head_dim
            past_kv = [
                (torch.zeros(B, n_heads, 0, head_dim, device=device),
                 torch.zeros(B, n_heads, 0, head_dim, device=device))
                for _ in range(n_layers)
            ]

        # Get retrieval context from current input tokens
        # NOTE: retrieval query uses batch 0 tokens (retrieval index is shared, not per-sample)
        query_tokens = input_ids[0].tolist() if B > 0 else []
        retrieval_context = self._get_retrieval_context(query_tokens, device)
        # Ensure (B, d_model) shape for TriMemoryBlock
        if retrieval_context is not None and retrieval_context.dim() == 1:
            retrieval_context = retrieval_context.unsqueeze(0).expand(B, -1)

        # Build hidden states -- cache embedding output for TRN state reuse
        pe_start = min(position, self.pe.size(0) - T)
        pe_end = min(pe_start + T, self.pe.size(0))
        param_dtype = next(self.parameters()).dtype
        emb = self.embedding(input_ids).to(param_dtype)  # (B, T, d_model)
        x = self.drop_emb(emb.clone())
        actual_pe_len = pe_end - pe_start
        if actual_pe_len > 0:
            x[:, :actual_pe_len] = x[:, :actual_pe_len] + self.pe[pe_start:pe_end]

        # Process through blocks with KV cache
        new_past_kv = []
        for layer_idx, block in enumerate(self.blocks):
            pk, pv = past_kv[layer_idx]
            x, new_pk, new_pv = block(
                x,
                retrieval_context=retrieval_context,
                past_k=pk,
                past_v=pv,
                position_offset=position,
            )
            new_past_kv.append((new_pk, new_pv))

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

        # Update TRN states via step_single -- reuse cached embedding
        for t_idx in range(T):
            x_tok = self.drop_emb(emb[:, t_idx])  # reuse cached embedding
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
            tok_id = input_ids[0, t_idx].item()
            self._eviction_buffer.append(tok_id)
            if len(self._eviction_buffer) >= self.chunk_size:
                evicted = list(self._eviction_buffer)[:self.chunk_size]
                # Remove consumed tokens from deque
                for _ in range(self.chunk_size):
                    self._eviction_buffer.popleft()
                # Compute hidden mean for evicted chunk
                with torch.no_grad():
                    evicted_tensor = torch.tensor(evicted, device=device).unsqueeze(0)
                    hidden = self.embedding(evicted_tensor).mean(dim=1).squeeze(0)
                self._process_eviction(evicted, hidden, self._global_step)
                self._global_step += 1

        return result, states_r, states_i, new_past_kv

    def collect_gate_telemetry(self) -> dict:
        """Collect gate ratios and retrieval stats from last forward pass.

        Call after a forward() to get telemetry for that batch.
        """
        gate_kv, gate_trn, gate_ret = 0.0, 0.0, 0.0
        n_blocks = len(self.blocks)
        for block in self.blocks:
            if hasattr(block, "_last_gates"):
                g = block._last_gates
                gate_kv += g[0].item()
                gate_trn += g[1].item()
                gate_ret += g[2].item()
        if n_blocks > 0:
            gate_kv /= n_blocks
            gate_trn /= n_blocks
            gate_ret /= n_blocks

        telemetry = getattr(self, "_telemetry", {})
        return {
            "router_kv_ratio": gate_kv,
            "router_trn_ratio": gate_trn,
            "router_ret_ratio": gate_ret,
            "retrieval_used": telemetry.get("retrieval_used", False),
            "archive_chunk_count": telemetry.get("n_chunks", 0),
            "state_bytes": self.state_memory_bytes,
            "retained_kv_tokens": self.window_size,
        }

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
        from trn.utils import configure_optimizer_param_groups
        return configure_optimizer_param_groups(self, weight_decay)

    def num_parameters(self, non_embedding: bool = True) -> int:
        from trn.utils import num_parameters
        return num_parameters(self, non_embedding)

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
